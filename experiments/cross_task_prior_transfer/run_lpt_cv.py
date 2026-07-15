#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


METRIC_NAMES = (
    "macro_f1",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "accuracy",
    "infection_attr_f1",
    "ischaemia_attr_f1",
)
LEGACY_REPO_ROOTS: tuple[Path, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-task lesion-prior transfer CV sequentially.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def remap_legacy_repo_path(path: Path, repo_root: Path) -> Path:
    for legacy_root in LEGACY_REPO_ROOTS:
        try:
            relative_path = path.relative_to(legacy_root)
        except ValueError:
            continue
        return repo_root / relative_path
    return path


def resolve_path(path_str: str | Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        return remap_legacy_repo_path(path, repo_root)
    return repo_root / path


def load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    with resolve_path(config_path, repo_root).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_paths(config: dict[str, Any], fold: int, repo_root: Path) -> dict[str, Path]:
    experiment_name = str(config["experiment_name"])
    checkpoint_dir = resolve_path(config["outputs"]["checkpoint_dir"], repo_root)
    log_dir = resolve_path(config["outputs"]["log_dir"], repo_root)
    prediction_dir = resolve_path(config["outputs"]["prediction_dir"], repo_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    return {
        "best_checkpoint": checkpoint_dir / str(config["naming"]["best_checkpoint"]).format(fold=fold),
        "last_checkpoint": checkpoint_dir / str(config["naming"]["last_checkpoint"]).format(fold=fold),
        "metrics": log_dir / f"{experiment_name}_fold{fold}_metrics.json",
        "cv_summary_json": log_dir / f"{experiment_name}_cv_summary.json",
        "cv_summary_csv": log_dir / f"{experiment_name}_cv_summary.csv",
        "best_overall_json": log_dir / f"{experiment_name}_best_overall.json",
        "best_overall_checkpoint": checkpoint_dir / f"{experiment_name}_best_overall.pt",
    }


def expected_counts_by_fold(config: dict[str, Any], repo_root: Path) -> dict[int, tuple[int, int]]:
    manifest_path = resolve_path(config["data"]["split_manifest"], repo_root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fold_counts: dict[int, int] = {}
    for row in rows:
        fold = int(row["fold"])
        fold_counts[fold] = fold_counts.get(fold, 0) + 1
    total = sum(fold_counts.values())
    return {fold: (total - count, count) for fold, count in fold_counts.items()}


def metrics_complete(metrics_path: Path, expected_train: int, expected_val: int, primary: str) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError:
        return False
    if int(payload.get("train_samples", -1)) != expected_train:
        return False
    if int(payload.get("val_samples", -1)) != expected_val:
        return False
    return primary in payload.get("best_metrics", {})


def run_fold(
    args: argparse.Namespace,
    train_script: Path,
    config_path: Path,
    fold: int,
    metrics_path: Path,
    expected_train: int,
    expected_val: int,
    primary: str,
) -> None:
    if args.skip_existing and metrics_complete(metrics_path, expected_train, expected_val, primary):
        print(f"[cv] skip fold {fold}: complete metrics already exist at {metrics_path}", flush=True)
        return
    command = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--fold",
        str(fold),
        "--device",
        args.device,
        "--amp",
        args.amp,
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_workers is not None:
        command.extend(["--num-workers", str(args.num_workers)])
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.disable_pretrained:
        command.append("--disable-pretrained")
    print(f"[cv] starting fold {fold}", flush=True)
    print("[cv] command:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load_fold_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cv_summary(results: list[dict[str, Any]], config: dict[str, Any], summary_json: Path, summary_csv: Path) -> None:
    aggregate = {
        metric: {
            "mean": float(np.mean([result["best_metrics"][metric] for result in results])),
            "std": float(np.std([result["best_metrics"][metric] for result in results])),
        }
        for metric in METRIC_NAMES
    }
    payload = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "fold_results": results,
        "aggregate": aggregate,
    }
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "train_samples", "val_samples", "best_epoch", *METRIC_NAMES, "best_checkpoint"])
        for result in results:
            writer.writerow(
                [
                    result["fold"],
                    result["train_samples"],
                    result["val_samples"],
                    result["best_epoch"],
                    *(f"{result['best_metrics'][metric]:.6f}" for metric in METRIC_NAMES),
                    result["best_checkpoint"],
                ]
            )
        writer.writerow(["mean", "", "", "", *(f"{aggregate[metric]['mean']:.6f}" for metric in METRIC_NAMES), ""])
        writer.writerow(["std", "", "", "", *(f"{aggregate[metric]['std']:.6f}" for metric in METRIC_NAMES), ""])


def save_best_overall(results: list[dict[str, Any]], best_json: Path, best_checkpoint: Path) -> dict[str, Any]:
    best_result = max(results, key=lambda item: item["best_metrics"]["macro_f1"])
    shutil.copy2(best_result["best_checkpoint"], best_checkpoint)
    payload = {
        "best_fold": best_result["fold"],
        "best_metrics": best_result["best_metrics"],
        "source_checkpoint": best_result["best_checkpoint"],
        "saved_checkpoint": str(best_checkpoint),
    }
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    config_path = resolve_path(args.config, repo_root)
    config = load_config(config_path, repo_root)
    train_script = repo_root / "experiments/cross_task_prior_transfer/train_lpt.py"
    expected_counts = expected_counts_by_fold(config, repo_root)
    primary = str(config["metrics"]["primary"])

    for fold in args.folds:
        if fold not in expected_counts:
            raise ValueError(f"Fold {fold} is missing from manifest.")
        paths = build_paths(config, fold, repo_root)
        expected_train, expected_val = expected_counts[fold]
        run_fold(args, train_script, config_path, fold, paths["metrics"], expected_train, expected_val, primary)

    results = []
    for fold in args.folds:
        metrics_path = build_paths(config, fold, repo_root)["metrics"]
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for fold {fold}: {metrics_path}")
        results.append(load_fold_metrics(metrics_path))

    shared_paths = build_paths(config, args.folds[0], repo_root)
    write_cv_summary(results, config, shared_paths["cv_summary_json"], shared_paths["cv_summary_csv"])
    best_payload = save_best_overall(results, shared_paths["best_overall_json"], shared_paths["best_overall_checkpoint"])
    print(
        f"[cv] completed {config['experiment_name']} folds {args.folds} | "
        f"best_fold={best_payload['best_fold']} "
        f"macro_f1={best_payload['best_metrics']['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
