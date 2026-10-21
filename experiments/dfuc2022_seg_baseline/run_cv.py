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


METRIC_NAMES = ("dice", "iou", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DFUC2022 5-fold cross-validation sequentially.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/dfuc2022_seg_deeplabv3p_cv5_v1.yaml",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold ids to run in order.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        choices=("auto", "on", "off"),
        default="auto",
        help="Automatic mixed precision. `auto` enables AMP only on CUDA.",
    )
    parser.add_argument("--disable-pretrained", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a fold when its metrics file already exists.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path_str: str | Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def build_paths(config: dict[str, Any], fold: int, repo_root: Path) -> dict[str, Path]:
    experiment_name = str(config["experiment_name"])
    checkpoint_dir = resolve_path(config["outputs"]["checkpoint_dir"], repo_root)
    log_dir = resolve_path(config["outputs"]["log_dir"], repo_root)
    figure_dir = resolve_path(config["outputs"]["figure_dir"], repo_root)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    best_name = str(config["naming"]["best_checkpoint"]).format(fold=fold)
    last_name = str(config["naming"]["last_checkpoint"]).format(fold=fold)

    return {
        "best_checkpoint": checkpoint_dir / best_name,
        "last_checkpoint": checkpoint_dir / last_name,
        "metrics": log_dir / f"{experiment_name}_fold{fold}_metrics.json",
        "cv_summary_json": log_dir / f"{experiment_name}_cv_summary.json",
        "cv_summary_csv": log_dir / f"{experiment_name}_cv_summary.csv",
        "best_overall_json": log_dir / f"{experiment_name}_best_overall.json",
        "best_overall_checkpoint": checkpoint_dir / f"{experiment_name}_best_overall.pt",
        "best_overall_figure": figure_dir / f"{experiment_name}_best_overall_examples.png",
    }


def run_fold(args: argparse.Namespace, train_script: Path, config_path: Path, fold: int, metrics_path: Path) -> None:
    if args.skip_existing and metrics_path.exists():
        print(f"[cv] skip fold {fold}: metrics already exist at {metrics_path}", flush=True)
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
    summary_payload = {
        "experiment_name": config["experiment_name"],
        "fold_results": results,
        "aggregate": {
            metric: {
                "mean": float(np.mean([result["best_metrics"][metric] for result in results])),
                "std": float(np.std([result["best_metrics"][metric] for result in results])),
            }
            for metric in METRIC_NAMES
        },
    }

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

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
        writer.writerow(
            [
                "mean",
                "",
                "",
                "",
                *(f"{summary_payload['aggregate'][metric]['mean']:.6f}" for metric in METRIC_NAMES),
                "",
            ]
        )
        writer.writerow(
            [
                "std",
                "",
                "",
                "",
                *(f"{summary_payload['aggregate'][metric]['std']:.6f}" for metric in METRIC_NAMES),
                "",
            ]
        )


def save_best_overall(
    results: list[dict[str, Any]],
    best_overall_json: Path,
    best_overall_checkpoint: Path,
    best_overall_figure: Path,
) -> dict[str, Any]:
    best_result = max(results, key=lambda item: item["best_metrics"]["dice"])

    shutil.copy2(best_result["best_checkpoint"], best_overall_checkpoint)

    best_figure = Path(best_result.get("best_figure", ""))
    if best_figure.exists():
        shutil.copy2(best_figure, best_overall_figure)

    payload = {
        "best_fold": best_result["fold"],
        "best_metrics": best_result["best_metrics"],
        "source_checkpoint": best_result["best_checkpoint"],
        "saved_checkpoint": str(best_overall_checkpoint),
        "source_figure": best_result.get("best_figure"),
        "saved_figure": str(best_overall_figure) if best_figure.exists() else None,
    }
    with best_overall_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repo_root = resolve_repo_root()
    train_script = repo_root / "experiments/dfuc2022_seg_baseline/train.py"

    for fold in args.folds:
        paths = build_paths(config, fold, repo_root)
        run_fold(args, train_script, args.config, fold, paths["metrics"])

    results = []
    for fold in args.folds:
        metrics_path = build_paths(config, fold, repo_root)["metrics"]
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics for fold {fold}: {metrics_path}")
        results.append(load_fold_metrics(metrics_path))

    shared_paths = build_paths(config, args.folds[0], repo_root)
    write_cv_summary(
        results=results,
        config=config,
        summary_json=shared_paths["cv_summary_json"],
        summary_csv=shared_paths["cv_summary_csv"],
    )
    best_payload = save_best_overall(
        results=results,
        best_overall_json=shared_paths["best_overall_json"],
        best_overall_checkpoint=shared_paths["best_overall_checkpoint"],
        best_overall_figure=shared_paths["best_overall_figure"],
    )

    print(
        f"[cv] completed folds {args.folds} | "
        f"best_fold={best_payload['best_fold']} "
        f"dice={best_payload['best_metrics']['dice']:.4f} "
        f"saved={best_payload['saved_checkpoint']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
