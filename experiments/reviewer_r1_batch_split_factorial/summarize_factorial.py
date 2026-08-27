#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


EXPERIMENT_VERSION = "reviewer_r1_batch_split_factorial_v1"
CLASS_NAMES = ["none", "infection", "ischaemia", "both"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed Reviewer 1 factorial cells.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / EXPERIMENT_VERSION,
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_cell(config_path: Path, root: Path, require_complete: bool) -> dict[str, Any] | None:
    config = load_yaml(config_path)
    design = config["reviewer_ablation"]
    name = config["experiment_name"]
    log_dir = resolve(config["outputs"]["log_dir"], root)
    prediction_dir = resolve(config["outputs"]["prediction_dir"], root)

    metrics_paths = [log_dir / f"{name}_fold{fold}_metrics.json" for fold in range(5)]
    prediction_paths = [
        prediction_dir / config["naming"]["predictions"].format(fold=fold) for fold in range(5)
    ]
    missing = [str(path) for path in [*metrics_paths, *prediction_paths] if not path.exists()]
    if missing:
        if require_complete:
            raise FileNotFoundError(f"Incomplete cell {name}; missing: {missing}")
        return None

    metrics = [load_json(path) for path in metrics_paths]
    fold_macro_f1 = np.array([item["best_metrics"]["macro_f1"] for item in metrics], dtype=float)
    predictions = pd.concat(
        [pd.read_csv(path).assign(fold=fold) for fold, path in enumerate(prediction_paths)],
        ignore_index=True,
    )
    if predictions["image"].duplicated().any():
        duplicates = predictions.loc[predictions["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Cell {name} contains duplicate OOF images: {duplicates}")

    y_true = predictions["true_label"].astype(str)
    y_pred = predictions["pred_label"].astype(str)
    return {
        "experiment_name": name,
        "model": design["model"],
        "split": design["split"],
        "batch_size": int(design["batch_size"]),
        "n_oof": len(predictions),
        "fold_macro_f1_mean": float(fold_macro_f1.mean()),
        "fold_macro_f1_sd": float(fold_macro_f1.std(ddof=0)),
        "pooled_oof_macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASS_NAMES, average="macro", zero_division=0)
        ),
        "pooled_oof_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pooled_oof_accuracy": float(accuracy_score(y_true, y_pred)),
    }


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = ["fold_macro_f1_mean", "pooled_oof_macro_f1"]

    for model in sorted(summary["model"].unique()):
        for split in sorted(summary["split"].unique()):
            subset = summary[(summary["model"] == model) & (summary["split"] == split)].set_index("batch_size")
            if {16, 128}.issubset(subset.index):
                for metric in metric_names:
                    rows.append(
                        {
                            "contrast": "batch_16_minus_128",
                            "model": model,
                            "split": split,
                            "batch_size": "",
                            "metric": metric,
                            "difference": float(subset.loc[16, metric] - subset.loc[128, metric]),
                        }
                    )

    for model in sorted(summary["model"].unique()):
        for batch_size in sorted(summary["batch_size"].unique()):
            subset = summary[(summary["model"] == model) & (summary["batch_size"] == batch_size)].set_index("split")
            if {"standard", "grouped"}.issubset(subset.index):
                for metric in metric_names:
                    rows.append(
                        {
                            "contrast": "grouped_minus_standard",
                            "model": model,
                            "split": "",
                            "batch_size": batch_size,
                            "metric": metric,
                            "difference": float(subset.loc["grouped", metric] - subset.loc["standard", metric]),
                        }
                    )

    for split in sorted(summary["split"].unique()):
        for batch_size in sorted(summary["batch_size"].unique()):
            subset = summary[(summary["split"] == split) & (summary["batch_size"] == batch_size)].set_index("model")
            if {"full_lpt", "resnet50"}.issubset(subset.index):
                for metric in metric_names:
                    rows.append(
                        {
                            "contrast": "full_lpt_minus_resnet50",
                            "model": "",
                            "split": split,
                            "batch_size": batch_size,
                            "metric": metric,
                            "difference": float(subset.loc["full_lpt", metric] - subset.loc["resnet50", metric]),
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_root = resolve(args.output_root, root)
    config_dir = output_root / "resolved_configs"
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No resolved configs found in {config_dir}")

    rows = []
    for config_path in config_paths:
        row = summarize_cell(config_path, root, args.require_complete)
        if row is not None:
            rows.append(row)
    if not rows:
        print("No complete cells are available yet.")
        return

    summary = pd.DataFrame(rows).sort_values(["batch_size", "model", "split"]).reset_index(drop=True)
    contrasts = build_contrasts(summary)
    summary.to_csv(output_root / "factorial_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    contrasts.to_csv(output_root / "factorial_contrasts.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    print(summary.to_string(index=False))
    if not contrasts.empty:
        print("\nContrasts")
        print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
