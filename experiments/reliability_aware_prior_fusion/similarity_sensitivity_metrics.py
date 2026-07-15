#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


METHOD_FILES = {
    "ResNet50": "outputs/predictions/dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv",
    "EfficientNetV2-S": "outputs/predictions/dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv",
    "Full LPT": "outputs/predictions/cross_task_lpt_resnet50_mobilenetv3_fold{fold}_val_predictions.csv",
    "Reliability-aware LPT fusion": "outputs/predictions/reliability_prior_fusion_subset_search_lpt_ablation_ensemble_oof_predictions.csv",
    "Strong cls ensemble": "outputs/predictions/reliability_prior_fusion_strong_strong_cls_ensemble_oof_predictions.csv",
    "Strong LPT output ensemble": "outputs/predictions/reliability_prior_fusion_strong_strong_lpt_output_ensemble_oof_predictions.csv",
    "Strong RA-LPT fusion": "outputs/predictions/reliability_prior_fusion_strong_strong_lpt_prior_fusion_oof_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute OOF metrics after excluding near-duplicate audit candidates.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit-pairs", type=Path, default=Path("outputs/paper_tables/similar_image_audit_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--risk-group", default="cross_fold_same_label")
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def load_predictions(repo_root: Path, pattern: str) -> pd.DataFrame:
    if "{fold}" in pattern:
        frames = []
        for fold in range(5):
            path = resolve(Path(pattern.format(fold=fold)), repo_root)
            fold_frame = pd.read_csv(path)
            fold_frame["fold"] = fold
            frames.append(fold_frame)
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.read_csv(resolve(Path(pattern), repo_root))
    if "label_name" not in frame.columns and "true_label" in frame.columns:
        frame = frame.rename(columns={"true_label": "label_name"})
    return frame


def compute_metrics(frame: pd.DataFrame, method: str, subset: str) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["label_name"],
        frame["pred_label"],
        labels=["none", "infection", "ischaemia", "both"],
        average="macro",
        zero_division=0,
    )
    return {
        "method": method,
        "subset": subset,
        "samples": int(len(frame)),
        "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(frame["label_name"], frame["pred_label"])),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "accuracy": float(accuracy_score(frame["label_name"], frame["pred_label"])),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve(args.output_dir, repo_root)
    pairs = pd.read_csv(resolve(args.audit_pairs, repo_root))
    risk_pairs = pairs[pairs["risk_group"] == args.risk_group]
    excluded_images = sorted(set(risk_pairs["image_a"]).union(set(risk_pairs["image_b"])))

    rows: list[dict[str, object]] = []
    for method, pattern in METHOD_FILES.items():
        frame = load_predictions(repo_root, pattern)
        rows.append(compute_metrics(frame, method, "all_labeled_oof"))
        filtered = frame[~frame["image"].isin(excluded_images)].copy()
        rows.append(compute_metrics(filtered, method, "excluding_cross_fold_same_label_near_duplicates"))

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "similarity_sensitivity_metrics.csv", index=False)
    summary = {
        "risk_group": args.risk_group,
        "risk_pair_count": int(len(risk_pairs)),
        "excluded_unique_images": int(len(excluded_images)),
        "remaining_images": int(result[result["subset"] == "excluding_cross_fold_same_label_near_duplicates"]["samples"].iloc[0]),
        "metrics_csv": "outputs/paper_tables/similarity_sensitivity_metrics.csv",
    }
    (output_dir / "similarity_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
