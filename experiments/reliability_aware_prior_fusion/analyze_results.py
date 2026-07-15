#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
METRIC_NAMES = ("macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy")


@dataclass(frozen=True)
class SummarySpec:
    method: str
    file_name: str
    group: str


SUMMARY_SPECS = (
    SummarySpec("ResNet50", "dfuc2021_cls_resnet50_cv5_v1_cv_summary.csv", "classification_baseline"),
    SummarySpec("EfficientNet-B0", "dfuc2021_cls_efficientnetb0_cv5_v1_cv_summary.csv", "classification_baseline"),
    SummarySpec("Full LPT", "cross_task_lpt_resnet50_mobilenetv3_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("LPT no alignment", "cross_task_lpt_no_alignment_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("LPT no prior modulation", "cross_task_lpt_no_prior_mod_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("LPT no ROI", "cross_task_lpt_no_roi_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("LPT stats only", "cross_task_lpt_stats_only_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("LPT random prior", "cross_task_lpt_random_prior_cv5_v1_cv_summary.csv", "spatial_prior_transfer"),
    SummarySpec("Pure cls ensemble", "reliability_prior_fusion_nobias_cls_ensemble_cv_summary.csv", "fusion"),
    SummarySpec("Cls + prior residual", "reliability_prior_fusion_nobias_cls_prior_residual_cv_summary.csv", "fusion"),
    SummarySpec("LPT output ensemble", "reliability_prior_fusion_subset_search_lpt_ablation_output_ensemble_cv_summary.csv", "fusion"),
    SummarySpec("LPT ablations + prior stats", "reliability_prior_fusion_subset_search_lpt_ablation_ensemble_cv_summary.csv", "fusion"),
)


PREDICTION_SPECS = {
    "ResNet50": "dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv",
    "Pure cls ensemble": "reliability_prior_fusion_nobias_cls_ensemble_oof_predictions.csv",
    "LPT output ensemble": "reliability_prior_fusion_subset_search_lpt_ablation_output_ensemble_oof_predictions.csv",
    "LPT ablations + prior stats": "reliability_prior_fusion_subset_search_lpt_ablation_ensemble_oof_predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready tables for reliability-aware prior fusion.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument(
        "--prior-meta",
        type=Path,
        default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def read_summary(path: Path, method: str, group: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    mean_row = next(row for row in rows if row["fold"] == "mean")
    std_row = next(row for row in rows if row["fold"] == "std")
    result: dict[str, Any] = {"method": method, "group": group}
    for metric in METRIC_NAMES:
        result[metric] = float(mean_row[metric])
        result[f"{metric}_std"] = float(std_row[metric])
    return result


def build_summary_tables(log_dir: Path, output_dir: Path) -> pd.DataFrame:
    rows = [read_summary(log_dir / spec.file_name, spec.method, spec.group) for spec in SUMMARY_SPECS]
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "dfuc2021_all_method_summary.csv", index=False)
    frame[frame["group"] == "classification_baseline"].to_csv(output_dir / "table_classification_baselines.csv", index=False)
    frame[frame["group"] == "spatial_prior_transfer"].to_csv(output_dir / "table_lpt_spatial_ablations.csv", index=False)
    frame[frame["group"] == "fusion"].to_csv(output_dir / "table_reliability_fusion.csv", index=False)
    return frame


def label_series_to_id(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.map(LABEL_TO_ID).astype(int).to_numpy()


def read_prediction_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        y_true = label_series_to_id(frame["label"])
    elif "true_label" in frame.columns:
        y_true = label_series_to_id(frame["true_label"])
    else:
        raise ValueError(f"Missing true label column in {path}")
    if "pred_label" not in frame.columns:
        raise ValueError(f"Missing pred_label column in {path}")
    y_pred = label_series_to_id(frame["pred_label"])
    result = pd.DataFrame(
        {
            "image": frame["image"],
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "fold" in frame.columns:
        result["fold"] = frame["fold"].astype(int)
    return result


def load_predictions(prediction_dir: Path, manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    manifest_fold = manifest[["image", "fold"]].copy()
    for method, pattern in PREDICTION_SPECS.items():
        if "{fold}" in pattern:
            parts = []
            for fold in range(5):
                parts.append(read_prediction_file(prediction_dir / pattern.format(fold=fold)))
            frame = pd.concat(parts, ignore_index=True)
        else:
            frame = read_prediction_file(prediction_dir / pattern)
        if "fold" not in frame.columns:
            frame = frame.merge(manifest_fold, on="image", how="left", validate="one_to_one")
        if frame["fold"].isnull().any():
            raise ValueError(f"Missing fold values for {method}")
        if frame["image"].duplicated().any():
            raise ValueError(f"Duplicate image predictions for {method}")
        outputs[method] = frame.sort_values("image").reset_index(drop=True)
    return outputs


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    values = fast_metrics(y_true, y_pred)
    return {
        "macro_f1": float(values["macro_f1"]),
        "balanced_accuracy": float(values["balanced_accuracy"]),
        "macro_precision": float(values["macro_precision"]),
        "macro_recall": float(values["macro_recall"]),
        "accuracy": float(values["accuracy"]),
    }


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int, copy=False)
    y_pred = y_pred.astype(int, copy=False)
    encoded = len(CLASS_NAMES) * y_true + y_pred
    cm = np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).astype(np.float64)
    tp = np.diag(cm)
    pred_count = cm.sum(axis=0)
    true_count = cm.sum(axis=1)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    total = max(1.0, float(cm.sum()))
    return {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "accuracy": float(tp.sum() / total),
    }


def per_class_metrics(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return [
        {
            "method": method,
            "class": class_name,
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(CLASS_NAMES)
    ]


def build_prediction_metrics(predictions: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, dict[str, Any]]:
    summary_rows = []
    per_class_rows = []
    confusion_payload: dict[str, Any] = {}
    metrics_by_method: dict[str, dict[str, Any]] = {}
    for method, frame in predictions.items():
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        metrics = compute_metrics(y_true, y_pred)
        metrics_by_method[method] = metrics
        summary_rows.append({"method": method, **metrics})
        per_class_rows.extend(per_class_metrics(method, y_true, y_pred))
        confusion_payload[method] = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ).astype(int).tolist()
    pd.DataFrame(summary_rows).to_csv(output_dir / "oof_prediction_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "oof_per_class_metrics.csv", index=False)
    with (output_dir / "oof_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump({"class_names": CLASS_NAMES, "confusion_matrices": confusion_payload}, handle, indent=2)
    return metrics_by_method


def paired_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    n_iters: int,
    seed: int,
) -> dict[str, Any]:
    merged = baseline[["image", "y_true", "y_pred"]].rename(columns={"y_pred": "baseline_pred"}).merge(
        candidate[["image", "y_true", "y_pred"]].rename(columns={"y_true": "candidate_true", "y_pred": "candidate_pred"}),
        on="image",
        how="inner",
        validate="one_to_one",
    )
    if not np.array_equal(merged["y_true"].to_numpy(), merged["candidate_true"].to_numpy()):
        raise ValueError(f"Label mismatch between {baseline_name} and {candidate_name}")
    y_true = merged["y_true"].to_numpy(dtype=int)
    baseline_pred = merged["baseline_pred"].to_numpy(dtype=int)
    candidate_pred = merged["candidate_pred"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    n_samples = len(merged)
    observed_candidate = fast_metrics(y_true, candidate_pred)
    observed_baseline = fast_metrics(y_true, baseline_pred)
    observed = {metric: observed_candidate[metric] - observed_baseline[metric] for metric in METRIC_NAMES}
    diff_by_metric = {metric: np.empty(n_iters, dtype=np.float64) for metric in ("macro_f1", "balanced_accuracy", "accuracy")}
    for idx in range(n_iters):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        candidate_metrics = fast_metrics(y_true[sample_idx], candidate_pred[sample_idx])
        baseline_metrics = fast_metrics(y_true[sample_idx], baseline_pred[sample_idx])
        for metric in diff_by_metric:
            diff_by_metric[metric][idx] = candidate_metrics[metric] - baseline_metrics[metric]
    rows = []
    for metric, diffs in diff_by_metric.items():
        one_sided_p = float((np.sum(diffs <= 0.0) + 1.0) / (n_iters + 1.0))
        two_sided_p = min(1.0, 2.0 * min(one_sided_p, float((np.sum(diffs >= 0.0) + 1.0) / (n_iters + 1.0))))
        rows.append(
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "metric": metric,
                "observed_delta": float(observed[metric]),
                "ci95_low": float(np.quantile(diffs, 0.025)),
                "ci95_high": float(np.quantile(diffs, 0.975)),
                "one_sided_p_delta_le_0": one_sided_p,
                "two_sided_p": two_sided_p,
                "bootstrap_iters": n_iters,
            }
        )
    return {"rows": rows}


def build_bootstrap_table(predictions: dict[str, pd.DataFrame], output_dir: Path, n_iters: int, seed: int) -> None:
    comparisons = [
        ("ResNet50", "Pure cls ensemble"),
        ("ResNet50", "LPT ablations + prior stats"),
        ("Pure cls ensemble", "LPT ablations + prior stats"),
        ("LPT output ensemble", "LPT ablations + prior stats"),
    ]
    rows: list[dict[str, Any]] = []
    for offset, (baseline_name, candidate_name) in enumerate(comparisons):
        result = paired_bootstrap(
            predictions[baseline_name],
            predictions[candidate_name],
            baseline_name,
            candidate_name,
            n_iters=n_iters,
            seed=seed + offset,
        )
        rows.extend(result["rows"])
    pd.DataFrame(rows).to_csv(output_dir / "paired_bootstrap_significance.csv", index=False)


def add_prior_groups(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    quality = frame["prior_confidence"].astype(float) * (1.0 - frame["prior_fallback"].astype(float))
    frame["prior_quality"] = quality
    frame["fallback_group"] = np.where(frame["prior_fallback"].astype(int) == 1, "fallback", "non_fallback")
    nonfallback = frame["prior_fallback"].astype(int) == 0
    median_quality = float(frame.loc[nonfallback, "prior_quality"].median()) if nonfallback.any() else 0.0
    frame["quality_group"] = "fallback"
    frame.loc[nonfallback & (frame["prior_quality"] < median_quality), "quality_group"] = "nonfallback_low_quality"
    frame.loc[nonfallback & (frame["prior_quality"] >= median_quality), "quality_group"] = "nonfallback_high_quality"
    area = frame["prior_area_ratio"].astype(float)
    frame["area_group"] = pd.cut(
        area,
        bins=[-1e-9, 0.02, 0.15, 0.45, 1.01],
        labels=["tiny", "small", "medium", "large"],
    ).astype(str)
    return frame


def group_metric_rows(
    predictions: dict[str, pd.DataFrame],
    prior_meta: pd.DataFrame,
    group_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = prior_meta[["image", group_column]].copy()
    for method, pred_frame in predictions.items():
        merged = pred_frame.merge(groups, on="image", how="left", validate="one_to_one")
        for group_value, group_frame in merged.groupby(group_column):
            y_true = group_frame["y_true"].to_numpy(dtype=int)
            y_pred = group_frame["y_pred"].to_numpy(dtype=int)
            metrics = compute_metrics(y_true, y_pred)
            rows.append(
                {
                    "method": method,
                    "group_type": group_column,
                    "group": str(group_value),
                    "samples": int(len(group_frame)),
                    **metrics,
                }
            )
    return rows


def build_prior_strata_table(predictions: dict[str, pd.DataFrame], prior_meta_path: Path, output_dir: Path) -> None:
    prior_meta = add_prior_groups(pd.read_csv(prior_meta_path))
    rows: list[dict[str, Any]] = []
    for group_column in ("fallback_group", "quality_group", "area_group"):
        rows.extend(group_metric_rows(predictions, prior_meta, group_column))
    pd.DataFrame(rows).to_csv(output_dir / "prior_reliability_strata_metrics.csv", index=False)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    main_methods = [
        "ResNet50",
        "EfficientNet-B0",
        "Full LPT",
        "LPT no alignment",
        "LPT no prior modulation",
        "LPT stats only",
        "Pure cls ensemble",
        "LPT output ensemble",
        "LPT ablations + prior stats",
    ]
    table = summary[summary["method"].isin(main_methods)].copy()
    table["order"] = table["method"].map({method: idx for idx, method in enumerate(main_methods)})
    table = table.sort_values("order")
    table = table[["method", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy"]]

    best = summary.loc[summary["method"] == "LPT ablations + prior stats"].iloc[0]
    resnet = summary.loc[summary["method"] == "ResNet50"].iloc[0]
    cls = summary.loc[summary["method"] == "Pure cls ensemble"].iloc[0]
    output_only = summary.loc[summary["method"] == "LPT output ensemble"].iloc[0]

    text = f"""# DFUC2021 Paper Result Summary

Generated by `experiments/reliability_aware_prior_fusion/analyze_results.py`.

## Main Table

{markdown_table(table, list(table.columns))}

## Main Claim

Best method: `LPT ablations + prior stats`.

- Macro-F1: `{best['macro_f1']:.6f}`
- Balanced Accuracy: `{best['balanced_accuracy']:.6f}`
- Accuracy: `{best['accuracy']:.6f}`
- Delta vs ResNet50 Macro-F1: `{best['macro_f1'] - resnet['macro_f1']:+.6f}`
- Delta vs pure classification ensemble Macro-F1: `{best['macro_f1'] - cls['macro_f1']:+.6f}`
- Delta vs LPT output ensemble without explicit prior stats Macro-F1: `{best['macro_f1'] - output_only['macro_f1']:+.6f}`

## Interpretation

The spatial LPT variants underperform the pure ResNet50 baseline when used directly. The stronger result comes from treating the cross-dataset lesion prior as reliability-aware residual evidence rather than as a hard spatial attention or ROI constraint.
"""
    (output_dir / "DFUC2021_paper_result_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    log_dir = resolve_path(args.log_dir, repo_root)
    prediction_dir = resolve_path(args.prediction_dir, repo_root)
    manifest_path = resolve_path(args.manifest, repo_root)
    prior_meta_path = resolve_path(args.prior_meta, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_summary_tables(log_dir, output_dir)
    manifest = pd.read_csv(manifest_path)[["image", "fold"]].copy()
    predictions = load_predictions(prediction_dir, manifest)
    build_prediction_metrics(predictions, output_dir)
    build_bootstrap_table(predictions, output_dir, n_iters=int(args.bootstrap_iters), seed=int(args.seed))
    build_prior_strata_table(predictions, prior_meta_path, output_dir)
    write_markdown_summary(summary, output_dir)
    print(f"Saved paper tables to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
