#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}
METRIC_NAMES = ("macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy")


@dataclass(frozen=True)
class PredictionSpec:
    method: str
    pattern: str


BASE_SPECS = (
    PredictionSpec("ResNet50 grouped", "dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv"),
    PredictionSpec("EfficientNet-B0 grouped", "dfuc2021_cls_efficientnetb0_fold{fold}_val_predictions.csv"),
    PredictionSpec("ConvNeXt-Tiny grouped", "dfuc2021_cls_convnext_tiny_fold{fold}_val_predictions.csv"),
    PredictionSpec("EfficientNetV2-S grouped", "dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv"),
    PredictionSpec("Full LPT grouped", "cross_task_lpt_resnet50_mobilenetv3_fold{fold}_val_predictions.csv"),
    PredictionSpec("LPT no alignment grouped", "cross_task_lpt_no_alignment_fold{fold}_val_predictions.csv"),
    PredictionSpec("LPT no prior modulation grouped", "cross_task_lpt_no_prior_mod_fold{fold}_val_predictions.csv"),
    PredictionSpec("LPT no ROI grouped", "cross_task_lpt_no_roi_fold{fold}_val_predictions.csv"),
    PredictionSpec("LPT stats only grouped", "cross_task_lpt_stats_only_fold{fold}_val_predictions.csv"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise similarity-grouped RA-LPT and strong-ensemble retraining results.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/grouped_retraining/predictions"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/grouped_retraining/tables"))
    parser.add_argument("--manifest", type=Path, default=Path("data_processed/dfuc2021_cls_v1/similarity_grouped_splits/dfuc2021_cls_cv5_similarity_grouped_manifest.csv"))
    parser.add_argument("--output-prefix", default="similarity_grouped_reliability_prior_fusion")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def labels_to_id(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map(LABEL_TO_ID).astype(int).to_numpy()


def read_prediction(path: Path, fold: int | None = None, manifest_fold: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        y_true = labels_to_id(frame["label"])
        label_name = frame.get("label_name", pd.Series([ID_TO_LABEL[int(v)] for v in y_true]))
    elif "true_label" in frame.columns:
        y_true = labels_to_id(frame["true_label"])
        label_name = pd.Series([ID_TO_LABEL[int(v)] for v in y_true])
    else:
        raise ValueError(f"Missing label/true_label in {path}")
    if "pred_label" not in frame.columns:
        raise ValueError(f"Missing pred_label in {path}")
    y_pred = labels_to_id(frame["pred_label"])
    output = pd.DataFrame(
        {
            "image": frame["image"].astype(str),
            "label": y_true,
            "label_name": label_name.astype(str),
            "pred_label": [ID_TO_LABEL[int(v)] for v in y_pred],
            "pred_id": y_pred,
        }
    )
    if "fold" in frame.columns:
        output["fold"] = frame["fold"].astype(int)
    elif fold is not None:
        output["fold"] = int(fold)
    elif manifest_fold is not None:
        output = output.merge(manifest_fold, on="image", how="left", validate="one_to_one")
    else:
        raise ValueError(f"No fold information for {path}")
    for class_name in CLASS_NAMES:
        column = f"prob_{class_name}"
        if column in frame.columns:
            output[column] = frame[column].astype(float)
    return output


def load_spec(spec: PredictionSpec, prediction_dir: Path, manifest: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    missing: list[str] = []
    if "{fold}" in spec.pattern:
        parts = []
        for fold in range(5):
            path = prediction_dir / spec.pattern.format(fold=fold)
            if not path.exists():
                missing.append(str(path))
                continue
            parts.append(read_prediction(path, fold=fold))
        if missing:
            return None, missing
        frame = pd.concat(parts, ignore_index=True)
    else:
        path = prediction_dir / spec.pattern
        if not path.exists():
            return None, [str(path)]
        frame = read_prediction(path, manifest_fold=manifest[["image", "fold"]].copy())
    if frame["image"].duplicated().any():
        duplicates = frame.loc[frame["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Duplicate images in {spec.method}: {duplicates}")
    return frame.sort_values("image").reset_index(drop=True), []


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    encoded = len(CLASS_NAMES) * y_true.astype(int) + y_pred.astype(int)
    matrix = np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(len(CLASS_NAMES), len(CLASS_NAMES)).astype(float)
    tp = np.diag(matrix)
    pred_count = matrix.sum(axis=0)
    true_count = matrix.sum(axis=1)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    total = max(1.0, float(matrix.sum()))
    return {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "accuracy": float(tp.sum() / total),
    }


def metric_tables(predictions: dict[str, pd.DataFrame], table_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for method, frame in predictions.items():
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = frame["pred_id"].to_numpy(dtype=int)
        rows.append({"method": method, "samples": int(len(frame)), **fast_metrics(y_true, y_pred)})
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES))), average=None, zero_division=0
        )
        for idx, class_name in enumerate(CLASS_NAMES):
            per_class_rows.append(
                {
                    "method": method,
                    "class": class_name,
                    "precision": float(precision[idx]),
                    "recall": float(recall[idx]),
                    "f1": float(f1[idx]),
                    "support": int(support[idx]),
                }
            )
        confusion_payload[method] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).tolist()
    metrics = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics.to_csv(table_dir / "similarity_grouped_oof_prediction_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(table_dir / "similarity_grouped_oof_per_class_metrics.csv", index=False)
    with (table_dir / "similarity_grouped_oof_confusion_matrices.json").open("w", encoding="utf-8") as handle:
        json.dump({"class_names": CLASS_NAMES, "confusion_matrices": confusion_payload}, handle, indent=2)
    return metrics


def paired_bootstrap(
    predictions: dict[str, pd.DataFrame],
    baseline_name: str,
    candidate_name: str,
    n_iters: int,
    seed: int,
) -> list[dict[str, Any]]:
    baseline = predictions[baseline_name][["image", "label", "pred_id"]].rename(columns={"pred_id": "baseline_pred"})
    candidate = predictions[candidate_name][["image", "label", "pred_id"]].rename(columns={"label": "candidate_label", "pred_id": "candidate_pred"})
    merged = baseline.merge(candidate, on="image", how="inner", validate="one_to_one")
    if not np.array_equal(merged["label"].to_numpy(), merged["candidate_label"].to_numpy()):
        raise ValueError(f"Label mismatch: {baseline_name} vs {candidate_name}")
    y_true = merged["label"].to_numpy(dtype=int)
    baseline_pred = merged["baseline_pred"].to_numpy(dtype=int)
    candidate_pred = merged["candidate_pred"].to_numpy(dtype=int)
    observed_base = fast_metrics(y_true, baseline_pred)
    observed_candidate = fast_metrics(y_true, candidate_pred)
    observed = {metric: observed_candidate[metric] - observed_base[metric] for metric in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    n_samples = len(merged)
    rows = []
    for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
        diffs = np.empty(n_iters, dtype=float)
        for idx in range(n_iters):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            diffs[idx] = fast_metrics(y_true[sample_idx], candidate_pred[sample_idx])[metric] - fast_metrics(y_true[sample_idx], baseline_pred[sample_idx])[metric]
        p_le_zero = float((np.sum(diffs <= 0.0) + 1.0) / (n_iters + 1.0))
        p_ge_zero = float((np.sum(diffs >= 0.0) + 1.0) / (n_iters + 1.0))
        rows.append(
            {
                "baseline": baseline_name,
                "candidate": candidate_name,
                "metric": metric,
                "observed_delta": float(observed[metric]),
                "ci95_low": float(np.quantile(diffs, 0.025)),
                "ci95_high": float(np.quantile(diffs, 0.975)),
                "two_sided_p": min(1.0, 2.0 * min(p_le_zero, p_ge_zero)),
                "bootstrap_iters": n_iters,
            }
        )
    return rows


def bootstrap_table(predictions: dict[str, pd.DataFrame], table_dir: Path, n_iters: int, seed: int) -> pd.DataFrame:
    comparisons = (
        ("ResNet50 grouped", "RA-LPT fusion grouped"),
        ("ResNet50 grouped", "RA-LPT full+ablation fusion grouped"),
        ("RA-LPT fusion grouped", "RA-LPT full+ablation fusion grouped"),
        ("RA-LPT fusion grouped", "Strong cls ensemble grouped"),
        ("Strong cls ensemble grouped", "Strong RA-LPT fusion grouped"),
        ("Strong cls ensemble grouped", "Strong full RA-LPT fusion grouped"),
        ("Strong RA-LPT fusion grouped", "Strong full RA-LPT fusion grouped"),
        ("Strong cls ensemble grouped", "Strong LPT output ensemble grouped"),
        ("Strong cls ensemble grouped", "Strong full LPT output ensemble grouped"),
        ("EfficientNetV2-S grouped", "Strong cls ensemble grouped"),
    )
    rows: list[dict[str, Any]] = []
    available = set(predictions)
    for idx, (baseline, candidate) in enumerate(comparisons):
        if baseline in available and candidate in available:
            rows.extend(paired_bootstrap(predictions, baseline, candidate, n_iters=n_iters, seed=seed + idx))
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "similarity_grouped_paired_bootstrap.csv", index=False)
    return frame


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".6f") -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    rows = [columns]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(row[idx]) for row in rows) for idx in range(len(columns))]
    header = "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, divider, *body])


def write_notes(metrics: pd.DataFrame, bootstrap: pd.DataFrame, missing: list[str], table_dir: Path) -> None:
    if missing:
        missing_text = "\n".join(f"- `{path}`" for path in missing)
        text = f"""# Similarity-Grouped Retraining Status

The grouped retraining summary is incomplete because required prediction files are missing.

## Missing files

{missing_text}
"""
    else:
        metric_md = markdown_table(metrics)
        boot_md = markdown_table(bootstrap[bootstrap["metric"] == "macro_f1"]) if not bootstrap.empty else "No bootstrap rows."
        text = f"""# Similarity-Grouped Retraining Summary

This file summarises the reviewer-facing grouped retraining requested for RA-LPT fusion and the strong pure-vision ensemble.

## Pooled OOF Metrics

{metric_md}

## Paired Bootstrap, Macro-F1

{boot_md}

## Interpretation Template

Use these results to update the manuscript limitation about similarity-grouped retraining. If RA-LPT fusion or the strong classification ensemble drops materially under grouped splitting, keep the original cautious conclusion and report the grouped split as the stricter estimate.
"""
    (table_dir / "similarity_grouped_retraining_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prediction_dir = resolve(args.prediction_dir, repo_root)
    table_dir = resolve(args.table_dir, repo_root)
    manifest_path = resolve(args.manifest, repo_root)
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)[["image", "fold"]].copy()
    manifest["image"] = manifest["image"].astype(str)

    stack_specs = (
        PredictionSpec("RA-LPT full+ablation fusion grouped", f"{args.output_prefix}_lpt_model_ensemble_oof_predictions.csv"),
        PredictionSpec("Full LPT output ensemble grouped", f"{args.output_prefix}_lpt_output_ensemble_oof_predictions.csv"),
        PredictionSpec("RA-LPT fusion grouped", f"{args.output_prefix}_lpt_ablation_ensemble_oof_predictions.csv"),
        PredictionSpec("LPT output ensemble grouped", f"{args.output_prefix}_lpt_ablation_output_ensemble_oof_predictions.csv"),
        PredictionSpec("Strong cls ensemble grouped", f"{args.output_prefix}_strong_cls_ensemble_oof_predictions.csv"),
        PredictionSpec("Strong cls + prior residual grouped", f"{args.output_prefix}_strong_cls_prior_residual_oof_predictions.csv"),
        PredictionSpec("Strong LPT output ensemble grouped", f"{args.output_prefix}_strong_lpt_output_ensemble_oof_predictions.csv"),
        PredictionSpec("Strong RA-LPT fusion grouped", f"{args.output_prefix}_strong_lpt_prior_fusion_oof_predictions.csv"),
        PredictionSpec("Strong full LPT output ensemble grouped", f"{args.output_prefix}_strong_lpt_full_output_ensemble_oof_predictions.csv"),
        PredictionSpec("Strong full RA-LPT fusion grouped", f"{args.output_prefix}_strong_lpt_full_prior_fusion_oof_predictions.csv"),
    )

    predictions: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for spec in (*BASE_SPECS, *stack_specs):
        frame, spec_missing = load_spec(spec, prediction_dir, manifest)
        if spec_missing:
            missing.extend(spec_missing)
            continue
        assert frame is not None
        predictions[spec.method] = frame

    if missing:
        write_notes(pd.DataFrame(), pd.DataFrame(), missing, table_dir)
        raise SystemExit(f"Missing {len(missing)} prediction files. Wrote status file to {table_dir}")

    metrics = metric_tables(predictions, table_dir)
    bootstrap = bootstrap_table(predictions, table_dir, n_iters=int(args.bootstrap_iters), seed=int(args.seed))
    write_notes(metrics, bootstrap, [], table_dir)
    print(f"Saved similarity-grouped retraining summary to {table_dir}", flush=True)


if __name__ == "__main__":
    main()
