#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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


PREDICTION_SPECS = (
    PredictionSpec("ResNet50", "dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv"),
    PredictionSpec("EfficientNet-B0", "dfuc2021_cls_efficientnetb0_fold{fold}_val_predictions.csv"),
    PredictionSpec("ConvNeXt-Tiny", "dfuc2021_cls_convnext_tiny_fold{fold}_val_predictions.csv"),
    PredictionSpec("EfficientNetV2-S", "dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv"),
    PredictionSpec("Pure cls ensemble", "reliability_prior_fusion_nobias_cls_ensemble_oof_predictions.csv"),
    PredictionSpec(
        "Reliability-aware LPT fusion",
        "reliability_prior_fusion_subset_search_lpt_ablation_ensemble_oof_predictions.csv",
    ),
    PredictionSpec(
        "LPT output ensemble",
        "reliability_prior_fusion_subset_search_lpt_ablation_output_ensemble_oof_predictions.csv",
    ),
    PredictionSpec("Strong cls ensemble", "reliability_prior_fusion_strong_strong_cls_ensemble_oof_predictions.csv"),
    PredictionSpec(
        "Strong cls + prior residual",
        "reliability_prior_fusion_strong_strong_cls_prior_residual_oof_predictions.csv",
    ),
    PredictionSpec(
        "Strong LPT output ensemble",
        "reliability_prior_fusion_strong_strong_lpt_output_ensemble_oof_predictions.csv",
    ),
    PredictionSpec("Strong RA-LPT fusion", "reliability_prior_fusion_strong_strong_lpt_prior_fusion_oof_predictions.csv"),
)


BOOTSTRAP_COMPARISONS = (
    ("EfficientNetV2-S", "Reliability-aware LPT fusion"),
    ("EfficientNetV2-S", "Strong cls ensemble"),
    ("EfficientNetV2-S", "Strong LPT output ensemble"),
    ("Reliability-aware LPT fusion", "Strong cls ensemble"),
    ("Reliability-aware LPT fusion", "Strong RA-LPT fusion"),
    ("Strong cls ensemble", "Strong RA-LPT fusion"),
    ("Strong cls ensemble", "Strong LPT output ensemble"),
    ("Strong LPT output ensemble", "Strong RA-LPT fusion"),
)


STRATA_METHODS = (
    "EfficientNetV2-S",
    "Reliability-aware LPT fusion",
    "Strong cls ensemble",
    "Strong LPT output ensemble",
    "Strong RA-LPT fusion",
)


ERROR_METHODS = (
    "ResNet50",
    "EfficientNetV2-S",
    "Reliability-aware LPT fusion",
    "Strong cls ensemble",
    "Strong RA-LPT fusion",
)


METHOD_SHORT = {
    "ResNet50": "ResNet50",
    "EfficientNet-B0": "EffNet-B0",
    "ConvNeXt-Tiny": "ConvNeXt",
    "EfficientNetV2-S": "EffNetV2-S",
    "Reliability-aware LPT fusion": "RA-LPT",
    "LPT output ensemble": "LPT outputs",
    "Strong cls ensemble": "Strong cls ens.",
    "Strong cls + prior residual": "Strong cls+prior",
    "Strong LPT output ensemble": "Strong LPT outputs",
    "Strong RA-LPT fusion": "Strong RA-LPT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong baseline bootstrap, strata, and error analysis.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--manifest", type=Path, default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"))
    parser.add_argument("--prior-meta", type=Path, default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv"))
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def label_series_to_id(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map(LABEL_TO_ID).astype(int).to_numpy()


def read_prediction_file(path: Path, manifest_fold: pd.DataFrame | None = None, fold: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        y_true = label_series_to_id(frame["label"])
        label_name = frame["label_name"] if "label_name" in frame.columns else pd.Series([ID_TO_LABEL[int(v)] for v in y_true])
    elif "true_label" in frame.columns:
        y_true = label_series_to_id(frame["true_label"])
        label_name = pd.Series([ID_TO_LABEL[int(v)] for v in y_true])
    else:
        raise ValueError(f"Missing label column in {path}")
    if "pred_label" not in frame.columns:
        raise ValueError(f"Missing pred_label column in {path}")
    y_pred = label_series_to_id(frame["pred_label"])
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
    if output["fold"].isnull().any():
        raise ValueError(f"Missing fold values after loading {path}")
    return output


def load_method_predictions(spec: PredictionSpec, prediction_dir: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    manifest_fold = manifest[["image", "fold"]].copy()
    if "{fold}" in spec.pattern:
        parts = []
        for fold in range(5):
            parts.append(read_prediction_file(prediction_dir / spec.pattern.format(fold=fold), fold=fold))
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = read_prediction_file(prediction_dir / spec.pattern, manifest_fold=manifest_fold)
    if frame["image"].duplicated().any():
        duplicate = frame.loc[frame["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Duplicate predictions for {spec.method}: {duplicate}")
    return frame.sort_values("image").reset_index(drop=True)


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(int, copy=False)
    y_pred = y_pred.astype(int, copy=False)
    encoded = len(CLASS_NAMES) * y_true + y_pred
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


def prediction_metrics(predictions: dict[str, pd.DataFrame], table_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_payload: dict[str, Any] = {}
    for method, frame in predictions.items():
        y_true = frame["label"].to_numpy(dtype=int)
        y_pred = frame["pred_id"].to_numpy(dtype=int)
        rows.append({"method": method, **fast_metrics(y_true, y_pred)})
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            average=None,
            zero_division=0,
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
        confusion_payload[method] = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).astype(int).tolist()
    metrics = pd.DataFrame(rows)
    metrics.to_csv(table_dir / "strong_oof_prediction_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(table_dir / "strong_oof_per_class_metrics.csv", index=False)
    with (table_dir / "strong_oof_confusion_matrices.json").open("w", encoding="utf-8") as handle:
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
    candidate = predictions[candidate_name][["image", "label", "pred_id"]].rename(
        columns={"label": "candidate_label", "pred_id": "candidate_pred"}
    )
    merged = baseline.merge(candidate, on="image", how="inner", validate="one_to_one")
    if not np.array_equal(merged["label"].to_numpy(), merged["candidate_label"].to_numpy()):
        raise ValueError(f"Label mismatch: {baseline_name} vs {candidate_name}")
    y_true = merged["label"].to_numpy(dtype=int)
    baseline_pred = merged["baseline_pred"].to_numpy(dtype=int)
    candidate_pred = merged["candidate_pred"].to_numpy(dtype=int)
    observed_baseline = fast_metrics(y_true, baseline_pred)
    observed_candidate = fast_metrics(y_true, candidate_pred)
    observed = {metric: observed_candidate[metric] - observed_baseline[metric] for metric in METRIC_NAMES}
    rng = np.random.default_rng(seed)
    n_samples = len(merged)
    diff_by_metric = {metric: np.empty(n_iters, dtype=float) for metric in ("macro_f1", "balanced_accuracy", "accuracy")}
    for idx in range(n_iters):
        sample_idx = rng.integers(0, n_samples, size=n_samples)
        candidate_metrics = fast_metrics(y_true[sample_idx], candidate_pred[sample_idx])
        baseline_metrics = fast_metrics(y_true[sample_idx], baseline_pred[sample_idx])
        for metric in diff_by_metric:
            diff_by_metric[metric][idx] = candidate_metrics[metric] - baseline_metrics[metric]
    rows = []
    for metric, diffs in diff_by_metric.items():
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
                "one_sided_p_delta_le_0": p_le_zero,
                "two_sided_p": min(1.0, 2.0 * min(p_le_zero, p_ge_zero)),
                "bootstrap_iters": n_iters,
            }
        )
    return rows


def build_bootstrap(predictions: dict[str, pd.DataFrame], table_dir: Path, n_iters: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for offset, (baseline, candidate) in enumerate(BOOTSTRAP_COMPARISONS):
        rows.extend(paired_bootstrap(predictions, baseline, candidate, n_iters, seed + offset))
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "strong_paired_bootstrap_significance.csv", index=False)
    return frame


def add_prior_groups(prior_meta: pd.DataFrame) -> pd.DataFrame:
    frame = prior_meta.copy()
    frame["prior_fallback"] = frame["prior_fallback"].astype(int)
    frame["prior_confidence"] = frame["prior_confidence"].astype(float)
    frame["prior_area_ratio"] = frame["prior_area_ratio"].astype(float)
    frame["prior_quality"] = frame["prior_confidence"] * (1.0 - frame["prior_fallback"])
    frame["fallback_group"] = np.where(frame["prior_fallback"] == 1, "fallback", "non_fallback")
    nonfallback = frame["prior_fallback"] == 0
    median_quality = float(frame.loc[nonfallback, "prior_quality"].median()) if nonfallback.any() else 0.0
    frame["quality_group"] = "fallback"
    frame.loc[nonfallback & (frame["prior_quality"] < median_quality), "quality_group"] = "nonfallback_low_quality"
    frame.loc[nonfallback & (frame["prior_quality"] >= median_quality), "quality_group"] = "nonfallback_high_quality"
    frame["area_group"] = pd.cut(
        frame["prior_area_ratio"],
        bins=[-1e-9, 0.02, 0.15, 0.45, 1.01],
        labels=["tiny", "small", "medium", "large"],
    ).astype(str)
    return frame


def build_strata(predictions: dict[str, pd.DataFrame], prior_meta: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    prior_groups = add_prior_groups(prior_meta)
    rows: list[dict[str, Any]] = []
    for method in STRATA_METHODS:
        pred = predictions[method]
        merged = pred.merge(
            prior_groups[["image", "fallback_group", "quality_group", "area_group"]],
            on="image",
            how="left",
            validate="one_to_one",
        )
        for group_column in ("fallback_group", "quality_group", "area_group"):
            for group_value, group_frame in merged.groupby(group_column):
                rows.append(
                    {
                        "method": method,
                        "group_type": group_column,
                        "group": str(group_value),
                        "samples": int(len(group_frame)),
                        **fast_metrics(
                            group_frame["label"].to_numpy(dtype=int),
                            group_frame["pred_id"].to_numpy(dtype=int),
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "strong_prior_reliability_strata_metrics.csv", index=False)
    return frame


def build_error_analysis(
    predictions: dict[str, pd.DataFrame],
    prior_meta: pd.DataFrame,
    table_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = predictions[ERROR_METHODS[0]][["image", "label", "label_name", "fold"]].copy()
    for method in ERROR_METHODS:
        pred = predictions[method][["image", "pred_label", "pred_id"]].rename(
            columns={"pred_label": f"{method}_pred", "pred_id": f"{method}_pred_id"}
        )
        base = base.merge(pred, on="image", how="inner", validate="one_to_one")
        base[f"{method}_correct"] = base[f"{method}_pred_id"] == base["label"]
    prior_columns = [
        "image",
        "prior_quality",
        "prior_confidence",
        "prior_fallback",
        "prior_area_ratio",
        "largest_component_area_ratio",
        "bbox_area_ratio",
    ]
    enriched_prior = add_prior_groups(prior_meta)
    base = base.merge(enriched_prior[prior_columns + ["fallback_group", "quality_group", "area_group"]], on="image", how="left", validate="one_to_one")
    base.to_csv(table_dir / "strong_error_cases.csv", index=False)

    pattern_columns = [f"{method}_correct" for method in ERROR_METHODS]
    pattern = base.groupby(pattern_columns + ["label_name"], dropna=False).size().reset_index(name="samples")
    pattern.to_csv(table_dir / "strong_error_pattern_summary.csv", index=False)

    contrasts = {
        "RA-LPT correct, ResNet50 wrong": base["Reliability-aware LPT fusion_correct"] & ~base["ResNet50_correct"],
        "RA-LPT correct, EfficientNetV2-S wrong": base["Reliability-aware LPT fusion_correct"] & ~base["EfficientNetV2-S_correct"],
        "EfficientNetV2-S correct, RA-LPT wrong": base["EfficientNetV2-S_correct"] & ~base["Reliability-aware LPT fusion_correct"],
        "Strong cls correct, RA-LPT wrong": base["Strong cls ensemble_correct"] & ~base["Reliability-aware LPT fusion_correct"],
        "Strong RA-LPT correct, Strong cls wrong": base["Strong RA-LPT fusion_correct"] & ~base["Strong cls ensemble_correct"],
        "Strong cls correct, Strong RA-LPT wrong": base["Strong cls ensemble_correct"] & ~base["Strong RA-LPT fusion_correct"],
        "All selected methods correct": np.logical_and.reduce([base[f"{method}_correct"].to_numpy(bool) for method in ERROR_METHODS]),
        "All selected methods wrong": ~np.logical_or.reduce([base[f"{method}_correct"].to_numpy(bool) for method in ERROR_METHODS]),
    }
    summary_rows: list[dict[str, Any]] = []
    example_rows: list[pd.DataFrame] = []
    for contrast, mask in contrasts.items():
        group = base.loc[mask].copy()
        class_counts = group["label_name"].value_counts().to_dict()
        summary_rows.append(
            {
                "contrast": contrast,
                "samples": int(len(group)),
                "fallback_ratio": float(group["prior_fallback"].mean()) if len(group) else 0.0,
                "prior_quality_mean": float(group["prior_quality"].mean()) if len(group) else 0.0,
                "prior_area_ratio_mean": float(group["prior_area_ratio"].mean()) if len(group) else 0.0,
                **{f"class_{class_name}": int(class_counts.get(class_name, 0)) for class_name in CLASS_NAMES},
            }
        )
        if len(group):
            examples = group.sort_values(["prior_quality", "prior_area_ratio"], ascending=False).head(12).copy()
            examples.insert(0, "contrast", contrast)
            example_rows.append(examples)
    contrast_summary = pd.DataFrame(summary_rows)
    contrast_summary.to_csv(table_dir / "strong_error_case_summary.csv", index=False)
    examples_frame = pd.concat(example_rows, ignore_index=True) if example_rows else pd.DataFrame()
    examples_frame.to_csv(table_dir / "strong_error_examples.csv", index=False)
    return base, contrast_summary, examples_frame


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_bootstrap(bootstrap: pd.DataFrame, figure_dir: Path) -> None:
    rows = bootstrap[bootstrap["metric"] == "macro_f1"].copy()
    rows["comparison"] = rows["candidate"].map(METHOD_SHORT).fillna(rows["candidate"]) + "\nvs " + rows["baseline"].map(METHOD_SHORT).fillna(rows["baseline"])
    rows = rows.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    x = rows["observed_delta"].to_numpy(dtype=float)
    xerr = np.vstack([x - rows["ci95_low"].to_numpy(dtype=float), rows["ci95_high"].to_numpy(dtype=float) - x])
    colors = np.where(x >= 0, "#2f6f9f", "#b5532f")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx in range(len(rows)):
        ax.errorbar(x[idx], y[idx], xerr=xerr[:, [idx]], fmt="o", color=colors[idx], ecolor=colors[idx], capsize=3)
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(rows["comparison"])
    ax.set_xlabel("Macro-F1 delta (candidate - baseline), paired bootstrap 95% CI")
    ax.set_title("Strong-baseline paired bootstrap")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    for xpos, ypos, pvalue in zip(x, y, rows["two_sided_p"], strict=True):
        text_x = xpos + (0.0008 if xpos >= 0 else -0.0008)
        ax.text(text_x, ypos + 0.12, f"p={pvalue:.3g}", fontsize=7.5, ha="left" if xpos >= 0 else "right")
    save_figure(fig, figure_dir, "fig_strong_bootstrap_macro_f1_delta")


def render_strata(strata: pd.DataFrame, figure_dir: Path) -> None:
    methods = ["EfficientNetV2-S", "Strong cls ensemble", "Strong LPT output ensemble", "Strong RA-LPT fusion"]
    quality_order = ["fallback", "nonfallback_low_quality", "nonfallback_high_quality"]
    labels = ["fallback", "low quality", "high quality"]
    frame = strata[(strata["group_type"] == "quality_group") & (strata["method"].isin(methods))].copy()
    x = np.arange(len(quality_order))
    width = 0.18
    colors = ["#909090", "#2f6f9f", "#d58936", "#4f8a5b"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for idx, method in enumerate(methods):
        values = frame[frame["method"] == method].set_index("group").loc[quality_order]["macro_f1"].to_numpy(dtype=float)
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=METHOD_SHORT.get(method, method), color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.88, 0.98)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Strong methods by prior reliability")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, figure_dir, "fig_strong_prior_reliability_strata")


def render_error_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    rows = summary[~summary["contrast"].isin(["All selected methods correct"])].copy()
    rows = rows.sort_values("samples", ascending=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.barh(np.arange(len(rows)), rows["samples"], color="#2f6f9f")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows["contrast"])
    ax.set_xlabel("Number of samples")
    ax.set_title("Strong-method error/correction contrasts")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    for ypos, value in enumerate(rows["samples"]):
        ax.text(value + 8, ypos, str(int(value)), va="center", fontsize=8)
    save_figure(fig, figure_dir, "fig_strong_error_summary")


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, float_digits: int = 6) -> str:
    if columns is None:
        columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |"
    lines = [header, separator]
    for _, row in frame[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append(f"{float(value):.{float_digits}f}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    strata: pd.DataFrame,
    error_summary: pd.DataFrame,
    table_dir: Path,
) -> None:
    best_row = metrics.loc[metrics["macro_f1"].idxmax()]
    eff = metrics.loc[metrics["method"] == "EfficientNetV2-S"].iloc[0]
    strong = metrics.loc[metrics["method"] == "Strong cls ensemble"].iloc[0]
    strong_ra = metrics.loc[metrics["method"] == "Strong RA-LPT fusion"].iloc[0]
    ra = metrics.loc[metrics["method"] == "Reliability-aware LPT fusion"].iloc[0]
    macro_rows = bootstrap[bootstrap["metric"] == "macro_f1"].copy()
    quality = strata[(strata["group_type"] == "quality_group") & (strata["method"].isin(STRATA_METHODS))].copy()
    quality_pivot = quality.pivot(index="group", columns="method", values="macro_f1").reset_index()
    bootstrap_md = markdown_table(
        macro_rows[["baseline", "candidate", "observed_delta", "ci95_low", "ci95_high", "two_sided_p"]],
        float_digits=6,
    )
    quality_md = markdown_table(quality_pivot, float_digits=6)
    error_md = markdown_table(error_summary, float_digits=4)
    text = f"""# Strong Follow-up Analysis Notes

Generated by `experiments/reliability_aware_prior_fusion/strong_followup_analysis.py`.

## Main Finding

The strongest completed method is `{best_row['method']}` with pooled OOF Macro-F1 `{best_row['macro_f1']:.6f}` and Balanced Accuracy `{best_row['balanced_accuracy']:.6f}`.

Key OOF values:

- EfficientNetV2-S: Macro-F1 `{eff['macro_f1']:.6f}`, Balanced Accuracy `{eff['balanced_accuracy']:.6f}`.
- Reliability-aware LPT fusion: Macro-F1 `{ra['macro_f1']:.6f}`, Balanced Accuracy `{ra['balanced_accuracy']:.6f}`.
- Strong classification ensemble: Macro-F1 `{strong['macro_f1']:.6f}`, Balanced Accuracy `{strong['balanced_accuracy']:.6f}`.
- Strong RA-LPT fusion: Macro-F1 `{strong_ra['macro_f1']:.6f}`, Balanced Accuracy `{strong_ra['balanced_accuracy']:.6f}`.

## Interpretation

EfficientNetV2-S is a strong single-backbone baseline and is close to the original reliability-aware LPT fusion. The strong classification ensemble becomes the best overall method. This means the paper should not claim that prior fusion beats every strong classifier. The better claim is that hard spatial prior transfer is fragile, while reliability-aware modeling makes lesion priors safer and more interpretable; however, the marginal gain of priors decreases once strong visual classifiers are ensembled.

## Strong Bootstrap Tests

{bootstrap_md}

## Prior Reliability Groups

{quality_md}

## Error Analysis Highlights

{error_md}

## Files

- `strong_oof_prediction_metrics.csv`
- `strong_paired_bootstrap_significance.csv`
- `strong_prior_reliability_strata_metrics.csv`
- `strong_error_case_summary.csv`
- `strong_error_cases.csv`
- `strong_error_examples.csv`
- `fig_strong_bootstrap_macro_f1_delta`
- `fig_strong_prior_reliability_strata`
- `fig_strong_error_summary`
"""
    (table_dir / "strong_followup_analysis_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prediction_dir = resolve_path(args.prediction_dir, repo_root)
    table_dir = resolve_path(args.table_dir, repo_root)
    figure_dir = resolve_path(args.figure_dir, repo_root)
    manifest_path = resolve_path(args.manifest, repo_root)
    prior_meta_path = resolve_path(args.prior_meta, repo_root)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    manifest = pd.read_csv(manifest_path)[["image", "fold"]].copy()
    manifest["image"] = manifest["image"].astype(str)
    prior_meta = pd.read_csv(prior_meta_path)
    prior_meta["image"] = prior_meta["image"].astype(str)
    predictions = {
        spec.method: load_method_predictions(spec, prediction_dir, manifest)
        for spec in PREDICTION_SPECS
    }

    metrics = prediction_metrics(predictions, table_dir)
    bootstrap = build_bootstrap(predictions, table_dir, n_iters=int(args.bootstrap_iters), seed=int(args.seed))
    strata = build_strata(predictions, prior_meta, table_dir)
    _, error_summary, _ = build_error_analysis(predictions, prior_meta, table_dir)

    render_bootstrap(bootstrap, figure_dir)
    render_strata(strata, figure_dir)
    render_error_summary(error_summary, figure_dir)
    write_markdown(metrics, bootstrap, strata, error_summary, table_dir)
    print(f"Saved strong follow-up analysis to {table_dir}", flush=True)


if __name__ == "__main__":
    main()
