#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}
PROB_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]
EPS = 1e-8
FINAL_TABLE_METHOD = "Selected prior-informed fusion"
FINAL_CANDIDATE_NAME = "Selected prior-informed probability-fusion candidate"
METRIC_COLUMNS = ["method", "macro_f1", "macro_f1_std", "balanced_accuracy", "accuracy", "macro_precision", "macro_recall"]


COMPARISONS = [
    ("ResNet50 grouped", ("dfuc2021_cls_resnet50_fold{fold}_val_predictions.csv",)),
    ("ConvNeXt-Tiny grouped", ("dfuc2021_cls_convnext_tiny_fold{fold}_val_predictions.csv",)),
    ("EfficientNetV2-S grouped", ("dfuc2021_cls_efficientnetv2s_fold{fold}_val_predictions.csv",)),
    (
        "Strong cls ensemble grouped",
        (
            "reliability_prior_fusion_strong_cls_ensemble_oof_predictions.csv",
            "similarity_grouped_reliability_prior_fusion_strong_cls_ensemble_oof_predictions.csv",
        ),
    ),
    (
        "Strong cls + prior residual grouped",
        (
            "reliability_prior_fusion_strong_cls_prior_residual_oof_predictions.csv",
            "similarity_grouped_reliability_prior_fusion_strong_cls_prior_residual_oof_predictions.csv",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build descriptive assets for the post hoc prior-informed full-OOF candidate."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/publication_grouped/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/publication_grouped/publication_final_assets"))
    parser.add_argument("--paper-figure-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--paper-table-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument(
        "--audited-metrics-table",
        type=Path,
        default=Path("paper_assets/source_tables/publication_final_metrics.csv"),
        help="Audited source table for Table 5 rows that are five-run means or retained single OOF estimates.",
    )
    parser.add_argument(
        "--lpt-output-predictions",
        default="reliability_prior_fusion_strong_lpt_full_output_ensemble_oof_predictions.csv",
        help="OOF prediction file for the strong LPT full-output ensemble component.",
    )
    parser.add_argument(
        "--convnext-predictions",
        default="dfuc2021_cls_convnext_tiny_seed2027_oof_predictions.csv",
        help="OOF prediction file for the selected ConvNeXt-Tiny component.",
    )
    parser.add_argument(
        "--efficientnetv2s-predictions",
        default="dfuc2021_cls_efficientnetv2s_seed3407_oof_predictions.csv",
        help="OOF prediction file for the selected EfficientNetV2-S component.",
    )
    parser.add_argument(
        "--group-manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/similarity_grouped_splits/dfuc2021_cls_cv5_similarity_grouped_manifest.csv"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def label_ids(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map(LABEL_TO_ID).astype(int).to_numpy()


def read_prediction(path: Path, fold: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        labels = label_ids(frame["label"])
    elif "true_label" in frame.columns:
        labels = label_ids(frame["true_label"])
    else:
        raise ValueError(f"Missing label column in {path}")
    output = pd.DataFrame({"image": frame["image"].astype(str), "label": labels})
    if "fold" in frame.columns:
        output["fold"] = frame["fold"].astype(int)
    elif fold is not None:
        output["fold"] = int(fold)
    probs = frame[PROB_COLUMNS].astype(float).to_numpy()
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), EPS, None)
    for idx, column in enumerate(PROB_COLUMNS):
        output[column] = probs[:, idx]
    output["pred_id"] = np.argmax(probs, axis=1)
    output["pred_label"] = [ID_TO_LABEL[int(value)] for value in output["pred_id"]]
    return output


def read_fold_pattern(prediction_dir: Path, pattern: str) -> pd.DataFrame:
    parts = []
    for fold in range(5):
        parts.append(read_prediction(prediction_dir / pattern.format(fold=fold), fold=fold))
    return pd.concat(parts, ignore_index=True).sort_values("image").reset_index(drop=True)


def read_oof(prediction_dir: Path, filename: str) -> pd.DataFrame:
    return read_prediction(prediction_dir / filename).sort_values("image").reset_index(drop=True)


def read_first_existing_oof(prediction_dir: Path, filenames: tuple[str, ...]) -> pd.DataFrame:
    tried = []
    for filename in filenames:
        path = prediction_dir / filename
        tried.append(str(path))
        if path.exists():
            return read_prediction(path).sort_values("image").reset_index(drop=True)
    raise FileNotFoundError("None of the expected OOF prediction files exists: " + "; ".join(tried))


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


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, EPS, 1.0)) + bias.reshape(1, -1)
    return np.argmax(logits, axis=1)


def build_selected_fusion(
    prediction_dir: Path,
    lpt_output_predictions: str,
    convnext_predictions: str,
    efficientnetv2s_predictions: str,
) -> pd.DataFrame:
    lpt = read_first_existing_oof(
        prediction_dir,
        (
            lpt_output_predictions,
            "similarity_grouped_reliability_prior_fusion_strong_lpt_full_output_ensemble_oof_predictions.csv",
        ),
    )
    conv = read_oof(prediction_dir, convnext_predictions)
    eff = read_oof(prediction_dir, efficientnetv2s_predictions)
    reference = lpt[["image", "label", "fold"]].reset_index(drop=True)
    for name, frame in (("ConvNeXt-Tiny", conv), ("EfficientNetV2-S", eff)):
        current = frame[["image", "label", "fold"]].reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f"Prediction alignment mismatch for {name}")
    probs = (
        0.400 * lpt[PROB_COLUMNS].to_numpy(dtype=float)
        + 0.350 * conv[PROB_COLUMNS].to_numpy(dtype=float)
        + 0.250 * eff[PROB_COLUMNS].to_numpy(dtype=float)
    )
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), EPS, None)
    bias = np.array([-0.35, -0.025, -0.25, 0.0], dtype=float)
    pred_id = apply_bias(probs, bias)
    output = reference.copy()
    output["label_name"] = [ID_TO_LABEL[int(value)] for value in output["label"]]
    output["pred_id"] = pred_id
    output["pred_label"] = [ID_TO_LABEL[int(value)] for value in pred_id]
    for idx, column in enumerate(PROB_COLUMNS):
        output[column] = probs[:, idx]
    return output


def paired_group_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    baseline_name: str,
    candidate_name: str,
    group_manifest: pd.DataFrame,
    n_iters: int,
    seed: int,
) -> dict[str, Any]:
    base = baseline[["image", "label", "pred_id"]].rename(columns={"pred_id": "baseline_pred"})
    cand = candidate[["image", "label", "pred_id"]].rename(columns={"label": "candidate_label", "pred_id": "candidate_pred"})
    merged = base.merge(cand, on="image", how="inner", validate="one_to_one")
    if not np.array_equal(merged["label"].to_numpy(), merged["candidate_label"].to_numpy()):
        raise ValueError(f"Label mismatch for {baseline_name}")
    merged = merged.merge(group_manifest[["image", "similarity_group"]], on="image", how="left", validate="one_to_one")
    if merged["similarity_group"].isna().any():
        missing = merged.loc[merged["similarity_group"].isna(), "image"].head().tolist()
        raise ValueError(f"Missing similarity_group for images: {missing}")
    y_true = merged["label"].to_numpy(dtype=int)
    y_base = merged["baseline_pred"].to_numpy(dtype=int)
    y_cand = merged["candidate_pred"].to_numpy(dtype=int)
    observed = fast_metrics(y_true, y_cand)["macro_f1"] - fast_metrics(y_true, y_base)["macro_f1"]
    merged = merged.reset_index(drop=True)
    group_to_indices = [group.index.to_numpy(dtype=int) for _, group in merged.groupby("similarity_group", sort=True)]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_iters, dtype=float)
    for idx in range(n_iters):
        sampled_groups = rng.integers(0, len(group_to_indices), size=len(group_to_indices))
        sample = np.concatenate([group_to_indices[group_idx] for group_idx in sampled_groups])
        diffs[idx] = fast_metrics(y_true[sample], y_cand[sample])["macro_f1"] - fast_metrics(y_true[sample], y_base[sample])["macro_f1"]
    p_le_zero = float((np.sum(diffs <= 0.0) + 1.0) / (n_iters + 1.0))
    p_ge_zero = float((np.sum(diffs >= 0.0) + 1.0) / (n_iters + 1.0))
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metric": "macro_f1",
        "observed_delta": float(observed),
        "ci95_low": float(np.quantile(diffs, 0.025)),
        "ci95_high": float(np.quantile(diffs, 0.975)),
        "two_sided_p": min(1.0, 2.0 * min(p_le_zero, p_ge_zero)),
        "bootstrap_iters": int(n_iters),
        "bootstrap_unit": "similarity_group",
        "groups_resampled": int(len(group_to_indices)),
    }


def configure_plot() -> None:
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
        }
    )


def render_main_figure(metrics: pd.DataFrame, figure_dir: Path, dpi: int) -> None:
    plot = metrics.copy()
    names = {
        "ResNet50": "ResNet50\n5-run mean",
        "EfficientNetV2-S": "EffNetV2-S\n5-run mean",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny\n5-run mean",
        "Full LPT": "Full LPT",
        "RA-LPT fusion": "RA-LPT",
        "Strong cls ensemble": "Strong cls\nensemble",
        "Strong cls + prior residual": "Strong cls\n+ prior residual",
        FINAL_TABLE_METHOD: "Selected\nprior-informed\nfusion",
    }
    colors = []
    for method in plot["method"]:
        if method == FINAL_TABLE_METHOD:
            colors.append("#1b7f3a")
        elif "LPT" in method or "prior" in method:
            colors.append("#4f8a5b")
        else:
            colors.append("#2f6f9f")
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(plot))
    values = plot["macro_f1"].to_numpy(dtype=float)
    bars = ax.bar(x, values, color=colors, width=0.72)
    if "macro_f1_std" in plot.columns:
        yerr = plot["macro_f1_std"].fillna(0).to_numpy(dtype=float)
        ax.errorbar(x, values, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0012, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_title("Similarity-grouped DFUC2021 performance")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.90, 0.972)
    ax.set_xticks(x)
    ax.set_xticklabels([names.get(method, method) for method in plot["method"]], rotation=25, ha="right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_publication_final_main_results.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figure_dir / "fig_publication_final_main_results.pdf", bbox_inches="tight")
    plt.close(fig)


def render_bootstrap_figure(bootstrap: pd.DataFrame, figure_dir: Path, dpi: int) -> None:
    rows = bootstrap.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    x = rows["observed_delta"].to_numpy(dtype=float)
    xerr = np.vstack([x - rows["ci95_low"].to_numpy(dtype=float), rows["ci95_high"].to_numpy(dtype=float) - x])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#1b7f3a", ecolor="#78a987", capsize=3)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["baseline"])
    ax.set_xlabel("Macro-F1 improvement of selected candidate")
    ax.set_title("Group-level paired bootstrap sensitivity")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    for xpos, ypos, pvalue in zip(x, y, rows["two_sided_p"]):
        ax.text(xpos + 0.0008, ypos + 0.12, f"p={pvalue:.3g}", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_publication_final_bootstrap.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figure_dir / "fig_publication_final_bootstrap.pdf", bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for column in show.columns:
        if pd.api.types.is_float_dtype(show[column]):
            show[column] = show[column].map(lambda value: f"{value:.6f}")
    widths = [max(len(str(column)), *(len(str(value)) for value in show[column])) for column in show.columns]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(show.columns)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(str(row[column]).ljust(widths[idx]) for idx, column in enumerate(show.columns)) + " |" for _, row in show.iterrows()]
    return "\n".join([header, divider, *body])


def load_audited_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing audited metrics table: {path}. "
            "Provide --audited-metrics-table or regenerate the paper source table before assembling final assets."
        )
    frame = pd.read_csv(path)
    required = {"method", "macro_f1", "balanced_accuracy", "accuracy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Audited metrics table is missing required columns: {sorted(missing)}")
    for column in METRIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[METRIC_COLUMNS].copy()


def assemble_metrics_from_audited_table(audited: pd.DataFrame, final_metrics: dict[str, float]) -> pd.DataFrame:
    base = audited[audited["method"] != FINAL_TABLE_METHOD].copy()
    if base.empty:
        raise ValueError("Audited metrics table does not contain comparator rows.")
    final_row = {
        "method": FINAL_TABLE_METHOD,
        "macro_f1": final_metrics["macro_f1"],
        "macro_f1_std": np.nan,
        "balanced_accuracy": final_metrics["balanced_accuracy"],
        "accuracy": final_metrics["accuracy"],
        "macro_precision": final_metrics["macro_precision"],
        "macro_recall": final_metrics["macro_recall"],
    }
    return pd.concat([base, pd.DataFrame([final_row])], ignore_index=True)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    prediction_dir = resolve(args.prediction_dir, repo_root)
    output_dir = resolve(args.output_dir, repo_root)
    figure_dir = resolve(args.paper_figure_dir, repo_root)
    table_dir = resolve(args.paper_table_dir, repo_root)
    group_manifest_path = resolve(args.group_manifest, repo_root)
    audited_metrics_path = resolve(args.audited_metrics_table, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    final = build_selected_fusion(
        prediction_dir,
        args.lpt_output_predictions,
        args.convnext_predictions,
        args.efficientnetv2s_predictions,
    ).sort_values("image").reset_index(drop=True)
    final.to_csv(output_dir / "publication_selected_prior_informed_probability_fusion_predictions.csv", index=False)

    comparison_frames: list[tuple[str, pd.DataFrame]] = []
    for name, patterns in COMPARISONS:
        if any("{fold}" in pattern for pattern in patterns):
            comparison_frames.append((name, read_fold_pattern(prediction_dir, patterns[0])))
        else:
            comparison_frames.append((name, read_first_existing_oof(prediction_dir, patterns)))

    final_metrics = fast_metrics(final["label"].to_numpy(dtype=int), final["pred_id"].to_numpy(dtype=int))
    audited_metrics = load_audited_metrics(audited_metrics_path)
    metrics = assemble_metrics_from_audited_table(audited_metrics, final_metrics)
    metrics.to_csv(output_dir / "publication_final_metrics.csv", index=False)
    metrics.to_csv(table_dir / "publication_final_metrics.csv", index=False)

    group_manifest = pd.read_csv(group_manifest_path)
    if "similarity_group" not in group_manifest.columns:
        raise ValueError(f"Missing similarity_group column in {group_manifest_path}")
    group_manifest = group_manifest[["image", "similarity_group"]].copy()
    group_manifest["image"] = group_manifest["image"].astype(str)

    bootstrap_rows = []
    for idx, (name, frame) in enumerate(comparison_frames):
        bootstrap_rows.append(
            paired_group_bootstrap(
                frame,
                final,
                name,
                FINAL_CANDIDATE_NAME,
                group_manifest,
                args.bootstrap_iters,
                args.seed + idx,
            )
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "publication_final_group_level_bootstrap.csv", index=False)
    bootstrap.to_csv(table_dir / "publication_final_group_level_bootstrap.csv", index=False)

    configure_plot()
    render_main_figure(metrics, figure_dir, args.dpi)
    render_bootstrap_figure(bootstrap, figure_dir, args.dpi)

    report = "\n".join(
        [
            "# Publication Final Assets",
            "",
            "## Final Metrics",
            "",
            markdown_table(metrics[["method", "macro_f1", "macro_f1_std", "balanced_accuracy", "accuracy"]]),
            "",
            "## Group-Level Bootstrap Sensitivity",
            "",
            markdown_table(bootstrap),
            "",
            "## Selected Fusion",
            "",
            "Selected prior-informed probability-fusion candidate: 0.400 x strong LPT full-output ensemble + 0.350 x ConvNeXt-Tiny prediction + 0.250 x EfficientNetV2-S prediction; class-logit bias [-0.35, -0.025, -0.25, 0.0].",
            "",
            f"Comparator rows were read from audited source table `{args.audited_metrics_table}`; the selected-candidate row was recomputed from the specified OOF prediction files.",
            "",
        ]
    )
    (output_dir / "publication_final_summary.md").write_text(report, encoding="utf-8")
    (table_dir / "publication_final_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
