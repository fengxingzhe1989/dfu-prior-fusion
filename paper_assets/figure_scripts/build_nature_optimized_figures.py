#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")

PALETTE = {
    "rgb": "#4B6FA8",
    "rgb_light": "#B9C7E6",
    "prior": "#2C8C84",
    "prior_light": "#BFE1DD",
    "selected": "#102A56",
    "negative": "#B64A4A",
    "neutral": "#6B7280",
    "neutral_light": "#D8DEE9",
    "grid": "#E6E8EF",
    "text": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Nature-style manuscript figures from audited DFU source tables."
    )
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "paper_assets" / "source_tables",
        help="Directory containing audited manuscript source tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "nature_optimized_figures",
        help="Directory for optimized figure exports.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_all(fig: plt.Figure, outdir: Path, stem: str, dpi: int, manifest: list[dict[str, str]], source: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(outdir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)
    manifest.append(
        {
            "figure": stem,
            "source_table": source,
            "exports": ", ".join(f"{stem}.{suffix}" for suffix in ("svg", "pdf", "png", "tiff")),
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=8,
        color=PALETTE["text"],
    )


def clean_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.5)
        ax.set_axisbelow(True)


def method_color(method: str) -> str:
    lower = method.lower()
    if "selected" in lower:
        return PALETTE["selected"]
    if "direct" in lower or "full lpt" in lower or "random" in lower:
        return PALETTE["negative"]
    if "prior" in lower or "lpt" in lower or "ra-" in lower:
        return PALETTE["prior"]
    return PALETTE["rgb"]


def short_method(method: str) -> str:
    return {
        "EfficientNetV2-S": "EffNetV2-S",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny",
        "Strong cls ensemble": "Strong RGB stack",
        "Strong cls + prior residual": "Strong RGB + prior",
        "Selected prior-informed fusion": "Selected prior-informed",
        "Pure cls ensemble": "RGB ensemble",
        "Cls + prior residual": "RGB + prior residual",
        "LPT output ensemble": "LPT output ensemble",
        "LPT ablations + prior stats": "RA-LPT stack",
        "Reliability-aware LPT fusion": "RA-LPT",
        "Selected prior-informed probability-fusion candidate": "Selected prior-informed",
        "Retained strong RGB comparator": "Strong RGB comparator",
        "Retained strong RGB + prior-informed residual comparator": "Strong RGB + prior comparator",
    }.get(method, method)


def read_csv(source_dir: Path, relative: str) -> pd.DataFrame:
    return pd.read_csv(source_dir / relative)


def plot_grouped_performance(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frame = read_csv(source_dir, "publication_final_metrics.csv")
    order = [
        "ResNet50",
        "EfficientNetV2-S",
        "ConvNeXt-Tiny",
        "Full LPT",
        "RA-LPT fusion",
        "Strong cls ensemble",
        "Strong cls + prior residual",
        "Selected prior-informed fusion",
    ]
    frame = frame.set_index("method").loc[order].reset_index()
    y_pos = np.arange(len(frame))
    y = frame["macro_f1"].astype(float).to_numpy()
    err = frame["macro_f1_std"].fillna(0).astype(float).to_numpy()
    colors = [method_color(method) for method in frame["method"]]

    fig, ax = plt.subplots(figsize=(6.9, 3.7))
    ax.barh(y_pos, y, xerr=err, color=colors, height=0.64, edgecolor="white", linewidth=0.6, capsize=2.2)
    for ypos, value in zip(y_pos, y):
        ax.text(value + 0.001, ypos, f"{value:.3f}", va="center", fontsize=6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_method(m) for m in frame["method"]])
    ax.set_xlim(0.90, 0.972)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Similarity-grouped internal validation")
    clean_axis(ax, "x")
    ax.text(
        0.98,
        0.04,
        "OOF estimates; selected fusion is internally optimised",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=PALETTE["neutral"],
    )
    fig.tight_layout()
    save_all(fig, outdir, "nature_fig_grouped_performance", dpi, manifest, "publication_final_metrics.csv")


def plot_group_bootstrap(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frame = read_csv(source_dir, "methodology_audit_v7/publication_final_group_level_bootstrap.csv")
    keep = frame[frame["baseline"].isin(["ConvNeXt-Tiny grouped", "EfficientNetV2-S grouped", "Strong cls ensemble grouped", "Strong cls + prior residual grouped"])].copy()
    if keep.empty:
        keep = frame.copy()
    keep = keep.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(keep))
    x = keep["observed_delta"].astype(float).to_numpy()
    xerr = np.vstack(
        [
            x - keep["ci95_low"].astype(float).to_numpy(),
            keep["ci95_high"].astype(float).to_numpy() - x,
        ]
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.errorbar(x, y, xerr=xerr, fmt="o", color=PALETTE["selected"], ecolor=PALETTE["prior"], capsize=2.5, elinewidth=1.0)
    ax.axvline(0, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v.replace(" grouped", "")) for v in keep["baseline"]])
    ax.set_xlabel("Observed Macro-F1 difference vs selected candidate")
    ax.set_title("Group-level paired bootstrap sensitivity")
    for xpos, ypos, pvalue in zip(x, y, keep["two_sided_p"]):
        ax.text(xpos + 0.001, ypos + 0.12, f"p={float(pvalue):.3g}", fontsize=6, color=PALETTE["text"])
    clean_axis(ax, "x")
    fig.tight_layout()
    save_all(fig, outdir, "nature_fig_group_level_bootstrap", dpi, manifest, "methodology_audit_v7/publication_final_group_level_bootstrap.csv")


def plot_selected_candidate_summary(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    cm = read_csv(source_dir, "publication_final_confusion_matrix.csv").set_index("Unnamed: 0")
    matrix = cm.to_numpy(dtype=float)
    row_sum = np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    normalized = matrix / row_sum
    per_class = read_csv(source_dir, "publication_final_per_class_metrics.csv")
    binary = read_csv(source_dir, "publication_final_binary_endpoint_auc_auprc.csv")

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.05, 0.95], wspace=0.45)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax0.set_xticks(np.arange(4))
    ax0.set_yticks(np.arange(4))
    ax0.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax0.set_yticklabels(CLASS_NAMES)
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_title("Confusion matrix")
    for i in range(4):
        for j in range(4):
            color = "white" if normalized[i, j] > 0.55 else PALETTE["text"]
            ax0.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j] * 100:.1f}%", ha="center", va="center", fontsize=5.6, color=color)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.045, pad=0.03)
    cbar.ax.tick_params(labelsize=5.5)
    cbar.set_label("Row proportion", fontsize=6)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(per_class))
    f1 = per_class["f1"].astype(float).to_numpy()
    ax1.bar(x, f1, color=PALETTE["selected"], width=0.58)
    for xpos, value in zip(x, f1):
        ax1.text(xpos, value + 0.001, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=5.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(per_class["class"], rotation=30, ha="right")
    ax1.set_ylim(0.93, 0.985)
    ax1.set_ylabel("F1")
    ax1.set_title("Per-class performance")
    clean_axis(ax1)
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[0, 2])
    endpoints = [v.replace(" detection", "") for v in binary["Endpoint"]]
    y = np.arange(len(binary))
    height = 0.32
    ax2.barh(y - height / 2, binary["AUROC"].astype(float), height=height, label="AUROC", color=PALETTE["rgb"])
    ax2.barh(y + height / 2, binary["AUPRC"].astype(float), height=height, label="AUPRC", color=PALETTE["prior"])
    for row_idx, (_, row) in enumerate(binary.iterrows()):
        ax2.text(float(row["AUROC"]) + 0.0008, row_idx - height / 2, f"{float(row['AUROC']):.3f}", va="center", fontsize=5.8)
        ax2.text(float(row["AUPRC"]) + 0.0008, row_idx + height / 2, f"{float(row['AUPRC']):.3f}", va="center", fontsize=5.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(endpoints)
    ax2.set_xlim(0.975, 1.003)
    ax2.set_xlabel("Score")
    ax2.set_title("Binary endpoints")
    ax2.legend(loc="lower right", fontsize=6)
    clean_axis(ax2, "x")
    panel_label(ax2, "c")

    fig.tight_layout()
    save_all(
        fig,
        outdir,
        "nature_fig_selected_candidate_summary",
        dpi,
        manifest,
        "publication_final_confusion_matrix.csv; publication_final_per_class_metrics.csv; publication_final_binary_endpoint_auc_auprc.csv",
    )


def plot_lpt_ablation(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frame = read_csv(source_dir, "table_lpt_spatial_ablations.csv")
    frame = frame.sort_values("macro_f1", ascending=True)
    y = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()
    err = frame["macro_f1_std"].astype(float).to_numpy()
    colors = [method_color(m) for m in frame["method"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.barh(y, values, xerr=err, color=colors, edgecolor="white", linewidth=0.5, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.set_xlim(0.90, max(0.947, values.max() + 0.008))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Direct structured prior injection and ablations")
    for ypos, value in zip(y, values):
        ax.text(value + 0.001, ypos, f"{value:.3f}", va="center", fontsize=6)
    clean_axis(ax, "x")
    fig.tight_layout()
    save_all(fig, outdir, "nature_supp_lpt_ablation", dpi, manifest, "table_lpt_spatial_ablations.csv")


def plot_reliability_fusion(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frame = read_csv(source_dir, "table_reliability_fusion.csv")
    order = [
        "Pure cls ensemble",
        "Cls + prior residual",
        "LPT output ensemble",
        "LPT ablations + prior stats",
    ]
    frame = frame.set_index("method").loc[[m for m in order if m in frame["method"].values]].reset_index()
    x = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()
    err = frame["macro_f1_std"].astype(float).to_numpy()
    colors = [method_color(m) for m in frame["method"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    ax.bar(x, values, yerr=err, capsize=2, color=colors, edgecolor="white", linewidth=0.6)
    for xpos, value in zip(x, values):
        ax.text(xpos, value + 0.001, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_method(v) for v in frame["method"]], rotation=28, ha="right")
    ax.set_ylim(0.94, max(0.962, values.max() + 0.007))
    ax.set_ylabel("Macro-F1")
    ax.set_title("Reliability-aware stacking in standard CV")
    clean_axis(ax)
    fig.tight_layout()
    save_all(fig, outdir, "nature_supp_reliability_fusion", dpi, manifest, "table_reliability_fusion.csv")


def plot_prior_reliability_strata(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frames = []
    for file_name in ("prior_reliability_strata_metrics.csv", "strong_prior_reliability_strata_metrics.csv"):
        frame = read_csv(source_dir, file_name)
        frame = frame[(frame["group_type"] == "fallback_group") & frame["method"].isin(["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong RA-LPT fusion"])].copy()
        if not frame.empty:
            frame["source_file"] = file_name
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    methods = list(dict.fromkeys(data["method"]))
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    x = np.arange(len(methods))
    width = 0.34
    for offset, group, color in [(-width / 2, "fallback", PALETTE["neutral"]), (width / 2, "non_fallback", PALETTE["prior"])]:
        vals = []
        for method in methods:
            row = data[(data["method"] == method) & (data["group"] == group)]
            vals.append(float(row["macro_f1"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, vals, width=width, color=color, label=group.replace("_", "-"), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_method(m) for m in methods], rotation=25, ha="right")
    ax.set_ylim(0.88, 0.975)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performance stratified by prior fallback status")
    ax.legend(loc="upper left", ncol=2)
    clean_axis(ax)
    fig.tight_layout()
    save_all(
        fig,
        outdir,
        "nature_supp_prior_reliability_strata",
        dpi,
        manifest,
        "prior_reliability_strata_metrics.csv; strong_prior_reliability_strata_metrics.csv",
    )


def plot_interpretability(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    summary = read_csv(source_dir, "cam_prior_occlusion_summary.csv")
    subset = summary[(summary["group_type"] == "fallback") & (summary["method"].isin(["ResNet50", "EfficientNetV2-S"]))].copy()
    metrics = [
        ("cam_prior_mass_ratio", "CAM-prior mass ratio"),
        ("pointing_game_accuracy", "Pointing-game accuracy"),
        ("prior_minus_random_drop", "Prior minus random occlusion drop"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for ax, (column, title) in zip(axes, metrics):
        pivot = subset.pivot(index="method", columns="group", values=column).loc[["ResNet50", "EfficientNetV2-S"]]
        x = np.arange(len(pivot))
        width = 0.34
        ax.bar(x - width / 2, pivot["fallback"], width=width, label="fallback", color=PALETTE["neutral"])
        ax.bar(x + width / 2, pivot["non_fallback"], width=width, label="non-fallback", color=PALETTE["prior"])
        ax.set_xticks(x)
        ax.set_xticklabels([short_method(v) for v in pivot.index], rotation=25, ha="right")
        ax.set_title(title)
        clean_axis(ax)
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    save_all(fig, outdir, "nature_supp_interpretability_summary", dpi, manifest, "cam_prior_occlusion_summary.csv")


def plot_overlap_sensitivity(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    base = "cross_dataset_overlap_exclusion_sensitivity_v1/"
    metrics = read_csv(source_dir, base + "cross_dataset_overlap_exclusion_overall_metrics.csv")
    diffs = read_csv(source_dir, base + "cross_dataset_overlap_exclusion_model_differences.csv")
    selected = metrics[metrics["model_key"] == "selected_prior_informed_probability_fusion"].copy()
    comparators = metrics[metrics["model_key"].isin(["retained_strong_rgb_logistic_stack", "retained_strong_rgb_prior_feature_stack"])].copy()

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.42)

    ax0 = fig.add_subplot(gs[0, 0])
    labels = ["full OOF", "exclude 241"]
    x = np.arange(2)
    ax0.plot(x, selected["macro_f1"], marker="o", color=PALETTE["selected"], label="Selected")
    for key, color in [("retained_strong_rgb_logistic_stack", PALETTE["rgb"]), ("retained_strong_rgb_prior_feature_stack", PALETTE["prior"])]:
        sub = comparators[comparators["model_key"] == key]
        ax0.plot(x, sub["macro_f1"], marker="o", color=color, label=short_method(sub["model"].iloc[0]))
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("Macro-F1")
    ax0.set_ylim(0.948, 0.967)
    ax0.set_title("Broad visual-overlap exclusion")
    ax0.legend(loc="lower right", fontsize=5.8)
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    keep = diffs[diffs["scope"] == "exclude_broad_cross_dataset_overlap_241"].copy()
    y = np.arange(len(keep))
    ax1.barh(y, keep["macro_f1_delta"].astype(float), color=[PALETTE["rgb"], PALETTE["prior"]][: len(keep)])
    ax1.axvline(0, color=PALETTE["neutral"], linestyle="--", linewidth=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels([short_method(v) for v in keep["comparator"]])
    ax1.set_xlabel("Macro-F1 difference")
    ax1.set_title("Difference after exclusion")
    for ypos, value in zip(y, keep["macro_f1_delta"].astype(float)):
        ax1.text(value + 0.00025, ypos, f"{value:.4f}", va="center", fontsize=6)
    clean_axis(ax1, "x")
    panel_label(ax1, "b")
    fig.tight_layout()
    save_all(
        fig,
        outdir,
        "nature_supp_cross_dataset_overlap_sensitivity",
        dpi,
        manifest,
        base + "cross_dataset_overlap_exclusion_overall_metrics.csv; " + base + "cross_dataset_overlap_exclusion_model_differences.csv",
    )


def plot_auc_summary(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict[str, str]]) -> None:
    frame = read_csv(source_dir, "roc_pr_auc_pivot.csv")
    methods = frame["method"].map(short_method)
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    y = np.arange(len(frame))
    height = 0.34
    ax.barh(y - height / 2, frame["roc_auc_macro"].astype(float), height=height, color=PALETTE["rgb"], label="Macro AUROC")
    ax.barh(y + height / 2, frame["average_precision_macro"].astype(float), height=height, color=PALETTE["prior"], label="Macro AUPRC")
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0.975, 1.000)
    ax.set_xlabel("Score")
    ax.set_title("Macro one-vs-rest discrimination summary")
    ax.legend(loc="lower right")
    clean_axis(ax, "x")
    fig.tight_layout()
    save_all(fig, outdir, "nature_supp_auc_summary", dpi, manifest, "roc_pr_auc_pivot.csv")


def write_audit(outdir: Path, manifest: list[dict[str, str]]) -> None:
    lines = [
        "# Nature-Style Figure Optimization Manifest",
        "",
        "These figures were regenerated from audited source tables using Python/matplotlib.",
        "They are intended as replacement candidates for manuscript and supplementary quantitative figures.",
        "Clinical photographs, Grad-CAM image plates, and workflow schematics should be handled separately as image-layout figures, not contrast-enhanced as quantitative plots.",
        "",
        "| Figure | Source table(s) | Exports |",
        "| --- | --- | --- |",
    ]
    for row in manifest:
        lines.append(f"| {row['figure']} | {row['source_table']} | {row['exports']} |")
    lines.extend(
        [
            "",
            "## Figure Contract",
            "",
            "- Core conclusion: DFUC2022-derived priors add value only when treated as uncertain auxiliary evidence; direct LPT can over-constrain the classifier.",
            "- Archetype: quantitative grid plus selected image/interpretability support.",
            "- Backend: Python/matplotlib only.",
            "- Export policy: SVG and PDF retain editable text; PNG/TIFF are 600 dpi raster previews/submission alternatives.",
            "- Review boundary: all selected fusion estimates remain internally optimised grouped OOF estimates, not external validation.",
        ]
    )
    (outdir / "figure_optimization_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "figure_optimization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    source_dir = args.source_dir.resolve()
    outdir = args.output_dir.resolve()
    manifest: list[dict[str, str]] = []
    builders: list[Callable[[Path, Path, int, list[dict[str, str]]], None]] = [
        plot_grouped_performance,
        plot_group_bootstrap,
        plot_selected_candidate_summary,
        plot_lpt_ablation,
        plot_reliability_fusion,
        plot_prior_reliability_strata,
        plot_interpretability,
        plot_overlap_sensitivity,
        plot_auc_summary,
    ]
    for builder in builders:
        builder(source_dir, outdir, args.dpi, manifest)
    write_audit(outdir, manifest)
    print(f"Wrote {len(manifest)} optimized figure candidates to {outdir}")


if __name__ == "__main__":
    main()
