#!/usr/bin/env python3
"""Build a source-grounded, Nature-style unified DFU extra-figure series."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw


CLASS_NAMES = ("none", "infection", "ischaemia", "both")

COLORS = {
    "selected": "#173A67",
    "rgb": "#4F78A8",
    "rgb_light": "#AFC4DD",
    "prior": "#2A8C82",
    "prior_light": "#BFDCD8",
    "negative": "#B94B4B",
    "negative_light": "#E4B4B1",
    "warning": "#A96F22",
    "neutral": "#747D8C",
    "neutral_light": "#D7DCE4",
    "text": "#172033",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "paper_assets" / "source_tables",
        help="Directory containing audited manuscript source tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "nature_unified_extra_figures",
        help="Directory for SVG/PDF/PNG/TIFF exports.",
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
            "font.size": 7.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_csv(source_dir: Path, relative: str, **kwargs) -> pd.DataFrame:
    path = source_dir / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def panel_label(ax: plt.Axes, label: str, x: float = -0.09, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.8, color=COLORS["text"])


def short_method(method: str) -> str:
    mapping = {
        "EfficientNetV2-S": "EffNetV2-S",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny",
        "RA-LPT fusion": "RA-LPT",
        "Reliability-aware LPT fusion": "RA-LPT",
        "LPT ablations + prior stats": "RA-LPT",
        "Pure cls ensemble": "RGB ensemble",
        "Pure classification ensemble": "RGB ensemble",
        "Cls + prior residual": "RGB + prior residual",
        "Strong cls ensemble": "Strong RGB stack",
        "Strong classification ensemble": "Strong RGB stack",
        "Strong cls + prior residual": "Strong RGB + prior",
        "Retained strong RGB comparator": "Retained strong RGB",
        "Retained strong RGB + prior-informed residual comparator": "Retained RGB + prior",
        "LPT output ensemble": "LPT output",
        "Strong LPT output ensemble": "Strong LPT output",
        "Strong RA-LPT fusion": "Strong RA-LPT",
        "Selected prior-informed fusion": "Selected prior-informed",
        "Selected prior-informed probability-fusion candidate": "Selected prior-informed",
    }
    return mapping.get(method, method)


def method_color(method: str) -> str:
    lower = method.lower()
    if "selected" in lower:
        return COLORS["selected"]
    if "random" in lower or "full lpt" in lower:
        return COLORS["negative"]
    if "prior" in lower or "lpt" in lower or "ra-" in lower:
        return COLORS["prior"]
    return COLORS["rgb"]


def save_all(
    fig: plt.Figure,
    outdir: Path,
    stem: str,
    dpi: int,
    manifest: list[dict],
    *,
    source: str,
    note: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    exports = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = outdir / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        exports.append(path.name)
    plt.close(fig)
    manifest.append({"figure": stem, "source": source, "note": note, "exports": exports})


def add_value(ax: plt.Axes, x: float, y: float, value: float, *, fmt: str = ".3f", color: str | None = None) -> None:
    ax.text(x, y, format(value, fmt), ha="left", va="center", fontsize=7.0, color=color or COLORS["text"])


def figure_u01_grouped_performance(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "publication_final_metrics.csv")
    order = [
        "Selected prior-informed fusion",
        "Strong cls + prior residual",
        "Strong cls ensemble",
        "ConvNeXt-Tiny",
        "EfficientNetV2-S",
        "RA-LPT fusion",
        "Full LPT",
        "ResNet50",
    ]
    frame = frame.set_index("method").loc[order].reset_index()
    y = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    bars = ax.barh(
        y,
        values,
        color=[method_color(v) for v in frame["method"]],
        height=0.62,
        edgecolor="white",
        linewidth=0.6,
    )
    for idx, row in frame.iterrows():
        std = row.get("macro_f1_std")
        if pd.notna(std) and float(std) > 0:
            ax.errorbar(
                float(row["macro_f1"]),
                idx,
                xerr=float(std),
                fmt="none",
                ecolor=COLORS["text"],
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=4,
            )
        label_x = float(row["macro_f1"]) + (float(std) if pd.notna(std) else 0.0) + 0.0008
        add_value(ax, label_x, idx, float(row["macro_f1"]))
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.invert_yaxis()
    ax.set_xlim(0.900, 0.972)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Similarity-grouped internal validation", pad=8)
    clean_axis(ax)
    ax.text(
        0.99,
        0.015,
        "Error bars: SD across five grouped runs where available; other rows are grouped OOF/internal estimates",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=COLORS["neutral"],
    )
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.18, top=0.88)
    save_all(
        fig,
        outdir,
        "U01_grouped_performance",
        dpi,
        manifest,
        source="publication_final_metrics.csv",
        note="Grouped performance hierarchy with repeated-run SD shown only where available.",
    )


def figure_u02_group_bootstrap(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "methodology_audit_v7/publication_final_group_level_bootstrap.csv")
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    estimate = frame["observed_delta"].astype(float).to_numpy()
    low = frame["ci95_low"].astype(float).to_numpy()
    high = frame["ci95_high"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for ypos, est, lo, hi in zip(y, estimate, low, high):
        ax.plot([lo, hi], [ypos, ypos], color=COLORS["prior"], linewidth=1.6)
        ax.plot(est, ypos, "o", color=COLORS["selected"], markersize=4.5)
    ax.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v.replace(" grouped", "")) for v in frame["baseline"]])
    ax.set_xlabel("Observed Macro-F1 difference: selected candidate minus comparator")
    ax.set_title("Group-level paired-bootstrap sensitivity", pad=8)
    x_pad = max(estimate) * 0.025
    for ypos, est, pvalue in zip(y, estimate, frame["two_sided_p"].astype(float)):
        ax.text(est + x_pad, ypos + 0.13, f"p={pvalue:.4f}", fontsize=6.7, color=COLORS["text"])
    ax.set_xlim(-0.003, max(high) + 0.009)
    clean_axis(ax)
    ax.text(
        0.99,
        0.03,
        "5,000 resamples of 5,670 reconstructed similarity groups",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.4,
        color=COLORS["neutral"],
    )
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.21, top=0.84)
    save_all(
        fig,
        outdir,
        "U02_group_level_bootstrap",
        dpi,
        manifest,
        source="methodology_audit_v7/publication_final_group_level_bootstrap.csv",
        note="Observed differences with group-level percentile confidence intervals and two-sided p values.",
    )


def figure_u03_selected_summary(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    cm = read_csv(source_dir, "publication_final_confusion_matrix.csv", index_col=0)
    matrix = cm.to_numpy(dtype=float)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    per_class = read_csv(source_dir, "publication_final_per_class_metrics.csv")
    binary = read_csv(source_dir, "publication_final_binary_endpoint_auc_auprc.csv")
    cmap = LinearSegmentedColormap.from_list("dfu_blues", ["#FFFFFF", "#DCE8F3", COLORS["selected"]])

    fig = plt.figure(figsize=(7.25, 3.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 0.95], wspace=0.47)

    ax0 = fig.add_subplot(gs[0, 0])
    im = ax0.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
    ax0.set_xticks(np.arange(4))
    ax0.set_yticks(np.arange(4))
    ax0.set_xticklabels(CLASS_NAMES, rotation=32, ha="right")
    ax0.set_yticklabels(CLASS_NAMES)
    ax0.set_xlabel("Predicted class")
    ax0.set_ylabel("True class")
    ax0.set_title("Confusion matrix")
    for i in range(4):
        for j in range(4):
            color = COLORS["white"] if normalized[i, j] > 0.55 else COLORS["text"]
            ax0.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j] * 100:.1f}%", ha="center", va="center", fontsize=5.7, color=color)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.047, pad=0.03)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.set_label("Row proportion", fontsize=6.3)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(per_class))
    width = 0.24
    for offset, column, label, color in (
        (-width, "precision", "Precision", COLORS["rgb_light"]),
        (0.0, "recall", "Recall", COLORS["prior"]),
        (width, "f1", "F1", COLORS["selected"]),
    ):
        vals = per_class[column].astype(float).to_numpy()
        bars = ax1.bar(x + offset, vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.4)
        for bar, value in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.0008, f"{value:.3f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax1.set_xticks(x)
    ax1.set_xticklabels(per_class["class"], rotation=28, ha="right")
    ax1.set_ylim(0.93, 1.004)
    ax1.set_ylabel("Score")
    ax1.set_title("Per-class performance")
    ax1.legend(loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.0, bbox_to_anchor=(0.0, 1.02))
    clean_axis(ax1)
    panel_label(ax1, "b")

    ax2 = fig.add_subplot(gs[0, 2])
    endpoints = [v.replace(" detection", "") for v in binary["Endpoint"]]
    y = np.arange(len(binary))
    height = 0.31
    auroc = binary["AUROC"].astype(float).to_numpy()
    auprc = binary["AUPRC"].astype(float).to_numpy()
    ax2.barh(y - height / 2, auroc, height=height, label="AUROC", color=COLORS["rgb"])
    ax2.barh(y + height / 2, auprc, height=height, label="AUPRC", color=COLORS["prior"])
    for ypos, value in zip(y - height / 2, auroc):
        ax2.text(value + 0.00045, ypos, f"{value:.4f}", va="center", fontsize=5.8)
    for ypos, value in zip(y + height / 2, auprc):
        ax2.text(value + 0.00045, ypos, f"{value:.4f}", va="center", fontsize=5.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(endpoints)
    ax2.set_xlim(0.975, 1.002)
    ax2.set_xlabel("Score")
    ax2.set_title("Binary endpoints")
    ax2.legend(loc="lower right")
    clean_axis(ax2)
    panel_label(ax2, "c")

    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.84)
    save_all(
        fig,
        outdir,
        "U03_selected_candidate_summary",
        dpi,
        manifest,
        source="publication_final_confusion_matrix.csv; publication_final_per_class_metrics.csv; publication_final_binary_endpoint_auc_auprc.csv",
        note="Selected internally optimised grouped OOF candidate; binary endpoints are derived from four-class probabilities.",
    )


def figure_u04_lpt_ablation(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "table_lpt_spatial_ablations.csv")
    order = ["LPT no alignment", "LPT no prior modulation", "LPT no ROI", "LPT stats only", "Full LPT", "LPT random prior"]
    frame = frame.set_index("method").loc[order].reset_index()
    y = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()
    errors = frame["macro_f1_std"].astype(float).to_numpy()
    colors = [COLORS["prior"] if method not in {"Full LPT", "LPT random prior"} else COLORS["negative"] for method in frame["method"]]

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white", linewidth=0.5)
    ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=COLORS["text"], elinewidth=1.0, capsize=2.6, capthick=1.0)
    for ypos, value, error in zip(y, values, errors):
        add_value(ax, value + error + 0.0008, ypos, value)
    reference = 0.9442
    ax.axvline(reference, color=COLORS["selected"], linestyle=(0, (3, 2)), linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.invert_yaxis()
    ax.set_xlim(0.900, 0.960)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Direct LPT ablation in standard five-fold CV", pad=8)
    clean_axis(ax)
    fig.text(
        0.28,
        0.055,
        "Dashed line: ResNet50 standard CV = 0.9442",
        color=COLORS["selected"],
        fontsize=6.3,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.98,
        0.055,
        "Bars: fold mean; error bars: fold SD",
        color=COLORS["neutral"],
        fontsize=6.3,
        ha="right",
    )
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.23, top=0.86)
    save_all(
        fig,
        outdir,
        "U04_direct_lpt_ablation",
        dpi,
        manifest,
        source="table_lpt_spatial_ablations.csv",
        note="Standard five-fold CV; direct prior-injection variants remain below the ResNet50 reference.",
    )


def figure_u05_ra_lpt_support(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "table_reliability_fusion.csv")
    order = ["Pure cls ensemble", "Cls + prior residual", "LPT output ensemble", "LPT ablations + prior stats"]
    frame = frame.set_index("method").loc[order].reset_index()
    paired = read_csv(source_dir, "paired_bootstrap_significance.csv")
    paired = paired[(paired["candidate"] == "LPT ablations + prior stats") & (paired["metric"] == "macro_f1")].copy()
    paired_order = ["ResNet50", "Pure cls ensemble", "LPT output ensemble"]
    paired = paired.set_index("baseline").loc[paired_order].reset_index().iloc[::-1]

    fig = plt.figure(figsize=(7.25, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.46)
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()
    errors = frame["macro_f1_std"].astype(float).to_numpy()
    ax0.bar(x, values, color=[method_color(v) for v in frame["method"]], width=0.62, edgecolor="white", linewidth=0.5)
    ax0.errorbar(x, values, yerr=errors, fmt="none", ecolor=COLORS["text"], elinewidth=1.0, capsize=2.5, capthick=1.0)
    for xpos, value in zip(x, values):
        ax0.text(xpos, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3, rotation=90)
    ax0.set_xticks(x)
    ax0.set_xticklabels([short_method(v) for v in frame["method"]], rotation=27, ha="right")
    ax0.set_ylim(0.938, 0.970)
    ax0.set_ylabel("Macro-F1")
    ax0.set_title("Reliability-aware stacking")
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(paired))
    estimate = paired["observed_delta"].astype(float).to_numpy()
    low = paired["ci95_low"].astype(float).to_numpy()
    high = paired["ci95_high"].astype(float).to_numpy()
    for ypos, est, lo, hi in zip(y, estimate, low, high):
        ax1.plot([lo, hi], [ypos, ypos], color=COLORS["prior"], linewidth=1.5)
        ax1.plot(est, ypos, "o", color=COLORS["selected"], markersize=4.3)
    ax1.axvline(0, color=COLORS["neutral"], linestyle="--", linewidth=0.9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([short_method(v) for v in paired["baseline"]])
    ax1.set_xlabel("Observed Macro-F1 difference vs RA-LPT")
    ax1.set_title("Paired bootstrap vs RA-LPT")
    for ypos, est, pvalue in zip(y, estimate, paired["two_sided_p"].astype(float)):
        ax1.text(est + 0.0005, ypos + 0.13, f"p={pvalue:.3f}", fontsize=6.3)
    ax1.set_xlim(-0.004, 0.019)
    clean_axis(ax1)
    panel_label(ax1, "b")

    fig.text(0.985, 0.02, "Standard five-fold support analysis; 2,000 pooled-OOF paired-bootstrap resamples", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.25, top=0.84)
    save_all(
        fig,
        outdir,
        "U05_ra_lpt_support",
        dpi,
        manifest,
        source="table_reliability_fusion.csv; paired_bootstrap_significance.csv",
        note="RA-LPT standard-CV support analysis kept separate from the selected grouped probability-fusion estimate.",
    )


def _find_metric(frame: pd.DataFrame, method: str, group_type: str, group: str) -> float:
    row = frame[(frame["method"] == method) & (frame["group_type"] == group_type) & (frame["group"] == group)]
    if row.empty:
        return float("nan")
    return float(row["macro_f1"].iloc[0])


def figure_u06_prior_reliability(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    base = read_csv(source_dir, "prior_reliability_strata_metrics.csv")
    strong = read_csv(source_dir, "strong_prior_reliability_strata_metrics.csv")
    combined = pd.concat([base, strong], ignore_index=True).drop_duplicates(subset=["method", "group_type", "group"], keep="last")

    methods_a = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong cls ensemble", "Strong LPT output ensemble"]
    methods_b = ["ResNet50", "Reliability-aware LPT fusion", "Strong cls ensemble", "Strong LPT output ensemble"]
    fig = plt.figure(figsize=(7.25, 3.55))
    gs = fig.add_gridspec(1, 2, wspace=0.36)

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods_a))
    width = 0.34
    fallback = [_find_metric(combined, m, "fallback_group", "fallback") for m in methods_a]
    nonfallback = [_find_metric(combined, m, "fallback_group", "non_fallback") for m in methods_a]
    ax0.bar(x - width / 2, fallback, width=width, color=COLORS["neutral"], label="Fallback")
    ax0.bar(x + width / 2, nonfallback, width=width, color=COLORS["prior"], label="Non-fallback")
    ax0.set_xticks(x)
    ax0.set_xticklabels([short_method(v) for v in methods_a], rotation=27, ha="right")
    ax0.set_ylim(0.88, 0.985)
    ax0.set_ylabel("Macro-F1")
    ax0.set_title("Fallback status")
    ax0.legend(loc="upper left", ncol=2)
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    x2 = np.arange(len(methods_b))
    width2 = 0.25
    groups = [
        ("fallback", "Fallback", COLORS["neutral"]),
        ("nonfallback_low_quality", "Low quality", COLORS["prior_light"]),
        ("nonfallback_high_quality", "High quality", COLORS["prior"]),
    ]
    for idx, (group, label, color) in enumerate(groups):
        vals = [_find_metric(combined, m, "quality_group", group) for m in methods_b]
        ax1.bar(x2 + (idx - 1) * width2, vals, width=width2, color=color, label=label)
    ax1.set_xticks(x2)
    ax1.set_xticklabels([short_method(v) for v in methods_b], rotation=27, ha="right")
    ax1.set_ylim(0.88, 0.985)
    ax1.set_title("Prior-quality strata")
    ax1.legend(loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.0)
    clean_axis(ax1)
    panel_label(ax1, "b")

    fig.text(0.985, 0.02, "Descriptive pooled-OOF stratification; association does not establish causality", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.84)
    save_all(
        fig,
        outdir,
        "U06_prior_reliability_strata",
        dpi,
        manifest,
        source="prior_reliability_strata_metrics.csv; strong_prior_reliability_strata_metrics.csv",
        note="Fallback and quality strata are descriptive difficulty markers, not causal effects.",
    )


def figure_u07_interpretability(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    summary = read_csv(source_dir, "cam_prior_occlusion_summary.csv")
    subset = summary[(summary["group_type"] == "fallback") & (summary["method"].isin(["ResNet50", "EfficientNetV2-S"]))].copy()
    metrics = [
        ("cam_prior_mass_ratio", "CAM-prior mass", (0, 0.62)),
        ("pointing_game_accuracy", "Pointing-game accuracy", (0, 0.80)),
        ("prior_minus_random_drop", "Prior minus random drop", (0, 0.115)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.95))
    for index, (ax, (column, title, ylim)) in enumerate(zip(axes, metrics)):
        pivot = subset.pivot(index="method", columns="group", values=column).loc[["ResNet50", "EfficientNetV2-S"]]
        x = np.arange(len(pivot))
        width = 0.34
        ax.bar(x - width / 2, pivot["fallback"], width=width, label="Fallback", color=COLORS["neutral"])
        ax.bar(x + width / 2, pivot["non_fallback"], width=width, label="Non-fallback", color=COLORS["prior"])
        ax.set_xticks(x)
        ax.set_xticklabels([short_method(v) for v in pivot.index], rotation=24, ha="right")
        ax.set_ylim(*ylim)
        ax.set_title(title)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    axes[0].legend(loc="upper left")
    fig.text(0.985, 0.02, "Post hoc interpretability analysis; priors are auxiliary maps, not DFUC2021 masks", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.82, wspace=0.36)
    save_all(
        fig,
        outdir,
        "U07_interpretability_summary",
        dpi,
        manifest,
        source="cam_prior_occlusion_summary.csv",
        note="Non-fallback priors align more strongly with attention and carry more occlusion signal than fallback priors.",
    )


def figure_u08_overlap_sensitivity(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    base = "cross_dataset_overlap_exclusion_sensitivity_v1/"
    metrics = read_csv(source_dir, base + "cross_dataset_overlap_exclusion_overall_metrics.csv")
    diffs = read_csv(source_dir, base + "cross_dataset_overlap_exclusion_model_differences.csv")
    model_order = [
        "selected_prior_informed_probability_fusion",
        "retained_strong_rgb_logistic_stack",
        "retained_strong_rgb_prior_feature_stack",
    ]
    model_colors = [COLORS["selected"], COLORS["rgb"], COLORS["prior"]]

    fig = plt.figure(figsize=(7.25, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.13, 0.87], wspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    for key, color in zip(model_order, model_colors):
        sub = metrics[metrics["model_key"] == key]
        ax0.plot(x, sub["macro_f1"].astype(float), marker="o", markersize=4.2, linewidth=1.5, color=color, label=short_method(sub["model"].iloc[0]))
    ax0.set_xticks(x)
    ax0.set_xticklabels(["Full grouped OOF", "Exclude 241 broad-overlap images"])
    ax0.set_ylim(0.952, 0.966)
    ax0.set_ylabel("Macro-F1")
    ax0.set_title("Broad visual-overlap exclusion")
    ax0.legend(loc="lower right", fontsize=6.2)
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    keep = diffs[diffs["scope"] == "exclude_broad_cross_dataset_overlap_241"].copy().iloc[::-1]
    y = np.arange(len(keep))
    vals = keep["macro_f1_delta"].astype(float).to_numpy()
    ax1.barh(y, vals, color=[COLORS["prior"], COLORS["rgb"]][: len(keep)], height=0.55)
    ax1.set_yticks(y)
    ax1.set_yticklabels([short_method(v) for v in keep["comparator"]])
    ax1.set_xlim(0, max(vals) + 0.002)
    ax1.set_xlabel("Macro-F1 difference")
    ax1.set_title("Selected minus comparator")
    for ypos, value in zip(y, vals):
        add_value(ax1, value + 0.0002, ypos, value, fmt=".4f")
    clean_axis(ax1)
    panel_label(ax1, "b")

    fig.text(0.985, 0.02, "Prediction-level sensitivity only; no retraining or fusion reselection", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.83)
    save_all(
        fig,
        outdir,
        "U08_cross_dataset_overlap_sensitivity",
        dpi,
        manifest,
        source=base + "cross_dataset_overlap_exclusion_overall_metrics.csv; " + base + "cross_dataset_overlap_exclusion_model_differences.csv",
        note="Excluding the 241 broad-overlap images did not reverse model ranking or the positive selected-candidate difference.",
    )


def figure_u09_discrimination(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "roc_pr_auc_pivot.csv")
    frame = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(frame))
    auroc = frame["roc_auc_macro"].astype(float).to_numpy()
    auprc = frame["average_precision_macro"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    for ypos, roc, pr in zip(y, auroc, auprc):
        ax.plot([pr, roc], [ypos, ypos], color=COLORS["neutral_light"], linewidth=2.0, zorder=1)
        ax.plot(roc, ypos, "o", color=COLORS["rgb"], markersize=5, label="Macro AUROC" if ypos == 0 else None)
        ax.plot(pr, ypos, "o", color=COLORS["prior"], markersize=5, label="Macro AUPRC" if ypos == 0 else None)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.set_xlim(0.978, 0.997)
    ax.set_xlabel("Score")
    ax.set_title("Standard-CV macro one-vs-rest discrimination", pad=8)
    handles, labels = ax.get_legend_handles_labels()
    clean_axis(ax)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.66, 0.062), ncol=2, frameon=False)
    fig.text(0.985, 0.018, "Summary metrics only; full ROC/PR curves require the original OOF probabilities", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.30, top=0.84)
    save_all(
        fig,
        outdir,
        "U09_discrimination_summary",
        dpi,
        manifest,
        source="roc_pr_auc_pivot.csv",
        note="Standard-CV support-model AUROC/AUPRC summary; legacy curve images were not redrawn without raw OOF probabilities.",
    )


def _error_label(row: pd.Series) -> str:
    return f"{row['true_class']} -> {row['predicted_class']}"


def _contrast_short(value: str) -> str:
    mapping = {
        "RA-LPT correct, ResNet50 wrong": "RA-LPT correct\nResNet50 wrong",
        "RA-LPT correct, EfficientNetV2-S wrong": "RA-LPT correct\nEffNetV2-S wrong",
        "EfficientNetV2-S correct, RA-LPT wrong": "EffNetV2-S correct\nRA-LPT wrong",
        "Strong cls correct, RA-LPT wrong": "Strong RGB correct\nRA-LPT wrong",
        "Strong RA-LPT correct, Strong cls wrong": "Strong RA-LPT correct\nStrong RGB wrong",
        "Strong cls correct, Strong RA-LPT wrong": "Strong RGB correct\nStrong RA-LPT wrong",
    }
    return mapping.get(value, value)


def figure_u10_error_patterns(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    errors = read_csv(source_dir, "publication_final_error_patterns.csv").head(8).iloc[::-1]
    cases = read_csv(source_dir, "strong_error_case_summary.csv")
    cases = cases[~cases["contrast"].isin(["All selected methods correct", "All selected methods wrong"])].copy().iloc[::-1]

    fig = plt.figure(figsize=(7.25, 3.65))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 1.18], wspace=0.72)
    ax0 = fig.add_subplot(gs[0, 0])
    y = np.arange(len(errors))
    counts = errors["count"].astype(int).to_numpy()
    ax0.barh(y, counts, color=COLORS["negative"], height=0.60)
    ax0.set_yticks(y)
    ax0.set_yticklabels([_error_label(row) for _, row in errors.iterrows()])
    ax0.set_xlabel("Error count")
    ax0.set_title("Selected-candidate residual errors")
    for ypos, value in zip(y, counts):
        ax0.text(value + 1.5, ypos, str(value), va="center", fontsize=6.6)
    ax0.set_xlim(0, max(counts) * 1.18)
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    y2 = np.arange(len(cases))
    height = 0.32
    fallback = cases["fallback_ratio"].astype(float).to_numpy()
    quality = cases["prior_quality_mean"].astype(float).to_numpy()
    ax1.barh(y2 - height / 2, fallback, height=height, color=COLORS["neutral"], label="Fallback ratio")
    ax1.barh(y2 + height / 2, quality, height=height, color=COLORS["prior"], label="Prior quality")
    ax1.set_yticks(y2)
    ax1.set_yticklabels([_contrast_short(v) for v in cases["contrast"]])
    ax1.tick_params(axis="y", labelsize=6.2)
    ax1.set_xlim(0, 0.82)
    ax1.set_xlabel("Proportion / mean")
    ax1.set_title("Difficult-case prior characteristics")
    ax1.legend(loc="lower right")
    clean_axis(ax1)
    panel_label(ax1, "b")

    fig.text(0.985, 0.02, "Pairwise disagreement rows are selected subsets, not the full dataset", ha="right", fontsize=6.2, color=COLORS["neutral"])
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.84)
    save_all(
        fig,
        outdir,
        "U10_error_patterns",
        dpi,
        manifest,
        source="publication_final_error_patterns.csv; strong_error_case_summary.csv",
        note="Residual-error and disagreement-subset analysis; subset definitions must remain explicit.",
    )


def parse_candidate_components(candidate: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in candidate.split("+"):
        item = part.strip()
        if "*" not in item:
            continue
        weight, name = item.split("*", 1)
        out[name.strip()] = float(weight.strip())
    return out


def component_short(name: str) -> str:
    if "strong_lpt_full_output" in name:
        return "Strong LPT output"
    if "convnext" in name:
        return "ConvNeXt-Tiny"
    if "efficientnetv2s" in name:
        return "EffNetV2-S"
    return name


def figure_u11_fusion_trace(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    fusion_dir = source_dir / "fusion_search"
    if not fusion_dir.exists():
        fusion_dir = source_dir.parent / "fusion_search"
    search = read_csv(fusion_dir, "publication_baseline5_full_oof_fusion_search.csv")
    trace = read_csv(fusion_dir, "publication_selected_prior_informed_candidate_trace.csv").iloc[0]
    ranked = search.head(40).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    weights = parse_candidate_components(str(trace["candidate"]))

    fig = plt.figure(figsize=(7.25, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 0.94], wspace=0.42)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ranked["rank"], ranked["macro_f1"], color=COLORS["rgb"], linewidth=1.5)
    ax0.scatter([int(trace["overall_rank"])], [float(trace["macro_f1"])], s=28, color=COLORS["selected"], zorder=3)
    ax0.annotate(
        "selected prior-informed\ncandidate",
        xy=(int(trace["overall_rank"]), float(trace["macro_f1"])),
        xytext=(int(trace["overall_rank"]) + 4, float(trace["macro_f1"]) - 0.00024),
        fontsize=6.4,
        color=COLORS["selected"],
        arrowprops={"arrowstyle": "-", "color": COLORS["selected"], "linewidth": 0.8},
    )
    ax0.set_xlim(1, 40)
    ax0.set_ylim(0.9628, 0.96445)
    ax0.set_xlabel("Rank among full-OOF fusion candidates")
    ax0.set_ylabel("Macro-F1")
    ax0.set_title("Fusion-search trace")
    clean_axis(ax0)
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    labels = [component_short(v) for v in weights]
    vals = list(weights.values())
    x = np.arange(len(vals))
    colors = [COLORS["prior"] if "LPT" in label else COLORS["rgb"] for label in labels]
    ax1.bar(x, vals, color=colors, width=0.58, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=24, ha="right")
    ax1.set_ylim(0, 0.47)
    ax1.set_ylabel("Selected weight")
    ax1.set_title("Selected candidate composition")
    for xpos, value in zip(x, vals):
        ax1.text(xpos, value + 0.014, f"{value:.3f}", ha="center", fontsize=6.8)
    clean_axis(ax1)
    panel_label(ax1, "b")

    fig.text(
        0.985,
        0.050,
        f"Macro-F1 {float(trace['macro_f1']):.4f}; overall rank {int(trace['overall_rank'])}; rank 1 among strong-LPT-output candidates",
        ha="right",
        fontsize=6.1,
        color=COLORS["neutral"],
    )
    fig.text(0.985, 0.018, "Internally optimised grouped OOF estimate; no independent tuning set", ha="right", fontsize=6.2, color=COLORS["negative"])
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.28, top=0.84)
    save_all(
        fig,
        outdir,
        "U11_fusion_search_trace",
        dpi,
        manifest,
        source="fusion_search/publication_baseline5_full_oof_fusion_search.csv; fusion_search/publication_selected_prior_informed_candidate_trace.csv",
        note="Search ranking and selected component weights with internal-selection limitation explicit.",
    )


def make_contact_sheet(outdir: Path, manifest: list[dict]) -> None:
    paths = [outdir / f"{row['figure']}.png" for row in manifest]
    thumbs: list[Image.Image] = []
    target_w = 1200
    label_h = 56
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = target_w / image.width
        resized = image.resize((target_w, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (target_w, resized.height + label_h), "white")
        canvas.paste(resized, (0, 0))
        ImageDraw.Draw(canvas).text((12, resized.height + 16), path.stem, fill=COLORS["text"])
        thumbs.append(canvas)
    columns = 2
    gutter = 36
    rows = math.ceil(len(thumbs) / columns)
    cell_h = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (columns * target_w + gutter, rows * cell_h + (rows - 1) * gutter), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * (target_w + gutter)
        y = (index // columns) * (cell_h + gutter)
        sheet.paste(thumb, (x, y))
    sheet.save(outdir / "nature_unified_extra_figures_contact_sheet.png", dpi=(300, 300))


def write_manifest(outdir: Path, manifest: list[dict]) -> None:
    replacement_map = {
        "E1": "U01",
        "E2": "U02",
        "E3": "U03",
        "E4/L1": "U04",
        "E5/L2/L3": "U05",
        "E6/L7/L8": "U06",
        "E7": "U07",
        "E8": "U08",
        "E9": "U09",
        "L9/L10/L11": "U10",
        "formal fusion-search detail": "U11",
        "L4/L5/L6 full ROC/PR curves": "not redrawn; raw OOF probabilities or per-threshold curve points required",
    }
    payload = {
        "figure_contract": {
            "backend": "Python/matplotlib",
            "palette": COLORS,
            "validation_boundary": "Internal validation only; selected fusion is an internally optimised grouped OOF estimate.",
            "exports": ["SVG", "PDF", "600 dpi PNG", "600 dpi TIFF"],
        },
        "replacement_map": replacement_map,
        "figures": manifest,
    }
    (outdir / "nature_unified_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Nature-style unified extra-figure manifest",
        "",
        "All quantitative figures were regenerated from audited source tables with one shared palette and typography contract.",
        "",
        "| Figure | Source | Note |",
        "| --- | --- | --- |",
    ]
    for row in manifest:
        lines.append(f"| {row['figure']} | {row['source']} | {row['note']} |")
    lines.extend(["", "## Replacement map", "", "| Previous figure(s) | Unified figure |", "| --- | --- |"])
    for old, new in replacement_map.items():
        lines.append(f"| {old} | {new} |")
    (outdir / "nature_unified_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    source_dir = args.source_dir.resolve()
    outdir = args.output_dir.resolve()
    manifest: list[dict] = []
    builders: list[Callable[[Path, Path, int, list[dict]], None]] = [
        figure_u01_grouped_performance,
        figure_u02_group_bootstrap,
        figure_u03_selected_summary,
        figure_u04_lpt_ablation,
        figure_u05_ra_lpt_support,
        figure_u06_prior_reliability,
        figure_u07_interpretability,
        figure_u08_overlap_sensitivity,
        figure_u09_discrimination,
        figure_u10_error_patterns,
        figure_u11_fusion_trace,
    ]
    for builder in builders:
        builder(source_dir, outdir, args.dpi, manifest)
    make_contact_sheet(outdir, manifest)
    write_manifest(outdir, manifest)
    print(f"Wrote {len(manifest)} unified Nature-style figures to {outdir}")


if __name__ == "__main__":
    main()
