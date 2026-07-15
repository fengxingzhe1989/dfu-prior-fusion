#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "navy": "#153562",
    "blue": "#5278B2",
    "teal": "#2F9188",
    "teal_light": "#79B9B3",
    "purple": "#735FA7",
    "purple_light": "#A090C5",
    "red": "#BD5050",
    "gray": "#6B7280",
    "gray_light": "#D7DCE5",
    "text": "#111827",
    "white": "#FFFFFF",
}


MATCHED_LABELS = {
    "resnet50_grouped": "ResNet50",
    "full_lpt_grouped": "Full LPT",
    "lpt_output_only_grouped": "Moderate output-only\nLPT stack",
    "ra_lpt_grouped": "RA-LPT",
    "strong_rgb_grouped": "Strong RGB stack",
    "strong_rgb_prior_grouped": "Strong RGB + explicit\nprior features",
}


MATCHED_COLORS = {
    "resnet50_grouped": COLORS["blue"],
    "full_lpt_grouped": COLORS["red"],
    "lpt_output_only_grouped": COLORS["teal_light"],
    "ra_lpt_grouped": COLORS["teal"],
    "strong_rgb_grouped": COLORS["purple_light"],
    "strong_rgb_prior_grouped": COLORS["purple"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild manuscript Figure 3 from audited source tables.")
    root = Path(__file__).resolve().parents[2]
    source = root / "paper_assets" / "source_tables"
    parser.add_argument("--source-dir", type=Path, default=source)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "outputs" / "manuscript_figures",
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
            "font.size": 7.5,
            "axes.titlesize": 8.7,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.10) -> None:
    ax.text(
        x,
        1.055,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=COLORS["text"],
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=3.5)


def save_all(fig: plt.Figure, output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "Figure3_validation_aware_model_comparisons"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def build_figure(source_dir: Path, output_dir: Path, dpi: int) -> None:
    revision_dir = source_dir / "reviewer_revision_v3" / "derived"
    matched = pd.read_csv(revision_dir / "matched_grouped_model_metrics.csv")
    repeated = pd.read_csv(revision_dir / "grouped_rgb_repeated_run_metrics.csv")
    bootstrap = pd.read_csv(revision_dir / "matched_grouped_bootstrap.csv")
    ablation = pd.read_csv(source_dir / "table_lpt_spatial_ablations.csv")

    matched_order = [
        "resnet50_grouped",
        "full_lpt_grouped",
        "lpt_output_only_grouped",
        "ra_lpt_grouped",
        "strong_rgb_grouped",
        "strong_rgb_prior_grouped",
    ]
    matched = matched.set_index("method").loc[matched_order].reset_index()

    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.03, 0.97])

    # a. Hero panel: matched grouped OOF point estimates.
    ax = fig.add_subplot(grid[0, 0])
    y = np.arange(len(matched))
    values = matched["macro_f1"].to_numpy(float)
    colors = [MATCHED_COLORS[method] for method in matched["method"]]
    bars = ax.barh(y, values, height=0.60, color=colors, edgecolor="white", linewidth=0.6)
    ax.invert_yaxis()
    for bar, value in zip(bars, values):
        ax.text(value + 0.0007, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=7.1)
    ax.set_yticks(y)
    ax.set_yticklabels([MATCHED_LABELS[method] for method in matched["method"]])
    ax.set_xlim(0.900, 0.958)
    ax.set_xticks([0.90, 0.92, 0.94, 0.96])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Matched similarity-grouped OOF estimates", loc="left", pad=7)
    ax.axhline(1.5, color=COLORS["gray_light"], linewidth=0.7)
    ax.axhline(3.5, color=COLORS["gray_light"], linewidth=0.7)
    clean_axis(ax)
    panel_label(ax, "a", x=-0.13)

    # b. Repeated grouped RGB runs.
    ax = fig.add_subplot(grid[0, 1])
    summary = repeated.loc[repeated["row_type"] == "summary"].copy()
    repeat_order = ["ResNet50", "ConvNeXt-Tiny", "EfficientNetV2-S"]
    summary = summary.set_index("model").loc[repeat_order].reset_index()
    y = np.arange(len(summary))
    means = summary["macro_f1"].to_numpy(float)
    sds = summary["macro_f1_sample_sd"].to_numpy(float)
    bars = ax.barh(
        y,
        means,
        xerr=sds,
        height=0.58,
        color=[COLORS["blue"], COLORS["navy"], COLORS["teal"]],
        edgecolor="white",
        linewidth=0.6,
        capsize=2.5,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": COLORS["text"]},
    )
    ax.invert_yaxis()
    for bar, mean, sd in zip(bars, means, sds):
        ax.text(
            mean + sd + 0.0010,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.4f} ± {sd:.4f}",
            va="center",
            fontsize=6.7,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(["ResNet50", "ConvNeXt-Tiny", "EfficientNetV2-S"])
    ax.set_xlim(0.900, 0.958)
    ax.set_xticks([0.90, 0.92, 0.94, 0.96])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Repeated grouped RGB runs", loc="left", pad=7)
    clean_axis(ax)
    panel_label(ax, "b", x=-0.14)

    # c. Matched paired group bootstrap comparisons.
    ax = fig.add_subplot(grid[1, 0])
    comparison_order = [
        ("resnet50_grouped", "full_lpt_grouped"),
        ("resnet50_grouped", "ra_lpt_grouped"),
        ("lpt_output_only_grouped", "ra_lpt_grouped"),
        ("strong_rgb_grouped", "strong_rgb_prior_grouped"),
    ]
    indexed = bootstrap.set_index(["baseline", "candidate"])
    rows = [indexed.loc[key] for key in comparison_order]
    deltas = np.array([row["observed_delta"] for row in rows], dtype=float)
    lows = np.array([row["ci95_low"] for row in rows], dtype=float)
    highs = np.array([row["ci95_high"] for row in rows], dtype=float)
    pvalues = np.array([row["two_sided_p"] for row in rows], dtype=float)
    y = np.arange(len(rows))
    labels = [
        "Full LPT vs ResNet50",
        "RA-LPT vs ResNet50",
        "RA-LPT vs moderate\noutput-only LPT stack",
        "Strong RGB + prior features\nvs strong RGB stack",
    ]
    point_colors = [COLORS["red"], COLORS["teal"], COLORS["teal"], COLORS["purple"]]
    for ypos, delta, low, high, color in zip(y, deltas, lows, highs, point_colors):
        ax.errorbar(
            delta,
            ypos,
            xerr=np.array([[delta - low], [high - delta]]),
            fmt="o",
            markersize=5.4,
            color=color,
            ecolor=color,
            capsize=2.5,
            elinewidth=1.2,
            capthick=1.0,
            zorder=3,
        )
    for ypos, high, pvalue in zip(y, highs, pvalues):
        ax.text(high + 0.0010, ypos, f"p = {pvalue:.3g}", va="center", fontsize=6.8)
    ax.axvline(0, color=COLORS["gray"], linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.015, 0.032)
    ax.set_xticks([-0.01, 0, 0.01, 0.02, 0.03])
    ax.set_xlabel("Observed Macro-F1 difference")
    ax.set_title("Matched group-level bootstrap comparisons", loc="left", pad=7)
    clean_axis(ax)
    panel_label(ax, "c", x=-0.13)

    # d. Standard-CV ablation panel to make validation-setting contrast explicit.
    ax = fig.add_subplot(grid[1, 1])
    method_order = [
        "LPT no alignment",
        "LPT no prior modulation",
        "LPT no ROI",
        "LPT stats only",
        "Full LPT",
        "LPT random prior",
    ]
    ablation = ablation.set_index("method").loc[method_order].reset_index()
    y = np.arange(len(ablation))
    means = ablation["macro_f1"].to_numpy(float)
    sds = ablation["macro_f1_std"].to_numpy(float)
    colors = [COLORS["teal"]] * 4 + [COLORS["red"], COLORS["red"]]
    bars = ax.barh(
        y,
        means,
        xerr=sds,
        height=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=2.2,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": COLORS["text"]},
    )
    ax.invert_yaxis()
    for bar, value in zip(bars, means):
        ax.text(
            0.865,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8.0,
            color=COLORS["white"],
            fontweight="bold",
        )
    reference = 0.9442
    ax.axvline(reference, color=COLORS["blue"], linestyle=(0, (2, 2)), linewidth=2.0)
    ax.text(
        0.949,
        2.7,
        "ResNet50 standard\nCV 0.9442",
        ha="left",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=COLORS["blue"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.6},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        ["No alignment", "No prior modulation", "No ROI branch", "Stats only", "Full LPT", "Random prior"]
    )
    ax.set_xlim(0.86, 0.98)
    ax.set_xticks([0.86, 0.90, 0.94, 0.98])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Direct LPT ablation in standard CV", loc="left", pad=7)
    clean_axis(ax)
    panel_label(ax, "d", x=-0.14)

    save_all(fig, output_dir, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    build_figure(args.source_dir, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
