#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageDraw


CLASS_NAMES = ("none", "infection", "ischaemia", "both")

COLORS = {
    "navy": "#102A56",
    "blue": "#4B6FA8",
    "blue_light": "#B9C7E6",
    "teal": "#2C8C84",
    "teal_light": "#BFE1DD",
    "purple": "#6D5FA7",
    "red": "#B64A4A",
    "red_light": "#EAC7C7",
    "gold": "#C7923E",
    "gray": "#6B7280",
    "gray_light": "#E6E8EF",
    "gray_mid": "#C9CEDA",
    "text": "#111827",
    "white": "#FFFFFF",
}


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    repo_root = script.parents[2]
    parser = argparse.ArgumentParser(
        description="Build supplementary figure candidates in the same style as the main manuscript figures."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "paper_assets" / "source_tables",
        help="Directory containing supplementary source-data tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "supplementary_figures",
        help="Output directory for style-aligned supplementary figures.",
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
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_csv(source_dir: Path, relative: str) -> pd.DataFrame:
    path = source_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing source table: {path}")
    return pd.read_csv(path)


def panel_label(ax: Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=10,
        color=COLORS["text"],
    )


def clean_axis(ax: Axes, grid_axis: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


def short_method(name: str) -> str:
    mapping = {
        "ResNet50": "ResNet50",
        "EfficientNetV2-S": "EffNetV2-S",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny",
        "Full LPT": "Full LPT",
        "LPT no alignment": "No alignment",
        "LPT no prior modulation": "No prior modulation",
        "LPT no ROI": "No ROI branch",
        "LPT stats only": "Stats only",
        "LPT random prior": "Random prior",
        "Random prior": "Random prior",
        "Pure cls ensemble": "RGB ensemble",
        "Pure classification ensemble": "RGB ensemble",
        "Cls + prior residual": "RGB + prior",
        "Classification plus prior residual": "RGB + prior",
        "LPT output ensemble": "LPT output",
        "LPT ablations + prior stats": "RA-LPT",
        "Reliability-aware LPT fusion": "RA-LPT",
        "Strong classification ensemble": "Strong RGB stack",
        "Strong cls ensemble": "Strong RGB stack",
        "Strong LPT output ensemble": "Strong LPT output",
        "Strong RA-LPT fusion": "Strong RA-LPT",
        "Retained strong RGB comparator": "Retained strong RGB",
        "Retained strong RGB + prior-informed residual comparator": "Retained RGB + prior",
        "Selected prior-informed probability-fusion candidate": "Selected prior-informed",
    }
    return mapping.get(name, name)


def method_color(name: str) -> str:
    lower = name.lower()
    if "selected" in lower:
        return COLORS["navy"]
    if "random" in lower or "full lpt" in lower or "direct" in lower:
        return COLORS["red"]
    if "prior" in lower or "lpt" in lower or "ra-" in lower:
        return COLORS["teal"]
    if "strong" in lower:
        return COLORS["purple"]
    return COLORS["blue"]


def save_all(fig: plt.Figure, outdir: Path, stem: str, dpi: int, manifest: list[dict]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    exports = []
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = outdir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        exports.append(path.name)
    plt.close(fig)
    manifest.append({"figure": stem, "exports": exports})


def schematic_box(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 6.6,
    weight: str = "normal",
    color: str | None = None,
    ha: str = "center",
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=ha,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color or COLORS["text"],
        linespacing=1.08,
    )
    return (x, y, w, h)


def box_anchor(
    box: tuple[float, float, float, float],
    side: str,
    *,
    frac: float = 0.5,
    inset: float = 0.018,
) -> tuple[float, float]:
    x, y, w, h = box
    if side == "left":
        return (x + inset, y + h * frac)
    if side == "right":
        return (x + w - inset, y + h * frac)
    if side == "top":
        return (x + w * frac, y + h - inset)
    if side == "bottom":
        return (x + w * frac, y + inset)
    raise ValueError(f"Unsupported side: {side}")


def box_arrow(
    ax: Axes,
    source: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    *,
    source_side: str = "right",
    target_side: str = "left",
    source_frac: float = 0.5,
    target_frac: float = 0.5,
    inset: float = 0.018,
    color: str = COLORS["gray"],
    lw: float = 1.05,
    rad: float = 0.0,
    linestyle: str = "-",
) -> None:
    start = box_anchor(source, source_side, frac=source_frac, inset=inset)
    end = box_anchor(target, target_side, frac=target_frac, inset=inset)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.2,
        linewidth=lw,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=f"arc3,rad={rad}",
        capstyle="butt",
        joinstyle="miter",
    )
    ax.add_patch(arrow)


def mini_bars(ax: Axes, x: float, y: float, w: float, h: float, color: str) -> None:
    vals = [0.35, 0.62, 0.48, 0.78]
    bar_w = w / 7
    for idx, val in enumerate(vals):
        bx = x + idx * bar_w * 1.45
        ax.add_patch(
            plt.Rectangle(
                (bx, y),
                bar_w,
                h * val,
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
            )
        )


def figure_s01_lpt_architecture(outdir: Path, dpi: int, manifest: list[dict]) -> None:
    navy = "#102A56"
    teal = "#207A74"
    amber = "#B37A2C"
    red = "#B64A4A"
    gray = "#6B7280"
    gray_light = "#D9DEE8"
    neutral_fill = "#F8FAFC"
    rgb_fill = "#EEF4FB"
    prior_fill = "#EAF7F5"
    aux_fill = "#FFF8EB"
    warn_fill = "#FFF1F1"

    fig = plt.figure(figsize=(7.55, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.08, 1.0],
        width_ratios=[1.0, 1.0],
        left=0.04,
        right=0.98,
        top=0.97,
        bottom=0.06,
        hspace=0.22,
        wspace=0.16,
    )
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "a", x=0.0, y=0.98)
    ax.text(0.04, 0.98, "Direct structured lesion-prior transfer", ha="left", va="top", fontsize=9.6, fontweight="bold", color=navy)
    ax.text(0.04, 0.89, "Three direct prior pathways are injected before four-class prediction", ha="left", va="top", fontsize=6.8, color=gray)

    # A: three clean lanes feeding one fusion head.
    rgb_box = schematic_box(ax, 0.04, 0.66, 0.12, 0.11, "RGB image", face=rgb_fill, edge=navy, weight="bold")
    global_branch_box = schematic_box(ax, 0.21, 0.66, 0.14, 0.11, "Global branch\nResNet50", face=neutral_fill, edge=navy, weight="bold")
    global_feature_box = schematic_box(ax, 0.41, 0.66, 0.12, 0.11, "Global\nfeature", face=neutral_fill, edge=navy, weight="bold")

    prior_box = schematic_box(ax, 0.04, 0.46, 0.12, 0.11, "Soft lesion\nprior M", face=prior_fill, edge=teal, weight="bold")
    prior_mod_box = schematic_box(ax, 0.21, 0.46, 0.14, 0.11, "Prior\nmodulation", face=prior_fill, edge=teal, weight="bold")
    guided_box = schematic_box(ax, 0.41, 0.46, 0.12, 0.11, "Guided\nfeature", face=prior_fill, edge=teal, weight="bold")

    roi_box = schematic_box(ax, 0.21, 0.25, 0.32, 0.11, "ROI crop + prior statistics", face=aux_fill, edge=amber, weight="bold")
    fusion_box = schematic_box(ax, 0.65, 0.44, 0.14, 0.18, "Feature\nfusion head", face=neutral_fill, edge=gray, weight="bold")
    output_box = schematic_box(ax, 0.86, 0.53, 0.10, 0.10, "Four-class\noutput", face=rgb_fill, edge=navy, weight="bold", fontsize=6.1)
    aux_box = schematic_box(ax, 0.86, 0.35, 0.10, 0.10, "Auxiliary /\nalignment\nheads", face=aux_fill, edge=amber, fontsize=5.8)

    arrow_kw = {"lw": 1.05}
    box_arrow(ax, rgb_box, global_branch_box, color=navy, **arrow_kw)
    box_arrow(ax, global_branch_box, global_feature_box, color=navy, **arrow_kw)
    box_arrow(ax, global_feature_box, fusion_box, target_frac=0.72, color=navy, **arrow_kw)
    box_arrow(ax, prior_box, prior_mod_box, color=teal, **arrow_kw)
    box_arrow(ax, prior_mod_box, guided_box, color=teal, **arrow_kw)
    box_arrow(ax, guided_box, fusion_box, target_frac=0.50, color=teal, **arrow_kw)
    box_arrow(ax, prior_box, roi_box, source_frac=0.20, target_frac=0.76, color=amber, **arrow_kw)
    box_arrow(ax, roi_box, fusion_box, target_frac=0.28, color=amber, **arrow_kw)
    box_arrow(ax, fusion_box, output_box, source_frac=0.70, color=navy, **arrow_kw)
    box_arrow(ax, fusion_box, aux_box, source_frac=0.30, color=amber, **arrow_kw)

    ax.text(0.04, 0.08, "Soft priors are auxiliary probability maps, not DFUC2021 ground-truth masks.", ha="left", va="center", fontsize=6.8, color=gray)

    ax = axes[1]
    panel_label(ax, "b", x=0.0, y=0.98)
    ax.text(0.06, 0.98, "Fallback prior handling", ha="left", va="top", fontsize=9.0, fontweight="bold", color=navy)
    nonfallback_box = schematic_box(ax, 0.06, 0.66, 0.26, 0.13, "Non-fallback prior\nprior_max >= 0.15\ncomponent >= 0.01", face=prior_fill, edge=teal, weight="bold", fontsize=6.1)
    roi_crop_box = schematic_box(ax, 0.40, 0.66, 0.20, 0.13, "ROI crop\nfrom prior bbox", face=neutral_fill, edge=navy, fontsize=6.2)
    retained_box = schematic_box(ax, 0.70, 0.66, 0.22, 0.13, "Spatial cue\nretained", face=neutral_fill, edge=navy, weight="bold", fontsize=6.2)
    fallback_box = schematic_box(ax, 0.06, 0.39, 0.26, 0.13, "Fallback prior\nlow confidence or\ntiny component", face=warn_fill, edge=red, weight="bold", fontsize=6.1)
    no_roi_box = schematic_box(ax, 0.40, 0.39, 0.20, 0.13, "No ROI\nconstraint", face=warn_fill, edge=red, fontsize=6.2)
    full_image_box = schematic_box(ax, 0.70, 0.39, 0.22, 0.13, "Full-image\nprior-enhanced input", face=neutral_fill, edge=navy, fontsize=6.0)
    box_arrow(ax, nonfallback_box, roi_crop_box, color=teal, **arrow_kw)
    box_arrow(ax, roi_crop_box, retained_box, color=navy, **arrow_kw)
    box_arrow(ax, fallback_box, no_roi_box, color=red, **arrow_kw)
    box_arrow(ax, no_roi_box, full_image_box, color=navy, **arrow_kw)
    ax.text(0.06, 0.18, "Fallback priors are not treated as reliable spatial constraints.", ha="left", va="center", fontsize=6.8, color=gray, fontweight="bold")

    ax = axes[2]
    panel_label(ax, "c", x=0.0, y=0.98)
    ax.text(0.06, 0.98, "RA-LPT and selected fusion are distinct", ha="left", va="top", fontsize=9.0, fontweight="bold", color=navy)
    ax.plot([0.50, 0.50], [0.18, 0.84], color=gray_light, linewidth=0.9)
    ax.text(0.06, 0.84, "RA-LPT stacking", ha="left", va="center", fontsize=7.0, color=navy, fontweight="bold")
    ax.text(0.56, 0.84, "Selected probability fusion", ha="left", va="center", fontsize=7.0, color=navy, fontweight="bold")

    model_prob_box = schematic_box(ax, 0.06, 0.66, 0.20, 0.10, "Model\nprobabilities", face=rgb_fill, edge=navy, weight="bold", fontsize=6.1)
    reliability_box = schematic_box(ax, 0.06, 0.50, 0.20, 0.10, "Prior reliability\nfeatures", face=prior_fill, edge=teal, weight="bold", fontsize=6.1)
    logistic_box = schematic_box(ax, 0.34, 0.55, 0.13, 0.14, "Logistic\nstacking", face=neutral_fill, edge=navy, weight="bold", fontsize=6.1)
    box_arrow(ax, model_prob_box, logistic_box, target_frac=0.70, color=navy, **arrow_kw)
    box_arrow(ax, reliability_box, logistic_box, target_frac=0.30, color=teal, **arrow_kw)
    ax.text(0.06, 0.33, "Explicit reliability statistics\nand interaction features", ha="left", va="center", fontsize=6.3, color=gray)

    strong_rgb_box = schematic_box(ax, 0.56, 0.66, 0.18, 0.10, "Strong RGB\npredictions", face=rgb_fill, edge=navy, weight="bold", fontsize=6.1)
    lpt_output_box = schematic_box(ax, 0.56, 0.48, 0.18, 0.10, "LPT-output\nevidence", face=prior_fill, edge=teal, weight="bold", fontsize=6.1)
    selected_box = schematic_box(ax, 0.82, 0.55, 0.14, 0.14, "Selected\nfusion\ncandidate", face=neutral_fill, edge=navy, weight="bold", fontsize=6.0)
    box_arrow(ax, strong_rgb_box, selected_box, target_frac=0.70, color=navy, **arrow_kw)
    box_arrow(ax, lpt_output_box, selected_box, target_frac=0.30, color=teal, **arrow_kw)
    ax.text(0.56, 0.33, "Uses LPT output probabilities;\nnot raw reliability statistics.", ha="left", va="center", fontsize=6.3, color=gray)
    ax.text(0.56, 0.18, "Internally optimised grouped OOF estimate; not external validation.", ha="left", va="center", fontsize=6.1, color=red, fontweight="bold")

    save_all(fig, outdir, "supp_fig_s01_lpt_ra_lpt_architecture_main_style", dpi, manifest)
    manifest[-1]["source"] = "matplotlib schematic generated by build_supplementary_main_style_figures.py"
    manifest[-1]["note"] = "Redrawn as an editable Nature-style schematic; no raster medical images are generated or altered."


def annotate_values(
    ax: Axes,
    values: Iterable[float],
    positions: Iterable[float],
    *,
    horizontal: bool = True,
    fixed_x: float | None = None,
) -> None:
    for pos, value in zip(positions, values):
        if horizontal:
            xpos = fixed_x if fixed_x is not None else value + 0.001
            ha = "right" if fixed_x is not None else "left"
            ax.text(xpos, pos, f"{value:.3f}", va="center", ha=ha, fontsize=7.0, color=COLORS["text"])
        else:
            ax.text(pos, value + 0.001, f"{value:.3f}", va="bottom", ha="center", fontsize=6.8, color=COLORS["text"], rotation=90)


def figure_s02_lpt_ablation(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "table_lpt_spatial_ablations.csv")
    order = ["LPT random prior", "Full LPT", "LPT no ROI", "LPT stats only", "LPT no prior modulation", "LPT no alignment"]
    order = [item for item in order if item in set(frame["method"])]
    frame = frame.set_index("method").loc[order].reset_index()

    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.16, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(frame))
    values = frame["macro_f1"].to_numpy(float)
    err = frame["macro_f1_std"].to_numpy(float)
    colors = [method_color(m) for m in frame["method"]]
    ax.barh(y, values, xerr=err, color=colors, edgecolor="white", linewidth=0.6, height=0.64, capsize=2.4)
    for ypos, value in zip(y, values):
        ax.text(
            value + 0.0012,
            ypos + 0.11,
            f"{value:.3f}",
            va="bottom",
            ha="left",
            fontsize=7.0,
            color=COLORS["text"],
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.set_xlim(0.850, 0.950)
    ax.set_xticks(np.arange(0.85, 0.951, 0.02))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Direct LPT ablation performance")
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(frame))
    width = 0.34
    balanced_accuracy = frame["balanced_accuracy"].to_numpy(float)
    accuracy = frame["accuracy"].to_numpy(float)
    ax2.bar(x - width / 2, balanced_accuracy, width=width, color=COLORS["blue"], label="Balanced accuracy")
    ax2.bar(x + width / 2, accuracy, width=width, color=COLORS["teal"], label="Accuracy")
    for xpos, value in zip(x - width / 2, balanced_accuracy):
        ax2.text(xpos - 0.10, value + 0.00045, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8, color=COLORS["text"])
    for xpos, value in zip(x + width / 2, accuracy):
        ax2.text(xpos + 0.10, value + 0.00045, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8, color=COLORS["text"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([short_method(v) for v in frame["method"]], rotation=28, ha="right")
    ax2.set_ylim(0.900, 0.935)
    ax2.set_ylabel("Score")
    ax2.set_title("Secondary metrics")
    ax2.legend(loc="upper left", ncol=1)
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s02_lpt_ablation_main_style", dpi, manifest)
    manifest[-1]["source"] = "table_lpt_spatial_ablations.csv"


def figure_s03_reliability_fusion(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    frame = read_csv(source_dir, "table_reliability_fusion.csv")
    order = ["Pure cls ensemble", "Cls + prior residual", "LPT output ensemble", "LPT ablations + prior stats"]
    frame = frame.set_index("method").loc[[m for m in order if m in set(frame["method"])]].reset_index()
    boot = read_csv(source_dir, "paired_bootstrap_significance.csv")
    boot = boot[(boot["candidate"] == "LPT ablations + prior stats") & (boot["metric"] == "macro_f1")].copy()
    boot["baseline_short"] = boot["baseline"].map(short_method)

    fig = plt.figure(figsize=(7.25, 3.25), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.06])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(frame))
    vals = frame["macro_f1"].to_numpy(float)
    err = frame["macro_f1_std"].to_numpy(float)
    ax.barh(y, vals, xerr=err, color=[method_color(m) for m in frame["method"]], edgecolor="white", linewidth=0.6, capsize=2.4, height=0.62)
    for ypos, value in zip(y, vals):
        ax.text(
            value + 0.0012,
            ypos + 0.11,
            f"{value:.3f}",
            va="bottom",
            ha="left",
            fontsize=7.0,
            color=COLORS["text"],
            zorder=4,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.set_xlim(0.900, 0.980)
    ax.set_xticks(np.arange(0.90, 0.981, 0.02))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Reliability-aware stacking")
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(boot))
    delta = boot["observed_delta"].to_numpy(float)
    xerr = np.vstack([delta - boot["ci95_low"].to_numpy(float), boot["ci95_high"].to_numpy(float) - delta])
    ax2.errorbar(delta, y, xerr=xerr, fmt="o", color=COLORS["navy"], ecolor=COLORS["teal"], capsize=2.8, elinewidth=1.2)
    ax2.set_yticks(y)
    ax2.set_yticklabels(boot["baseline_short"])
    ax2.set_ylim(-0.45, len(boot) - 1 + 0.45)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:.4f}"))
    ax2.set_xlabel("Observed Macro-F1 difference")
    ax2.set_title("Paired bootstrap vs RA-LPT")
    for xpos, ypos, pvalue in zip(delta, y, boot["two_sided_p"]):
        ax2.text(xpos + 0.0007, ypos + 0.12, f"p={float(pvalue):.3g}", fontsize=6.6, color=COLORS["gray"])
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s03_reliability_fusion_main_style", dpi, manifest)
    manifest[-1]["source"] = "table_reliability_fusion.csv; paired_bootstrap_significance.csv"


def figure_s04_auc_summary(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    summary = read_csv(source_dir, "roc_pr_auc_pivot.csv")
    metrics = read_csv(source_dir, "roc_pr_auc_metrics.csv")
    method_order = [
        "EfficientNetV2-S",
        "Reliability-aware LPT fusion",
        "Strong classification ensemble",
        "Strong LPT output ensemble",
        "Strong RA-LPT fusion",
    ]
    summary = summary.set_index("method").loc[[m for m in method_order if m in set(summary["method"])]].reset_index()

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(summary))
    macro_auroc = summary["roc_auc_macro"].to_numpy(float)
    macro_auprc = summary["average_precision_macro"].to_numpy(float)
    ax.hlines(y, macro_auprc, macro_auroc, color=COLORS["gray_light"], linewidth=2.2, zorder=1)
    ax.scatter(macro_auroc, y, s=30, color=COLORS["blue"], label="AUROC", zorder=3)
    ax.scatter(macro_auprc, y, s=30, color=COLORS["teal"], label="AUPRC", zorder=3)
    for ypos, auroc, auprc in zip(y, macro_auroc, macro_auprc):
        ax.text(auprc - 0.00022, ypos + 0.13, f"{auprc:.4f}", ha="right", va="bottom", fontsize=5.7, color=COLORS["teal"])
        ax.text(auroc + 0.00022, ypos + 0.13, f"{auroc:.4f}", ha="left", va="bottom", fontsize=5.7, color=COLORS["blue"])
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in summary["method"]])
    ax.set_ylim(-0.5, len(summary) - 0.5)
    ax.set_xlim(0.970, 1.000)
    ax.set_xticks(np.arange(0.97, 1.001, 0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.set_xlabel("Score")
    ax.set_title("Standard-CV macro discrimination")
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    strong = metrics[(metrics["method"] == "Strong classification ensemble") & (metrics["curve"] == "one_vs_rest")].copy()
    strong = strong.set_index("class").loc[list(CLASS_NAMES)].reset_index()
    y2 = np.arange(len(strong))
    class_auroc = strong["roc_auc"].to_numpy(float)
    class_auprc = strong["average_precision"].to_numpy(float)
    ax2.hlines(y2, class_auprc, class_auroc, color=COLORS["gray_light"], linewidth=2.2, zorder=1)
    ax2.scatter(class_auroc, y2, s=30, color=COLORS["blue"], zorder=3, clip_on=False)
    ax2.scatter(class_auprc, y2, s=30, color=COLORS["teal"], zorder=3, clip_on=False)
    for ypos, auroc, auprc in zip(y2, class_auroc, class_auprc):
        ax2.text(auprc, ypos - 0.14, f"{auprc:.4f}", ha="center", va="bottom", fontsize=5.7, color=COLORS["teal"])
        ax2.text(auroc, ypos + 0.14, f"{auroc:.4f}", ha="center", va="top", fontsize=5.7, color=COLORS["blue"])
    ax2.set_yticks(y2)
    ax2.set_yticklabels(strong["class"])
    ax2.set_ylim(len(strong) - 0.5, -0.5)
    ax2.set_xlim(0.980, 1.000)
    ax2.set_xticks(np.arange(0.98, 1.001, 0.01))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax2.set_xlabel("Score")
    ax2.set_title("Strong RGB stack: class-wise")
    clean_axis(ax2)
    panel_label(ax2, "b")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=2)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.24, top=0.84, wspace=0.43)

    save_all(fig, outdir, "supp_fig_s04_auc_discrimination_main_style", dpi, manifest)
    manifest[-1]["source"] = "roc_pr_auc_pivot.csv; roc_pr_auc_metrics.csv"


def figure_s05_prior_reliability(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    standard = read_csv(source_dir, "prior_reliability_strata_metrics.csv")
    strong = read_csv(source_dir, "strong_prior_reliability_strata_metrics.csv")
    data = pd.concat([standard, strong], ignore_index=True)
    fallback_color = "#7C8797"
    low_quality_color = "#7FA6C9"
    high_quality_color = COLORS["teal"]

    methods = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong cls ensemble", "Strong LPT output ensemble"]
    fig = plt.figure(figsize=(7.25, 3.65), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    fallback = data[(data["group_type"] == "fallback_group") & (data["method"].isin(methods))].copy()
    x = np.arange(len(methods))
    width = 0.34
    for offset, group, color, label in [(-width / 2, "fallback", fallback_color, "Fallback"), (width / 2, "non_fallback", high_quality_color, "Non-fallback")]:
        vals = []
        for method in methods:
            row = fallback[(fallback["method"] == method) & (fallback["group"] == group)]
            vals.append(float(row["macro_f1"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(x + offset, vals, width=width, color=color, label=label, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0010, f"{value:.3f}", ha="center", va="bottom", fontsize=5.7, color=COLORS["text"])
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet50", "EffNetV2-S", "RA-LPT", "Strong RGB\nstack", "Strong LPT\noutput"])
    ax.set_ylim(0.88, 0.985)
    ax.set_yticks(np.arange(0.88, 0.981, 0.02))
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fallback status")
    ax.legend(loc="upper left", ncol=2)
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    quality_methods = ["ResNet50", "Reliability-aware LPT fusion", "Strong cls ensemble"]
    quality_groups = ["fallback", "nonfallback_low_quality", "nonfallback_high_quality"]
    labels = ["Fallback", "Low quality", "High quality"]
    colors = [fallback_color, low_quality_color, high_quality_color]
    qdata = data[(data["group_type"] == "quality_group") & (data["method"].isin(quality_methods))].copy()
    x = np.arange(len(quality_methods))
    width = 0.25
    for idx, (group, label, color) in enumerate(zip(quality_groups, labels, colors)):
        vals = []
        for method in quality_methods:
            row = qdata[(qdata["method"] == method) & (qdata["group"] == group)]
            vals.append(float(row["macro_f1"].iloc[0]) if len(row) else np.nan)
        bars = ax2.bar(x + (idx - 1) * width, vals, width=width, color=color, label=label, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, value + 0.0010, f"{value:.3f}", ha="center", va="bottom", fontsize=5.7, color=COLORS["text"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["ResNet50", "RA-LPT", "Strong RGB\nstack"])
    ax2.set_ylim(0.88, 0.985)
    ax2.set_yticks(np.arange(0.88, 0.981, 0.02))
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Prior-quality strata")
    ax2.legend(loc="upper left", ncol=1)
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s05_prior_reliability_strata_main_style", dpi, manifest)
    manifest[-1]["source"] = "prior_reliability_strata_metrics.csv; strong_prior_reliability_strata_metrics.csv"


def figure_s06_similarity_sensitivity(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    sim = read_csv(source_dir, "similarity_sensitivity_metrics.csv")
    overlap_base = "cross_dataset_overlap_exclusion_sensitivity_v1/"
    overlap = read_csv(source_dir, overlap_base + "cross_dataset_overlap_exclusion_overall_metrics.csv")
    diffs = read_csv(source_dir, overlap_base + "cross_dataset_overlap_exclusion_model_differences.csv")

    fig = plt.figure(figsize=(7.25, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    methods = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong cls ensemble", "Strong LPT output ensemble"]
    label_map = {"RA-LPT fusion": "Reliability-aware LPT fusion"}
    x = np.arange(len(methods))
    width = 0.34
    series = [
        (-width / 2, "all_labeled_oof", COLORS["blue"], "All OOF (n = 5,955)"),
        (
            width / 2,
            "excluding_cross_fold_same_label_near_duplicates",
            COLORS["teal"],
            "Exclude near-duplicates (n = 5,625)",
        ),
    ]
    for offset, subset, color, label in series:
        vals = []
        for method in methods:
            row = sim[(sim["method"] == method) & (sim["subset"] == subset)]
            if row.empty and method in label_map:
                row = sim[(sim["method"] == label_map[method]) & (sim["subset"] == subset)]
            vals.append(float(row["macro_f1"].iloc[0]) if len(row) else np.nan)
        bars = ax.bar(
            x + offset,
            vals,
            width=width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.0014,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=COLORS["text"],
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet50", "EffNetV2-S", "RA-LPT", "Strong RGB\nstack", "Strong LPT\noutput"])
    ax.set_ylim(0.90, 0.98)
    ax.set_yticks(np.arange(0.90, 0.981, 0.02))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model")
    ax.set_title("Internal near-duplicate exclusion")
    ax.legend(loc="upper left", ncol=1, fontsize=5.9, handlelength=1.8)
    ax.tick_params(axis="both", labelsize=7.1)
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    keep_models = [
        "selected_prior_informed_probability_fusion",
        "retained_strong_rgb_logistic_stack",
        "retained_strong_rgb_prior_feature_stack",
    ]
    model_labels = {
        "selected_prior_informed_probability_fusion": "Selected",
        "retained_strong_rgb_logistic_stack": "Retained RGB",
        "retained_strong_rgb_prior_feature_stack": "Retained RGB + prior",
    }
    colors = {
        "selected_prior_informed_probability_fusion": COLORS["navy"],
        "retained_strong_rgb_logistic_stack": COLORS["blue"],
        "retained_strong_rgb_prior_feature_stack": COLORS["teal"],
    }
    value_offsets = {
        "selected_prior_informed_probability_fusion": -0.00048,
        "retained_strong_rgb_logistic_stack": 0.00038,
        "retained_strong_rgb_prior_feature_stack": -0.00050,
    }
    for key in keep_models:
        sub = overlap[overlap["model_key"] == key].copy()
        sub["scope"] = pd.Categorical(
            sub["scope"],
            categories=["full_oof", "exclude_broad_cross_dataset_overlap_241"],
            ordered=True,
        )
        sub = sub.sort_values("scope")
        xvals = [0, 1]
        values = sub["macro_f1"].to_numpy(float)
        ax2.plot(
            xvals,
            values,
            marker="o",
            markersize=5.5,
            linewidth=1.8,
            color=colors[key],
            label=model_labels[key],
        )
        for xvalue, value in zip(xvals, values):
            ax2.text(
                xvalue,
                value + value_offsets[key],
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=5.9,
                color=colors[key],
                fontweight="bold" if key == "selected_prior_informed_probability_fusion" else "normal",
            )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Full OOF\n(n = 5,955)", "Exclude 241\n(n = 5,714)"], rotation=0)
    ax2.set_xlim(-0.08, 1.08)
    ax2.set_ylim(0.950, 0.966)
    ax2.set_yticks(np.arange(0.950, 0.9661, 0.003))
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}"))
    ax2.set_ylabel("Macro-F1")
    ax2.set_xlabel("Analysis set")
    ax2.set_title("Cross-dataset broad-overlap sensitivity")
    ax2.legend(loc="lower left", fontsize=5.9, handlelength=2.0)
    delta_rows = diffs[diffs["comparator_key"] == "retained_strong_rgb_logistic_stack"].set_index("scope")
    full_delta = float(delta_rows.loc["full_oof", "macro_f1_delta"])
    excluded_delta = float(delta_rows.loc["exclude_broad_cross_dataset_overlap_241", "macro_f1_delta"])
    ax2.text(
        0.50,
        0.9652,
        f"Selected - retained RGB: +{full_delta:.4f} to +{excluded_delta:.4f}",
        ha="center",
        va="center",
        fontsize=6.0,
        color=COLORS["gray"],
    )
    ax2.tick_params(axis="both", labelsize=7.1)
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s06_similarity_sensitivity_main_style", dpi, manifest)
    manifest[-1]["source"] = "similarity_sensitivity_metrics.csv; cross_dataset_overlap_exclusion_sensitivity_v1/*.csv"


def figure_s07_interpretability(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    data = read_csv(source_dir, "cam_prior_occlusion_summary.csv")
    subset = data[(data["group_type"] == "fallback") & (data["method"].isin(["ResNet50", "EfficientNetV2-S"]))].copy()
    metrics = [
        ("cam_prior_mass_ratio", "CAM-prior mass", "log"),
        ("pointing_game_accuracy", "Pointing-game accuracy", "log"),
        ("prior_minus_random_drop", "Prior minus random drop", "linear"),
    ]

    fig = plt.figure(figsize=(7.25, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 0.04], hspace=0.0)
    group_styles = {
        "fallback": {"label": "Fallback", "color": COLORS["gray"]},
        "non_fallback": {"label": "Non-fallback", "color": COLORS["teal"]},
    }
    method_order = ["ResNet50", "EfficientNetV2-S"]
    x = np.arange(len(method_order))
    width = 0.34
    legend_handles = []
    legend_labels = []

    for i, (column, title, scale) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        pivot = subset.pivot(index="method", columns="group", values=column).loc[method_order]

        for group, position in [("fallback", x - width / 2), ("non_fallback", x + width / 2)]:
            style = group_styles[group]
            values = pivot[group].to_numpy(float)
            if scale == "log":
                baseline = 0.003 if column == "cam_prior_mass_ratio" else 0.008
                heights = values - baseline
            else:
                baseline = 0.0
                heights = values
            bars = ax.bar(
                position,
                heights,
                width=width,
                bottom=baseline,
                color=style["color"],
                edgecolor="white",
                linewidth=0.5,
                label=style["label"],
            )
            if i == 0:
                legend_handles.append(bars[0])
                legend_labels.append(style["label"])
            for bar, value in zip(bars, values):
                yvalue = value * 1.07 if scale == "log" else value + 0.0020
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    yvalue,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.1,
                    color=style["color"],
                    fontweight="bold" if group == "non_fallback" else "normal",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(["ResNet50", "EffNetV2-S"])
        ax.set_xlabel("Model")
        if scale == "log":
            ax.set_yscale("log")
            lower_limit = 0.003 if column == "cam_prior_mass_ratio" else 0.008
            ax.set_ylim(lower_limit, 1.0)
            ticks = (
                [0.005, 0.01, 0.05, 0.10, 0.50, 1.0]
                if column == "cam_prior_mass_ratio"
                else [0.01, 0.05, 0.10, 0.50, 1.0]
            )
            ax.set_yticks(ticks)
            ax.set_ylim(lower_limit, 1.0)
            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _: (
                        f"{value:.3f}"
                        if value < 0.01
                        else f"{value:.2f}"
                        if value < 0.10
                        else f"{value:.1f}"
                    )
                )
            )
            ax.set_ylabel("Value (log scale)")
        else:
            ax.axhline(0, color=COLORS["gray_mid"], linewidth=0.8)
            ax.set_ylim(0, 0.12)
            ax.set_yticks(np.arange(0, 0.121, 0.02))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:.2f}"))
            ax.set_ylabel("Confidence-drop difference")
        ax.set_title(title)
        ax.tick_params(axis="both", labelsize=7.0)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + i))

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=2,
        fontsize=6.4,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    save_all(fig, outdir, "supp_fig_s07_interpretability_occlusion_main_style", dpi, manifest)
    manifest[-1]["source"] = "cam_prior_occlusion_summary.csv"


def figure_s08_error_patterns(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    errors = read_csv(source_dir, "publication_final_error_patterns.csv").head(8).copy()
    cases = read_csv(source_dir, "strong_error_case_summary.csv").copy()

    fig = plt.figure(figsize=(7.25, 3.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    labels = [f"{r.true_class}->{r.predicted_class}" for r in errors.itertuples()]
    y = np.arange(len(errors))[::-1]
    vals = errors["count"].to_numpy(int)
    ax.barh(y, vals, color=COLORS["blue"], height=0.62, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Error count")
    ax.set_title("Selected-candidate residual errors")
    for ypos, value in zip(y, vals):
        ax.text(value + 1.5, ypos, str(value), va="center", fontsize=6.8)
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    keep = cases.head(5).copy()
    short_labels = [
        "RA-LPT correct,\nResNet50 wrong",
        "RA-LPT correct,\nEffNet wrong",
        "EffNet correct,\nRA-LPT wrong",
        "Strong RGB correct,\nRA-LPT wrong",
        "Strong RA-LPT correct,\nStrong RGB wrong",
    ][: len(keep)]
    x = np.arange(len(keep))
    fallback_bars = ax2.bar(
        x - 0.18,
        keep["fallback_ratio"],
        width=0.36,
        color=COLORS["gray_mid"],
        label="Fallback ratio",
    )
    quality_bars = ax2.bar(
        x + 0.18,
        keep["prior_quality_mean"],
        width=0.36,
        color=COLORS["teal"],
        label="Prior quality",
    )
    for bars, values, color in [
        (fallback_bars, keep["fallback_ratio"], COLORS["gray"]),
        (quality_bars, keep["prior_quality_mean"], COLORS["teal"]),
    ]:
        for bar, value in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.010,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=5.9,
                color=color,
                fontweight="bold" if color == COLORS["teal"] else "normal",
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=24, ha="right")
    ax2.set_ylim(0, 0.82)
    ax2.set_ylabel("Proportion / mean")
    ax2.set_title("Difficult-case prior characteristics")
    ax2.legend(loc="upper right")
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s08_error_patterns_main_style", dpi, manifest)
    manifest[-1]["source"] = "publication_final_error_patterns.csv; strong_error_case_summary.csv"


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
        return "ConvNeXt"
    if "efficientnetv2s" in name:
        return "EffNetV2-S"
    if "strong_cls" in name:
        return "Strong RGB"
    return name.replace("dfuc2021_cls_", "").replace("_seed2027", "").replace("_seed3407", "")


def figure_s09_fusion_search_trace(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    base = source_dir / "fusion_search"
    if not base.exists():
        base = source_dir.parent / "fusion_search"
    search = pd.read_csv(base / "publication_baseline5_full_oof_fusion_search.csv")
    trace = pd.read_csv(base / "publication_selected_prior_informed_candidate_trace.csv").iloc[0]
    single = pd.read_csv(base / "publication_baseline5_single_metrics.csv").head(10).copy()

    fig = plt.figure(figsize=(7.25, 3.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    ranked = search.head(40).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ax.plot(ranked["rank"], ranked["macro_f1"], color=COLORS["blue"], linewidth=1.5)
    ax.scatter([1], [float(ranked.iloc[0]["macro_f1"])], s=32, color=COLORS["blue"], zorder=3)
    ax.text(
        2.0,
        float(ranked.iloc[0]["macro_f1"]) + 0.00007,
        "unrestricted RGB-only winner",
        fontsize=6.2,
        color=COLORS["blue"],
        va="bottom",
    )
    ax.scatter([int(trace["overall_rank"])], [float(trace["macro_f1"])], s=36, color=COLORS["navy"], zorder=3)
    ax.text(
        int(trace["overall_rank"]) + 1.2,
        float(trace["macro_f1"]) + 0.00012,
        "post hoc prior-informed\ncandidate",
        fontsize=6.6,
        color=COLORS["navy"],
        va="bottom",
    )
    ax.set_xlabel("Rank among full-OOF fusion candidates")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fusion-search trace")
    ax.set_xlim(1, 40)
    ax.set_ylim(0.9628, 0.96445)
    clean_axis(ax)
    panel_label(ax, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    weights = parse_candidate_components(str(trace["candidate"]))
    labels = [component_short(k) for k in weights]
    vals = list(weights.values())
    colors = [COLORS["teal"] if "LPT" in label else COLORS["blue"] for label in labels]
    ax2.bar(np.arange(len(vals)), vals, color=colors, width=0.58, edgecolor="white", linewidth=0.6)
    ax2.set_xticks(np.arange(len(vals)))
    ax2.set_xticklabels(labels, rotation=24, ha="right")
    ax2.set_ylim(0, 0.46)
    ax2.set_ylabel("Selected weight")
    ax2.set_title("Post hoc prior-informed composition")
    for xpos, value in enumerate(vals):
        ax2.text(xpos, value + 0.015, f"{value:.3f}", ha="center", fontsize=6.8)
    ax2.text(
        0.98,
        0.96,
        f"Full-OOF Macro-F1 {float(trace['macro_f1']):.4f}; overall rank {int(trace['overall_rank'])}\nFold-held-out OOF trace Macro-F1 0.9529",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color=COLORS["text"],
    )
    clean_axis(ax2)
    panel_label(ax2, "b")

    save_all(fig, outdir, "supp_fig_s09_fusion_search_trace_main_style", dpi, manifest)
    manifest[-1]["source"] = "fusion_search/publication_baseline5_full_oof_fusion_search.csv; fusion_search/publication_selected_prior_informed_candidate_trace.csv"


def make_contact_sheet(outdir: Path, manifest: list[dict], dpi: int) -> None:
    png_files = [outdir / f"{row['figure']}.png" for row in manifest if (outdir / f"{row['figure']}.png").exists()]
    if not png_files:
        return
    thumbs = []
    width = 1200
    for path in png_files:
        img = Image.open(path).convert("RGB")
        ratio = width / img.width
        thumb = img.resize((width, max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, thumb.height + 52), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, thumb.height + 14), path.stem, fill=COLORS["text"])
        thumbs.append(canvas)
    cols = 2
    gutter = 34
    rows = math.ceil(len(thumbs) / cols)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * width + (cols - 1) * gutter, rows * cell_h + (rows - 1) * gutter), "white")
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = col * (width + gutter)
        y = row * (cell_h + gutter)
        sheet.paste(thumb, (x, y))
    sheet.save(outdir / "supplementary_main_style_contact_sheet.png")


def write_manifest(outdir: Path, manifest: list[dict]) -> None:
    lines = [
        "# Supplementary Figure Main-Style Alignment Manifest",
        "",
        "These supplementary figure candidates were generated in Python/matplotlib using the same restrained color and typography style as the main composite manuscript figures.",
        "",
        "## Figure Contract",
        "",
        "- Core conclusion: DFUC2022-derived lesion priors are useful as uncertain auxiliary evidence; direct spatial prior injection is not the final success path.",
        "- Archetype: quantitative-grid supplementary evidence with one editable architecture schematic redrawn for continuity.",
        "- Validation boundary: internal validation and internally optimised grouped OOF estimates only.",
        "- Export policy: SVG/PDF retain editable text; PNG/TIFF are high-resolution previews/submission alternatives.",
        "",
        "| Proposed figure | Source | Exports | Note |",
        "| --- | --- | --- | --- |",
    ]
    for row in manifest:
        lines.append(
            f"| {row['figure']} | {row.get('source', '')} | {', '.join(row.get('exports', []))} | {row.get('note', '')} |"
        )
    (outdir / "supplementary_main_style_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (outdir / "supplementary_main_style_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    outdir = args.output_dir.resolve()
    source_dir = args.source_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    figure_s01_lpt_architecture(outdir, args.dpi, manifest)
    figure_s02_lpt_ablation(source_dir, outdir, args.dpi, manifest)
    figure_s03_reliability_fusion(source_dir, outdir, args.dpi, manifest)
    figure_s04_auc_summary(source_dir, outdir, args.dpi, manifest)
    figure_s05_prior_reliability(source_dir, outdir, args.dpi, manifest)
    figure_s06_similarity_sensitivity(source_dir, outdir, args.dpi, manifest)
    figure_s07_interpretability(source_dir, outdir, args.dpi, manifest)
    figure_s08_error_patterns(source_dir, outdir, args.dpi, manifest)
    figure_s09_fusion_search_trace(source_dir, outdir, args.dpi, manifest)

    write_manifest(outdir, manifest)
    make_contact_sheet(outdir, manifest, args.dpi)
    print(f"Wrote {len(manifest)} style-aligned supplementary figures to {outdir}")


if __name__ == "__main__":
    main()
