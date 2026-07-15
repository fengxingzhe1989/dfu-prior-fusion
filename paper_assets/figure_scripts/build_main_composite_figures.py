#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.axes import Axes
from PIL import Image, ImageOps


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
        description="Build composite manuscript figures from audited DFU source tables."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "paper_assets" / "source_tables",
        help="Audited source-table directory distributed with the code package.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=repo_root / "paper_assets" / "figure_inputs",
        help="Directory containing representative image assets.",
    )
    parser.add_argument(
        "--legacy-image-dir",
        type=Path,
        default=repo_root / "paper_assets" / "figure_inputs" / "legacy",
        help="Optional directory containing original Grad-CAM/prior image plates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "figures" / "main_composite",
        help="Output directory for composite figures.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--make-pptx", action="store_true", help="Also create a review PPTX with exported PNGs.")
    parser.add_argument(
        "--require-qualitative-assets",
        action="store_true",
        help="Fail instead of skipping the Grad-CAM example figure when licensed image assets are absent.",
    )
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


def read_json(source_dir: Path, relative: str) -> dict:
    path = source_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing source JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def clean_axis(ax: Axes, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_axisbelow(True)


def short_method(name: str) -> str:
    return {
        "ResNet50": "ResNet50",
        "EfficientNetV2-S": "EffNetV2-S",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny",
        "Full LPT": "Full LPT",
        "RA-LPT fusion": "RA-LPT",
        "Strong cls ensemble": "Strong RGB stack",
        "Strong cls + prior residual": "Strong RGB + prior",
        "Selected prior-informed fusion": "Selected prior-informed",
        "Selected prior-informed probability-fusion candidate": "Selected prior-informed",
        "Retained strong RGB comparator": "Retained strong RGB",
        "Retained strong RGB + prior-informed residual comparator": "Retained strong RGB + prior",
        "Strong cls ensemble grouped": "Retained RGB stack",
        "Strong cls + prior residual grouped": "Retained RGB + prior",
        "ConvNeXt-Tiny grouped": "ConvNeXt-Tiny",
        "EfficientNetV2-S grouped": "EffNetV2-S",
        "ResNet50 grouped": "ResNet50",
        "Full LPT grouped": "Full LPT",
        "RA-LPT fusion grouped": "RA-LPT",
        "Fold-held-out fusion-selection trace": "Fold-held-out\nOOF trace",
        "Unrestricted best RGB-only fusion": "Best RGB-only\nfull-OOF",
        "Prior-informed fusion containing strong LPT output": "Prior-informed\nfull-OOF",
        "LPT no alignment": "No alignment",
        "LPT no prior modulation": "No prior modulation",
        "LPT no ROI": "No ROI branch",
        "LPT stats only": "Stats only",
        "Random prior": "Random prior",
        "Reliability-aware LPT fusion": "RA-LPT",
        "Strong RA-LPT fusion": "Strong RA-LPT",
        "Pure cls ensemble": "RGB ensemble",
        "LPT output ensemble": "LPT output",
    }.get(name, name)


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
        kwargs = {"bbox_inches": "tight"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = dpi
        path = outdir / f"{stem}.{suffix}"
        fig.savefig(path, **kwargs)
        exports.append(str(path.name))
    plt.close(fig)
    manifest.append({"figure": stem, "exports": exports})


def add_note(ax: Axes, text: str, loc: tuple[float, float] = (0.02, 0.02), width: float = 0.96) -> None:
    x, y = loc
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        0.13,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=ax.transAxes,
        facecolor="#F7F9FC",
        edgecolor=COLORS["gray_mid"],
        linewidth=0.7,
        zorder=0,
    )
    ax.add_patch(box)
    ax.text(
        x + 0.02,
        y + 0.065,
        text,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )


def metric_card(ax: Axes, title: str, items: Iterable[tuple[str, str]], accent: str = "navy") -> None:
    ax.set_axis_off()
    ax.text(0.0, 0.98, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color=COLORS[accent], transform=ax.transAxes)
    y = 0.78
    for label, value in items:
        ax.text(0.0, y, value, ha="left", va="center", fontsize=13, fontweight="bold", color=COLORS[accent], transform=ax.transAxes)
        ax.text(0.0, y - 0.13, label, ha="left", va="center", fontsize=6.7, color=COLORS["gray"], transform=ax.transAxes)
        y -= 0.30


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.exif_transpose(img)


def show_image(ax: Axes, img: Image.Image, title: str | None = None) -> None:
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.4, pad=3)


def crop_fraction(img: Image.Image, left: float, upper: float, right: float, lower: float) -> Image.Image:
    w, h = img.size
    return img.crop((int(w * left), int(h * upper), int(w * right), int(h * lower)))


def figure2_performance_and_fusion(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    hierarchy = read_csv(source_dir, "publication_reporting_hierarchy.csv")
    bootstrap = read_csv(source_dir, "reviewer_revision_v2/derived/grouped_ra_lpt_paired_bootstrap.csv")
    ablation = read_csv(source_dir, "table_lpt_spatial_ablations.csv")

    fig = plt.figure(figsize=(7.35, 6.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.22, 1.0], height_ratios=[1.04, 0.96])

    ax = fig.add_subplot(gs[0, 0])
    order = [
        "ResNet50 grouped",
        "Full LPT grouped",
        "RA-LPT fusion grouped",
        "EfficientNetV2-S grouped",
        "ConvNeXt-Tiny grouped",
        "Fold-held-out fusion-selection trace",
    ]
    frame = hierarchy.set_index("method").loc[order].reset_index().iloc[::-1]
    y = np.arange(len(frame))
    values = frame["macro_f1"].astype(float).to_numpy()
    std = frame["macro_f1_std"].fillna(0).astype(float).to_numpy()
    ax.barh(
        y,
        values,
        xerr=std,
        color=[method_color(v) for v in frame["method"]],
        height=0.62,
        edgecolor="white",
        linewidth=0.6,
        capsize=2,
    )
    for ypos, value, spread in zip(y, values, std):
        ax.text(value + spread + 0.0012, ypos, f"{value:.3f}", va="center", fontsize=6.4)
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in frame["method"]])
    ax.set_xlim(0.90, 0.97)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Similarity-grouped estimates and OOF trace")
    ax.tick_params(axis="x", labelsize=7.8)
    clean_axis(ax, "x")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    fusion_order = [
        "Fold-held-out fusion-selection trace",
        "Unrestricted best RGB-only fusion",
        "Prior-informed fusion containing strong LPT output",
    ]
    fusion = hierarchy.set_index("method").loc[fusion_order].reset_index()
    x = np.arange(len(fusion))
    fusion_colors = [COLORS["teal"], COLORS["blue"], COLORS["navy"]]
    bars = ax.bar(x, fusion["macro_f1"], color=fusion_colors, width=0.62, edgecolor="white", linewidth=0.6)
    for bar, value in zip(bars, fusion["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.0008, f"{float(value):.4f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Fold-held-out\nOOF trace", "Best RGB-only\nfull-OOF", "Prior-informed\nfull-OOF"])
    ax.set_ylim(0.94, 0.97)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fusion-selection hierarchy")
    ax.tick_params(axis="y", labelsize=7.8)
    clean_axis(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    keep = bootstrap.iloc[::-1].copy()
    y = np.arange(len(keep))
    delta = keep["observed_delta"].astype(float).to_numpy()
    xerr = np.vstack([delta - keep["ci95_low"].astype(float).to_numpy(), keep["ci95_high"].astype(float).to_numpy() - delta])
    ax.errorbar(delta, y, xerr=xerr, fmt="o", color=COLORS["navy"], ecolor=COLORS["teal"], capsize=2.5, elinewidth=1.1)
    ax.set_yticks(y)
    comparison_labels = {
        "resnet50_grouped": "RA-LPT vs ResNet50",
        "lpt_output_only_grouped": "RA-LPT vs output-only LPT stack",
        "strong_rgb_grouped": "Strong RGB + prior vs strong RGB",
    }
    ax.set_yticklabels([comparison_labels[v] for v in keep["baseline"]])
    ax.set_xlim(-0.015, 0.032)
    ax.set_xticks([-0.01, 0, 0.01, 0.02, 0.03])
    ax.set_ylim(-0.55, len(keep) - 0.45)
    ax.set_xlabel("Observed Macro-F1 difference")
    ax.set_title("Matched group-level bootstrap comparisons")
    ax.tick_params(axis="x", labelsize=8.3)
    for xpos, ypos, pvalue in zip(delta, y, keep["two_sided_p"].astype(float)):
        ax.text(
            xpos + 0.0010,
            ypos + 0.18,
            f"p={pvalue:.3g}",
            va="center",
            fontsize=7.4,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.35},
        )
    clean_axis(ax, "x")
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    ab = ablation.sort_values("macro_f1")
    y = np.arange(len(ab))
    ax.barh(y, ab["macro_f1"], xerr=ab["macro_f1_std"], color=[method_color(v) for v in ab["method"]], edgecolor="white", linewidth=0.6, capsize=2)
    ax.scatter(ab["macro_f1"], y, s=18, color="black", zorder=5)
    for ypos, value in zip(y, ab["macro_f1"]):
        ax.text(0.875, ypos, f"{value:.3f}", va="center", ha="left", fontsize=8.3, color="white", fontweight="bold")
    ax.axvline(0.9442, color=COLORS["blue"], linestyle=":", linewidth=1.8)
    ax.text(
        0.949,
        2.55,
        "ResNet50 standard\nCV 0.9442",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=COLORS["blue"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.7},
    )
    ax.set_yticks(y)
    ax.set_yticklabels([short_method(v) for v in ab["method"]])
    ax.set_xlim(0.86, 0.98)
    ax.set_xticks([0.86, 0.90, 0.94, 0.98])
    ax.set_xlabel("Macro-F1")
    ax.set_title("Direct LPT ablation in standard CV")
    ax.tick_params(axis="x", labelsize=7.8)
    clean_axis(ax, "x")
    panel_label(ax, "d")

    save_all(fig, outdir, "figure2_model_performance_and_fusion", dpi, manifest)


def figure3_selected_candidate_classification(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    cm = read_csv(source_dir, "publication_final_confusion_matrix.csv").set_index("Unnamed: 0")
    per_class = read_csv(source_dir, "publication_final_per_class_metrics.csv")
    binary = read_csv(source_dir, "publication_final_binary_endpoint_auc_auprc.csv")
    errors = read_csv(source_dir, "publication_final_error_patterns.csv").head(6)

    matrix = cm.to_numpy(dtype=float)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig = plt.figure(figsize=(7.35, 5.85), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.12, 0.88])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Post hoc full-OOF candidate confusion matrix")
    for i in range(4):
        for j in range(4):
            color = "white" if normalized[i, j] > 0.55 else COLORS["text"]
            ax.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j] * 100:.1f}%", ha="center", va="center", fontsize=6.0, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row proportion", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(per_class))
    width = 0.24
    metric_specs = [
        (-width, "precision", COLORS["blue"], "Precision"),
        (0, "recall", COLORS["teal"], "Recall"),
        (width, "f1", COLORS["navy"], "F1"),
    ]
    for offset, column, color, label in metric_specs:
        ax.bar(x + offset, per_class[column], width=width, color=color, label=label, edgecolor="white", linewidth=0.5)
        for xpos, value in zip(x + offset, per_class[column]):
            ax.text(xpos, value + 0.0010, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(per_class["class"], rotation=25, ha="right", fontsize=7.4)
    ax.set_ylim(0.93, 1.00)
    ax.set_yticks(np.arange(0.93, 1.001, 0.01))
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_ylabel("Score", fontsize=8.0)
    ax.set_title("Per-class metrics", pad=13)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.045),
        ncol=3,
        fontsize=6.8,
        handlelength=1.2,
        columnspacing=1.0,
    )
    clean_axis(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(binary))
    bar_height = 0.28
    bar_offset = 0.18
    labels = [v.replace(" detection", "") for v in binary["Endpoint"]]
    ax.barh(y - bar_offset, binary["AUROC"], height=bar_height, color=COLORS["blue"], label="AUROC")
    ax.barh(y + bar_offset, binary["AUPRC"], height=bar_height, color=COLORS["teal"], label="AUPRC")
    for ypos, row in enumerate(binary.itertuples(index=False)):
        ax.text(0.9612, ypos - bar_offset, "AUROC", va="center", ha="left", fontsize=5.8, color="white", fontweight="bold")
        ax.text(0.9612, ypos + bar_offset, "AUPRC", va="center", ha="left", fontsize=5.8, color="white", fontweight="bold")
        ax.text(float(row.AUROC) + 0.0007, ypos - bar_offset, f"{float(row.AUROC):.4f}", va="center", fontsize=6.1)
        ax.text(float(row.AUPRC) + 0.0007, ypos + bar_offset, f"{float(row.AUPRC):.4f}", va="center", fontsize=6.1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.52, len(binary) - 0.12)
    ax.set_xlim(0.96, 1.002)
    ax.set_xticks([0.96, 0.97, 0.98, 0.99, 1.00])
    ax.set_xlabel("Score")
    ax.set_title("Binary endpoint AUROC and AUPRC", pad=8)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=6.1,
        handlelength=1.4,
        columnspacing=1.0,
    )
    clean_axis(ax, "x")
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    errors_plot = errors.iloc[::-1].reset_index(drop=True)
    pairs = [f"{r.true_class} -> {r.predicted_class}" for r in errors_plot.itertuples(index=False)]
    y = np.arange(len(errors_plot))
    ax.barh(y, errors_plot["count"], color=COLORS["teal"], edgecolor="#1F625E", linewidth=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(pairs)
    ax.set_xlim(0, 130)
    ax.set_xlabel("Error count")
    ax.set_title("Most frequent residual errors")
    for ypos, value, share in zip(y, errors_plot["count"], errors_plot["share_of_errors"]):
        ax.text(value + 2, ypos, f"{int(value)} ({share * 100:.1f}%)", va="center", fontsize=6.8)
    clean_axis(ax, "x")
    panel_label(ax, "d")

    save_all(fig, outdir, "figure3_selected_candidate_classification_summary", dpi, manifest)


def figure4_prior_reliability_interpretability(
    source_dir: Path, image_dir: Path, legacy_image_dir: Path, outdir: Path, dpi: int, manifest: list[dict]
) -> None:
    cam = read_csv(source_dir, "reviewer_revision_v2/derived/interpretability_mask_support_audit.csv")
    priors = read_csv(source_dir, "prior_reliability_strata_metrics.csv")
    strong = read_csv(source_dir, "strong_prior_reliability_strata_metrics.csv")

    fig = plt.figure(figsize=(7.35, 6.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.08, 0.92])

    ax_panel = fig.add_subplot(gs[0, 0])
    ax_panel.set_axis_off()
    panel_label(ax_panel, "a", -0.04, 1.02)
    ax_panel.text(
        0.0,
        1.02,
        "Illustrative DFUC2021 prior examples",
        transform=ax_panel.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.6,
        color=COLORS["text"],
    )
    asset_dir = image_dir / "representative_images_word_assets"
    case_assets = [
        ("infection", "dfuc2021_infection_304467"),
        ("both", "dfuc2021_both_303735"),
    ]
    column_specs = [("RGB", "rgb"), ("Soft prior", "prior"), ("Prior overlay", "overlay")]
    x0 = 0.045
    y_positions = [0.50, 0.055]
    width = 0.27
    height = 0.40
    x_gap = 0.055
    for row_idx, (case_label, stem) in enumerate(case_assets):
        for col_idx, (title, suffix) in enumerate(column_specs):
            ax_img = ax_panel.inset_axes([x0 + col_idx * (width + x_gap), y_positions[row_idx], width, height])
            path = asset_dir / f"{stem}_{suffix}.png"
            if path.exists():
                show_image(ax_img, load_image(path), title if row_idx == 0 else None)
            else:
                ax_img.set_axis_off()
                ax_img.text(0.5, 0.5, "missing", ha="center", va="center", color=COLORS["gray"], transform=ax_img.transAxes)
            if col_idx == 0:
                ax_img.text(
                    -0.05,
                    0.5,
                    case_label,
                    transform=ax_img.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=6.6,
                    color=COLORS["gray"],
                )

    ax = fig.add_subplot(gs[0, 1])
    rel = pd.concat([priors, strong], ignore_index=True)
    rel = rel[(rel["group_type"] == "fallback_group") & rel["method"].isin(["ResNet50", "EfficientNetV2-S", "Strong RA-LPT fusion", "Reliability-aware LPT fusion"])]
    methods = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong RA-LPT fusion"]
    methods = [m for m in methods if m in set(rel["method"])]
    x = np.arange(len(methods))
    width = 0.34
    for offset, group, color, label in [(-width / 2, "fallback", COLORS["gray"], "fallback"), (width / 2, "non_fallback", COLORS["teal"], "non-fallback")]:
        values = []
        for method in methods:
            row = rel[(rel["method"] == method) & (rel["group"] == group)]
            values.append(float(row["macro_f1"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, color=color, label=label, edgecolor="white", linewidth=0.6)
        for xpos, value in zip(x + offset, values):
            if np.isfinite(value):
                ax.text(xpos, value + 0.0017, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_method(m).replace("Strong RA-LPT", "Strong\nRA-LPT") for m in methods])
    ax.set_ylim(0.88, 0.974)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Fallback status marks prior uncertainty")
    ax.legend(loc="upper left", ncol=2, fontsize=6.2)
    clean_axis(ax)
    panel_label(ax, "b")

    ax_c = fig.add_subplot(gs[1, 0])
    subset = cam[
        cam["stratum"].isin(["fallback_nonempty_mask", "nonfallback"])
        & cam["method"].isin(["ResNet50", "EfficientNetV2-S"])
    ].copy()
    pivot = subset.pivot(index="method", columns="stratum", values="mean_cam_mass_enrichment_nonempty").loc[["ResNet50", "EfficientNetV2-S"]]
    x = np.arange(len(pivot))
    width = 0.34
    fallback_x = x - width / 2
    nonfallback_x = x + width / 2
    fallback_values = pivot["fallback_nonempty_mask"].to_numpy(dtype=float)
    nonfallback_values = pivot["nonfallback"].to_numpy(dtype=float)

    ax_c.bar(
        fallback_x,
        fallback_values,
        width=width,
        color=COLORS["gray"],
        label="fallback, non-empty",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_c.bar(
        nonfallback_x,
        nonfallback_values,
        width=width,
        color=COLORS["teal"],
        label="non-fallback",
        edgecolor="white",
        linewidth=0.6,
    )
    for xpos, value in zip(fallback_x, fallback_values):
        ax_c.text(xpos, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=5.8)
    for xpos, value in zip(nonfallback_x, nonfallback_values):
        ax_c.text(xpos, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=5.8)
    ax_c.set_ylim(0, 3.8)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([short_method(v) for v in pivot.index])
    ax_c.set_ylabel("CAM-mass enrichment over mask area")
    ax_c.set_title("Area-normalised CAM enrichment\n(non-empty masks only)")
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.2)
    clean_axis(ax_c)
    panel_label(ax_c, "c")

    ax = fig.add_subplot(gs[1, 1])
    p2 = subset.pivot(index="method", columns="stratum", values="prior_minus_random_drop").loc[["ResNet50", "EfficientNetV2-S"]]
    low = subset.pivot(index="method", columns="stratum", values="delta_ci95_low").loc[["ResNet50", "EfficientNetV2-S"]]
    high = subset.pivot(index="method", columns="stratum", values="delta_ci95_high").loc[["ResNet50", "EfficientNetV2-S"]]
    x = np.arange(len(p2))
    fb = p2["fallback_nonempty_mask"].to_numpy(dtype=float)
    nf = p2["nonfallback"].to_numpy(dtype=float)
    fb_err = np.vstack([fb - low["fallback_nonempty_mask"].to_numpy(dtype=float), high["fallback_nonempty_mask"].to_numpy(dtype=float) - fb])
    nf_err = np.vstack([nf - low["nonfallback"].to_numpy(dtype=float), high["nonfallback"].to_numpy(dtype=float) - nf])
    ax.bar(x - width / 2, fb, yerr=fb_err, capsize=2.5, width=width, color=COLORS["gray"], label="fallback, non-empty", edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, nf, yerr=nf_err, capsize=2.5, width=width, color=COLORS["teal"], label="non-fallback", edgecolor="white", linewidth=0.6)
    for xpos, value in zip(x - width / 2, fb):
        ax.text(xpos, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    for xpos, value in zip(x + width / 2, nf):
        ax.text(xpos, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_method(v) for v in p2.index])
    ax.set_ylim(0, 0.115)
    ax.set_ylabel("Prior minus random confidence drop")
    ax.set_title("Prior-minus-random occlusion sensitivity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.1)
    clean_axis(ax)
    panel_label(ax, "d")

    save_all(fig, outdir, "figure4_prior_reliability_and_interpretability", dpi, manifest)


def figure5_robustness_and_similarity(source_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    component = read_json(source_dir, "methodology_audit_v7/similarity_group_component_summary.json")
    cross = read_json(source_dir, "methodology_audit_v7/dfuc2021_dfuc2022_cross_dataset_duplicate_summary.json")
    folds = read_csv(source_dir, "methodology_audit_v7/similarity_grouped_fold_class_counts.csv")
    metrics = read_csv(source_dir, "cross_dataset_overlap_exclusion_sensitivity_v1/cross_dataset_overlap_exclusion_overall_metrics.csv")
    diffs = read_csv(source_dir, "cross_dataset_overlap_exclusion_sensitivity_v1/cross_dataset_overlap_exclusion_model_differences.csv")

    fig = plt.figure(figsize=(7.35, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.12], height_ratios=[0.95, 1.05])

    ax = fig.add_subplot(gs[0, 0])
    metric_card(
        ax,
        "Image-similarity grouping",
        [
            ("valid DFUC2021 images", f"{component['samples']:,}"),
            ("similarity groups", f"{component['similarity_groups']:,}"),
            ("cross-fold near-duplicate pairs after grouping", f"{component['near_duplicate_pairs_cross_fold_after_grouping']}"),
        ],
        "navy",
    )
    panel_label(ax, "a", -0.02, 1.02)

    ax = fig.add_subplot(gs[0, 1])
    class_colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["purple"]]
    bottom = np.zeros(len(folds))
    x = np.arange(len(folds))
    for cls, color in zip(CLASS_NAMES, class_colors):
        ax.bar(x, folds[cls], bottom=bottom, color=color, label=cls, edgecolor="white", linewidth=0.4)
        bottom += folds[cls].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {int(v)}" for v in folds["fold"]])
    ax.set_ylabel("Images")
    ax.set_title("Similarity-grouped fold class distribution")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=6.2)
    clean_axis(ax)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    metric_card(
        ax,
        "DFUC2021-DFUC2022 overlap audit",
        [
            ("SHA256 exact duplicate pairs", f"{cross['cross_dataset_exact_duplicate_pairs']}"),
            ("broad pHash/dHash candidate pairs", f"{cross['cross_dataset_near_duplicate_candidate_pairs']:,}"),
            ("DFUC2021 images in broad candidates", f"{cross['dfuc2021_images_in_near_duplicate_pairs']:,}"),
        ],
        "teal",
    )
    panel_label(ax, "c", -0.02, 1.02)

    ax = fig.add_subplot(gs[1, 1])
    scope_order = ["full_oof", "exclude_broad_cross_dataset_overlap_241"]
    x = np.arange(2)
    labels = ["Full grouped\nOOF", "Exclude 241\nbroad-overlap images"]
    series_specs = [
        ("selected_prior_informed_probability_fusion", "Selected prior-informed", COLORS["navy"], 1.8),
        ("retained_strong_rgb_logistic_stack", "Retained strong RGB", COLORS["blue"], 1.35),
        ("retained_strong_rgb_prior_feature_stack", "Retained RGB + prior", COLORS["teal"], 1.35),
    ]
    for key, label, color, linewidth in series_specs:
        sub = metrics[metrics["model_key"] == key].set_index("scope").loc[scope_order]
        values = sub["macro_f1"].to_numpy(dtype=float)
        ax.plot(x, values, marker="o", markersize=5.2, linewidth=linewidth, color=color)
        ax.text(
            1.04,
            values[-1],
            f"{label}\n{values[-1]:.4f}",
            ha="left",
            va="center",
            fontsize=5.9,
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.08, 1.46)
    ax.set_ylim(0.954, 0.965)
    ax.set_yticks([0.954, 0.957, 0.960, 0.963])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Broad-overlap exclusion sensitivity")
    clean_axis(ax)
    panel_label(ax, "d")

    save_all(fig, outdir, "figure5_robustness_and_similarity_audit", dpi, manifest)


def figure6_gradcam_prior_examples(legacy_image_dir: Path, outdir: Path, dpi: int, manifest: list[dict]) -> None:
    src = legacy_image_dir / "fig_cam_prior_occlusion_examples.png"
    if not src.exists():
        raise FileNotFoundError(f"Missing Grad-CAM/prior example source plate: {src}")

    plate = load_image(src)
    col_boxes = [(58, 551), (761, 1254), (1464, 1957)]
    row_boxes = [(125, 618), (795, 1288), (1464, 1957), (2134, 2627)]
    examples = [
        ("infection", "306452.jpg", "CAM-prior mass 0.984\npointing hit 1\nprior-region drop 0.995"),
        ("both", "301422.jpg", "CAM-prior mass 0.964\npointing hit 1\nprior-region drop 0.999"),
        ("none", "306636.jpg", "CAM-prior mass 0.969\npointing hit 1\nprior-region drop 0.990"),
        ("ischaemia", "302687.jpg", "CAM-prior mass 0.965\npointing hit 1\nprior-region drop 0.989"),
    ]
    headers = ["RGB image", "Lesion-prior overlay", "Grad-CAM overlay", "Example metrics"]

    fig = plt.figure(figsize=(7.35, 7.15), constrained_layout=True)
    gs = fig.add_gridspec(
        len(examples) + 1,
        4,
        height_ratios=[0.12, 1, 1, 1, 1],
        width_ratios=[1, 1, 1, 0.94],
        hspace=0.08,
        wspace=0.10,
    )

    for col, header in enumerate(headers):
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.set_axis_off()
        ax_header.text(
            0.5,
            0.46,
            header,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color=COLORS["text"],
            transform=ax_header.transAxes,
        )

    for row_idx, (label, image_id, metric_text) in enumerate(examples):
        top, bottom = row_boxes[row_idx]
        for col_idx, (left, right) in enumerate(col_boxes):
            ax_img = fig.add_subplot(gs[row_idx + 1, col_idx])
            crop = plate.crop((left, top, right, bottom))
            show_image(ax_img, crop)
            if col_idx == 0:
                ax_img.text(
                    -0.07,
                    0.5,
                    label,
                    transform=ax_img.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9.4,
                    fontweight="bold",
                    color=COLORS["text"],
                )
                ax_img.text(
                    0.02,
                    0.04,
                    image_id,
                    transform=ax_img.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.0,
                    color=COLORS["white"],
                    bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": 0.65, "pad": 1.8},
                )
        ax_text = fig.add_subplot(gs[row_idx + 1, 3])
        ax_text.set_axis_off()
        ax_text.text(
            0.02,
            0.52,
            metric_text,
            transform=ax_text.transAxes,
            ha="left",
            va="center",
            fontsize=8.4,
            fontweight="medium",
            color=COLORS["text"],
            linespacing=1.28,
        )

    fig.suptitle("Illustrative high-overlap Grad-CAM/prior examples", fontsize=10.2, fontweight="bold", y=1.01)
    save_all(fig, outdir, "figure6_gradcam_prior_examples", dpi, manifest)


def build_contact_sheet(outdir: Path, manifest: list[dict], dpi: int) -> None:
    png_paths = [outdir / f"{entry['figure']}.png" for entry in manifest]
    images = [Image.open(path).convert("RGB") for path in png_paths if path.exists()]
    if not images:
        return
    thumb_w = 1100
    thumbs = []
    for img in images:
        ratio = thumb_w / img.width
        thumb = img.resize((thumb_w, int(img.height * ratio)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    pad = 34
    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(t.height for t in thumbs) + pad * 2
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * cell_h + pad), "white")
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = pad + col * (thumb_w + pad)
        y = pad + row * cell_h
        sheet.paste(thumb, (x, y))
    sheet.save(outdir / "main_composite_contact_sheet.png", dpi=(dpi, dpi))


def build_review_pptx(outdir: Path, manifest: list[dict]) -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except Exception:
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    for entry in manifest:
        png = outdir / f"{entry['figure']}.png"
        if not png.exists():
            continue
        slide = prs.slides.add_slide(blank)
        title_box = slide.shapes.add_textbox(Inches(0.25), Inches(0.12), Inches(12.7), Inches(0.35))
        title_box.text_frame.text = entry["figure"].replace("_", " ")
        title_box.text_frame.paragraphs[0].font.size = Pt(14)
        title_box.text_frame.paragraphs[0].font.bold = True
        slide.shapes.add_picture(str(png), Inches(0.28), Inches(0.55), width=Inches(12.8))
    prs.save(outdir / "main_composite_figures_review.pptx")


def write_manifest(outdir: Path, manifest: list[dict]) -> None:
    source_map = {
        "figure2_model_performance_and_fusion": [
            "publication_final_metrics.csv",
            "publication_final_fusion_step_summary.csv",
            "reviewer_revision_v2/derived/grouped_ra_lpt_paired_bootstrap.csv",
            "table_lpt_spatial_ablations.csv",
        ],
        "figure3_selected_candidate_classification_summary": [
            "publication_final_confusion_matrix.csv",
            "publication_final_per_class_metrics.csv",
            "publication_final_binary_endpoint_auc_auprc.csv",
            "publication_final_error_patterns.csv",
        ],
        "figure4_prior_reliability_and_interpretability": [
            "reviewer_revision_v2/derived/interpretability_mask_support_audit.csv",
            "prior_reliability_strata_metrics.csv",
            "strong_prior_reliability_strata_metrics.csv",
            "original Grad-CAM/prior image plate asset",
        ],
        "figure5_robustness_and_similarity_audit": [
            "methodology_audit_v7/similarity_group_component_summary.json",
            "methodology_audit_v7/similarity_grouped_fold_class_counts.csv",
            "methodology_audit_v7/dfuc2021_dfuc2022_cross_dataset_duplicate_summary.json",
            "cross_dataset_overlap_exclusion_sensitivity_v1/*.csv",
        ],
        "figure6_gradcam_prior_examples": [
            "original Grad-CAM/prior image plate asset",
            "illustrative post hoc examples only",
        ],
    }
    enriched = []
    for entry in manifest:
        row = dict(entry)
        row["sources"] = source_map.get(entry["figure"], [])
        enriched.append(row)
    (outdir / "main_composite_manifest.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    lines = [
        "# Main Composite Figure Manifest",
        "",
        "Figure 1 is intentionally not regenerated by this script.",
        "All quantitative panels are built from audited source tables in the code package.",
        "Selected probability-fusion results are labelled and interpreted as internally optimised grouped OOF estimates.",
        "",
        "| Figure | Source material | Exports |",
        "| --- | --- | --- |",
    ]
    for row in enriched:
        lines.append(f"| {row['figure']} | {'; '.join(row['sources'])} | {', '.join(row['exports'])} |")
    (outdir / "main_composite_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    source_dir = args.source_dir.resolve()
    image_dir = args.image_dir.resolve()
    legacy_image_dir = args.legacy_image_dir.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    figure2_performance_and_fusion(source_dir, outdir, args.dpi, manifest)
    figure3_selected_candidate_classification(source_dir, outdir, args.dpi, manifest)
    figure4_prior_reliability_interpretability(source_dir, image_dir, legacy_image_dir, outdir, args.dpi, manifest)
    figure5_robustness_and_similarity(source_dir, outdir, args.dpi, manifest)
    qualitative_source = legacy_image_dir / "fig_cam_prior_occlusion_examples.png"
    if qualitative_source.exists():
        figure6_gradcam_prior_examples(legacy_image_dir, outdir, args.dpi, manifest)
    elif args.require_qualitative_assets:
        raise FileNotFoundError(
            f"Missing Grad-CAM/prior example source plate: {qualitative_source}"
        )
    else:
        print(
            "Skipped the qualitative Grad-CAM example figure because licensed image assets "
            f"were not found at {qualitative_source}."
        )
    build_contact_sheet(outdir, manifest, args.dpi)
    if args.make_pptx:
        build_review_pptx(outdir, manifest)
    write_manifest(outdir, manifest)
    print(f"Wrote {len(manifest)} composite figures to {outdir}")


if __name__ == "__main__":
    main()
