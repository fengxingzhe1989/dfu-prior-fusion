#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper_assets/source_tables"
MAIN_OUT = ROOT / "outputs/manuscript_figures"
SUPP_OUT = ROOT / "outputs/supplementary_figures"
AUDIT_OUT = ROOT / "outputs/figure_qa"
IMAGE_ASSETS = ROOT / "paper_assets/figure_inputs"
DPI = 600

COLORS = {
    "navy": "#102A56",
    "blue": "#4B6FA8",
    "teal": "#2C8C84",
    "teal_light": "#BFE1DD",
    "purple": "#6D5FA7",
    "gray": "#737B8A",
    "gray_light": "#E6E8EF",
    "text": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the prior-mask support figure and selected supplementary panels."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--main-output-dir", type=Path, default=MAIN_OUT)
    parser.add_argument("--supp-output-dir", type=Path, default=SUPP_OUT)
    parser.add_argument("--audit-output-dir", type=Path, default=AUDIT_OUT)
    parser.add_argument(
        "--image-assets",
        type=Path,
        default=IMAGE_ASSETS,
        help=(
            "Directory containing licensed DFUC-derived RGB/prior/overlay PNGs. "
            "These images are not redistributed with the public code package."
        ),
    )
    parser.add_argument(
        "--require-image-assets",
        action="store_true",
        help=(
            "Fail when the licensed DFUC-derived image inputs for the qualitative "
            "main-figure panel are unavailable. By default that panel is skipped "
            "and the quantitative supplementary panels are still generated."
        ),
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
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def panel_label(ax: Axes, label: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color=COLORS["text"],
    )


def save_all(fig: plt.Figure, outdir: Path, stem: str) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for suffix in ("svg", "pdf", "png", "tiff"):
        path = outdir / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if suffix in {"png", "tiff"}:
            kwargs["dpi"] = DPI
        fig.savefig(path, **kwargs)
        paths[suffix] = str(path)
    plt.close(fig)
    return paths


def build_s3(source_dir: Path, output_dir: Path) -> dict[str, str]:
    frame = pd.read_csv(source_dir / "table_reliability_fusion.csv")
    boot = pd.read_csv(source_dir / "paired_bootstrap_significance.csv")

    method_map = {
        "Pure cls ensemble": "Moderate RGB stack",
        "Cls + prior residual": "Moderate RGB + prior features",
        "LPT output ensemble": "Moderate output-only LPT stack",
        "LPT ablations + prior stats": "RA-LPT",
    }
    order = list(method_map)
    frame = frame.set_index("method").loc[order].reset_index()
    frame["label"] = frame["method"].map(method_map)

    boot = boot[(boot["candidate"] == "LPT ablations + prior stats") & (boot["metric"] == "macro_f1")].copy()
    baseline_order = ["ResNet50", "Pure cls ensemble", "LPT output ensemble"]
    boot = boot.set_index("baseline").loc[baseline_order].reset_index()
    boot["label"] = boot["baseline"].map(
        {
            "ResNet50": "ResNet50",
            "Pure cls ensemble": "Moderate RGB stack",
            "LPT output ensemble": "Moderate output-only LPT stack",
        }
    )

    fig = plt.figure(figsize=(7.25, 3.35), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.08])

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(frame))
    values = frame["macro_f1"].to_numpy(float)
    errors = frame["macro_f1_std"].to_numpy(float)
    colors = [COLORS["blue"], COLORS["teal_light"], COLORS["teal"], COLORS["navy"]]
    ax.barh(y, values, xerr=errors, color=colors, edgecolor="white", linewidth=0.6, capsize=2.5, height=0.62)
    for ypos, value in zip(y, values):
        ax.text(value + 0.0012, ypos + 0.10, f"{value:.3f}", ha="left", va="bottom", fontsize=7.0)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"])
    ax.set_xlim(0.90, 0.98)
    ax.set_xticks(np.arange(0.90, 0.981, 0.02))
    ax.set_xlabel("Macro-F1")
    ax.set_title("Standard-CV moderate-model stacks")
    clean_axis(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(len(boot))
    delta = boot["observed_delta"].to_numpy(float)
    errors = np.vstack(
        [
            delta - boot["ci95_low"].to_numpy(float),
            boot["ci95_high"].to_numpy(float) - delta,
        ]
    )
    ax.axvline(0, color=COLORS["gray"], linewidth=0.8, linestyle="--", zorder=0)
    ax.errorbar(
        delta,
        y,
        xerr=errors,
        fmt="o",
        markersize=5.0,
        color=COLORS["navy"],
        ecolor=COLORS["teal"],
        capsize=3.0,
        elinewidth=1.25,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(boot["label"])
    ax.set_ylim(-0.5, len(boot) - 0.5)
    ax.set_xlim(-0.003, 0.018)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "0" if abs(value) < 1e-12 else f"{value:.3f}"))
    ax.set_xlabel("Observed Macro-F1 difference")
    ax.set_title("RA-LPT minus comparator")
    for xpos, ypos, pvalue in zip(delta, y, boot["two_sided_p"]):
        ax.text(xpos + 0.00055, ypos + 0.14, f"p = {float(pvalue):.3g}", fontsize=6.4, color=COLORS["gray"])
    clean_axis(ax)
    panel_label(ax, "b")

    return save_all(fig, output_dir, "supp_fig_s03_reliability_fusion_main_style")


def build_s6(source_dir: Path, output_dir: Path) -> dict[str, str]:
    sim = pd.read_csv(source_dir / "similarity_sensitivity_metrics.csv")
    overlap = pd.read_csv(
        source_dir / "reviewer_revision_v4/derived/cross_dataset_overlap_exclusion_overall_metrics_matched.csv"
    )

    fig = plt.figure(figsize=(7.25, 3.60), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.02])

    ax = fig.add_subplot(gs[0, 0])
    methods = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong cls ensemble", "Strong LPT output ensemble"]
    labels = ["ResNet50", "EffNetV2-S", "RA-LPT", "Strong RGB\nstack", "Strong LPT\noutput"]
    x = np.arange(len(methods))
    width = 0.34
    series = [
        (-width / 2, "all_labeled_oof", COLORS["blue"], "All standard-CV OOF (n = 5,955)"),
        (width / 2, "excluding_cross_fold_same_label_near_duplicates", COLORS["teal"], "Exclude screened images (n = 5,625)"),
    ]
    for offset, subset, color, label in series:
        values = []
        for method in methods:
            row = sim[(sim["method"] == method) & (sim["subset"] == subset)]
            values.append(float(row["macro_f1"].iloc[0]))
        bars = ax.bar(x + offset, values, width=width, color=color, label=label, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0013, f"{value:.3f}", ha="center", va="bottom", fontsize=5.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.90, 0.98)
    ax.set_yticks(np.arange(0.90, 0.981, 0.02))
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model")
    ax.set_title("Standard-CV near-duplicate exclusion")
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8)
    clean_axis(ax)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    method_specs = [
        ("Post hoc prior-informed full-OOF candidate", "Prior-informed full-OOF", COLORS["navy"]),
        ("Matched strong RGB stack", "Matched RGB stack", COLORS["blue"]),
        ("Matched strong RGB + explicit prior features", "Matched RGB + prior", COLORS["teal"]),
    ]
    cohort_order = ["all_5955", "exclude_241_broad_overlap"]
    x = np.array([0.0, 1.0])
    for method, label, color in method_specs:
        sub = overlap[overlap["method"] == method].set_index("cohort").loc[cohort_order]
        values = sub["macro_f1"].to_numpy(float)
        ax.plot(x, values, marker="o", markersize=5.2, linewidth=1.7, color=color, label=label)
        for xpos, value in zip(x, values):
            offset = 0.00045 if method == "Matched strong RGB stack" else -0.00048
            ax.text(
                xpos,
                value + offset,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color=color,
                fontweight="bold" if method.startswith("Post hoc") else "normal",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["All grouped OOF\n(n = 5,955)", "Exclude 241\n(n = 5,714)"])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(0.945, 0.966)
    ax.set_yticks(np.arange(0.945, 0.9661, 0.005))
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Analysis set")
    ax.set_title("Cross-dataset broad-overlap exclusion")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.50), fontsize=5.7, handlelength=1.8)
    ax.text(0.5, 0.9650, "Prediction-level only; no retraining or re-selection", ha="center", va="center", fontsize=5.8, color=COLORS["gray"])
    clean_axis(ax)
    panel_label(ax, "b")

    return save_all(fig, output_dir, "supp_fig_s06_similarity_sensitivity_main_style")


def build_main_figure3(source_dir: Path, image_assets: Path, output_dir: Path) -> dict[str, str]:
    cam = pd.read_csv(source_dir / "reviewer_revision_v2/derived/interpretability_mask_support_audit.csv")
    priors = pd.read_csv(source_dir / "prior_reliability_strata_metrics.csv")
    strong = pd.read_csv(source_dir / "strong_prior_reliability_strata_metrics.csv")

    fig = plt.figure(figsize=(7.35, 6.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0], height_ratios=[1.08, 0.92])

    panel = fig.add_subplot(gs[0, 0])
    panel.set_axis_off()
    panel_label(panel, "a", -0.04, 1.02)
    panel.text(0.0, 1.02, "Illustrative DFUC2021 prior examples", transform=panel.transAxes, ha="left", va="bottom", fontsize=8.6)
    cases = [("infection", "dfuc2021_infection_304467"), ("both", "dfuc2021_both_303735")]
    columns = [("RGB", "rgb"), ("Soft prior", "prior"), ("Prior overlay", "overlay")]
    x0, width, height, gap = 0.045, 0.27, 0.40, 0.055
    y_positions = [0.50, 0.055]
    for row_index, (case_label, stem) in enumerate(cases):
        for column_index, (title, suffix) in enumerate(columns):
            ax_img = panel.inset_axes([x0 + column_index * (width + gap), y_positions[row_index], width, height])
            image_path = image_assets / f"{stem}_{suffix}.png"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing licensed qualitative figure input: {image_path}. "
                    "Regenerate it from the DFUC data before building this panel."
                )
            with Image.open(image_path) as image:
                ax_img.imshow(image.convert("RGB"))
            ax_img.set_axis_off()
            if row_index == 0:
                ax_img.set_title(title, fontsize=7.2, pad=3)
            if column_index == 0:
                ax_img.text(-0.05, 0.5, case_label, transform=ax_img.transAxes, ha="right", va="center", rotation=90, fontsize=6.6, color=COLORS["gray"])

    ax = fig.add_subplot(gs[0, 1])
    rel = pd.concat([priors, strong], ignore_index=True)
    rel = rel[(rel["group_type"] == "fallback_group") & rel["method"].isin(["ResNet50", "EfficientNetV2-S", "Strong RA-LPT fusion", "Reliability-aware LPT fusion"])]
    methods = ["ResNet50", "EfficientNetV2-S", "Reliability-aware LPT fusion", "Strong RA-LPT fusion"]
    labels = ["ResNet50", "EffNetV2-S", "RA-LPT", "Strong\nRA-LPT"]
    x = np.arange(len(methods))
    width = 0.34
    for offset, group, color, label in [(-width / 2, "fallback", COLORS["gray"], "fallback"), (width / 2, "non_fallback", COLORS["teal"], "non-fallback")]:
        values = [float(rel[(rel["method"] == method) & (rel["group"] == group)]["macro_f1"].iloc[0]) for method in methods]
        ax.bar(x + offset, values, width=width, color=color, label=label, edgecolor="white", linewidth=0.6)
        for xpos, value in zip(x + offset, values):
            ax.text(xpos, value + 0.0020, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=7.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0.88, 1.00)
    ax.set_yticks(np.arange(0.88, 1.001, 0.02))
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_ylabel("Macro-F1", fontsize=8.0)
    ax.set_title("Standard-CV OOF performance by prior status")
    ax.legend(loc="upper left", ncol=2, fontsize=6.8)
    clean_axis(ax)
    panel_label(ax, "b")

    subset = cam[cam["stratum"].isin(["fallback_nonempty_mask", "nonfallback"]) & cam["method"].isin(["ResNet50", "EfficientNetV2-S"])].copy()
    method_order = ["ResNet50", "EfficientNetV2-S"]
    x = np.arange(2)
    width = 0.34

    ax = fig.add_subplot(gs[1, 0])
    enrichment = subset.pivot(index="method", columns="stratum", values="mean_cam_mass_enrichment_nonempty").loc[method_order]
    fallback = enrichment["fallback_nonempty_mask"].to_numpy(float)
    nonfallback = enrichment["nonfallback"].to_numpy(float)
    ax.bar(x - width / 2, fallback, width=width, color=COLORS["gray"], label="fallback, non-empty", edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, nonfallback, width=width, color=COLORS["teal"], label="non-fallback", edgecolor="white", linewidth=0.6)
    for xpos, value in zip(np.r_[x - width / 2, x + width / 2], np.r_[fallback, nonfallback]):
        ax.text(xpos, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(0, 3.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet50", "EffNetV2-S"], fontsize=7.6)
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_ylabel("CAM-mass enrichment over mask area", fontsize=8.0)
    ax.set_title("Area-normalised CAM enrichment\n(non-empty masks only)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.8)
    clean_axis(ax)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    effects = subset.pivot(index="method", columns="stratum", values="prior_minus_random_drop").loc[method_order]
    lower = subset.pivot(index="method", columns="stratum", values="delta_ci95_low").loc[method_order]
    upper = subset.pivot(index="method", columns="stratum", values="delta_ci95_high").loc[method_order]
    fallback = effects["fallback_nonempty_mask"].to_numpy(float)
    nonfallback = effects["nonfallback"].to_numpy(float)
    fallback_error = np.vstack([fallback - lower["fallback_nonempty_mask"].to_numpy(float), upper["fallback_nonempty_mask"].to_numpy(float) - fallback])
    nonfallback_error = np.vstack([nonfallback - lower["nonfallback"].to_numpy(float), upper["nonfallback"].to_numpy(float) - nonfallback])
    ax.bar(x - width / 2, fallback, yerr=fallback_error, capsize=2.5, width=width, color=COLORS["gray"], label="fallback, non-empty", edgecolor="white", linewidth=0.6)
    ax.bar(x + width / 2, nonfallback, yerr=nonfallback_error, capsize=2.5, width=width, color=COLORS["teal"], label="non-fallback", edgecolor="white", linewidth=0.6)
    for xpos, value in zip(np.r_[x - width / 2, x + width / 2], np.r_[fallback, nonfallback]):
        ax.text(
            xpos,
            value - 0.004,
            f"{value:.3f}",
            ha="center",
            va="top",
            fontsize=7.2,
            color="white",
            fontweight="bold",
        )
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["ResNet50", "EffNetV2-S"], fontsize=7.6)
    ax.set_ylim(0, 0.12)
    ax.set_yticks(np.arange(0, 0.121, 0.02))
    ax.tick_params(axis="y", labelsize=7.4)
    ax.set_ylabel("Prior minus random confidence drop", fontsize=8.0)
    ax.set_title("Prior-minus-random occlusion sensitivity")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.8)
    clean_axis(ax)
    panel_label(ax, "d")

    return save_all(fig, output_dir, "Figure5_prior_mask_support_and_occlusion")


def missing_qualitative_inputs(image_assets: Path) -> list[Path]:
    stems = ("dfuc2021_infection_304467", "dfuc2021_both_303735")
    suffixes = ("rgb", "prior", "overlay")
    return [
        image_assets / f"{stem}_{suffix}.png"
        for stem in stems
        for suffix in suffixes
        if not (image_assets / f"{stem}_{suffix}.png").exists()
    ]


def main() -> None:
    args = parse_args()
    configure_style()
    source_dir = args.source_dir.resolve()
    main_output_dir = args.main_output_dir.resolve()
    supp_output_dir = args.supp_output_dir.resolve()
    audit_output_dir = args.audit_output_dir.resolve()
    image_assets = args.image_assets.resolve()
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    exports: dict[str, object] = {
        "supplementary_reliability_panel": build_s3(source_dir, supp_output_dir),
        "supplementary_similarity_panel": build_s6(source_dir, supp_output_dir),
    }
    missing_inputs = missing_qualitative_inputs(image_assets)
    if missing_inputs:
        message = (
            "Licensed qualitative DFUC image inputs are unavailable; skipped the "
            "qualitative main-figure panel. Missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )
        if args.require_image_assets:
            raise FileNotFoundError(message)
        exports["figure5"] = {"status": "skipped", "reason": message}
    else:
        exports["figure5"] = build_main_figure3(source_dir, image_assets, main_output_dir)
    (audit_output_dir / "figure_export_manifest.json").write_text(
        json.dumps(exports, indent=2), encoding="utf-8"
    )
    print(json.dumps(exports, indent=2))


if __name__ == "__main__":
    main()
