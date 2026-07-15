#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.lines import Line2D


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
COLORS = {
    "navy": "#173B6C",
    "blue": "#557DB8",
    "teal": "#2E9188",
    "purple": "#735FA7",
    "gold": "#C9943B",
    "gray": "#727B8B",
    "gray_light": "#D8DEE7",
    "text": "#111827",
    "blue_fill": "#F1F5FB",
    "teal_fill": "#EEF8F6",
    "gold_fill": "#FCF7EC",
    "gray_fill": "#F5F7FA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild manuscript Figure 4 from audited source tables.")
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
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str, x: float = -0.075, y: float = 1.035) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
        color=COLORS["text"],
        clip_on=False,
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(width=2.0, length=4.5)


def flow_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    edge: str,
    face: str,
    title_size: float = 9.0,
    subtitle_size: float = 8.0,
    linewidth: float = 2.0,
) -> None:
    x, y = xy
    box = patches.FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * (0.68 if subtitle else 0.50),
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["text"],
        linespacing=0.95,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.18,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["gray"],
            linespacing=0.92,
        )


def flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    *,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 2.1,
            "mutation_scale": 12,
            "shrinkA": 2,
            "shrinkB": 2,
            "connectionstyle": connectionstyle,
        },
    )


def save_all(fig: plt.Figure, output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure4_similarity_grouping_and_robustness"
    metadata = {"Creator": "Matplotlib", "Title": "Similarity grouping and robustness audits"}
    fig.savefig(stem.with_suffix(".svg"), metadata=metadata)
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, metadata=metadata)
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, pil_kwargs={"compression": "tiff_lzw"})


def load_inputs(source_dir: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    audit_dir = source_dir / "methodology_audit_v7"
    with (audit_dir / "similarity_group_component_summary.json").open("r", encoding="utf-8") as handle:
        component = json.load(handle)
    with (audit_dir / "dfuc2021_dfuc2022_cross_dataset_duplicate_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        cross = json.load(handle)
    mixed = pd.read_csv(
        source_dir / "reviewer_revision_v3" / "derived" / "mixed_label_component_exclusion_sensitivity.csv"
    )
    cross_exclusion = pd.read_csv(
        source_dir
        / "reviewer_revision_v4"
        / "derived"
        / "cross_dataset_overlap_exclusion_overall_metrics_matched.csv"
    )
    return component, cross, mixed, cross_exclusion


def draw_grouping_panel(ax: plt.Axes, component: dict, mixed_excluded_images: int) -> None:
    ax.set_axis_off()
    ax.set_title("Image-similarity grouping", loc="left", pad=7, color=COLORS["navy"], fontweight="bold")

    flow_box(
        ax,
        (0.00, 0.63),
        0.29,
        0.23,
        f"{component['samples']:,} images",
        "DFUC2021",
        edge=COLORS["blue"],
        face=COLORS["blue_fill"],
        title_size=8.4,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.36, 0.63),
        0.26,
        0.23,
        f"{component['near_duplicate_pairs_used']:,} candidate\nlinks",
        "pHash <= 4\nor dHash <= 6",
        edge=COLORS["gray"],
        face=COLORS["gray_fill"],
        title_size=8.0,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.69, 0.63),
        0.30,
        0.23,
        f"{component['similarity_groups']:,} groups",
        "image-similarity\ncomponents",
        edge=COLORS["teal"],
        face=COLORS["teal_fill"],
        title_size=8.4,
        subtitle_size=8.0,
    )
    flow_arrow(ax, (0.29, 0.745), (0.36, 0.745), COLORS["navy"])
    flow_arrow(ax, (0.62, 0.745), (0.69, 0.745), COLORS["navy"])

    singleton_groups = component["similarity_groups"] - component["non_singleton_groups"]
    summaries = [
        (
            (0.00, 0.32),
            f"{singleton_groups:,} singleton\ngroups",
            f"{component['non_singleton_groups']:,} non-singletons\n{component['images_in_non_singleton_groups']:,} images",
            COLORS["blue"],
            COLORS["blue_fill"],
        ),
        (
            (0.51, 0.32),
            f"Largest group: {component['largest_group_size']}",
            f"{component['different_label_near_duplicate_pairs']} different-label\nlinks",
            COLORS["gray"],
            COLORS["gray_fill"],
        ),
        (
            (0.00, 0.02),
            f"{component['mixed_label_non_singleton_groups']} mixed-label\ngroups",
            f"{mixed_excluded_images} images\nexcluded",
            COLORS["gold"],
            COLORS["gold_fill"],
        ),
        (
            (0.51, 0.02),
            "0 candidate links\nacross folds",
            f"{component['near_duplicate_pairs_same_fold_after_grouping']} links\nwithin fold",
            COLORS["teal"],
            COLORS["teal_fill"],
        ),
    ]
    for xy, title, subtitle, edge, face in summaries:
        flow_box(
            ax,
            xy,
            0.48,
            0.21,
            title,
            subtitle,
            edge=edge,
            face=face,
            title_size=8.4,
            subtitle_size=8.0,
        )
    panel_label(ax, "A", x=-0.065, y=1.04)


def draw_fold_panel(ax: plt.Axes, folds: pd.DataFrame) -> None:
    class_colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["purple"]]
    bottom = np.zeros(len(folds))
    x = np.arange(len(folds))
    for class_name, color in zip(CLASS_NAMES, class_colors):
        values = folds[class_name].to_numpy(float)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.72,
            color=color,
            label=class_name,
            edgecolor="white",
            linewidth=2.0,
        )
        text_color = COLORS["text"] if class_name == "ischaemia" else "white"
        for bar, value, base in zip(bars, values, bottom):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                base + value / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=8.1,
                color=text_color,
                fontweight="bold",
            )
        bottom += values
    for xpos, total in zip(x, bottom):
        ax.text(
            xpos,
            total + 22,
            f"n={int(total):,}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=COLORS["gray"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {int(fold)}" for fold in folds["fold"]])
    ax.set_ylabel("Images")
    ax.set_ylim(0, 1280)
    ax.set_yticks([0, 400, 800, 1200])
    ax.set_title("Class balance after similarity grouping", loc="left", pad=7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.46, -0.13),
        ncol=4,
        fontsize=8.1,
        handlelength=1.6,
        columnspacing=1.15,
    )
    clean_axis(ax)
    panel_label(ax, "B", x=-0.10, y=1.04)


def draw_cross_dataset_panel(ax: plt.Axes, cross: dict) -> None:
    ax.set_axis_off()
    ax.set_title("DFUC2021-DFUC2022 cross-dataset audit", loc="left", pad=7, color=COLORS["teal"], fontweight="bold")

    flow_box(
        ax,
        (0.00, 0.30),
        0.19,
        0.34,
        f"{cross['dfuc2021_valid_train_images']:,} x {cross['dfuc2022_seg_train_images']:,}\nimages screened",
        "DFUC2021 x DFUC2022",
        edge=COLORS["navy"],
        face=COLORS["blue_fill"],
        title_size=9.4,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.27, 0.55),
        0.20,
        0.27,
        f"SHA256 exact\n{cross['cross_dataset_exact_duplicate_pairs']} pairs",
        "",
        edge=COLORS["blue"],
        face=COLORS["blue_fill"],
        title_size=8.6,
    )
    flow_box(
        ax,
        (0.27, 0.09),
        0.20,
        0.27,
        f"Broad screen\n{cross['cross_dataset_near_duplicate_candidate_pairs']:,} pairs",
        "pHash <= 4\nor dHash <= 6",
        edge=COLORS["teal"],
        face=COLORS["teal_fill"],
        title_size=8.5,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.55, 0.09),
        0.20,
        0.27,
        f"Images involved\n{cross['dfuc2021_images_in_near_duplicate_pairs']} / {cross['dfuc2022_images_in_near_duplicate_pairs']}",
        "DFUC2021 / DFUC2022",
        edge=COLORS["teal"],
        face=COLORS["teal_fill"],
        title_size=8.5,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.78, 0.55),
        0.20,
        0.27,
        "Zero-distance screen\n0 pairs",
        "pHash = 0\nor dHash = 0",
        edge=COLORS["gray"],
        face=COLORS["gray_fill"],
        title_size=8.2,
        subtitle_size=8.0,
    )
    flow_box(
        ax,
        (0.78, 0.09),
        0.20,
        0.27,
        "Strict combined\n0 pairs",
        "pHash <= 2\nand dHash <= 4",
        edge=COLORS["gold"],
        face=COLORS["gold_fill"],
        title_size=8.8,
        subtitle_size=8.0,
    )
    flow_arrow(ax, (0.19, 0.47), (0.27, 0.68), COLORS["blue"], connectionstyle="arc3,rad=-0.08")
    flow_arrow(ax, (0.19, 0.47), (0.27, 0.23), COLORS["teal"], connectionstyle="arc3,rad=0.08")
    flow_arrow(ax, (0.47, 0.23), (0.55, 0.23), COLORS["teal"])
    flow_arrow(ax, (0.75, 0.23), (0.78, 0.23), COLORS["gold"])
    flow_arrow(ax, (0.47, 0.68), (0.78, 0.68), COLORS["gray"])
    panel_label(ax, "B", x=-0.03, y=1.04)


def draw_dumbbell_panel(
    ax: plt.Axes,
    rows: list[dict[str, float | str]],
    *,
    title: str,
    panel: str,
    xlim: tuple[float, float],
    xticks: list[float],
    show_xlabel: bool,
) -> None:
    y = np.arange(len(rows))
    all_values = np.array([float(row["full"]) for row in rows])
    retained_values = np.array([float(row["retained"]) for row in rows])

    for ypos, start, end in zip(y, all_values, retained_values):
        ax.plot([start, end], [ypos, ypos], color=COLORS["gray"], linewidth=2.2, zorder=1)
    ax.scatter(
        all_values,
        y,
        s=48,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=2.0,
        marker="o",
        zorder=3,
    )
    ax.scatter(
        retained_values,
        y,
        s=50,
        color=COLORS["teal"],
        edgecolors="white",
        linewidths=0.8,
        marker="D",
        zorder=4,
    )

    text_x = xlim[1] - (xlim[1] - xlim[0]) * 0.006
    for ypos, row in zip(y, rows):
        change = float(row["change"])
        change_text = f"{change:+.5f}" if abs(change) < 0.0001 else f"{change:+.4f}"
        ax.text(
            text_x,
            ypos,
            f"{float(row['full']):.4f} -> {float(row['retained']):.4f}  ({change_text})",
            ha="right",
            va="center",
            fontsize=8.4,
            color=COLORS["text"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.88},
        )
    ax.set_yticks(y)
    ax.set_yticklabels([str(row["label"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel("Macro-F1" if show_xlabel else "")
    ax.set_title(title, loc="left", pad=7, fontweight="bold")
    clean_axis(ax)
    panel_label(ax, panel, x=-0.10, y=1.04)


def build_figure(source_dir: Path, output_dir: Path, dpi: int) -> None:
    component, cross, mixed, cross_exclusion = load_inputs(source_dir)
    mixed_excluded_images = int(mixed["excluded_images"].iloc[0])
    cross_excluded_images = int(
        cross_exclusion.loc[cross_exclusion["cohort"] == "all_5955", "samples"].iloc[0]
        - cross_exclusion.loc[cross_exclusion["cohort"] == "exclude_241_broad_overlap", "samples"].iloc[0]
    )

    order = [
        "resnet50_grouped",
        "full_lpt_grouped",
        "lpt_output_only_grouped",
        "ra_lpt_grouped",
        "strong_rgb_grouped",
        "strong_rgb_prior_grouped",
    ]
    labels = {
        "resnet50_grouped": "ResNet50",
        "full_lpt_grouped": "Full LPT",
        "lpt_output_only_grouped": "Output-only LPT",
        "ra_lpt_grouped": "RA-LPT",
        "strong_rgb_grouped": "Strong RGB",
        "strong_rgb_prior_grouped": "Strong RGB + prior",
    }
    mixed_frame = mixed.set_index("method").loc[order].reset_index()
    mixed_rows = [
        {
            "label": labels[row.method],
            "full": float(row.full_macro_f1),
            "retained": float(row.retained_macro_f1),
            "change": float(row.macro_f1_change),
        }
        for row in mixed_frame.itertuples()
    ]

    cross_labels = {
        "Post hoc prior-informed full-OOF candidate": "Post hoc candidate",
        "Matched strong RGB stack": "Strong RGB",
        "Matched strong RGB + explicit prior features": "Strong RGB + prior",
    }
    cross_pivot = cross_exclusion.pivot(index="method", columns="cohort", values="macro_f1")
    cross_order = [
        "Post hoc prior-informed full-OOF candidate",
        "Matched strong RGB stack",
        "Matched strong RGB + explicit prior features",
    ]
    cross_rows = []
    for method in cross_order:
        start = float(cross_pivot.loc[method, "all_5955"])
        end = float(cross_pivot.loc[method, "exclude_241_broad_overlap"])
        cross_rows.append(
            {
                "label": cross_labels[method],
                "full": start,
                "retained": end,
                "change": end - start,
            }
        )

    width_in = 180 / 25.4
    height_in = 205 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[0.96, 0.72, 1.04, 0.68],
        left=0.18,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.50,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[2, 0])
    ax_d = fig.add_subplot(outer[3, 0])

    draw_grouping_panel(ax_a, component, mixed_excluded_images)
    draw_cross_dataset_panel(ax_b, cross)
    draw_dumbbell_panel(
        ax_c,
        mixed_rows,
        title=f"Mixed-label component exclusion ({mixed_excluded_images} images; n = 5,827 retained)",
        panel="C",
        xlim=(0.905, 0.995),
        xticks=[0.91, 0.93, 0.95, 0.97],
        show_xlabel=False,
    )
    draw_dumbbell_panel(
        ax_d,
        cross_rows,
        title=f"Broad cross-dataset candidate exclusion ({cross_excluded_images} images; n = 5,714 retained)",
        panel="D",
        xlim=(0.905, 0.995),
        xticks=[0.91, 0.93, 0.95, 0.97],
        show_xlabel=True,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="white",
            markeredgecolor=COLORS["gray"],
            markeredgewidth=2.0,
            markersize=6.5,
            label="All 5,955 images",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["teal"],
            markeredgecolor="white",
            markersize=6.5,
            label="After scenario-specific exclusion",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.010),
        ncol=2,
        fontsize=8.4,
        handletextpad=0.55,
        columnspacing=1.6,
    )

    save_all(fig, output_dir, dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_style()
    build_figure(args.source_dir, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
