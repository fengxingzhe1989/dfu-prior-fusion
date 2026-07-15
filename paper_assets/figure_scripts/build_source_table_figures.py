#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript audit figures from public source tables.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source_tables",
        help="Directory containing manuscript source CSV/JSON tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/source_table_figures"),
        help="Directory for generated figure files.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


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
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_grouped_performance(source_dir: Path, output_dir: Path, dpi: int) -> None:
    metrics = pd.read_csv(source_dir / "publication_final_metrics.csv")
    labels = {
        "ResNet50": "ResNet50\n5-run",
        "EfficientNetV2-S": "EffNetV2-S\n5-run",
        "ConvNeXt-Tiny": "ConvNeXt-Tiny\n5-run",
        "Full LPT": "Full LPT",
        "RA-LPT fusion": "RA-LPT",
        "Strong cls ensemble": "Strong RGB\nstack",
        "Strong cls + prior residual": "Strong RGB\n+ prior",
        "Selected prior-informed fusion": "Selected\nprior-informed",
    }
    colors = ["#2f6f9f" if "prior" not in row.lower() and "lpt" not in row.lower() else "#4f8a5b" for row in metrics["method"]]
    colors = ["#1b7f3a" if row == "Selected prior-informed fusion" else color for row, color in zip(metrics["method"], colors)]
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(metrics))
    y = metrics["macro_f1"].astype(float).to_numpy()
    ax.bar(x, y, color=colors, width=0.72)
    if "macro_f1_std" in metrics.columns:
        err = metrics["macro_f1_std"].fillna(0).astype(float).to_numpy()
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2)
    for xpos, value in zip(x, y):
        ax.text(xpos, value + 0.0012, f"{value:.3f}", ha="center", va="bottom", rotation=90, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(method, method) for method in metrics["method"]], rotation=25, ha="right")
    ax.set_ylim(0.90, max(0.972, float(y.max()) + 0.006))
    ax.set_ylabel("Macro-F1")
    ax.set_title("Similarity-grouped DFUC2021 performance")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, output_dir, "source_table_grouped_performance", dpi)


def build_bootstrap(source_dir: Path, output_dir: Path, dpi: int) -> None:
    boot = pd.read_csv(source_dir / "methodology_audit_v7" / "publication_final_group_level_bootstrap.csv")
    rows = boot.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    x = rows["observed_delta"].astype(float).to_numpy()
    xerr = np.vstack([x - rows["ci95_low"].astype(float).to_numpy(), rows["ci95_high"].astype(float).to_numpy() - x])
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
    save(fig, output_dir, "source_table_group_level_bootstrap", dpi)


def build_confusion(source_dir: Path, output_dir: Path, dpi: int) -> None:
    cm = pd.read_csv(source_dir / "publication_final_confusion_matrix.csv", index_col=0)
    matrix = cm.to_numpy(dtype=float)
    row_sum = np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    normalized = matrix / row_sum
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(normalized, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Selected-candidate confusion matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}\n{normalized[i, j] * 100:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
    fig.tight_layout()
    save(fig, output_dir, "source_table_selected_confusion_matrix", dpi)


def build_interpretability(source_dir: Path, output_dir: Path, dpi: int) -> None:
    summary = pd.read_csv(source_dir / "cam_prior_occlusion_summary.csv")
    subset = summary[(summary["group_type"] == "fallback") & (summary["method"].isin(["ResNet50", "EfficientNetV2-S"]))].copy()
    metrics = [
        ("cam_prior_mass_ratio", "CAM-prior mass ratio"),
        ("pointing_game_accuracy", "Pointing-game accuracy"),
        ("prior_minus_random_drop", "Prior minus random occlusion drop"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10.5, 3.2), sharex=False)
    for ax, (column, title) in zip(axes, metrics):
        pivot = subset.pivot(index="method", columns="group", values=column).loc[["ResNet50", "EfficientNetV2-S"]]
        x = np.arange(len(pivot.index))
        width = 0.36
        ax.bar(x - width / 2, pivot["fallback"], width=width, label="fallback", color="#8f9aa3")
        ax.bar(x + width / 2, pivot["non_fallback"], width=width, label="non-fallback", color="#1b7f3a")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, output_dir, "source_table_interpretability_summary", dpi)


def main() -> None:
    args = parse_args()
    configure_style()
    build_grouped_performance(args.source_dir, args.output_dir, args.dpi)
    build_bootstrap(args.source_dir, args.output_dir, args.dpi)
    build_confusion(args.source_dir, args.output_dir, args.dpi)
    build_interpretability(args.source_dir, args.output_dir, args.dpi)
    print(f"Wrote source-table figures to {args.output_dir}")


if __name__ == "__main__":
    main()
