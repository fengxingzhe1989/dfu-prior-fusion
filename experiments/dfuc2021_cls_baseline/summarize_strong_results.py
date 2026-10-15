#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_NAMES = ("macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy")


@dataclass(frozen=True)
class SummarySpec:
    method: str
    file_name: str
    group: str


SUMMARY_SPECS = (
    SummarySpec("ResNet50", "dfuc2021_cls_resnet50_cv5_v1_cv_summary.csv", "reference"),
    SummarySpec("EfficientNet-B0", "dfuc2021_cls_efficientnetb0_cv5_v1_cv_summary.csv", "reference"),
    SummarySpec(
        "Reliability-aware LPT fusion",
        "reliability_prior_fusion_subset_search_lpt_ablation_ensemble_cv_summary.csv",
        "reference",
    ),
    SummarySpec("ConvNeXt-Tiny", "dfuc2021_cls_convnext_tiny_cv5_v1_cv_summary.csv", "strong_classifier"),
    SummarySpec("EfficientNetV2-S", "dfuc2021_cls_efficientnetv2s_cv5_v1_cv_summary.csv", "strong_classifier"),
    SummarySpec(
        "Strong cls ensemble",
        "reliability_prior_fusion_strong_strong_cls_ensemble_cv_summary.csv",
        "strong_fusion",
    ),
    SummarySpec(
        "Strong cls + prior residual",
        "reliability_prior_fusion_strong_strong_cls_prior_residual_cv_summary.csv",
        "strong_fusion",
    ),
    SummarySpec(
        "Strong LPT output ensemble",
        "reliability_prior_fusion_strong_strong_lpt_output_ensemble_cv_summary.csv",
        "strong_fusion",
    ),
    SummarySpec(
        "Strong RA-LPT fusion",
        "reliability_prior_fusion_strong_strong_lpt_prior_fusion_cv_summary.csv",
        "strong_fusion",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize strong DFUC2021 classifier follow-up results.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/paper_figures"))
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def read_summary(path: Path, spec: SummarySpec) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    mean_row = next((row for row in rows if row["fold"] == "mean"), None)
    std_row = next((row for row in rows if row["fold"] == "std"), None)
    oof_row = next((row for row in rows if row["fold"] == "oof"), None)
    if mean_row is None or std_row is None:
        return None
    result: dict[str, Any] = {"method": spec.method, "group": spec.group}
    for metric in METRIC_NAMES:
        result[metric] = float(mean_row[metric])
        result[f"{metric}_std"] = float(std_row[metric])
        if oof_row is not None and oof_row.get(metric, "") != "":
            result[f"{metric}_oof"] = float(oof_row[metric])
    return result


def build_table(log_dir: Path, table_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in SUMMARY_SPECS:
        row = read_summary(log_dir / spec.file_name, spec)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No summary files found in {log_dir}")
    table_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "table_strong_followup_summary.csv", index=False)
    return frame


def render_figure(frame: pd.DataFrame, figure_dir: Path) -> None:
    methods = frame["method"].tolist()
    labels = {
        "ResNet50": "ResNet50",
        "EfficientNet-B0": "EffNet-B0",
        "Reliability-aware LPT fusion": "RA-LPT",
        "ConvNeXt-Tiny": "ConvNeXt",
        "EfficientNetV2-S": "EffNetV2-S",
        "Strong cls ensemble": "Strong cls ens.",
        "Strong cls + prior residual": "Strong cls+prior",
        "Strong LPT output ensemble": "Strong LPT outputs",
        "Strong RA-LPT fusion": "Strong RA-LPT",
    }
    x = np.arange(len(methods))
    width = 0.36
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.2, len(methods) * 0.78), 3.6))
    for offset, metric, color in (
        (-width / 2.0, "macro_f1", "#2f6f9f"),
        (width / 2.0, "balanced_accuracy", "#d58936"),
    ):
        values = frame[metric].to_numpy(dtype=float)
        errors = frame[f"{metric}_std"].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=width, yerr=errors, capsize=2.5, label=metric.replace("_", " "), color=color)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.001, f"{value:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_ylim(0.92, 0.975)
    ax.set_ylabel("5-fold mean score")
    ax.set_title("Strong classifier follow-up")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(method, method) for method in methods], rotation=25, ha="right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_strong_followup_summary.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "fig_strong_followup_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def format_metric(row: pd.Series, metric: str) -> str:
    return f"{float(row[metric]):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def write_markdown(frame: pd.DataFrame, table_dir: Path) -> None:
    best = frame.loc[frame["macro_f1"].idxmax()]
    rows = [
        "| Method | Group | Macro-F1 | Balanced Acc | Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["method"]),
                    str(row["group"]),
                    format_metric(row, "macro_f1"),
                    format_metric(row, "balanced_accuracy"),
                    format_metric(row, "accuracy"),
                ]
            )
            + " |"
        )
    text = f"""# Strong Classifier Follow-up Summary

This file is generated after ConvNeXt-Tiny and EfficientNetV2-S finish. It is meant to decide whether strong classifiers belong in the main paper table or supplementary material.

## Table

{chr(10).join(rows)}

## Current Best Completed Method

- Method: `{best['method']}`
- Macro-F1: `{format_metric(best, 'macro_f1')}`
- Balanced Accuracy: `{format_metric(best, 'balanced_accuracy')}`
- Accuracy: `{format_metric(best, 'accuracy')}`

## How To Use

If the strong classifier follow-up beats the current reliability-aware LPT fusion, report it honestly as a stronger classification baseline and use the prior-fusion results to support the main methodological point: lesion priors are useful when treated as reliability-aware residual evidence, but hard spatial prior transfer is fragile.
"""
    (table_dir / "strong_followup_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    log_dir = resolve_path(args.log_dir, repo_root)
    table_dir = resolve_path(args.table_dir, repo_root)
    figure_dir = resolve_path(args.figure_dir, repo_root)
    frame = build_table(log_dir, table_dir)
    render_figure(frame, figure_dir)
    write_markdown(frame, table_dir)
    print(f"Saved strong follow-up summary to {table_dir}", flush=True)


if __name__ == "__main__":
    main()
