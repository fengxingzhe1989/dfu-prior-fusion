#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit near-duplicate DFUC2021 images across CV folds.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def phash_bits(path: Path) -> np.ndarray:
    image = read_gray(path, (32, 32))
    dct = cv2.dct(image)
    low = dct[:8, :8].copy()
    values = low.reshape(-1)
    median = np.median(values[1:])
    return values > median


def dhash_bits(path: Path) -> np.ndarray:
    image = read_gray(path, (9, 8))
    return (image[:, 1:] > image[:, :-1]).reshape(-1)


def collect_pairs(
    frame: pd.DataFrame,
    phash_matrix: np.ndarray,
    dhash_matrix: np.ndarray,
    phash_threshold: int,
    dhash_threshold: int,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n = len(frame)
    for idx in range(n - 1):
        p_dist = np.count_nonzero(phash_matrix[idx + 1 :] != phash_matrix[idx], axis=1)
        d_dist = np.count_nonzero(dhash_matrix[idx + 1 :] != dhash_matrix[idx], axis=1)
        hits = np.flatnonzero((p_dist <= phash_threshold) | (d_dist <= dhash_threshold))
        for rel in hits:
            j = idx + 1 + int(rel)
            rows.append(
                {
                    "image_a": frame.at[idx, "image"],
                    "image_b": frame.at[j, "image"],
                    "fold_a": int(frame.at[idx, "fold"]),
                    "fold_b": int(frame.at[j, "fold"]),
                    "label_a": frame.at[idx, "label_name"],
                    "label_b": frame.at[j, "label_name"],
                    "same_fold": int(frame.at[idx, "fold"]) == int(frame.at[j, "fold"]),
                    "same_label": frame.at[idx, "label_name"] == frame.at[j, "label_name"],
                    "phash_distance": int(p_dist[rel]),
                    "dhash_distance": int(d_dist[rel]),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "image_a",
                "image_b",
                "fold_a",
                "fold_b",
                "label_a",
                "label_b",
                "same_fold",
                "same_label",
                "phash_distance",
                "dhash_distance",
            ]
        )
    pairs = pd.DataFrame(rows)
    pairs["risk_group"] = np.select(
        [
            (~pairs["same_fold"]) & pairs["same_label"],
            (~pairs["same_fold"]) & (~pairs["same_label"]),
            pairs["same_fold"] & pairs["same_label"],
        ],
        ["cross_fold_same_label", "cross_fold_different_label", "within_fold_same_label"],
        default="within_fold_different_label",
    )
    return pairs.sort_values(["phash_distance", "dhash_distance", "risk_group"]).head(top_k)


def render_similarity_figure(summary: dict[str, object], output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        "exact duplicate\ngroups",
        "near-duplicate\npairs",
        "cross-fold\nnear pairs",
        "cross-fold\nsame-label pairs",
    ]
    values = [
        int(summary["exact_duplicate_groups"]),
        int(summary["near_duplicate_pairs"]),
        int(summary["cross_fold_near_duplicate_pairs"]),
        int(summary["cross_fold_same_label_near_duplicate_pairs"]),
    ]
    colors = ["#6c757d", "#2f6f9f", "#d58936", "#9b3d3d"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values + [1]) * 0.03, str(value), ha="center", va="bottom")
    ax.set_ylabel("Count")
    ax.set_title("DFUC2021 CV Split Image-Similarity Audit")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_similarity_leakage_audit.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "fig_similarity_leakage_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    text = f"""# DFUC2021 Similar-Image Leakage Audit

Generated by `experiments/reliability_aware_prior_fusion/audit_image_similarity.py`.

## Scope

This audit checks the 5-fold DFUC2021 classification manifest used in the paper. It is designed to detect obvious leakage risk from exact duplicate files or near-duplicate images being split across different validation folds.

## Method

- Exact duplicates were detected by SHA256 over image bytes.
- Near duplicates were screened with two low-level perceptual hashes computed from the image pixels:
  - pHash from an 8x8 low-frequency DCT signature, threshold <= {summary["phash_threshold"]}.
  - dHash from adjacent grayscale intensity differences, threshold <= {summary["dhash_threshold"]}.
- A pair is considered higher risk when it is both cross-fold and same-label.

These checks are conservative screening tools. They do not prove patient independence and they may miss semantically similar but visually transformed images.

## Summary

| Item | Value |
|---|---:|
| Labeled images audited | {summary["samples"]} |
| Exact duplicate groups | {summary["exact_duplicate_groups"]} |
| Images involved in exact duplicate groups | {summary["exact_duplicate_images"]} |
| Near-duplicate candidate pairs | {summary["near_duplicate_pairs"]} |
| Cross-fold near-duplicate candidate pairs | {summary["cross_fold_near_duplicate_pairs"]} |
| Cross-fold same-label near-duplicate candidate pairs | {summary["cross_fold_same_label_near_duplicate_pairs"]} |
| Cross-fold different-label near-duplicate candidate pairs | {summary["cross_fold_different_label_near_duplicate_pairs"]} |

## Interpretation

If the cross-fold same-label count is zero or very small, the current split has no obvious low-level duplicate leakage under this audit. If it is non-zero, inspect `similar_image_audit_pairs.csv` manually and consider a sensitivity analysis that removes or groups those images before cross-validation.

The manuscript should still describe the experiment as internal cross-validation, not as external validation, because this audit does not establish patient-level independence.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = resolve(args.manifest, repo_root)
    output_dir = resolve(args.output_dir, repo_root)
    figure_dir = resolve(args.figure_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(manifest_path)
    frame["abs_image_path"] = frame["image_path"].map(lambda value: str(resolve(Path(value), repo_root)))
    frame = frame.reset_index(drop=True)

    sha_values: list[str] = []
    phash_rows: list[np.ndarray] = []
    dhash_rows: list[np.ndarray] = []
    for path_str in frame["abs_image_path"]:
        path = Path(path_str)
        sha_values.append(sha256_file(path))
        phash_rows.append(phash_bits(path))
        dhash_rows.append(dhash_bits(path))

    frame["sha256"] = sha_values
    phash_matrix = np.vstack(phash_rows)
    dhash_matrix = np.vstack(dhash_rows)

    exact_groups = frame.groupby("sha256").filter(lambda group: len(group) > 1)
    exact_group_count = int(exact_groups["sha256"].nunique()) if not exact_groups.empty else 0
    exact_groups[["image", "fold", "label_name", "sha256"]].to_csv(output_dir / "similar_image_audit_exact_duplicates.csv", index=False)

    pairs = collect_pairs(
        frame,
        phash_matrix,
        dhash_matrix,
        args.phash_threshold,
        args.dhash_threshold,
        args.top_k,
    )
    pairs.to_csv(output_dir / "similar_image_audit_pairs.csv", index=False)

    cross_fold = pairs[pairs["same_fold"] == False] if not pairs.empty else pairs
    cross_fold_same = cross_fold[cross_fold["same_label"] == True] if not cross_fold.empty else cross_fold
    cross_fold_diff = cross_fold[cross_fold["same_label"] == False] if not cross_fold.empty else cross_fold
    summary: dict[str, object] = {
        "samples": int(len(frame)),
        "manifest": str(args.manifest),
        "phash_threshold": int(args.phash_threshold),
        "dhash_threshold": int(args.dhash_threshold),
        "exact_duplicate_groups": exact_group_count,
        "exact_duplicate_images": int(len(exact_groups)),
        "near_duplicate_pairs": int(len(pairs)),
        "cross_fold_near_duplicate_pairs": int(len(cross_fold)),
        "cross_fold_same_label_near_duplicate_pairs": int(len(cross_fold_same)),
        "cross_fold_different_label_near_duplicate_pairs": int(len(cross_fold_diff)),
    }
    (output_dir / "similar_image_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "similar_image_audit_summary.md")
    render_similarity_figure(summary, figure_dir, args.dpi)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
