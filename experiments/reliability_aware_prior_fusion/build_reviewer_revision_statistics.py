#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reviewer-requested grouped bootstrap and prior-mask support audits."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("paper_assets/source_tables/reviewer_revision_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_assets/source_tables/reviewer_revision_v2/derived"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def metadata_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def normalise_label(value: object) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().lower()
    if text in LABEL_TO_ID:
        return LABEL_TO_ID[text]
    return int(text)


def read_predictions(path: Path, prediction_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    true_column = next((column for column in ("label_name", "true_label", "label") if column in frame), None)
    if true_column is None or "pred_label" not in frame or "image" not in frame:
        raise ValueError(f"Prediction file lacks image/true/predicted labels: {path}")
    output = frame[["image", true_column, "pred_label"]].copy()
    output.columns = ["image", "label", prediction_name]
    output["image"] = output["image"].astype(str)
    output["label"] = output["label"].map(normalise_label)
    output[prediction_name] = output[prediction_name].map(normalise_label)
    if output["image"].duplicated().any():
        raise ValueError(f"Duplicate image rows in {path}")
    return output


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    encoded = len(CLASS_NAMES) * y_true.astype(int) + y_pred.astype(int)
    return np.bincount(encoded, minlength=len(CLASS_NAMES) ** 2).reshape(4, 4).astype(float)


def macro_f1_from_matrix(matrix: np.ndarray) -> float:
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    class_f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return float(class_f1.mean())


def group_matrices(frame: pd.DataFrame, prediction_column: str) -> np.ndarray:
    matrices = []
    for _, group in frame.groupby("similarity_group", sort=True):
        matrices.append(confusion_matrix(group["label"].to_numpy(), group[prediction_column].to_numpy()))
    return np.stack(matrices)


def paired_group_bootstrap(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int | str]:
    baseline_groups = group_matrices(frame, baseline_column)
    candidate_groups = group_matrices(frame, candidate_column)
    if baseline_groups.shape != candidate_groups.shape:
        raise ValueError("Baseline and candidate group matrices are not aligned.")
    groups = baseline_groups.shape[0]
    baseline_point = macro_f1_from_matrix(baseline_groups.sum(axis=0))
    candidate_point = macro_f1_from_matrix(candidate_groups.sum(axis=0))
    observed = candidate_point - baseline_point
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    probabilities = np.full(groups, 1.0 / groups)
    for index in range(iterations):
        counts = rng.multinomial(groups, probabilities)
        baseline_matrix = np.tensordot(counts, baseline_groups, axes=(0, 0))
        candidate_matrix = np.tensordot(counts, candidate_groups, axes=(0, 0))
        deltas[index] = macro_f1_from_matrix(candidate_matrix) - macro_f1_from_matrix(baseline_matrix)
    low, high = np.percentile(deltas, [2.5, 97.5])
    lower_tail = (np.count_nonzero(deltas <= 0) + 1) / (iterations + 1)
    upper_tail = (np.count_nonzero(deltas >= 0) + 1) / (iterations + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "baseline": baseline_column,
        "candidate": candidate_column,
        "samples": int(len(frame)),
        "similarity_groups": int(groups),
        "baseline_macro_f1": baseline_point,
        "candidate_macro_f1": candidate_point,
        "observed_delta": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "two_sided_p": float(p_value),
        "bootstrap_iterations": int(iterations),
        "seed": int(seed),
    }


def build_bootstrap_table(source_dir: Path, group_manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    specs = {
        "resnet50_grouped": "resnet50_grouped_oof_predictions.csv",
        "ra_lpt_grouped": "ra_lpt_grouped_oof_predictions.csv",
        "lpt_output_only_grouped": "lpt_ablation_output_grouped_oof_predictions.csv",
        "strong_rgb_grouped": "strong_cls_grouped_oof_predictions.csv",
        "strong_rgb_prior_grouped": "strong_cls_prior_grouped_oof_predictions.csv",
    }
    frames = {
        name: read_predictions(source_dir / filename, name)
        for name, filename in specs.items()
    }
    reference = group_manifest[["image", "similarity_group"]].copy()
    reference["image"] = reference["image"].astype(str)
    for name, frame in frames.items():
        label_column = f"label_{name}"
        frame = frame.rename(columns={"label": label_column})
        reference = reference.merge(frame, on="image", how="inner", validate="one_to_one")
    label_columns = [column for column in reference if column.startswith("label_")]
    for column in label_columns[1:]:
        if not np.array_equal(reference[label_columns[0]].to_numpy(), reference[column].to_numpy()):
            raise ValueError(f"True labels disagree between prediction files: {column}")
    reference["label"] = reference[label_columns[0]].astype(int)
    comparisons = [
        ("resnet50_grouped", "ra_lpt_grouped"),
        ("lpt_output_only_grouped", "ra_lpt_grouped"),
        ("strong_rgb_grouped", "strong_rgb_prior_grouped"),
    ]
    rows = []
    for offset, (baseline, candidate) in enumerate(comparisons):
        rows.append(
            paired_group_bootstrap(
                reference,
                baseline,
                candidate,
                args.bootstrap_iters,
                args.seed + offset,
            )
        )
    return pd.DataFrame(rows)


def bootstrap_group_mean(
    frame: pd.DataFrame,
    value_column: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    grouped = frame.groupby("similarity_group")[value_column].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    group_count = len(grouped)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    probabilities = np.full(group_count, 1.0 / group_count)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        weights = rng.multinomial(group_count, probabilities)
        estimates[index] = float(np.dot(weights, sums) / np.dot(weights, counts))
    low, high = np.percentile(estimates, [2.5, 97.5])
    lower_tail = (np.count_nonzero(estimates <= 0) + 1) / (iterations + 1)
    upper_tail = (np.count_nonzero(estimates >= 0) + 1) / (iterations + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return point, float(low), float(high), float(p_value)


def build_interpretability_audit(
    source_dir: Path,
    group_manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    detail = pd.read_csv(source_dir / "cam_prior_occlusion_detail.csv")
    detail["image"] = detail["image"].astype(str)
    detail = detail.merge(
        group_manifest[["image", "similarity_group"]],
        on="image",
        how="inner",
        validate="many_to_one",
    )
    detail["mask_nonempty"] = detail["prior_area_ratio"] > 0
    detail["cam_mass_enrichment"] = np.where(
        detail["mask_nonempty"],
        detail["cam_prior_mass_ratio"] / detail["prior_area_ratio"],
        np.nan,
    )
    detail["prior_minus_random_drop"] = (
        detail["prior_occlusion_drop_pred"] - detail["random_occlusion_drop_pred"]
    )
    strata = {
        "fallback_all": detail["prior_fallback"] == 1,
        "fallback_empty_mask": (detail["prior_fallback"] == 1) & ~detail["mask_nonempty"],
        "fallback_nonempty_mask": (detail["prior_fallback"] == 1) & detail["mask_nonempty"],
        "nonfallback": detail["prior_fallback"] == 0,
    }
    rows = []
    seed_offset = 100
    for method in sorted(detail["method"].unique()):
        method_frame = detail[detail["method"] == method]
        for stratum, mask in strata.items():
            subset = method_frame.loc[mask.loc[method_frame.index]].copy()
            if subset.empty:
                continue
            point, low, high, p_value = bootstrap_group_mean(
                subset,
                "prior_minus_random_drop",
                args.bootstrap_iters,
                args.seed + seed_offset,
            )
            seed_offset += 1
            rows.append(
                {
                    "method": method,
                    "stratum": stratum,
                    "samples": int(len(subset)),
                    "similarity_groups": int(subset["similarity_group"].nunique()),
                    "empty_mask_samples": int((~subset["mask_nonempty"]).sum()),
                    "mean_prior_area_ratio": float(subset["prior_area_ratio"].mean()),
                    "mean_cam_prior_mass_ratio": float(subset["cam_prior_mass_ratio"].mean()),
                    "mean_cam_mass_enrichment_nonempty": float(subset["cam_mass_enrichment"].mean()),
                    "pointing_game_accuracy": float(subset["pointing_hit"].mean()),
                    "mean_prior_occlusion_drop": float(subset["prior_occlusion_drop_pred"].mean()),
                    "mean_random_occlusion_drop": float(subset["random_occlusion_drop_pred"].mean()),
                    "prior_minus_random_drop": point,
                    "delta_ci95_low": low,
                    "delta_ci95_high": high,
                    "delta_two_sided_p": p_value,
                    "bootstrap_iterations": int(args.bootstrap_iters),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    source_dir = resolve(args.source_dir, repo_root)
    output_dir = resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_manifest = pd.read_csv(source_dir / "dfuc2021_similarity_grouped_manifest.csv")
    required = {"image", "similarity_group"}
    if not required.issubset(group_manifest.columns):
        raise ValueError(f"Group manifest is missing columns: {sorted(required - set(group_manifest.columns))}")

    bootstrap = build_bootstrap_table(source_dir, group_manifest, args)
    interpretability = build_interpretability_audit(source_dir, group_manifest, args)
    bootstrap.to_csv(output_dir / "grouped_ra_lpt_paired_bootstrap.csv", index=False)
    interpretability.to_csv(output_dir / "interpretability_mask_support_audit.csv", index=False)
    metadata = {
        "bootstrap_iterations": args.bootstrap_iters,
        "seed": args.seed,
        "source_dir": metadata_path(source_dir, repo_root),
        "group_manifest": metadata_path(
            source_dir / "dfuc2021_similarity_grouped_manifest.csv", repo_root
        ),
        "notes": [
            "All model comparisons resample reconstructed image-similarity groups.",
            "Fallback overlap summaries separate empty and non-empty thresholded prior masks.",
            "CAM mass enrichment is CAM-prior mass ratio divided by prior mask area and is undefined for empty masks.",
        ],
    }
    (output_dir / "reviewer_revision_statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(bootstrap.to_string(index=False))
    print(interpretability.to_string(index=False))


if __name__ == "__main__":
    main()
