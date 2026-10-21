#!/usr/bin/env python3
"""Audit and summarize the frozen 5-seed grouped interpretability rerun.

No checkpoints are fitted, no images are selected, and no source artifact is
modified. All 50 job completion markers and their asset hashes must pass before
any outcome is summarized. Case/placement selection is never based on results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


SEEDS = tuple(range(20260925, 20260930))
MODELS = ("resnet50", "lpt_full")
FOLDS = tuple(range(5))
N_IMAGES = 5955
N_GROUPS = 5670
BOOTSTRAP_SEED = 20261001
CLASS_NAMES = ("none", "infection", "ischaemia", "both")
CAM_METRICS = ("cam_prior_mass_ratio", "cam_prior_iou", "pointing_hit")
CONTRAST_METRICS = ("prior_minus_random_pred", "prior_minus_random_true")
METRICS = (
    *CAM_METRICS,
    "baseline_prob_pred",
    "prior_drop_pred", "random_drop_pred", "prior_minus_random_pred",
    "prior_drop_true", "random_drop_true", "prior_minus_random_true",
    "background_drop_pred", "baseline_fp32_vs_archived_maxdiff",
)
REQUIRED = {
    "image", "label", "similarity_group", "outer_fold", "seed", "model",
    "mask_status", "n_placements", "cam_valid", *METRICS,
    "baseline_fp32_pred_disagree",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csv_write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, na_rep="NA", float_format="%.12g")


def _bool_column(series: pd.Series, name: str) -> pd.Series:
    translated = series.astype(str).str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    })
    if translated.isna().any():
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return translated.astype(bool)


def _asset_path(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Asset path escapes its job directory: {name}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_and_audit(
    repo_root: Path, output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate frozen inputs and every completed job before reading outcomes."""
    interpretation = output_root / "interpretability"
    registry_path = interpretation / "source_registry.json"
    cases_path = interpretation / "qualitative_cases.csv"
    protocol_path = repo_root / "experiments/associate_editor_revision_v1/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    if tuple(protocol["training_seeds"]) != SEEDS:
        raise ValueError("Protocol seeds differ from the fixed five-seed matrix")
    registry = json.loads(registry_path.read_text())
    if "seeds" in registry and tuple(registry["seeds"]) != SEEDS:
        raise ValueError("Source registry seeds differ from the fixed matrix")
    if "models" in registry and set(registry["models"]) != set(MODELS):
        raise ValueError("Source registry models differ from the fixed two-model analysis")
    registry_hash = sha256(registry_path)
    manifest_path = repo_root / protocol["data"]["manifest"]
    if sha256(manifest_path) != protocol["data"]["server_manifest_sha256"]:
        raise ValueError("Grouped manifest hash differs from the frozen training protocol")
    reference = pd.read_csv(manifest_path, dtype={"image": str, "similarity_group": str})
    columns = ["image", "label", "similarity_group", "fold"]
    if not set(columns).issubset(reference):
        raise ValueError("Grouped manifest lacks image/label/similarity_group/fold")
    reference = reference[columns].sort_values("image").reset_index(drop=True)
    if (len(reference) != N_IMAGES or reference.image.nunique() != N_IMAGES
            or reference.similarity_group.nunique() != N_GROUPS):
        raise ValueError("Expected 5,955 unique images and 5,670 similarity groups")
    if set(reference.label) != set(range(4)) or set(reference.fold) != set(FOLDS):
        raise ValueError("Unexpected grouped-manifest labels or folds")
    if (reference.groupby("similarity_group").fold.nunique() != 1).any():
        raise ValueError("Similarity group spans outer folds")
    cases = pd.read_csv(cases_path, dtype={"image": str, "similarity_group": str})
    if not set(["case_rank", *columns]).issubset(cases):
        raise ValueError("Qualitative case manifest lacks required metadata")
    if (len(cases) != 20 or cases.image.nunique() != 20
            or cases.similarity_group.nunique() != 20
            or cases.groupby("label").size().to_dict() != {i: 5 for i in range(4)}
            or set(cases.case_rank) != set(range(1, 21))):
        raise ValueError("Case manifest must contain five globally distinct groups per class")
    case_reference = reference.set_index("image").loc[cases.image]
    for name in ["label", "similarity_group", "fold"]:
        if not np.array_equal(case_reference[name].to_numpy(), cases[name].to_numpy()):
            raise ValueError(f"Case metadata differs from grouped manifest: {name}")
    input_rows = []

    def record(path: Path, role: str) -> None:
        input_rows.append({"role": role, "path": str(path), "sha256": sha256(path)})

    for path, role in [(protocol_path, "protocol"), (manifest_path, "grouped_manifest"),
                       (registry_path, "source_registry"), (cases_path, "qualitative_cases")]:
        record(path, role)
    registry_assets = registry.get("assets", {})
    if "qualitative_cases.csv" in registry_assets:
        if registry_assets["qualitative_cases.csv"] != sha256(cases_path):
            raise ValueError("Case manifest does not match source-registry hash")
    audited_jobs = []
    # First pass: verify ALL marker and artifact hashes. No partial results.
    for seed in SEEDS:
        for fold in FOLDS:
            for model in MODELS:
                job = interpretation / "jobs" / f"seed_{seed}" / f"outer_{fold}" / model
                marker_path = job / "complete.json"
                marker = json.loads(marker_path.read_text())
                if marker.get("status") != "complete":
                    raise ValueError(f"Job not complete: {marker_path}")
                if marker.get("source_registry_sha256") != registry_hash:
                    raise ValueError(f"Source registry changed for job: {marker_path}")
                assets = marker.get("assets", {})
                if "metrics.csv" not in assets:
                    raise ValueError(f"metrics.csv absent from marker assets: {marker_path}")
                record(marker_path, "job_completion")
                for name, expected in assets.items():
                    path = _asset_path(job, name)
                    actual = sha256(path)
                    if actual != expected:
                        raise ValueError(f"Asset hash mismatch: {path}")
                    input_rows.append({"role": "job_asset", "path": str(path), "sha256": actual})
                audited_jobs.append((seed, fold, model, job / "metrics.csv"))
    frames = []
    for seed, fold, model, path in audited_jobs:
        frame = pd.read_csv(path, dtype={"image": str, "similarity_group": str})
        if not REQUIRED.issubset(frame):
            raise ValueError(f"Missing metric columns in {path}: {sorted(REQUIRED - set(frame))}")
        if frame.image.duplicated().any():
            raise ValueError(f"Duplicate image in {path}")
        for column, expected in [("seed", seed), ("outer_fold", fold), ("model", model)]:
            if not frame[column].eq(expected).all():
                raise ValueError(f"Wrong {column} metadata in {path}")
        expected = reference.loc[reference.fold == fold].sort_values("image").reset_index(drop=True)
        frame = frame.sort_values("image").reset_index(drop=True)
        for column in ["image", "label", "similarity_group"]:
            if not np.array_equal(frame[column].to_numpy(), expected[column].to_numpy()):
                raise ValueError(f"Outer evaluation coverage/metadata mismatch: {column}, {path}")
        if not frame.mask_status.isin(["empty", "immovable", "movable"]).all():
            raise ValueError(f"Unknown mask_status in {path}")
        counts = pd.to_numeric(frame.n_placements, errors="raise")
        if (counts.isna().any() or not np.equal(counts, np.floor(counts)).all()
                or ((counts < 0) | (counts > 30)).any()):
            raise ValueError(f"Invalid placement count in {path}")
        movable = frame.mask_status.eq("movable")
        if counts.loc[movable].lt(1).any() or counts.loc[~movable].ne(0).any():
            raise ValueError(f"Mask status and placement counts disagree in {path}")
        frame["n_placements"] = counts.astype(int)
        frame["cam_valid"] = _bool_column(frame.cam_valid, "cam_valid")
        frame["baseline_fp32_pred_disagree"] = _bool_column(
            frame.baseline_fp32_pred_disagree, "baseline_fp32_pred_disagree")
        for metric in METRICS:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
            if np.isinf(frame[metric]).any():
                raise ValueError(f"Infinite metric {metric} in {path}")
        if frame.loc[~movable, list(CONTRAST_METRICS)].notna().any().any():
            raise ValueError(f"Empty/immovable masks have spurious prior-random contrasts: {path}")
        if frame.loc[~frame.cam_valid, list(CAM_METRICS)].notna().any().any():
            raise ValueError(f"Invalid CAM rows have nonmissing CAM metrics: {path}")
        for suffix in ["pred", "true"]:
            delta = frame[f"prior_drop_{suffix}"] - frame[f"random_drop_{suffix}"]
            observed = frame[f"prior_minus_random_{suffix}"]
            comparable = delta.notna() & observed.notna()
            if not np.allclose(delta[comparable], observed[comparable], rtol=1e-7, atol=1e-8):
                raise ValueError(f"Prior-minus-random arithmetic disagrees in {path}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if len(data) != len(SEEDS) * len(MODELS) * N_IMAGES:
        raise ValueError("Total rows do not match the complete five-seed/two-model matrix")
    for column in ["label", "similarity_group", "outer_fold", "mask_status", "n_placements"]:
        if (data.groupby("image")[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"Input mask/metadata differs across model/seed: {column}")
    return data, reference, pd.DataFrame(input_rows), {
        "source_registry_sha256": registry_hash,
        "qualitative_cases_sha256": sha256(cases_path),
        "audited_jobs": len(audited_jobs),
    }


def summarize_descriptives(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Average images within each seed, then summarize the five seed estimates."""
    metric_rows, coverage_rows = [], []
    for (model, seed), full in data.groupby(["model", "seed"], sort=True):
        for scope, label in [("all", None), *[(name, i) for i, name in enumerate(CLASS_NAMES)]]:
            subset = full if label is None else full.loc[full.label == label]
            movable = subset.mask_status.eq("movable")
            placement = subset.loc[movable, "n_placements"]
            base = {"model": model, "seed": int(seed), "scope": scope,
                    "n_images": len(subset), "n_groups": subset.similarity_group.nunique()}
            coverage_rows.append({
                **base,
                **{f"n_mask_{status}": int(subset.mask_status.eq(status).sum())
                   for status in ["empty", "immovable", "movable"]},
                "n_invalid_cam": int((~subset.cam_valid).sum()),
                "n_fp32_prediction_disagreements": int(subset.baseline_fp32_pred_disagree.sum()),
                "max_fp32_archived_probability_difference": subset.baseline_fp32_vs_archived_maxdiff.max(),
                "n_probability_difference_available": int(subset.baseline_fp32_vs_archived_maxdiff.notna().sum()),
                "placements_min_movable": placement.min(), "placements_max_movable": placement.max(),
                "n_distinct_placements_sum": int(placement.sum()),
            })
            for metric in METRICS:
                valid = subset[metric].notna()
                metric_rows.append({
                    **base, "metric": metric,
                    "n_valid_images": int(valid.sum()), "n_missing_images": int((~valid).sum()),
                    "n_valid_groups": subset.loc[valid, "similarity_group"].nunique(),
                    "image_mean": subset.loc[valid, metric].mean(),
                })
    seed_metrics = pd.DataFrame(metric_rows)
    aggregate_rows = []
    for (model, scope, metric), subset in seed_metrics.groupby(["model", "scope", "metric"], sort=True):
        estimates = subset.image_mean.dropna()
        aggregate_rows.append({
            "model": model, "scope": scope, "metric": metric,
            "n_seeds_total": len(subset), "n_seeds_estimable": len(estimates),
            "mean_across_seed_image_means": estimates.mean(),
            "sample_sd_across_seed_image_means": estimates.std(ddof=1),
            "n_valid_images_min_per_seed": int(subset.n_valid_images.min()),
            "n_valid_images_max_per_seed": int(subset.n_valid_images.max()),
            "n_valid_groups_min_per_seed": int(subset.n_valid_groups.min()),
            "n_valid_groups_max_per_seed": int(subset.n_valid_groups.max()),
        })
    return seed_metrics, pd.DataFrame(aggregate_rows), pd.DataFrame(coverage_rows)


def crossed_group_bootstrap(
    values: np.ndarray, group_codes: np.ndarray, n_groups: int,
    replicates: int = 5000, seed: int = BOOTSTRAP_SEED, chunk_size: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Crossed seed × group bootstrap for series × seed × image values.

    Every replicate shares one group resample across all seeds and all outcome
    series, retaining every image in a sampled group with its multiplicity.
    It then resamples seed indices and averages the selected *seed-level image
    means*. Neither groups, image×seed rows, nor mask placements are treated as
    independent image observations. NaNs are excluded within a seed with explicit
    denominators; if a selected seed has no valid sampled image, the replicate is
    NA and is neither replaced nor silently reweighted across remaining seeds.
    """
    values = np.asarray(values, dtype=np.float64)
    group_codes = np.asarray(group_codes, dtype=np.int64)
    if (values.ndim != 3 or values.shape[2] != len(group_codes)
            or values.shape[1] < 1 or replicates < 1 or chunk_size < 1
            or n_groups < 1 or len(group_codes) < 1
            or group_codes.min() < 0 or group_codes.max() >= n_groups
            or np.isinf(values).any()):
        raise ValueError("Invalid crossed bootstrap arrays or dimensions")
    n_series, n_seeds, _ = values.shape
    sums = np.zeros((n_series, n_seeds, n_groups), dtype=np.float64)
    counts = np.zeros_like(sums)
    for series in range(n_series):
        for fitted_seed in range(n_seeds):
            valid = np.isfinite(values[series, fitted_seed])
            sums[series, fitted_seed] = np.bincount(
                group_codes[valid], weights=values[series, fitted_seed, valid], minlength=n_groups)
            counts[series, fitted_seed] = np.bincount(group_codes[valid], minlength=n_groups)
    rng = np.random.default_rng(seed)
    samples = np.full((replicates, n_series), np.nan)
    draw_digest = hashlib.sha256()
    for start in range(0, replicates, chunk_size):
        size = min(chunk_size, replicates - start)
        group_weights = np.zeros((size, n_groups), dtype=np.float64)
        seed_weights = np.zeros((size, n_seeds), dtype=np.float64)
        # Fixed per-replicate draw order makes results independent of chunk size.
        for repeat in range(size):
            chosen_groups = rng.integers(0, n_groups, size=n_groups, dtype=np.int64)
            chosen_seeds = rng.integers(0, n_seeds, size=n_seeds, dtype=np.int64)
            draw_digest.update(chosen_groups.astype("<i8").tobytes())
            draw_digest.update(chosen_seeds.astype("<i8").tobytes())
            group_weights[repeat] = np.bincount(chosen_groups, minlength=n_groups)
            seed_weights[repeat] = np.bincount(chosen_seeds, minlength=n_seeds)
        numerators = (group_weights @ sums.reshape(-1, n_groups).T).reshape(size, n_series, n_seeds)
        denominators = (group_weights @ counts.reshape(-1, n_groups).T).reshape(size, n_series, n_seeds)
        means = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators > 0)
        selected_empty = ((denominators == 0) & (seed_weights[:, None, :] > 0)).any(axis=2)
        chunk = (means * seed_weights[:, None, :]).sum(axis=2) / n_seeds
        chunk[selected_empty] = np.nan
        samples[start:start + size] = chunk
    return samples, {"draw_sequence_sha256": draw_digest.hexdigest(), "seed": seed,
                     "replicates": replicates, "n_seed_draws": n_seeds,
                     "n_group_draws": n_groups, "draw_order": "group_indices_then_seed_indices_per_replicate"}


def bootstrap_summary(
    data: pd.DataFrame, reference: pd.DataFrame, replicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    image_order = reference.image.to_list()
    groups = sorted(reference.similarity_group.unique())
    group_map = {group: index for index, group in enumerate(groups)}
    codes = reference.similarity_group.map(group_map).to_numpy()
    labels = reference.label.to_numpy()
    values, definitions, observed_rows = [], [], []
    for metric in CONTRAST_METRICS:
        model_arrays = {}
        for model in MODELS:
            model_arrays[model] = np.stack([
                data.loc[(data.model == model) & (data.seed == seed)].set_index("image").loc[image_order, metric].to_numpy(float)
                for seed in SEEDS
            ])
        arrays = {**model_arrays, "lpt_full_minus_resnet50": model_arrays["lpt_full"] - model_arrays["resnet50"]}
        for entity, array in arrays.items():
            for scope, label in [("all", None), *[(name, i) for i, name in enumerate(CLASS_NAMES)]]:
                scoped = array.copy()
                if label is not None:
                    scoped[:, labels != label] = np.nan
                series_id = f"{entity}__{metric}__{scope}"
                values.append(scoped)
                definitions.append({"series_id": series_id, "entity": entity, "metric": metric, "scope": scope,
                                    "endpoint_role": "primary_interpretability_score" if metric.endswith("_pred") else "descriptive_true_label_score"})
                for index, seed in enumerate(SEEDS):
                    valid = np.isfinite(scoped[index])
                    observed_rows.append({**definitions[-1], "seed": seed,
                                          "image_mean": scoped[index, valid].mean() if valid.any() else np.nan,
                                          "n_valid_images": int(valid.sum()),
                                          "n_valid_groups": len(np.unique(codes[valid]))})
    replicate_values, metadata = crossed_group_bootstrap(np.stack(values), codes, len(groups), replicates)
    observed = pd.DataFrame(observed_rows)
    interval_rows = []
    for index, definition in enumerate(definitions):
        seed_estimates = observed.loc[observed.series_id == definition["series_id"]]
        valid_seeds = seed_estimates.image_mean.dropna()
        valid_replicates = replicate_values[:, index][np.isfinite(replicate_values[:, index])]
        quantiles = np.quantile(valid_replicates, [.025, .975]) if len(valid_replicates) else [np.nan, np.nan]
        interval_rows.append({**definition,
                              "mean_across_seed_image_means": valid_seeds.mean(),
                              "sample_sd_across_seed_image_means": valid_seeds.std(ddof=1),
                              "n_seeds_estimable": len(valid_seeds),
                              "n_valid_images_min_per_seed": int(seed_estimates.n_valid_images.min()),
                              "n_valid_images_max_per_seed": int(seed_estimates.n_valid_images.max()),
                              "n_valid_groups_min_per_seed": int(seed_estimates.n_valid_groups.min()),
                              "n_valid_groups_max_per_seed": int(seed_estimates.n_valid_groups.max()),
                              "ci95_low": quantiles[0], "ci95_high": quantiles[1],
                              "n_replicates_requested": replicates,
                              "n_replicates_estimable": len(valid_replicates),
                              "n_replicates_undefined": replicates - len(valid_replicates)})
    draws = pd.DataFrame(replicate_values, columns=[definition["series_id"] for definition in definitions])
    draws.insert(0, "replicate", np.arange(1, replicates + 1))
    return observed, pd.DataFrame(interval_rows), draws, metadata


def _markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    def show(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.5f}"
        return str(value)
    return "\n".join([
        "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(show(record[c]) for c in columns) + " |" for record in frame.to_dict("records")],
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1"),
                        help="Experiment artifact root; inputs and summary are under its interpretability/ directory")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError("replicates must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    output_root = (output_root if output_root.is_absolute() else repo_root / output_root).resolve()
    final_dir = output_root / "interpretability/summary"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {final_dir}")
    data, reference, inputs, audit = load_and_audit(repo_root, output_root)
    seed_metrics, model_summary, coverage = summarize_descriptives(data)
    paired, intervals, replicates, draw_metadata = bootstrap_summary(data, reference, args.replicates)
    # Write only after all input audits and calculations succeed; publish atomically.
    temporary = Path(tempfile.mkdtemp(prefix=".summary_pending_", dir=final_dir.parent))
    for filename, frame in [
        ("seed_metrics.csv", seed_metrics), ("model_summary.csv", model_summary),
        ("coverage_audit.csv", coverage), ("paired_seed_metrics.csv", paired),
        ("bootstrap_intervals.csv", intervals), ("bootstrap_replicates.csv", replicates),
        ("input_registry.csv", inputs),
    ]:
        csv_write(frame, temporary / filename)
    write_json(temporary / "bootstrap_design.json", draw_metadata)
    primary = intervals.loc[(intervals.scope == "all") & (intervals.metric == "prior_minus_random_pred")]
    all_coverage = coverage.loc[coverage.scope == "all"]
    text = (
        "# Frozen grouped interpretability rerun: audited summary\n\n"
        f"All {audit['audited_jobs']} job markers and their listed assets passed SHA256 verification. "
        f"The complete matrix contains {len(data):,} image × seed × model rows: "
        "5,955 images, 5,670 similarity groups, five training seeds and two corresponding outer-fold CNN models.\n\n"
        "## Primary descriptive score\n\n"
        "Positive values mean a larger decrease in the original predicted-class probability after prior-region "
        "occlusion than after the mean of the same shape/area translated controls. Probabilities are not logits. "
        "Each fitted model's original predicted class is held fixed for its own perturbations; the two models may "
        "therefore have different predicted-class targets. The true-label score is reported separately.\n\n"
        + _markdown(primary, ["entity", "mean_across_seed_image_means", "sample_sd_across_seed_image_means",
                              "ci95_low", "ci95_high", "n_valid_images_min_per_seed", "n_valid_images_max_per_seed"])
        + "\n\n## Denominators and numerical agreement\n\n"
        + _markdown(all_coverage, ["model", "seed", "n_images", "n_groups", "n_mask_empty", "n_mask_immovable",
                                 "n_mask_movable", "n_invalid_cam", "n_fp32_prediction_disagreements",
                                 "max_fp32_archived_probability_difference"])
        + "\n\n## Reporting boundaries\n\n"
        "- Image means are computed separately within each seed, then summarized across all five seeds using sample SD. "
        "Image × seed rows are not treated as independent observations.\n"
        f"- {args.replicates:,} paired crossed seed × similarity-group bootstrap replicates share the same group and seed draws "
        "across models, targets and classes. Sampling groups retains all their images with multiplicity. Each seed's "
        "denominator is its valid sampled image count, not its number of groups or mask placements.\n"
        "- The model difference uses the intersection of valid paired image outcomes within each seed. Empty and immovable "
        "masks are reported separately and have no prior-minus-random estimate. Invalid CAM metrics are NA. Other missing "
        "metrics are excluded only from their own denominators.\n"
        "- Undefined bootstrap replicates are reported and are not replaced. Percentile 95% intervals are descriptive, "
        "not simultaneous confidence intervals. No p values or confirmatory hypothesis tests are reported.\n"
        "- CAM concentration and occlusion scores describe model behaviour relative to generated priors, not the anatomical "
        "accuracy of a mask, causal clinical features, a validated diagnosis, or an explanation of the entire RA-LPT stack.\n"
        "- Intervals reflect the available fitted seeds and image-similarity-group resampling, not patient independence, "
        "external validation, complete bootstrap retraining or prior-generator uncertainty.\n"
        "- Qualitative cases were frozen independently of prediction quality. Their 20-image subset is not the quantitative sample. "
        "This program creates summaries only; it does not edit manuscripts or select favourable cases.\n"
    )
    (temporary / "summary.md").write_text(text)
    assets = {path.name: sha256(path) for path in sorted(temporary.iterdir()) if path.is_file()}
    complete = {"status": "complete", "analysis": "frozen_grouped_interpretability",
                "seeds": list(SEEDS), "models": list(MODELS), "n_images": N_IMAGES, "n_groups": N_GROUPS,
                "n_image_seed_model_rows": len(data), "bootstrap": draw_metadata,
                "audited_inputs": audit, "assets": assets,
                "p_values_reported": False, "manuscript_modified": False}
    write_json(temporary / "summary_complete.json", complete)
    temporary.rename(final_dir)
    print(json.dumps({"status": "complete", "summary_dir": str(final_dir), **audit}, indent=2))


if __name__ == "__main__":
    main()
