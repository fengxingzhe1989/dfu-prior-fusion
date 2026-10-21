#!/usr/bin/env python3
"""Audit and summarize the frozen associate-editor revision experiment.

This script is deliberately read-only with respect to training artifacts.  It
validates the completed outer-fold prediction files, computes the prespecified
seed-level and fold-level metrics, performs the crossed seed x similarity-group
bootstrap, and writes a new atomic summary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


LABELS = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
PROBABILITY_COLUMNS = tuple(f"prob_{label}" for label in LABELS)


@dataclass(frozen=True)
class ArtifactSpec:
    report_name: str
    artifact_model: str
    source: str
    stage: str | None = None
    filename: str | None = None


ARTIFACT_SPECS = (
    ArtifactSpec("resnet50", "resnet50", "classification", stage="refit"),
    ArtifactSpec("efficientnetb0", "efficientnetb0", "classification", stage="refit"),
    ArtifactSpec("convnext_tiny", "convnext_tiny", "classification", stage="refit"),
    ArtifactSpec("efficientnetv2s", "efficientnetv2s", "classification", stage="refit"),
    ArtifactSpec("lpt_no_alignment", "lpt_no_alignment", "classification", stage="refit"),
    ArtifactSpec("lpt_no_prior_mod", "lpt_no_prior_mod", "classification", stage="refit"),
    ArtifactSpec("lpt_no_roi", "lpt_no_roi", "classification", stage="refit"),
    ArtifactSpec("lpt_stats_only", "lpt_stats_only", "classification", stage="refit"),
    ArtifactSpec("full_lpt", "lpt_full", "classification", stage="direct_refit"),
    ArtifactSpec("neutral_full_lpt", "lpt_neutral", "classification", stage="direct_refit"),
    ArtifactSpec(
        "ra_lpt_stack",
        "lpt_ablation_ensemble",
        "stack",
        filename="lpt_ablation_ensemble_predictions.csv",
    ),
    ArtifactSpec(
        "output_only_lpt_stack",
        "lpt_ablation_output_ensemble",
        "stack",
        filename="lpt_ablation_output_ensemble_predictions.csv",
    ),
    ArtifactSpec(
        "strong_rgb_stack",
        "strong_cls_ensemble",
        "stack",
        filename="strong_cls_ensemble_predictions.csv",
    ),
    ArtifactSpec(
        "strong_rgb_plus_prior",
        "strong_cls_prior_residual",
        "stack",
        filename="strong_cls_prior_residual_predictions.csv",
    ),
)

PRIMARY_CONTRASTS = (
    ("full_minus_neutral", "full_lpt", "neutral_full_lpt"),
    ("ra_minus_output_only", "ra_lpt_stack", "output_only_lpt_stack"),
    ("rgb_prior_minus_rgb", "strong_rgb_plus_prior", "strong_rgb_stack"),
)

REFERENCE_CONTRASTS = (
    ("full_minus_resnet50", "full_lpt", "resnet50"),
    ("ra_minus_resnet50", "ra_lpt_stack", "resnet50"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/associate_editor_revision_v1/summary"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_complete(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"Completion marker is not complete: {path}")
    return payload


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> tuple[dict[str, float], list[dict[str, Any]]]:
    tp = np.diag(matrix).astype(np.float64)
    fp = matrix.sum(axis=0).astype(np.float64) - tp
    fn = matrix.sum(axis=1).astype(np.float64) - tp
    support = matrix.sum(axis=1).astype(np.int64)
    precision = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) > 0)
    f1_denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, f1_denominator, out=np.zeros_like(tp), where=f1_denominator > 0)
    summary = {
        "n": int(matrix.sum()),
        "accuracy": float(tp.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()),
    }
    class_rows = []
    for index, label in enumerate(LABELS):
        class_rows.append(
            {
                "class": label,
                "support": int(support[index]),
                "tp": int(tp[index]),
                "fp": int(fp[index]),
                "fn": int(fn[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )
    return summary, class_rows


def macro_f1_from_confusions(matrices: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1).astype(np.float64)
    fp = matrices.sum(axis=-2).astype(np.float64) - diagonal
    fn = matrices.sum(axis=-1).astype(np.float64) - diagonal
    denominator = 2.0 * diagonal + fp + fn
    f1 = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    return f1.mean(axis=-1)


def bootstrap_p_value(values: np.ndarray) -> float:
    replicates = len(values)
    lower = (int(np.count_nonzero(values <= 0.0)) + 1.0) / (replicates + 1.0)
    upper = (int(np.count_nonzero(values >= 0.0)) + 1.0) / (replicates + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(count, dtype=np.float64)
    previous = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[int(original_index)])
        previous = max(previous, candidate)
        adjusted[int(original_index)] = previous
    return adjusted.tolist()


def csv_write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, na_rep="NA", float_format="%.10g")


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{decimals}f}"
        return str(value)

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(render(record[column]) for column in columns) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    experiment_dir = repo_root / "experiments/associate_editor_revision_v1"
    protocol_path = experiment_dir / "protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output_root = repo_root / "outputs/associate_editor_revision_v1"
    output_dir = resolve_path(args.output_dir, repo_root)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary directory: {output_dir}")

    training_marker = require_complete(output_root / "training_pipeline_complete.json")
    classification_marker = require_complete(output_root / "classification_complete.json")
    integrity_marker = require_complete(output_root / "classification_prediction_integrity.json")
    stack_marker = require_complete(output_root / "stack_complete.json")
    segmentation_marker = require_complete(output_root / "segmentation_complete.json")
    if int(classification_marker["jobs"]) != 900:
        raise RuntimeError("Expected 900 completed classification jobs.")
    if int(stack_marker["jobs"]) != 25:
        raise RuntimeError("Expected 25 completed stack jobs.")
    if int(segmentation_marker["jobs"]) != 25:
        raise RuntimeError("Expected 25 completed segmentation jobs.")

    seeds = [int(seed) for seed in protocol["training_seeds"]]
    folds = list(range(5))
    if classification_marker["seeds"] != seeds:
        raise RuntimeError("Classification seeds do not match the frozen protocol.")
    bootstrap_replicates = int(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else protocol["statistics"]["bootstrap"]["replicates"]
    )
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    manifest_path = resolve_path(Path(protocol["data"]["manifest"]), repo_root)
    if sha256(manifest_path) != protocol["data"]["server_manifest_sha256"]:
        raise RuntimeError("Grouped manifest SHA256 does not match the frozen protocol.")
    manifest = pd.read_csv(manifest_path)
    required_manifest_columns = {"image", "label_name", "fold", "similarity_group"}
    if not required_manifest_columns.issubset(manifest.columns):
        raise RuntimeError("Grouped manifest is missing required columns.")
    if len(manifest) != int(protocol["data"]["n_images"]):
        raise RuntimeError("Grouped manifest image count does not match the frozen protocol.")
    if manifest["image"].duplicated().any():
        raise RuntimeError("Grouped manifest contains duplicate images.")
    if manifest["similarity_group"].nunique() != int(protocol["data"]["n_groups"]):
        raise RuntimeError("Grouped manifest group count does not match the frozen protocol.")
    if set(manifest["label_name"].astype(str)) != set(LABELS):
        raise RuntimeError("Unexpected class labels in grouped manifest.")
    manifest = manifest.reset_index(drop=True)
    manifest["label_id"] = manifest["label_name"].map(LABEL_TO_ID).astype(int)
    image_to_position = pd.Series(manifest.index, index=manifest["image"]).to_dict()
    expected_by_fold = {
        fold: set(manifest.loc[manifest["fold"] == fold, "image"].astype(str)) for fold in folds
    }

    report_names = [spec.report_name for spec in ARTIFACT_SPECS]
    prediction_arrays = {
        model: np.full((len(seeds), len(manifest)), -1, dtype=np.int8) for model in report_names
    }
    filled_arrays = {
        model: np.zeros((len(seeds), len(manifest)), dtype=bool) for model in report_names
    }
    fold_metric_rows: list[dict[str, Any]] = []
    fold_class_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        for outer_fold in folds:
            stack_complete = require_complete(
                output_root / f"stacks/seed_{seed}/outer_{outer_fold}/complete.json"
            )
            for spec in ARTIFACT_SPECS:
                if spec.source == "classification":
                    path = (
                        output_root
                        / f"classification/seed_{seed}/outer_{outer_fold}/{spec.artifact_model}/{spec.stage}/predictions.csv"
                    )
                    complete = require_complete(path.parent / "complete.json")
                    recorded_hash = str(complete["prediction_sha256"])
                else:
                    path = output_root / f"stacks/seed_{seed}/outer_{outer_fold}/{spec.filename}"
                    feature_payload = stack_complete["feature_sets"][spec.artifact_model]
                    recorded_hash = str(feature_payload["prediction_sha256"])
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual_hash = sha256(path)
                if actual_hash != recorded_hash:
                    raise RuntimeError(f"Prediction SHA256 mismatch: {path}")
                frame = pd.read_csv(path)
                required_columns = {
                    "image",
                    "true_label",
                    "pred_label",
                    "base_seed",
                    "outer_fold",
                    "model",
                    *PROBABILITY_COLUMNS,
                }
                if not required_columns.issubset(frame.columns):
                    raise RuntimeError(f"Prediction file is missing required columns: {path}")
                if frame["image"].duplicated().any():
                    raise RuntimeError(f"Duplicate images within prediction file: {path}")
                images = set(frame["image"].astype(str))
                if images != expected_by_fold[outer_fold]:
                    raise RuntimeError(f"Outer-fold image coverage mismatch: {path}")
                if set(frame["base_seed"].astype(int)) != {seed}:
                    raise RuntimeError(f"Seed metadata mismatch: {path}")
                if set(frame["outer_fold"].astype(int)) != {outer_fold}:
                    raise RuntimeError(f"Outer-fold metadata mismatch: {path}")
                if set(frame["model"].astype(str)) != {spec.artifact_model}:
                    raise RuntimeError(f"Model metadata mismatch: {path}")
                probabilities = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
                if not np.isfinite(probabilities).all():
                    raise FloatingPointError(f"Non-finite outer probabilities: {path}")
                if (probabilities < -1e-8).any() or (probabilities > 1.0 + 1e-8).any():
                    raise RuntimeError(f"Probability outside [0, 1]: {path}")
                if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
                    raise RuntimeError(f"Probability rows do not sum to one: {path}")
                expected_labels = frame["image"].map(
                    manifest.set_index("image")["label_name"]
                )
                if expected_labels.isna().any() or not np.array_equal(
                    expected_labels.to_numpy(str), frame["true_label"].to_numpy(str)
                ):
                    raise RuntimeError(f"True labels do not match the frozen manifest: {path}")
                predicted_ids = frame["pred_label"].map(LABEL_TO_ID)
                if predicted_ids.isna().any():
                    raise RuntimeError(f"Unknown predicted label: {path}")
                if spec.source == "stack":
                    bias_payload = stack_complete["feature_sets"][spec.artifact_model]["biases"]
                    biases = np.asarray([bias_payload[label] for label in LABELS], dtype=np.float64)
                    expected_predictions = np.argmax(
                        np.log(np.clip(probabilities, 1e-8, 1.0)) + biases.reshape(1, -1),
                        axis=1,
                    )
                else:
                    expected_predictions = np.argmax(probabilities, axis=1)
                if not np.array_equal(predicted_ids.to_numpy(dtype=int), expected_predictions):
                    raise RuntimeError(f"Stored predictions are inconsistent with probabilities/biases: {path}")

                positions = frame["image"].map(image_to_position).to_numpy(dtype=int)
                if filled_arrays[spec.report_name][seed_index, positions].any():
                    raise RuntimeError(
                        f"An image has more than one outer prediction for {seed}/{spec.report_name}."
                    )
                prediction_arrays[spec.report_name][seed_index, positions] = predicted_ids.to_numpy(
                    dtype=np.int8
                )
                filled_arrays[spec.report_name][seed_index, positions] = True
                true_ids = frame["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
                matrix = confusion_matrix(true_ids, predicted_ids.to_numpy(dtype=int))
                metrics, class_metrics = metrics_from_confusion(matrix)
                fold_metric_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": spec.report_name,
                        **metrics,
                    }
                )
                for row in class_metrics:
                    fold_class_rows.append(
                        {
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "model": spec.report_name,
                            **row,
                        }
                    )
                registry_rows.append(
                    {
                        "artifact_type": "classification_outer_prediction",
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": spec.report_name,
                        "path": str(path.relative_to(repo_root)),
                        "rows": len(frame),
                        "sha256": actual_hash,
                        "recorded_sha256": recorded_hash,
                        "sha256_match": True,
                    }
                )

    for model in report_names:
        if not filled_arrays[model].all():
            missing = int(np.size(filled_arrays[model]) - filled_arrays[model].sum())
            raise RuntimeError(f"Missing {missing} outer predictions for {model}.")

    true_ids_all = manifest["label_id"].to_numpy(dtype=int)
    seed_metric_rows: list[dict[str, Any]] = []
    seed_class_rows: list[dict[str, Any]] = []
    for model in report_names:
        for seed_index, seed in enumerate(seeds):
            matrix = confusion_matrix(true_ids_all, prediction_arrays[model][seed_index].astype(int))
            metrics, class_metrics = metrics_from_confusion(matrix)
            seed_metric_rows.append({"seed": seed, "model": model, **metrics})
            for row in class_metrics:
                seed_class_rows.append({"seed": seed, "model": model, **row})

    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(["model", "seed", "outer_fold"])
    fold_class_metrics = pd.DataFrame(fold_class_rows).sort_values(
        ["model", "seed", "outer_fold", "class"]
    )
    seed_metrics = pd.DataFrame(seed_metric_rows).sort_values(["model", "seed"])
    seed_class_metrics = pd.DataFrame(seed_class_rows).sort_values(["model", "seed", "class"])
    model_summary = (
        seed_metrics.groupby("model", sort=False)
        .agg(
            seeds=("seed", "size"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
        )
        .reset_index()
    )

    seed_metric_lookup = seed_metrics.set_index(["seed", "model"])["macro_f1"]
    primary_seed_rows: list[dict[str, Any]] = []
    for contrast, model_a, model_b in PRIMARY_CONTRASTS:
        for seed in seeds:
            value_a = float(seed_metric_lookup.loc[(seed, model_a)])
            value_b = float(seed_metric_lookup.loc[(seed, model_b)])
            primary_seed_rows.append(
                {
                    "contrast": contrast,
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "macro_f1_a": value_a,
                    "macro_f1_b": value_b,
                    "difference": value_a - value_b,
                }
            )
    primary_seed_contrasts = pd.DataFrame(primary_seed_rows)

    reference_rows: list[dict[str, Any]] = []
    for contrast, model_a, model_b in REFERENCE_CONTRASTS:
        differences = []
        for seed in seeds:
            value_a = float(seed_metric_lookup.loc[(seed, model_a)])
            value_b = float(seed_metric_lookup.loc[(seed, model_b)])
            difference = value_a - value_b
            differences.append(difference)
            reference_rows.append(
                {
                    "contrast": contrast,
                    "row_type": "seed",
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "macro_f1_a": value_a,
                    "macro_f1_b": value_b,
                    "difference": difference,
                    "difference_sd_across_seeds": np.nan,
                }
            )
        reference_rows.append(
            {
                "contrast": contrast,
                "row_type": "across_seed_summary",
                "seed": np.nan,
                "model_a": model_a,
                "model_b": model_b,
                "macro_f1_a": float(
                    np.mean([seed_metric_lookup.loc[(seed, model_a)] for seed in seeds])
                ),
                "macro_f1_b": float(
                    np.mean([seed_metric_lookup.loc[(seed, model_b)] for seed in seeds])
                ),
                "difference": float(np.mean(differences)),
                "difference_sd_across_seeds": float(np.std(differences, ddof=1)),
            }
        )
    reference_contrasts = pd.DataFrame(reference_rows)

    group_names = sorted(manifest["similarity_group"].astype(str).unique())
    group_to_id = {group: index for index, group in enumerate(group_names)}
    group_ids = manifest["similarity_group"].astype(str).map(group_to_id).to_numpy(dtype=int)
    bootstrap_models = list(
        dict.fromkeys(
            [model for _, model_a, model_b in PRIMARY_CONTRASTS for model in (model_a, model_b)]
        )
    )
    group_confusions = np.zeros(
        (len(bootstrap_models), len(seeds), len(group_names), len(LABELS), len(LABELS)),
        dtype=np.int32,
    )
    for model_index, model in enumerate(bootstrap_models):
        for seed_index in range(len(seeds)):
            np.add.at(
                group_confusions[model_index, seed_index],
                (group_ids, true_ids_all, prediction_arrays[model][seed_index].astype(int)),
                1,
            )

    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    crossed_distributions = {
        contrast: np.empty(bootstrap_replicates, dtype=np.float64)
        for contrast, _, _ in PRIMARY_CONTRASTS
    }
    group_only_distributions = {
        contrast: np.empty((bootstrap_replicates, len(seeds)), dtype=np.float64)
        for contrast, _, _ in PRIMARY_CONTRASTS
    }
    flat_confusions = group_confusions.reshape(
        len(bootstrap_models), len(seeds), len(group_names), len(LABELS) ** 2
    )
    model_to_bootstrap_index = {model: index for index, model in enumerate(bootstrap_models)}
    for start in range(0, bootstrap_replicates, args.bootstrap_chunk_size):
        stop = min(bootstrap_replicates, start + args.bootstrap_chunk_size)
        size = stop - start
        group_counts = np.empty((size, len(group_names)), dtype=np.int16)
        for row in range(size):
            sampled_groups = rng.integers(0, len(group_names), size=len(group_names))
            group_counts[row] = np.bincount(sampled_groups, minlength=len(group_names))
        sampled_confusions = np.einsum(
            "bg,msgk->bmsk", group_counts, flat_confusions, optimize=True
        ).reshape(size, len(bootstrap_models), len(seeds), len(LABELS), len(LABELS))
        sampled_scores = macro_f1_from_confusions(sampled_confusions)
        sampled_seed_indices = rng.integers(0, len(seeds), size=(size, len(seeds)))
        batch_indices = np.arange(size)[:, None, None]
        model_indices = np.arange(len(bootstrap_models))[None, :, None]
        selected_scores = sampled_scores[
            batch_indices, model_indices, sampled_seed_indices[:, None, :]
        ].mean(axis=2)
        for contrast, model_a, model_b in PRIMARY_CONTRASTS:
            index_a = model_to_bootstrap_index[model_a]
            index_b = model_to_bootstrap_index[model_b]
            crossed_distributions[contrast][start:stop] = (
                selected_scores[:, index_a] - selected_scores[:, index_b]
            )
            group_only_distributions[contrast][start:stop] = (
                sampled_scores[:, index_a, :] - sampled_scores[:, index_b, :]
            )

    bootstrap_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for contrast, model_a, model_b in PRIMARY_CONTRASTS:
        point_values = primary_seed_contrasts.loc[
            primary_seed_contrasts["contrast"] == contrast, "difference"
        ].to_numpy(dtype=np.float64)
        distribution = crossed_distributions[contrast]
        raw_p = bootstrap_p_value(distribution)
        raw_p_values.append(raw_p)
        bootstrap_rows.append(
            {
                "contrast": contrast,
                "model_a": model_a,
                "model_b": model_b,
                "estimate_mean_seed_difference": float(point_values.mean()),
                "sd_seed_difference": float(point_values.std(ddof=1)),
                "ci_low": float(np.quantile(distribution, 0.025)),
                "ci_high": float(np.quantile(distribution, 0.975)),
                "bootstrap_replicates": bootstrap_replicates,
                "approximate_two_sided_p": raw_p,
            }
        )
    adjusted_p_values = holm_adjust(raw_p_values)
    for row, adjusted in zip(bootstrap_rows, adjusted_p_values, strict=True):
        row["holm_adjusted_p"] = adjusted
    primary_bootstrap = pd.DataFrame(bootstrap_rows)

    per_seed_bootstrap_rows: list[dict[str, Any]] = []
    for contrast, model_a, model_b in PRIMARY_CONTRASTS:
        distribution = group_only_distributions[contrast]
        for seed_index, seed in enumerate(seeds):
            values = distribution[:, seed_index]
            point = float(
                primary_seed_contrasts.loc[
                    (primary_seed_contrasts["contrast"] == contrast)
                    & (primary_seed_contrasts["seed"] == seed),
                    "difference",
                ].iloc[0]
            )
            per_seed_bootstrap_rows.append(
                {
                    "contrast": contrast,
                    "seed": seed,
                    "model_a": model_a,
                    "model_b": model_b,
                    "estimate": point,
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "approximate_two_sided_p_unadjusted": bootstrap_p_value(values),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    per_seed_bootstrap = pd.DataFrame(per_seed_bootstrap_rows)

    bootstrap_replicate_data: dict[str, Any] = {
        "replicate": np.arange(1, bootstrap_replicates + 1)
    }
    for contrast, _, _ in PRIMARY_CONTRASTS:
        bootstrap_replicate_data[f"crossed_{contrast}"] = crossed_distributions[contrast]
        for seed_index, seed in enumerate(seeds):
            bootstrap_replicate_data[f"group_only_{contrast}_seed_{seed}"] = (
                group_only_distributions[contrast][:, seed_index]
            )
    bootstrap_replicates_frame = pd.DataFrame(bootstrap_replicate_data)

    segmentation_fold_rows: list[dict[str, Any]] = []
    segmentation_registry_rows: list[dict[str, Any]] = []
    segmentation_frames: dict[tuple[int, str], list[pd.DataFrame]] = {}
    for seed in seeds:
        for outer_fold in folds:
            job_dir = output_root / f"segmentation/seed_{seed}/outer_{outer_fold}"
            complete = require_complete(job_dir / "complete.json")
            for grid in ("512x384", "224x224"):
                path = job_dir / f"outer_metrics_{grid}.csv"
                recorded_hash = str(complete["outer_metrics"][grid]["sha256"])
                actual_hash = sha256(path)
                if actual_hash != recorded_hash:
                    raise RuntimeError(f"Segmentation SHA256 mismatch: {path}")
                frame = pd.read_csv(path)
                required_columns = {
                    "image_id",
                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "seed",
                    "outer_fold",
                    "width",
                    "height",
                }
                if not required_columns.issubset(frame.columns):
                    raise RuntimeError(f"Segmentation file is missing required columns: {path}")
                if len(frame) != int(complete["outer_metrics"][grid]["n"]):
                    raise RuntimeError(f"Segmentation row count mismatch: {path}")
                if frame["image_id"].duplicated().any():
                    raise RuntimeError(f"Duplicate segmentation image IDs: {path}")
                values = frame[["dice", "iou", "precision", "recall"]].to_numpy(float)
                if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
                    raise RuntimeError(f"Invalid segmentation metric values: {path}")
                if set(frame["seed"].astype(int)) != {seed}:
                    raise RuntimeError(f"Segmentation seed metadata mismatch: {path}")
                if set(frame["outer_fold"].astype(int)) != {outer_fold}:
                    raise RuntimeError(f"Segmentation outer-fold metadata mismatch: {path}")
                means = frame[["dice", "iou", "precision", "recall"]].mean()
                segmentation_fold_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "grid": grid,
                        "n": len(frame),
                        **{metric: float(means[metric]) for metric in means.index},
                    }
                )
                segmentation_frames.setdefault((seed, grid), []).append(frame)
                segmentation_registry_rows.append(
                    {
                        "artifact_type": "segmentation_outer_metrics",
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "model": f"deeplabv3plus_{grid}",
                        "path": str(path.relative_to(repo_root)),
                        "rows": len(frame),
                        "sha256": actual_hash,
                        "recorded_sha256": recorded_hash,
                        "sha256_match": True,
                    }
                )

    segmentation_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        ids_by_grid: dict[str, set[str]] = {}
        for grid in ("512x384", "224x224"):
            frame = pd.concat(segmentation_frames[(seed, grid)], ignore_index=True)
            if len(frame) != 2000 or frame["image_id"].duplicated().any():
                raise RuntimeError(f"Segmentation outer folds do not cover 2,000 images once: {seed}/{grid}")
            ids_by_grid[grid] = set(frame["image_id"].astype(str))
            means = frame[["dice", "iou", "precision", "recall"]].mean()
            segmentation_seed_rows.append(
                {
                    "seed": seed,
                    "grid": grid,
                    "n": len(frame),
                    **{metric: float(means[metric]) for metric in means.index},
                }
            )
        if ids_by_grid["512x384"] != ids_by_grid["224x224"]:
            raise RuntimeError(f"Segmentation grids use different image sets for seed {seed}.")

    segmentation_fold_metrics = pd.DataFrame(segmentation_fold_rows).sort_values(
        ["grid", "seed", "outer_fold"]
    )
    segmentation_seed_metrics = pd.DataFrame(segmentation_seed_rows).sort_values(["grid", "seed"])
    segmentation_grid_summary = (
        segmentation_seed_metrics.groupby("grid", sort=False)
        .agg(
            seeds=("seed", "size"),
            dice_mean=("dice", "mean"),
            dice_sd=("dice", "std"),
            iou_mean=("iou", "mean"),
            iou_sd=("iou", "std"),
            precision_mean=("precision", "mean"),
            precision_sd=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
        .reset_index()
    )
    segmentation_differences_rows: list[dict[str, Any]] = []
    for seed in seeds:
        high = segmentation_seed_metrics.loc[
            (segmentation_seed_metrics["seed"] == seed)
            & (segmentation_seed_metrics["grid"] == "512x384")
        ].iloc[0]
        low = segmentation_seed_metrics.loc[
            (segmentation_seed_metrics["seed"] == seed)
            & (segmentation_seed_metrics["grid"] == "224x224")
        ].iloc[0]
        segmentation_differences_rows.append(
            {
                "row_type": "seed",
                "seed": seed,
                **{
                    f"{metric}_512x384_minus_224x224": float(high[metric] - low[metric])
                    for metric in ("dice", "iou", "precision", "recall")
                },
            }
        )
    segmentation_differences = pd.DataFrame(segmentation_differences_rows)
    segmentation_differences = pd.concat(
        [
            segmentation_differences,
            pd.DataFrame(
                [
                    {
                        "row_type": "across_seed_mean",
                        "seed": np.nan,
                        **{
                            column: float(segmentation_differences[column].mean())
                            for column in segmentation_differences.columns
                            if column.endswith("minus_224x224")
                        },
                    },
                    {
                        "row_type": "across_seed_sample_sd",
                        "seed": np.nan,
                        **{
                            column: float(segmentation_differences[column].std(ddof=1))
                            for column in segmentation_differences.columns
                            if column.endswith("minus_224x224")
                        },
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    registry = pd.DataFrame(registry_rows + segmentation_registry_rows).sort_values(
        ["artifact_type", "model", "seed", "outer_fold"]
    )
    if len(registry) != 400:
        raise RuntimeError(f"Expected 400 final input artifacts, found {len(registry)}.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".summary_tmp_", dir=output_dir.parent))
    try:
        output_frames = {
            "classification_fold_seed_metrics.csv": fold_metrics,
            "classification_fold_seed_class_metrics.csv": fold_class_metrics,
            "classification_seed_metrics.csv": seed_metrics,
            "classification_seed_class_metrics.csv": seed_class_metrics,
            "classification_model_summary.csv": model_summary,
            "primary_seed_contrasts.csv": primary_seed_contrasts,
            "primary_contrast_bootstrap.csv": primary_bootstrap,
            "primary_per_seed_group_bootstrap.csv": per_seed_bootstrap,
            "reference_seed_contrasts.csv": reference_contrasts,
            "bootstrap_replicates.csv": bootstrap_replicates_frame,
            "segmentation_fold_seed_metrics.csv": segmentation_fold_metrics,
            "segmentation_seed_metrics.csv": segmentation_seed_metrics,
            "segmentation_grid_summary.csv": segmentation_grid_summary,
            "segmentation_grid_differences.csv": segmentation_differences,
            "summary_input_registry.csv": registry,
        }
        for name, frame in output_frames.items():
            csv_write(frame, temporary_dir / name)

        summary_lines = [
            "# Frozen associate-editor revision experiment: final summary",
            "",
            "All results below were computed after the classification, stacking, and segmentation completion markers were present. No training artifact was modified by the summary step.",
            "",
            "## Three prespecified incremental comparisons",
            "",
            markdown_table(
                primary_bootstrap,
                [
                    "contrast",
                    "estimate_mean_seed_difference",
                    "sd_seed_difference",
                    "ci_low",
                    "ci_high",
                    "approximate_two_sided_p",
                    "holm_adjusted_p",
                ],
            ),
            "",
            "The estimate is the mean of the five seed-specific pooled-OOF Macro-F1 differences. Confidence intervals use 5,000 paired crossed seed x similarity-group bootstrap replicates unless a non-default replicate count is explicitly recorded in the completion marker. Approximate p values use plus-one two-sided bootstrap tails; Holm adjustment is across only these three prespecified new contrasts. Confidence intervals are not simultaneous.",
            "",
            "## Classification systems",
            "",
            markdown_table(
                model_summary,
                ["model", "macro_f1_mean", "macro_f1_sd", "accuracy_mean", "accuracy_sd"],
            ),
            "",
            "## Descriptive complete-system reference contrasts",
            "",
            markdown_table(
                reference_contrasts.loc[
                    reference_contrasts["row_type"] == "across_seed_summary"
                ],
                ["contrast", "difference", "difference_sd_across_seeds"],
            ),
            "",
            "These reference contrasts compare complete systems and do not isolate the effect of the spatial prior from architecture or system composition.",
            "",
            "## Source-domain segmentation audit",
            "",
            markdown_table(
                segmentation_grid_summary,
                [
                    "grid",
                    "dice_mean",
                    "dice_sd",
                    "iou_mean",
                    "iou_sd",
                    "precision_mean",
                    "precision_sd",
                    "recall_mean",
                    "recall_sd",
                ],
            ),
            "",
            "## Reporting boundaries",
            "",
            "- These are internal model-development estimates using image-similarity groups, not patient- or case-independent validation.",
            "- The classification experiment keeps the original transferred prior generator fixed and therefore does not include prior-generator retraining uncertainty.",
            "- Source-domain segmentation accuracy does not establish target-domain DFUC2021 prior accuracy because target-domain lesion masks were unavailable.",
            "- The bootstrap resamples existing similarity groups and training seeds; it does not retrain the complete pipeline within each replicate and does not capture historical model-development uncertainty.",
            "- Six inner-OOF EfficientNetV2-S prediction files containing 16 non-finite rows were rerun under the same frozen configuration and seed before stack fitting. The original files were archived and the repair is recorded in classification_prediction_integrity.json.",
            "- Non-significance is not evidence of equivalence. These estimates should not be presented as external validation or clinical readiness.",
            "",
        ]
        (temporary_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

        marker_paths = {
            "training_pipeline_complete": output_root / "training_pipeline_complete.json",
            "classification_complete": output_root / "classification_complete.json",
            "classification_prediction_integrity": output_root
            / "classification_prediction_integrity.json",
            "stack_complete": output_root / "stack_complete.json",
            "segmentation_complete": output_root / "segmentation_complete.json",
        }
        assets = {
            path.name: sha256(path)
            for path in sorted(temporary_dir.iterdir())
            if path.is_file() and path.name != "summary_complete.json"
        }
        completion_payload = {
            "status": "complete",
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256(protocol_path),
            "summarize_script_sha256": sha256(Path(__file__).resolve()),
            "training_pipeline_completed_at": training_marker["completed_at"],
            "seeds": seeds,
            "outer_folds": folds,
            "classification_models": report_names,
            "classification_images_per_seed_model": len(manifest),
            "similarity_groups": len(group_names),
            "bootstrap_seed": int(protocol["bootstrap_seed"]),
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "paired crossed seed x similarity_group",
            "prediction_repair_issue_files": int(integrity_marker["initial_issue_files"]),
            "prediction_repair_bad_rows": int(integrity_marker["initial_bad_rows"]),
            "input_artifacts": len(registry),
            "input_registry_sha256": sha256(temporary_dir / "summary_input_registry.csv"),
            "upstream_markers": {
                name: {"path": str(path.relative_to(repo_root)), "sha256": sha256(path)}
                for name, path in marker_paths.items()
            },
            "assets": assets,
        }
        write_json(temporary_dir / "summary_complete.json", completion_payload)
        temporary_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"SUMMARY_COMPLETE {output_dir}")
    print(primary_bootstrap.to_string(index=False))
    print(segmentation_grid_summary.to_string(index=False))


if __name__ == "__main__":
    main()
