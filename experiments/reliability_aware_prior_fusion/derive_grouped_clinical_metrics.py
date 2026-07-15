#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
ENDPOINTS = {
    "infection detection": ((1, 3), (1, 3)),
    "ischaemia detection": ((2, 3), (2, 3)),
}
EPS = 1e-12


@dataclass(frozen=True)
class PredictionInput:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive per-class and clinical binary-endpoint metrics with "
            "similarity-group bootstrap confidence intervals from fixed OOF predictions."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Repeat once per model. NAME is the publication display name.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-batch-size", type=int, default=100)
    return parser.parse_args()


def parse_prediction(value: str) -> PredictionInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Prediction inputs must use NAME=CSV syntax.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip()).expanduser().resolve()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("Prediction inputs must contain a non-empty name and path.")
    return PredictionInput(name=name, path=path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_to_ids(series: pd.Series, source: Path) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int).to_numpy()
    else:
        normalized = series.astype(str).str.strip().str.lower()
        mapped = normalized.map(LABEL_TO_ID)
        if mapped.isna().any():
            unknown = sorted(normalized[mapped.isna()].unique().tolist())
            raise ValueError(f"Unknown labels in {source}: {unknown}")
        values = mapped.astype(int).to_numpy()
    if not np.isin(values, np.arange(len(CLASS_NAMES))).all():
        raise ValueError(f"Labels outside 0-{len(CLASS_NAMES) - 1} in {source}")
    return values


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image", "similarity_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns {sorted(missing)} in {path}")
    if "label" in frame.columns:
        labels = labels_to_ids(frame["label"], path)
    elif "label_name" in frame.columns:
        labels = labels_to_ids(frame["label_name"], path)
    else:
        raise ValueError(f"Missing label or label_name in {path}")
    output = pd.DataFrame(
        {
            "image": frame["image"].astype(str),
            "label": labels,
            "similarity_group": frame["similarity_group"].astype(str),
        }
    )
    if output["image"].duplicated().any():
        duplicates = output.loc[output["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Duplicate manifest image names: {duplicates}")
    return output.sort_values("image").reset_index(drop=True)


def read_predictions(item: PredictionInput, manifest: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(item.path)
    required = {"image", *PROB_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns {sorted(missing)} in {item.path}")
    if "label" in frame.columns:
        labels = labels_to_ids(frame["label"], item.path)
    elif "true_label" in frame.columns:
        labels = labels_to_ids(frame["true_label"], item.path)
    elif "label_name" in frame.columns:
        labels = labels_to_ids(frame["label_name"], item.path)
    else:
        raise ValueError(f"Missing true-label column in {item.path}")

    probabilities = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(f"Invalid probabilities in {item.path}")
    row_sums = probabilities.sum(axis=1)
    if (row_sums <= 0).any():
        raise ValueError(f"Zero-sum probability rows in {item.path}")
    if (np.abs(row_sums - 1.0) > 0.05).any():
        raise ValueError(f"Probability row sums differ materially from one in {item.path}")

    if "pred_id" in frame.columns:
        predictions = labels_to_ids(frame["pred_id"], item.path)
    elif "pred_label" in frame.columns:
        predictions = labels_to_ids(frame["pred_label"], item.path)
    else:
        predictions = np.argmax(probabilities, axis=1).astype(int)

    output = pd.DataFrame({"image": frame["image"].astype(str), "label": labels, "pred_id": predictions})
    for index, column in enumerate(PROB_COLUMNS):
        output[column] = probabilities[:, index]
    if output["image"].duplicated().any():
        duplicates = output.loc[output["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Duplicate prediction image names in {item.path}: {duplicates}")

    aligned = manifest.merge(output, on="image", how="left", suffixes=("_manifest", "_prediction"), validate="one_to_one")
    if aligned["pred_id"].isna().any():
        missing_images = aligned.loc[aligned["pred_id"].isna(), "image"].head().tolist()
        raise ValueError(f"Missing prediction rows in {item.path}: {missing_images}")
    if len(output) != len(manifest):
        extra = sorted(set(output["image"]) - set(manifest["image"]))[:5]
        raise ValueError(f"Prediction image set differs from manifest in {item.path}; extra examples: {extra}")
    if not np.array_equal(aligned["label_manifest"].to_numpy(dtype=int), aligned["label_prediction"].to_numpy(dtype=int)):
        raise ValueError(f"True labels disagree with the manifest in {item.path}")
    aligned = aligned.rename(columns={"label_manifest": "label"})
    return aligned[["image", "label", "similarity_group", "pred_id", *PROB_COLUMNS]].copy()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_bootstrap_group_counts(
    group_count: int,
    iterations: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    output = np.empty((iterations, group_count), dtype=np.uint16)
    for start in range(0, iterations, batch_size):
        stop = min(iterations, start + batch_size)
        sampled = rng.integers(0, group_count, size=(stop - start, group_count))
        for offset, row in enumerate(sampled):
            output[start + offset] = np.bincount(row, minlength=group_count).astype(np.uint16)
    return output


def group_confusion_matrices(group_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrices = np.zeros((int(group_ids.max()) + 1, len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int32)
    np.add.at(matrices, (group_ids, y_true, y_pred), 1)
    return matrices


def bootstrap_confusion_matrices(group_counts: np.ndarray, group_matrices: np.ndarray, batch_size: int) -> np.ndarray:
    output = np.empty((len(group_counts), len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.float64)
    flattened = group_matrices.reshape(len(group_matrices), -1).astype(np.float64)
    for start in range(0, len(group_counts), batch_size):
        stop = min(len(group_counts), start + batch_size)
        weights = group_counts[start:stop].astype(np.float64)
        # Explicit accumulation avoids platform-specific BLAS warnings seen for
        # a highly rectangular count-matrix product on macOS Accelerate.
        weighted = np.sum(weights[:, :, None] * flattened[None, :, :], axis=1)
        output[start:stop] = weighted.reshape(stop - start, len(CLASS_NAMES), len(CLASS_NAMES))
    return output


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def class_metrics_from_matrices(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    predicted = matrices.sum(axis=-2)
    actual = matrices.sum(axis=-1)
    precision = safe_divide(true_positive, predicted)
    recall = safe_divide(true_positive, actual)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def binary_metrics_from_matrices(
    matrices: np.ndarray,
    positive_classes: tuple[int, ...],
) -> dict[str, np.ndarray]:
    positive = np.array(positive_classes, dtype=int)
    negative = np.array([index for index in range(len(CLASS_NAMES)) if index not in positive_classes], dtype=int)
    tp = matrices[..., positive[:, None], positive].sum(axis=(-2, -1))
    fn = matrices[..., positive[:, None], negative].sum(axis=(-2, -1))
    fp = matrices[..., negative[:, None], positive].sum(axis=(-2, -1))
    tn = matrices[..., negative[:, None], negative].sum(axis=(-2, -1))
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    ppv = safe_divide(tp, tp + fp)
    npv = safe_divide(tn, tn + fn)
    f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def weighted_rank_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    group_ids: np.ndarray,
    group_counts: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order].astype(float)
    sorted_groups = group_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    auroc = np.empty(len(group_counts), dtype=float)
    auprc = np.empty(len(group_counts), dtype=float)

    for start in range(0, len(group_counts), batch_size):
        stop = min(len(group_counts), start + batch_size)
        sample_weights = group_counts[start:stop, sorted_groups].astype(float)
        positive_weights = np.add.reduceat(sample_weights * sorted_labels, starts, axis=1)
        negative_weights = np.add.reduceat(sample_weights * (1.0 - sorted_labels), starts, axis=1)
        total_positive = positive_weights.sum(axis=1)
        total_negative = negative_weights.sum(axis=1)

        cumulative_positive = np.cumsum(positive_weights, axis=1)
        cumulative_negative = np.cumsum(negative_weights, axis=1)
        lower_negative = total_negative[:, None] - cumulative_negative
        auc_numerator = np.sum(positive_weights * (lower_negative + 0.5 * negative_weights), axis=1)
        auroc[start:stop] = safe_divide(auc_numerator, total_positive * total_negative)

        cumulative_total = cumulative_positive + cumulative_negative
        precision_at_threshold = safe_divide(cumulative_positive, cumulative_total)
        recall_increment = safe_divide(positive_weights, total_positive[:, None])
        auprc[start:stop] = np.sum(recall_increment * precision_at_threshold, axis=1)
    return auroc, auprc


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def metric_columns(row: dict[str, object], name: str, observed: float, bootstrap: np.ndarray) -> None:
    low, high = interval(bootstrap)
    row[name] = float(observed)
    row[f"{name}_ci_low"] = low
    row[f"{name}_ci_high"] = high


def build_tables(
    item: PredictionInput,
    frame: pd.DataFrame,
    group_ids: np.ndarray,
    group_counts: np.ndarray,
    batch_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y_true = frame["label"].to_numpy(dtype=int)
    y_pred = frame["pred_id"].to_numpy(dtype=int)
    probabilities = frame[list(PROB_COLUMNS)].to_numpy(dtype=float)
    grouped_matrices = group_confusion_matrices(group_ids, y_true, y_pred)
    observed_matrix = grouped_matrices.sum(axis=0, keepdims=True).astype(float)
    bootstrap_matrices = bootstrap_confusion_matrices(group_counts, grouped_matrices, batch_size)
    observed_precision, observed_recall, observed_f1 = class_metrics_from_matrices(observed_matrix)
    bootstrap_precision, bootstrap_recall, bootstrap_f1 = class_metrics_from_matrices(bootstrap_matrices)

    per_class_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_truth = (y_true == class_index).astype(int)
        class_auroc, class_auprc = weighted_rank_metrics(
            class_truth,
            probabilities[:, class_index],
            group_ids,
            group_counts,
            batch_size,
        )
        observed_rank = weighted_rank_metrics(
            class_truth,
            probabilities[:, class_index],
            group_ids,
            np.ones((1, len(grouped_matrices)), dtype=np.uint16),
            1,
        )
        tp = float(observed_matrix[0, class_index, class_index])
        support = float(observed_matrix[0, class_index].sum())
        predicted = float(observed_matrix[0, :, class_index].sum())
        total = float(observed_matrix.sum())
        row: dict[str, object] = {
            "method": item.name,
            "class": class_name,
            "images": int(len(frame)),
            "similarity_groups": int(len(grouped_matrices)),
            "support": int(support),
            "tp": int(tp),
            "fp": int(predicted - tp),
            "fn": int(support - tp),
            "tn": int(total - support - predicted + tp),
        }
        metric_columns(row, "precision", observed_precision[0, class_index], bootstrap_precision[:, class_index])
        metric_columns(row, "recall", observed_recall[0, class_index], bootstrap_recall[:, class_index])
        metric_columns(row, "f1", observed_f1[0, class_index], bootstrap_f1[:, class_index])
        metric_columns(row, "auroc", observed_rank[0][0], class_auroc)
        metric_columns(row, "auprc", observed_rank[1][0], class_auprc)
        per_class_rows.append(row)

    binary_rows: list[dict[str, object]] = []
    for endpoint_name, (positive_classes, probability_classes) in ENDPOINTS.items():
        observed_binary = binary_metrics_from_matrices(observed_matrix, positive_classes)
        bootstrap_binary = binary_metrics_from_matrices(bootstrap_matrices, positive_classes)
        endpoint_truth = np.isin(y_true, positive_classes).astype(int)
        endpoint_scores = probabilities[:, probability_classes].sum(axis=1)
        bootstrap_auroc, bootstrap_auprc = weighted_rank_metrics(
            endpoint_truth,
            endpoint_scores,
            group_ids,
            group_counts,
            batch_size,
        )
        observed_rank = weighted_rank_metrics(
            endpoint_truth,
            endpoint_scores,
            group_ids,
            np.ones((1, len(grouped_matrices)), dtype=np.uint16),
            1,
        )
        row = {
            "method": item.name,
            "endpoint": endpoint_name,
            "images": int(len(frame)),
            "similarity_groups": int(len(grouped_matrices)),
            "positive_images": int(endpoint_truth.sum()),
            "decision_rule": "positive when the saved four-class OOF decision is in either positive class",
            "score_rule": "+".join(PROB_COLUMNS[index] for index in probability_classes),
            "tp": int(observed_binary["tp"][0]),
            "fp": int(observed_binary["fp"][0]),
            "fn": int(observed_binary["fn"][0]),
            "tn": int(observed_binary["tn"][0]),
        }
        for metric_name in ("sensitivity", "specificity", "ppv", "npv", "f1", "balanced_accuracy"):
            metric_columns(
                row,
                metric_name,
                observed_binary[metric_name][0],
                bootstrap_binary[metric_name],
            )
        metric_columns(row, "auroc", observed_rank[0][0], bootstrap_auroc)
        metric_columns(row, "auprc", observed_rank[1][0], bootstrap_auprc)
        binary_rows.append(row)
    return per_class_rows, binary_rows


def main() -> None:
    args = parse_args()
    if args.bootstrap_resamples < 1:
        raise ValueError("--bootstrap-resamples must be positive")
    if args.bootstrap_batch_size < 1:
        raise ValueError("--bootstrap-batch-size must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    predictions = [parse_prediction(value) for value in args.prediction]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path)
    group_labels, group_ids = np.unique(manifest["similarity_group"].to_numpy(), return_inverse=True)
    group_counts = build_bootstrap_group_counts(
        len(group_labels),
        args.bootstrap_resamples,
        args.seed,
        args.bootstrap_batch_size,
    )

    per_class_rows: list[dict[str, object]] = []
    binary_rows: list[dict[str, object]] = []
    input_records: list[dict[str, str]] = []
    for item in predictions:
        frame = read_predictions(item, manifest)
        class_rows, endpoint_rows = build_tables(
            item,
            frame,
            group_ids,
            group_counts,
            args.bootstrap_batch_size,
        )
        per_class_rows.extend(class_rows)
        binary_rows.extend(endpoint_rows)
        input_records.append(
            {
                "method": item.name,
                "source_file": item.path.name,
                "source_id": slugify(item.name),
                "sha256": file_sha256(item.path),
            }
        )

    per_class_path = output_dir / "main_reported_grouped_per_class_metrics_with_ci.csv"
    binary_path = output_dir / "main_reported_grouped_binary_endpoint_metrics_with_ci.csv"
    metadata_path = output_dir / "main_reported_grouped_clinical_metrics_metadata.json"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)
    pd.DataFrame(binary_rows).to_csv(binary_path, index=False)
    metadata = {
        "analysis": "clinical metrics derived from fixed similarity-grouped OOF predictions",
        "images": int(len(manifest)),
        "similarity_groups": int(len(group_labels)),
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "bootstrap_seed": int(args.seed),
        "bootstrap_unit": "reconstructed image-similarity group",
        "interval": "2.5th and 97.5th percentiles",
        "paired_draws_across_models": True,
        "threshold_selection": "none; saved four-class OOF decisions were retained",
        "binary_score_definitions": {
            "infection detection": "prob_infection + prob_both",
            "ischaemia detection": "prob_ischaemia + prob_both",
        },
        "manifest": {"source_file": manifest_path.name, "sha256": file_sha256(manifest_path)},
        "prediction_inputs": input_records,
        "outputs": [per_class_path.name, binary_path.name],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"per_class": str(per_class_path), "binary": str(binary_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
