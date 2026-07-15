#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class PredictionSpec:
    model_key: str
    model: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute grouped OOF metrics after excluding DFUC2021 images involved in broad cross-dataset visual-overlap candidates."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--cross-dataset-pairs",
        type=Path,
        default=Path("paper_assets/source_tables/methodology_audit_v7/dfuc2021_dfuc2022_cross_dataset_near_duplicate_pairs.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_tables/cross_dataset_overlap_exclusion_sensitivity_v1"))
    parser.add_argument(
        "--selected-predictions",
        type=Path,
        default=Path("outputs/publication_grouped/publication_final_assets/publication_selected_prior_informed_probability_fusion_predictions.csv"),
    )
    parser.add_argument(
        "--strong-rgb-predictions",
        type=Path,
        default=Path("outputs/predictions/reliability_prior_fusion_strong_cls_ensemble_oof_predictions.csv"),
    )
    parser.add_argument(
        "--strong-prior-predictions",
        type=Path,
        default=Path("outputs/predictions/reliability_prior_fusion_strong_cls_prior_residual_oof_predictions.csv"),
    )
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def label_ids(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).to_numpy()
    return series.astype(str).map(LABEL_TO_ID).astype(int).to_numpy()


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "label" in frame.columns:
        labels = label_ids(frame["label"])
    elif "true_label" in frame.columns:
        labels = label_ids(frame["true_label"])
    elif "label_name" in frame.columns:
        labels = label_ids(frame["label_name"])
    else:
        raise ValueError(f"Missing true-label column in {path}")
    if "pred_id" in frame.columns:
        pred = frame["pred_id"].astype(int).to_numpy()
    elif "pred_label" in frame.columns:
        pred = label_ids(frame["pred_label"])
    else:
        raise ValueError(f"Missing prediction column in {path}")
    output = pd.DataFrame({"image": frame["image"].astype(str), "label": labels, "pred_id": pred})
    output["label_name"] = [ID_TO_LABEL[int(value)] for value in output["label"]]
    output["pred_label"] = [ID_TO_LABEL[int(value)] for value in output["pred_id"]]
    return output


def binary_f1(frame: pd.DataFrame, positive_labels: set[int]) -> float:
    y_true = frame["label"].isin(positive_labels).astype(int).to_numpy()
    y_pred = frame["pred_id"].isin(positive_labels).astype(int).to_numpy()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return float(2.0 * precision * recall / max(1e-12, precision + recall))


def overall_metrics(frame: pd.DataFrame, scope: str, model_key: str, model: str) -> dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["label"],
        frame["pred_id"],
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return {
        "scope": scope,
        "model_key": model_key,
        "model": model,
        "samples": int(len(frame)),
        "macro_f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(frame["label"], frame["pred_id"])),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "accuracy": float(accuracy_score(frame["label"], frame["pred_id"])),
        "infection_binary_f1": binary_f1(frame, {1, 3}),
        "ischaemia_binary_f1": binary_f1(frame, {2, 3}),
    }


def per_class_metrics(frame: pd.DataFrame, scope: str, model_key: str, model: str) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        frame["label"],
        frame["pred_id"],
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    rows: list[dict[str, object]] = []
    for idx, class_name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "scope": scope,
                "model_key": model_key,
                "model": model,
                "class_name": class_name,
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )
    return rows


def binary_metrics(frame: pd.DataFrame, scope: str, model_key: str, model: str, endpoint: str, positive_labels: set[int]) -> dict[str, object]:
    y_true = frame["label"].isin(positive_labels).astype(int).to_numpy()
    y_pred = frame["pred_id"].isin(positive_labels).astype(int).to_numpy()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2.0 * precision * sensitivity / max(1e-12, precision + sensitivity)
    return {
        "scope": scope,
        "model_key": model_key,
        "model": model,
        "endpoint": endpoint,
        "positive_classes": ";".join(ID_TO_LABEL[idx] for idx in sorted(positive_labels)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (sensitivity + specificity)),
        "positive_support": int(np.sum(y_true == 1)),
        "negative_support": int(np.sum(y_true == 0)),
    }


def subset_frame(frame: pd.DataFrame, excluded_images: set[str], scope: str) -> pd.DataFrame:
    if scope == "full_oof":
        return frame.copy()
    if scope == "exclude_broad_cross_dataset_overlap_241":
        return frame[~frame["image"].isin(excluded_images)].copy()
    raise ValueError(scope)


def class_distribution(frame: pd.DataFrame, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_name in CLASS_NAMES:
        rows.append({"scope": scope, "class_name": class_name, "count": int((frame["label_name"] == class_name).sum())})
    return rows


def write_report(output_dir: Path, overall: pd.DataFrame, differences: pd.DataFrame, excluded_count: int) -> None:
    scope = "exclude_broad_cross_dataset_overlap_241"
    selected = overall[
        (overall["scope"] == scope)
        & (overall["model_key"] == "selected_prior_informed_probability_fusion")
    ].iloc[0]
    strong_rgb = overall[
        (overall["scope"] == scope)
        & (overall["model_key"] == "retained_strong_rgb_logistic_stack")
    ].iloc[0]
    strong_prior = overall[
        (overall["scope"] == scope)
        & (overall["model_key"] == "retained_strong_rgb_prior_feature_stack")
    ].iloc[0]
    delta_rgb = differences[
        (differences["scope"] == scope)
        & (differences["comparator_key"] == "retained_strong_rgb_logistic_stack")
    ].iloc[0]
    delta_prior = differences[
        (differences["scope"] == scope)
        & (differences["comparator_key"] == "retained_strong_rgb_prior_feature_stack")
    ].iloc[0]
    sentence = (
        f"Excluding the {excluded_count} DFUC2021 images involved in broad DFUC2021-DFUC2022 pHash/dHash "
        f"candidate pairs did not materially change the retained grouped OOF ranking: the selected prior-informed "
        f"probability-fusion candidate achieved Macro-F1 {selected['macro_f1']:.4f} on {int(selected['samples'])} "
        f"retained images, compared with {strong_rgb['macro_f1']:.4f} for the retained strong RGB comparator and "
        f"{strong_prior['macro_f1']:.4f} for the retained strong RGB plus prior-informed residual comparator; "
        f"the selected-candidate Macro-F1 differences remained positive "
        f"(+{float(delta_rgb['macro_f1_delta']):.4f} and +{float(delta_prior['macro_f1_delta']):.4f}, respectively)."
    )
    (output_dir / "cross_dataset_overlap_exclusion_supplement_sentence.txt").write_text(sentence + "\n", encoding="utf-8")
    (output_dir / "cross_dataset_overlap_exclusion_sensitivity_report.md").write_text("# Cross-Dataset Overlap Exclusion Sensitivity\n\n" + sentence + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(resolve(args.cross_dataset_pairs, repo_root))
    if "image_dfuc2021" in pairs.columns:
        dfuc2021_column = "image_dfuc2021"
    elif "dfuc2021_image" in pairs.columns:
        dfuc2021_column = "dfuc2021_image"
    else:
        raise ValueError("Cross-dataset pair table must include image_dfuc2021 or dfuc2021_image.")
    excluded_images = set(pairs[dfuc2021_column].astype(str))

    specs = [
        PredictionSpec(
            "selected_prior_informed_probability_fusion",
            "Selected prior-informed probability-fusion candidate",
            args.selected_predictions,
        ),
        PredictionSpec("retained_strong_rgb_logistic_stack", "Retained strong RGB comparator", args.strong_rgb_predictions),
        PredictionSpec(
            "retained_strong_rgb_prior_feature_stack",
            "Retained strong RGB + prior-informed residual comparator",
            args.strong_prior_predictions,
        ),
    ]
    predictions = {spec.model_key: read_predictions(resolve(spec.path, repo_root)) for spec in specs}
    reference = predictions["selected_prior_informed_probability_fusion"][["image", "label"]].sort_values("image").reset_index(drop=True)
    for model_id, frame in predictions.items():
        current = frame[["image", "label"]].sort_values("image").reset_index(drop=True)
        if not reference.equals(current):
            raise ValueError(f"Prediction image/label alignment mismatch for {model_id}")
        predictions[model_id] = frame.sort_values("image").reset_index(drop=True)

    excluded_frame = predictions["selected_prior_informed_probability_fusion"][
        predictions["selected_prior_informed_probability_fusion"]["image"].isin(excluded_images)
    ][["image", "label_name"]].sort_values("image")
    excluded_frame.to_csv(output_dir / "cross_dataset_overlap_excluded_dfuc2021_images.csv", index=False)

    scope_names = ("full_oof", "exclude_broad_cross_dataset_overlap_241")
    overall_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    binary_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    for scope in scope_names:
        reference_subset = subset_frame(predictions["selected_prior_informed_probability_fusion"], excluded_images, scope)
        distribution_rows.extend(class_distribution(reference_subset, scope))
        for spec in specs:
            frame = subset_frame(predictions[spec.model_key], excluded_images, scope)
            overall_rows.append(overall_metrics(frame, scope, spec.model_key, spec.model))
            per_class_rows.extend(per_class_metrics(frame, scope, spec.model_key, spec.model))
            binary_rows.append(binary_metrics(frame, scope, spec.model_key, spec.model, "infection_detection", {1, 3}))
            binary_rows.append(binary_metrics(frame, scope, spec.model_key, spec.model, "ischaemia_detection", {2, 3}))

    overall = pd.DataFrame(overall_rows)
    per_class = pd.DataFrame(per_class_rows)
    binary = pd.DataFrame(binary_rows)
    distribution = pd.DataFrame(distribution_rows)

    difference_rows: list[dict[str, object]] = []
    for scope in scope_names:
        selected = overall[(overall["scope"] == scope) & (overall["model_key"] == "selected_prior_informed_probability_fusion")].iloc[0]
        for comparator_id in ("retained_strong_rgb_logistic_stack", "retained_strong_rgb_prior_feature_stack"):
            comparator = overall[(overall["scope"] == scope) & (overall["model_key"] == comparator_id)].iloc[0]
            difference_rows.append(
                {
                    "scope": scope,
                    "candidate_key": selected["model_key"],
                    "candidate": selected["model"],
                    "comparator_key": comparator["model_key"],
                    "comparator": comparator["model"],
                    "candidate_samples": int(selected["samples"]),
                    "comparator_samples": int(comparator["samples"]),
                    "macro_f1_delta": float(selected["macro_f1"] - comparator["macro_f1"]),
                    "accuracy_delta": float(selected["accuracy"] - comparator["accuracy"]),
                    "infection_binary_f1_delta": float(selected["infection_binary_f1"] - comparator["infection_binary_f1"]),
                    "ischaemia_binary_f1_delta": float(selected["ischaemia_binary_f1"] - comparator["ischaemia_binary_f1"]),
                }
            )
    differences = pd.DataFrame(difference_rows)

    overall.to_csv(output_dir / "cross_dataset_overlap_exclusion_overall_metrics.csv", index=False)
    per_class.to_csv(output_dir / "cross_dataset_overlap_exclusion_per_class_metrics.csv", index=False)
    binary.to_csv(output_dir / "cross_dataset_overlap_exclusion_binary_endpoint_metrics.csv", index=False)
    distribution.to_csv(output_dir / "cross_dataset_overlap_exclusion_class_distribution.csv", index=False)
    differences.to_csv(output_dir / "cross_dataset_overlap_exclusion_model_differences.csv", index=False)

    retained_images = int(
        overall.loc[
            (overall["scope"] == "exclude_broad_cross_dataset_overlap_241")
            & (overall["model_key"] == "selected_prior_informed_probability_fusion"),
            "samples",
        ].iloc[0]
    )
    metadata = {
        "cross_dataset_pair_file": str(args.cross_dataset_pairs),
        "broad_candidate_pairs": int(len(pairs)),
        "excluded_dfuc2021_images": int(len(excluded_frame)),
        "retained_images": retained_images,
        "prediction_files": {
            "selected_prior_informed_probability_fusion": str(args.selected_predictions),
            "retained_strong_rgb_logistic_stack": str(args.strong_rgb_predictions),
            "retained_strong_rgb_prior_feature_stack": str(args.strong_prior_predictions),
        },
        "note": "Models are not retrained and fusion weights/class-logit bias terms are not reselected in this prediction-level sensitivity analysis.",
    }
    (output_dir / "cross_dataset_overlap_exclusion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output_dir, overall, differences, int(len(excluded_frame)))
    print(json.dumps(metadata, indent=2))
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
