#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
LABEL_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
PROB_COLUMNS = tuple(f"prob_{name}" for name in CLASS_NAMES)
METRIC_NAMES = ("macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy")
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
EPS = 1e-6


MODEL_PREFIXES = {
    "resnet50": "dfuc2021_cls_resnet50",
    "efficientnetb0": "dfuc2021_cls_efficientnetb0",
    "convnext_tiny": "dfuc2021_cls_convnext_tiny",
    "efficientnetv2s": "dfuc2021_cls_efficientnetv2s",
    "lpt_full": "cross_task_lpt_resnet50_mobilenetv3",
    "lpt_no_alignment": "cross_task_lpt_no_alignment",
    "lpt_no_prior_mod": "cross_task_lpt_no_prior_mod",
    "lpt_no_roi": "cross_task_lpt_no_roi",
    "lpt_stats_only": "cross_task_lpt_stats_only",
    "lpt_random_prior": "cross_task_lpt_random_prior",
}


PRIOR_NUMERIC_COLUMNS = (
    "prior_mean",
    "prior_max",
    "prior_area_ratio",
    "largest_component_area_ratio",
    "bbox_area_ratio",
    "prior_confidence",
    "prior_fallback",
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    model_names: tuple[str, ...]
    use_prior_stats: bool
    use_interactions: bool = True
    description: str = ""
    default: bool = True


FEATURE_SETS = (
    FeatureSet(
        name="resnet50_calibrated",
        model_names=("resnet50",),
        use_prior_stats=False,
        use_interactions=False,
        description="Fold-wise calibrated ResNet50 OOF predictions.",
    ),
    FeatureSet(
        name="cls_ensemble",
        model_names=("resnet50", "efficientnetb0"),
        use_prior_stats=False,
        use_interactions=False,
        description="Pure classification ensemble from ResNet50 and EfficientNet-B0.",
    ),
    FeatureSet(
        name="resnet50_prior_residual",
        model_names=("resnet50",),
        use_prior_stats=True,
        description="Reliability-aware residual fusion on top of ResNet50.",
    ),
    FeatureSet(
        name="cls_prior_residual",
        model_names=("resnet50", "efficientnetb0"),
        use_prior_stats=True,
        description="Reliability-aware residual fusion on top of the two pure classification baselines.",
    ),
    FeatureSet(
        name="strong_cls_ensemble",
        model_names=("resnet50", "efficientnetb0", "convnext_tiny", "efficientnetv2s"),
        use_prior_stats=False,
        use_interactions=False,
        description="Pure classification ensemble including ConvNeXt-Tiny and EfficientNetV2-S.",
        default=False,
    ),
    FeatureSet(
        name="strong_cls_prior_residual",
        model_names=("resnet50", "efficientnetb0", "convnext_tiny", "efficientnetv2s"),
        use_prior_stats=True,
        description="Reliability-aware residual fusion on top of the strong classification ensemble.",
        default=False,
    ),
    FeatureSet(
        name="lpt_model_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_full",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=True,
        description="Ensemble of non-random prior-transfer variants plus prior reliability statistics.",
    ),
    FeatureSet(
        name="lpt_output_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_full",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=False,
        use_interactions=False,
        description="Ensemble of non-random prior-transfer variants without explicit prior reliability statistics.",
    ),
    FeatureSet(
        name="lpt_ablation_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=True,
        description="Ensemble of non-random LPT ablations plus prior reliability statistics, excluding the full LPT model.",
    ),
    FeatureSet(
        name="lpt_ablation_output_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=False,
        use_interactions=False,
        description="Ensemble of non-random LPT ablation outputs without explicit prior reliability statistics.",
    ),
    FeatureSet(
        name="compact_prior_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_stats_only",
        ),
        use_prior_stats=True,
        description="Compact prior-aware ensemble with the strongest non-random prior-transfer variants.",
    ),
    FeatureSet(
        name="compact_prior_output_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_stats_only",
        ),
        use_prior_stats=False,
        use_interactions=False,
        description="Compact ensemble with strongest prior-transfer outputs and no explicit prior reliability statistics.",
    ),
    FeatureSet(
        name="strong_lpt_output_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "convnext_tiny",
            "efficientnetv2s",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=False,
        use_interactions=False,
        description="Strong classifier and LPT-ablation output ensemble without explicit prior reliability statistics.",
        default=False,
    ),
    FeatureSet(
        name="strong_lpt_prior_fusion",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "convnext_tiny",
            "efficientnetv2s",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=True,
        description="Strong classifier and non-random LPT-ablation ensemble plus prior reliability statistics.",
        default=False,
    ),
    FeatureSet(
        name="strong_lpt_full_output_ensemble",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "convnext_tiny",
            "efficientnetv2s",
            "lpt_full",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=False,
        use_interactions=False,
        description="Strong classifier and full non-random LPT output ensemble without explicit prior reliability statistics.",
        default=False,
    ),
    FeatureSet(
        name="strong_lpt_full_prior_fusion",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "convnext_tiny",
            "efficientnetv2s",
            "lpt_full",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
        ),
        use_prior_stats=True,
        description="Strong classifier and full non-random LPT ensemble plus prior reliability statistics.",
        default=False,
    ),
    FeatureSet(
        name="diagnostic_with_random_prior",
        model_names=(
            "resnet50",
            "efficientnetb0",
            "lpt_full",
            "lpt_no_alignment",
            "lpt_no_prior_mod",
            "lpt_no_roi",
            "lpt_stats_only",
            "lpt_random_prior",
        ),
        use_prior_stats=True,
        description="Diagnostic upper-bound style stack including the random-prior control.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-strict OOF stacking for reliability-aware lesion-prior fusion.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument(
        "--prior-meta",
        type=Path,
        default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv"),
    )
    parser.add_argument("--prediction-dir", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/logs"))
    parser.add_argument("--output-prefix", default="reliability_prior_fusion")
    parser.add_argument("--folds", type=int, nargs="+", default=list(DEFAULT_FOLDS))
    parser.add_argument("--feature-sets", nargs="+", default=None)
    parser.add_argument("--no-bias-search", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image", "label", "label_name", "fold"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame[["image", "label", "label_name", "fold"]].copy()


def read_prior_meta(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"image", *PRIOR_NUMERIC_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prior metadata missing required columns: {sorted(missing)}")
    keep_columns = ["image", *PRIOR_NUMERIC_COLUMNS]
    prior = frame[keep_columns].copy()
    for column in PRIOR_NUMERIC_COLUMNS:
        prior[column] = pd.to_numeric(prior[column], errors="coerce").fillna(0.0).astype(float)
    return prior


def probability_features(prefix: str, probs: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probs.astype(float), EPS, 1.0)
    clipped = clipped / clipped.sum()
    order = np.sort(clipped)
    entropy = float(-np.sum(clipped * np.log(clipped)) / math.log(len(CLASS_NAMES)))
    features: dict[str, float] = {
        f"{prefix}_confidence": float(order[-1]),
        f"{prefix}_margin": float(order[-1] - order[-2]),
        f"{prefix}_entropy": entropy,
        f"{prefix}_uncertainty": float(1.0 - order[-1]),
    }
    for idx, class_name in enumerate(CLASS_NAMES):
        features[f"{prefix}_prob_{class_name}"] = float(clipped[idx])
        features[f"{prefix}_logprob_{class_name}"] = float(np.log(clipped[idx]))
    return features


def load_model_predictions(prediction_dir: Path, model_name: str, folds: tuple[int, ...]) -> pd.DataFrame:
    prefix = MODEL_PREFIXES[model_name]
    rows: list[dict[str, Any]] = []
    for fold in folds:
        path = prediction_dir / f"{prefix}_fold{fold}_val_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file for {model_name} fold {fold}: {path}")
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                probs = np.array([float(row[column]) for column in PROB_COLUMNS], dtype=np.float64)
                item: dict[str, Any] = {
                    "image": row["image"],
                    f"{model_name}_pred_label": LABEL_TO_ID[str(row["pred_label"])],
                    f"{model_name}_true_label": LABEL_TO_ID[str(row["true_label"])],
                }
                item.update(probability_features(model_name, probs))
                for attr_column in ("prob_infection_attr", "prob_ischaemia_attr"):
                    if attr_column in row:
                        item[f"{model_name}_{attr_column}"] = float(row[attr_column])
                rows.append(item)
    frame = pd.DataFrame(rows)
    if frame["image"].duplicated().any():
        duplicates = frame.loc[frame["image"].duplicated(), "image"].head().tolist()
        raise ValueError(f"Duplicate predictions for {model_name}: {duplicates}")
    return frame


def add_prior_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    area = np.clip(frame["prior_area_ratio"].to_numpy(dtype=float), EPS, 1.0)
    largest = np.clip(frame["largest_component_area_ratio"].to_numpy(dtype=float), 0.0, 1.0)
    bbox_area = np.clip(frame["bbox_area_ratio"].to_numpy(dtype=float), EPS, 1.0)
    confidence = np.clip(frame["prior_confidence"].to_numpy(dtype=float), 0.0, 1.0)
    fallback = np.clip(frame["prior_fallback"].to_numpy(dtype=float), 0.0, 1.0)

    frame["prior_quality"] = confidence * (1.0 - fallback)
    frame["prior_bad_quality"] = 1.0 - frame["prior_quality"]
    frame["prior_component_fraction"] = largest / area
    frame["prior_bbox_fill"] = area / bbox_area
    frame["prior_area_logit"] = np.log(area / np.clip(1.0 - area, EPS, 1.0))
    frame["prior_confidence_x_area"] = confidence * area
    frame["prior_fallback_x_area"] = fallback * area
    return frame


def build_base_frame(
    manifest: pd.DataFrame,
    prior_meta: pd.DataFrame,
    prediction_dir: Path,
    folds: tuple[int, ...],
    model_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    frame = manifest.merge(add_prior_derived_features(prior_meta), on="image", how="left", validate="one_to_one")
    if frame[list(PRIOR_NUMERIC_COLUMNS)].isnull().any().any():
        raise ValueError("Some manifest rows did not receive prior metadata.")

    load_model_names = model_names if model_names is not None else tuple(MODEL_PREFIXES)
    for model_name in load_model_names:
        predictions = load_model_predictions(prediction_dir, model_name, folds)
        frame = frame.merge(predictions, on="image", how="left", validate="one_to_one")
    missing_prediction = [column for column in frame.columns if column.endswith("_confidence") and frame[column].isnull().any()]
    if missing_prediction:
        raise ValueError(f"Missing model predictions in columns: {missing_prediction}")

    if "resnet50_confidence" in frame and "efficientnetb0_confidence" in frame:
        res_pred = frame["resnet50_pred_label"].to_numpy(dtype=int)
        eff_pred = frame["efficientnetb0_pred_label"].to_numpy(dtype=int)
        frame["cls_models_agree"] = (res_pred == eff_pred).astype(float)
        frame["cls_models_disagree"] = 1.0 - frame["cls_models_agree"]
        frame["cls_confidence_mean"] = 0.5 * (frame["resnet50_confidence"] + frame["efficientnetb0_confidence"])
        frame["cls_entropy_mean"] = 0.5 * (frame["resnet50_entropy"] + frame["efficientnetb0_entropy"])
        diff_columns = [
            np.abs(frame[f"resnet50_prob_{class_name}"] - frame[f"efficientnetb0_prob_{class_name}"])
            for class_name in CLASS_NAMES
        ]
        frame["cls_prob_l1_disagreement"] = np.sum(np.vstack(diff_columns), axis=0)

    return frame


def feature_columns_for(frame: pd.DataFrame, feature_set: FeatureSet) -> list[str]:
    columns: list[str] = []
    for model_name in feature_set.model_names:
        model_prefixes = (
            f"{model_name}_prob_",
            f"{model_name}_logprob_",
        )
        columns.extend(
            column
            for column in frame.columns
            if column.startswith(model_prefixes)
            or column in {
                f"{model_name}_confidence",
                f"{model_name}_margin",
                f"{model_name}_entropy",
                f"{model_name}_uncertainty",
                f"{model_name}_prob_infection_attr",
                f"{model_name}_prob_ischaemia_attr",
            }
        )
    if "resnet50" in feature_set.model_names and "efficientnetb0" in feature_set.model_names:
        columns.extend(
            [
                "cls_models_agree",
                "cls_models_disagree",
                "cls_confidence_mean",
                "cls_entropy_mean",
                "cls_prob_l1_disagreement",
            ]
        )
    if feature_set.use_prior_stats:
        prior_columns = [
            column
            for column in frame.columns
            if column.startswith("prior_") or column in {"largest_component_area_ratio", "bbox_area_ratio"}
        ]
        columns.extend(prior_columns)
        if feature_set.use_interactions:
            interaction_columns = add_interaction_columns(frame, feature_set)
            columns.extend(interaction_columns)
    deduped = list(dict.fromkeys(columns))
    return [column for column in deduped if column in frame.columns]


def add_interaction_columns(frame: pd.DataFrame, feature_set: FeatureSet) -> list[str]:
    prior_quality = frame["prior_quality"].to_numpy(dtype=float)
    prior_bad_quality = frame["prior_bad_quality"].to_numpy(dtype=float)
    interaction_columns: list[str] = []
    for model_name in feature_set.model_names:
        for signal in ("confidence", "margin", "entropy", "uncertainty"):
            source = f"{model_name}_{signal}"
            if source not in frame.columns:
                continue
            quality_column = f"{source}_x_prior_quality"
            bad_quality_column = f"{source}_x_prior_bad_quality"
            if quality_column not in frame.columns:
                frame[quality_column] = frame[source].to_numpy(dtype=float) * prior_quality
            if bad_quality_column not in frame.columns:
                frame[bad_quality_column] = frame[source].to_numpy(dtype=float) * prior_bad_quality
            interaction_columns.extend([quality_column, bad_quality_column])
    if "cls_models_disagree" in frame.columns:
        for prior_column in ("prior_quality", "prior_confidence", "prior_area_ratio", "prior_fallback"):
            interaction = f"cls_disagree_x_{prior_column}"
            if interaction not in frame.columns:
                frame[interaction] = frame["cls_models_disagree"].to_numpy(dtype=float) * frame[prior_column].to_numpy(dtype=float)
            interaction_columns.append(interaction)
    return interaction_columns


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).astype(int).tolist(),
    }


def build_estimator(c_value: float, class_weight: str | None) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=5000,
            solver="lbfgs",
            random_state=20260501,
        ),
    )


def predict_proba_aligned(estimator: Any, x_values: np.ndarray) -> np.ndarray:
    probs = estimator.predict_proba(x_values)
    classes = estimator.named_steps["logisticregression"].classes_
    aligned = np.zeros((len(x_values), len(CLASS_NAMES)), dtype=np.float64)
    for idx, class_id in enumerate(classes):
        aligned[:, int(class_id)] = probs[:, idx]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(row_sums, EPS, None)


def apply_biases(probs: np.ndarray, biases: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, EPS, 1.0)) + biases.reshape(1, -1)
    return np.argmax(logits, axis=1)


def optimize_biases(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float]:
    biases = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    best_pred = apply_biases(probs, biases)
    best_score = float(precision_recall_fscore_support(y_true, best_pred, average="macro", zero_division=0)[2])
    for step in (0.5, 0.25, 0.1, 0.05):
        improved = True
        while improved:
            improved = False
            for class_idx in range(len(CLASS_NAMES)):
                for delta in (-step, step):
                    candidate = biases.copy()
                    candidate[class_idx] = float(np.clip(candidate[class_idx] + delta, -2.0, 2.0))
                    pred = apply_biases(probs, candidate)
                    score = float(precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)[2])
                    if score > best_score + 1e-9:
                        biases = candidate
                        best_score = score
                        improved = True
    return biases, best_score


def inner_select_params(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    feature_columns: list[str],
    folds: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray, float]:
    x_all = frame[feature_columns].to_numpy(dtype=np.float64)
    y_all = frame["label"].to_numpy(dtype=int)
    fold_values = frame["fold"].to_numpy(dtype=int)
    candidate_params = [
        {"C": c_value, "class_weight": class_weight}
        for class_weight in (None, "balanced")
        for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    ]
    best_params = candidate_params[0]
    best_score = -float("inf")
    best_oof_probs = np.zeros((int(train_mask.sum()), len(CLASS_NAMES)), dtype=np.float64)
    train_indices = np.flatnonzero(train_mask)

    for params in candidate_params:
        oof_probs = np.zeros((len(train_indices), len(CLASS_NAMES)), dtype=np.float64)
        for inner_fold in folds:
            inner_val_mask = train_mask & (fold_values == inner_fold)
            if not inner_val_mask.any():
                continue
            inner_fit_mask = train_mask & (fold_values != inner_fold)
            estimator = build_estimator(params["C"], params["class_weight"])
            estimator.fit(x_all[inner_fit_mask], y_all[inner_fit_mask])
            val_indices = np.flatnonzero(inner_val_mask)
            positions = np.searchsorted(train_indices, val_indices)
            oof_probs[positions] = predict_proba_aligned(estimator, x_all[inner_val_mask])
        y_train = y_all[train_mask]
        pred = np.argmax(oof_probs, axis=1)
        score = float(precision_recall_fscore_support(y_train, pred, average="macro", zero_division=0)[2])
        if score > best_score:
            best_score = score
            best_params = params
            best_oof_probs = oof_probs
    return best_params, best_oof_probs, best_score


def run_feature_set(
    frame: pd.DataFrame,
    feature_set: FeatureSet,
    folds: tuple[int, ...],
    prediction_dir: Path,
    log_dir: Path,
    output_prefix: str,
    use_bias_search: bool,
) -> dict[str, Any]:
    feature_columns = feature_columns_for(frame, feature_set)
    if not feature_columns:
        raise ValueError(f"No features selected for {feature_set.name}")

    x_all = frame[feature_columns].to_numpy(dtype=np.float64)
    y_all = frame["label"].to_numpy(dtype=int)
    fold_values = frame["fold"].to_numpy(dtype=int)
    fold_results: list[dict[str, Any]] = []
    all_rows: list[pd.DataFrame] = []

    for fold in folds:
        train_mask = fold_values != fold
        val_mask = fold_values == fold
        inner_folds = tuple(inner_fold for inner_fold in folds if inner_fold != fold)
        best_params, inner_oof_probs, inner_score = inner_select_params(frame, train_mask, feature_columns, inner_folds)
        if use_bias_search:
            biases, bias_train_macro_f1 = optimize_biases(inner_oof_probs, y_all[train_mask])
        else:
            biases = np.zeros(len(CLASS_NAMES), dtype=np.float64)
            bias_train_macro_f1 = inner_score

        estimator = build_estimator(best_params["C"], best_params["class_weight"])
        estimator.fit(x_all[train_mask], y_all[train_mask])
        val_probs = predict_proba_aligned(estimator, x_all[val_mask])
        val_pred = apply_biases(val_probs, biases)
        metrics = compute_metrics(y_all[val_mask], val_pred)

        fold_frame = frame.loc[val_mask, ["image", "label", "label_name", "fold"]].copy()
        fold_frame["pred_label"] = [CLASS_NAMES[int(label)] for label in val_pred]
        for class_idx, class_name in enumerate(CLASS_NAMES):
            fold_frame[f"prob_{class_name}"] = val_probs[:, class_idx]
        fold_frame["selected_C"] = float(best_params["C"])
        fold_frame["selected_class_weight"] = "none" if best_params["class_weight"] is None else str(best_params["class_weight"])
        prediction_path = prediction_dir / f"{output_prefix}_{feature_set.name}_fold{fold}_val_predictions.csv"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        fold_frame.to_csv(prediction_path, index=False)
        all_rows.append(fold_frame)

        fold_results.append(
            {
                "fold": int(fold),
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "inner_macro_f1": float(inner_score),
                "bias_train_macro_f1": float(bias_train_macro_f1),
                "selected_C": float(best_params["C"]),
                "selected_class_weight": best_params["class_weight"],
                "biases": {class_name: float(biases[idx]) for idx, class_name in enumerate(CLASS_NAMES)},
                "metrics": metrics,
                "prediction_path": str(prediction_path),
            }
        )

    stacked_predictions = pd.concat(all_rows, ignore_index=True)
    aggregate_metrics = compute_metrics(stacked_predictions["label"].to_numpy(dtype=int), np.array([LABEL_TO_ID[label] for label in stacked_predictions["pred_label"]]))
    fold_aggregate = {
        metric: {
            "mean": float(np.mean([result["metrics"][metric] for result in fold_results])),
            "std": float(np.std([result["metrics"][metric] for result in fold_results])),
        }
        for metric in METRIC_NAMES
    }
    payload = {
        "experiment_name": f"{output_prefix}_{feature_set.name}",
        "feature_set": feature_set.__dict__,
        "feature_columns": feature_columns,
        "fold_results": fold_results,
        "fold_aggregate": fold_aggregate,
        "oof_metrics": aggregate_metrics,
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / f"{output_prefix}_{feature_set.name}_cv_summary.json"
    csv_path = log_dir / f"{output_prefix}_{feature_set.name}_cv_summary.csv"
    all_predictions_path = prediction_dir / f"{output_prefix}_{feature_set.name}_oof_predictions.csv"
    stacked_predictions.to_csv(all_predictions_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "fold",
                "train_samples",
                "val_samples",
                "inner_macro_f1",
                "bias_train_macro_f1",
                "selected_C",
                "selected_class_weight",
                *METRIC_NAMES,
            ]
        )
        for result in fold_results:
            writer.writerow(
                [
                    result["fold"],
                    result["train_samples"],
                    result["val_samples"],
                    f"{result['inner_macro_f1']:.6f}",
                    f"{result['bias_train_macro_f1']:.6f}",
                    f"{result['selected_C']:.6g}",
                    "none" if result["selected_class_weight"] is None else result["selected_class_weight"],
                    *(f"{result['metrics'][metric]:.6f}" for metric in METRIC_NAMES),
                ]
            )
        writer.writerow(["mean", "", "", "", "", "", "", *(f"{fold_aggregate[metric]['mean']:.6f}" for metric in METRIC_NAMES)])
        writer.writerow(["std", "", "", "", "", "", "", *(f"{fold_aggregate[metric]['std']:.6f}" for metric in METRIC_NAMES)])
        writer.writerow(["oof", "", "", "", "", "", "", *(f"{aggregate_metrics[metric]:.6f}" for metric in METRIC_NAMES)])

    return {
        "feature_set": feature_set.name,
        "description": feature_set.description,
        "macro_f1": aggregate_metrics["macro_f1"],
        "balanced_accuracy": aggregate_metrics["balanced_accuracy"],
        "macro_precision": aggregate_metrics["macro_precision"],
        "macro_recall": aggregate_metrics["macro_recall"],
        "accuracy": aggregate_metrics["accuracy"],
        "fold_macro_f1_mean": fold_aggregate["macro_f1"]["mean"],
        "fold_macro_f1_std": fold_aggregate["macro_f1"]["std"],
        "feature_count": len(feature_columns),
        "summary_json": str(json_path),
        "summary_csv": str(csv_path),
        "predictions_csv": str(all_predictions_path),
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = resolve_path(args.manifest, repo_root)
    prior_meta_path = resolve_path(args.prior_meta, repo_root)
    prediction_dir = resolve_path(args.prediction_dir, repo_root)
    log_dir = resolve_path(args.log_dir, repo_root)
    folds = tuple(int(fold) for fold in args.folds)

    requested_feature_sets = args.feature_sets or [feature_set.name for feature_set in FEATURE_SETS if feature_set.default]
    selected_feature_sets = [feature_set for feature_set in FEATURE_SETS if feature_set.name in set(requested_feature_sets)]
    missing = set(requested_feature_sets).difference(feature_set.name for feature_set in selected_feature_sets)
    if missing:
        raise ValueError(f"Unknown feature sets: {sorted(missing)}")
    required_model_names = tuple(
        dict.fromkeys(model_name for feature_set in selected_feature_sets for model_name in feature_set.model_names)
    )

    manifest = read_manifest(manifest_path)
    prior_meta = read_prior_meta(prior_meta_path)
    frame = build_base_frame(manifest, prior_meta, prediction_dir, folds, required_model_names)

    comparison_rows = []
    for feature_set in selected_feature_sets:
        result = run_feature_set(
            frame=frame,
            feature_set=feature_set,
            folds=folds,
            prediction_dir=prediction_dir,
            log_dir=log_dir,
            output_prefix=args.output_prefix,
            use_bias_search=not args.no_bias_search,
        )
        comparison_rows.append(result)
        print(
            f"{result['feature_set']}: macro_f1={result['macro_f1']:.6f} "
            f"balanced_accuracy={result['balanced_accuracy']:.6f} accuracy={result['accuracy']:.6f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False)
    comparison_path = log_dir / f"{args.output_prefix}_comparison.csv"
    comparison_json_path = log_dir / f"{args.output_prefix}_comparison.json"
    comparison.to_csv(comparison_path, index=False)
    with comparison_json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison_rows, handle, ensure_ascii=False, indent=2)
    print(f"Saved comparison: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
