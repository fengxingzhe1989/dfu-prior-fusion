#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.reliability_aware_prior_fusion import oof_stack as stack  # noqa: E402


MODEL_NAMES = (
    "resnet50",
    "efficientnetb0",
    "lpt_no_alignment",
    "lpt_no_prior_mod",
    "lpt_no_roi",
    "lpt_stats_only",
    "convnext_tiny",
    "efficientnetv2s",
)
FEATURE_SETS = tuple(
    feature_set
    for feature_set in stack.FEATURE_SETS
    if feature_set.name
    in {
        "lpt_ablation_output_ensemble",
        "lpt_ablation_ensemble",
        "strong_cls_ensemble",
        "strong_cls_prior_residual",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_prediction(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        probabilities = np.array([float(row[column]) for column in stack.PROB_COLUMNS], dtype=np.float64)
        item: dict[str, Any] = {
            "image": row["image"],
            f"{model_name}_pred_label": stack.LABEL_TO_ID[str(row["pred_label"])],
        }
        item.update(stack.probability_features(model_name, probabilities))
        for column in ("prob_infection_attr", "prob_ischaemia_attr"):
            if column in row and not pd.isna(row[column]):
                item[f"{model_name}_{column}"] = float(row[column])
        rows.append(item)
    result = pd.DataFrame(rows)
    if result["image"].duplicated().any():
        raise ValueError(f"Duplicate predictions in {path}")
    return result


def add_classifier_disagreement(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    res_pred = frame["resnet50_pred_label"].to_numpy(dtype=int)
    eff_pred = frame["efficientnetb0_pred_label"].to_numpy(dtype=int)
    frame["cls_models_agree"] = (res_pred == eff_pred).astype(float)
    frame["cls_models_disagree"] = 1.0 - frame["cls_models_agree"]
    frame["cls_confidence_mean"] = 0.5 * (frame["resnet50_confidence"] + frame["efficientnetb0_confidence"])
    frame["cls_entropy_mean"] = 0.5 * (frame["resnet50_entropy"] + frame["efficientnetb0_entropy"])
    differences = [
        np.abs(frame[f"resnet50_prob_{name}"] - frame[f"efficientnetb0_prob_{name}"])
        for name in stack.CLASS_NAMES
    ]
    frame["cls_prob_l1_disagreement"] = np.sum(np.vstack(differences), axis=0)
    return frame


def build_frame(
    manifest: pd.DataFrame,
    prior_meta: pd.DataFrame,
    paths: dict[str, list[Path]],
) -> pd.DataFrame:
    frame = manifest.merge(stack.add_prior_derived_features(prior_meta), on="image", how="left", validate="one_to_one")
    for model_name in MODEL_NAMES:
        predictions = pd.concat([load_prediction(path, model_name) for path in paths[model_name]], ignore_index=True)
        if predictions["image"].duplicated().any():
            raise ValueError(f"Duplicate combined predictions for {model_name}")
        frame = frame.merge(predictions, on="image", how="left", validate="one_to_one")
    if frame.isnull().any().any():
        missing = frame.columns[frame.isnull().any()].tolist()
        raise ValueError(f"Missing stack inputs: {missing}")
    return add_classifier_disagreement(frame)


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    classification_root = (
        args.classification_root if args.classification_root.is_absolute() else repo_root / args.classification_root
    )
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    job_dir = output_root / f"seed_{args.seed}" / f"outer_{args.outer}"
    complete_path = job_dir / "complete.json"
    if complete_path.exists() and not args.force:
        print(f"SKIP completed {job_dir}", flush=True)
        return json.loads(complete_path.read_text(encoding="utf-8"))
    job_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(repo_root / "data_processed/associate_editor_revision_v1/nested_grouped_assignments.csv")
    assignments = assignments.loc[assignments["outer_fold"].eq(args.outer)].copy()
    prior_meta = stack.read_prior_meta(repo_root / "data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv")
    train_manifest = assignments.loc[assignments["outer_role"].eq("train"), ["image", "label", "label_name", "inner_fold"]].rename(
        columns={"inner_fold": "fold"}
    )
    test_manifest = assignments.loc[assignments["outer_role"].eq("test"), ["image", "label", "label_name"]].copy()
    test_manifest["fold"] = -1

    seed_root = classification_root / f"seed_{args.seed}" / f"outer_{args.outer}"
    train_paths = {
        model: [seed_root / model / f"inner_{inner}" / "predictions.csv" for inner in range(3)]
        for model in MODEL_NAMES
    }
    test_paths = {model: [seed_root / model / "refit" / "predictions.csv"] for model in MODEL_NAMES}
    missing = [str(path) for collection in (train_paths, test_paths) for paths in collection.values() for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} stack inputs; first: {missing[0]}")

    train_frame = build_frame(train_manifest, prior_meta, train_paths)
    test_frame = build_frame(test_manifest, prior_meta, test_paths)
    if len(train_frame) + len(test_frame) != len(assignments):
        raise RuntimeError("Stack frame size mismatch.")

    payload: dict[str, Any] = {"status": "complete", "seed": args.seed, "outer_fold": args.outer, "feature_sets": {}}
    for feature_set in FEATURE_SETS:
        train_work = train_frame.copy()
        test_work = test_frame.copy()
        feature_columns = stack.feature_columns_for(train_work, feature_set)
        test_columns = stack.feature_columns_for(test_work, feature_set)
        if feature_columns != test_columns:
            raise RuntimeError(f"Feature mismatch for {feature_set.name}")
        train_mask = np.ones(len(train_work), dtype=bool)
        best_params, inner_probabilities, inner_score = stack.inner_select_params(
            train_work, train_mask, feature_columns, (0, 1, 2)
        )
        biases, bias_score = stack.optimize_biases(inner_probabilities, train_work["label"].to_numpy(dtype=int))
        estimator = stack.build_estimator(best_params["C"], best_params["class_weight"])
        estimator.fit(train_work[feature_columns].to_numpy(dtype=np.float64), train_work["label"].to_numpy(dtype=int))
        probabilities = stack.predict_proba_aligned(estimator, test_work[feature_columns].to_numpy(dtype=np.float64))
        predicted = stack.apply_biases(probabilities, biases)
        rows: list[dict[str, Any]] = []
        for index, image in enumerate(test_work["image"]):
            rows.append(
                {
                    "image": image,
                    "true_label": str(test_work.iloc[index]["label_name"]),
                    "pred_label": stack.CLASS_NAMES[int(predicted[index])],
                    **{f"prob_{name}": float(probabilities[index, class_index]) for class_index, name in enumerate(stack.CLASS_NAMES)},
                    "base_seed": args.seed,
                    "outer_fold": args.outer,
                    "model": feature_set.name,
                }
            )
        prediction_path = job_dir / f"{feature_set.name}_predictions.csv"
        temporary = prediction_path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(prediction_path)
        payload["feature_sets"][feature_set.name] = {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "selected_C": float(best_params["C"]),
            "selected_class_weight": best_params["class_weight"],
            "inner_macro_f1": float(inner_score),
            "bias_train_macro_f1": float(bias_score),
            "biases": {name: float(biases[index]) for index, name in enumerate(stack.CLASS_NAMES)},
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256(prediction_path),
        }
    write_json(complete_path, payload)
    print(f"COMPLETE {job_dir}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the four prespecified nested stackers for one seed and outer fold.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--classification-root", type=Path, default=Path("outputs/associate_editor_revision_v1/classification")
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1/stacks"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--force", action="store_true")
    run_job(parser.parse_args())


if __name__ == "__main__":
    main()
