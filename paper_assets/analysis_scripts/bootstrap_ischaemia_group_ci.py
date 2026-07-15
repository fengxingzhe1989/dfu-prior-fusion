#!/usr/bin/env python3
"""Compute similarity-group bootstrap CIs for ischaemia-only metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


CLASS_NAME = "ischaemia"


@dataclass(frozen=True)
class ModelInput:
    name: str
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_predictions(model: ModelInput, manifest: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(model.path)
    required = {"image", "pred_label", "prob_ischaemia"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{model.path} is missing columns: {sorted(missing)}")
    if frame["image"].duplicated().any():
        raise ValueError(f"Duplicate image names in {model.path}")

    true_col = "label_name" if "label_name" in frame.columns else "true_label"
    if true_col not in frame.columns:
        raise ValueError(f"No true-label column in {model.path}")

    selected = frame[["image", true_col, "pred_label", "prob_ischaemia"]].rename(
        columns={true_col: "prediction_true_label"}
    )
    merged = manifest.merge(selected, on="image", how="left", validate="one_to_one")
    if merged[["prediction_true_label", "pred_label", "prob_ischaemia"]].isna().any().any():
        raise ValueError(f"Missing aligned predictions in {model.path}")

    expected = merged["label_name"].astype(str).str.lower()
    observed = merged["prediction_true_label"].astype(str).str.lower()
    if not expected.equals(observed):
        mismatch = merged.loc[expected != observed, ["image", "label_name", "prediction_true_label"]]
        raise ValueError(f"True-label mismatch in {model.path}:\n{mismatch.head()}")

    return pd.DataFrame(
        {
            "image": merged["image"].astype(str),
            "similarity_group": merged["similarity_group"].astype(str),
            "y_true": (expected == CLASS_NAME).astype(np.int8),
            "y_pred": (merged["pred_label"].astype(str).str.lower() == CLASS_NAME).astype(np.int8),
            "score": merged["prob_ischaemia"].astype(float),
        }
    )


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        sample_weight=sample_weight,
        zero_division=0,
    )
    auprc = average_precision_score(y_true, score, sample_weight=sample_weight)
    return float(precision), float(recall), float(f1), float(auprc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resnet50", type=Path, required=True)
    parser.add_argument("--full-lpt", type=Path, required=True)
    parser.add_argument("--moderate-output", type=Path, required=True)
    parser.add_argument("--ra-lpt", type=Path, required=True)
    parser.add_argument("--strong-rgb", type=Path, required=True)
    parser.add_argument("--strong-rgb-prior", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--output-resamples", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [
        ModelInput("ResNet50", args.resnet50),
        ModelInput("Full LPT", args.full_lpt),
        ModelInput("Moderate output-only LPT stack", args.moderate_output),
        ModelInput("RA-LPT", args.ra_lpt),
        ModelInput("Strong RGB stack", args.strong_rgb),
        ModelInput("Strong RGB + explicit prior features", args.strong_rgb_prior),
    ]
    for path in [args.manifest, *(model.path for model in models)]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(args.manifest)
    required_manifest = {"image", "label_name", "similarity_group"}
    missing = required_manifest.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest["image"].duplicated().any():
        raise ValueError("Manifest contains duplicate image names")
    manifest = manifest[["image", "label_name", "similarity_group"]].sort_values("image").reset_index(drop=True)

    aligned = {model.name: normalize_predictions(model, manifest) for model in models}
    reference = aligned[models[0].name]
    for model in models[1:]:
        current = aligned[model.name]
        if not reference[["image", "similarity_group", "y_true"]].equals(
            current[["image", "similarity_group", "y_true"]]
        ):
            raise ValueError(f"Alignment mismatch for {model.name}")

    group_codes, group_names = pd.factorize(reference["similarity_group"], sort=True)
    n_groups = len(group_names)
    if len(reference) != 5955 or n_groups != 5670:
        raise ValueError(f"Unexpected cohort dimensions: {len(reference)} images, {n_groups} groups")

    rng = np.random.default_rng(args.seed)
    observed: dict[str, tuple[float, float, float, float]] = {}
    distributions = {
        model.name: np.empty((args.n_bootstrap, 4), dtype=np.float64) for model in models
    }
    for model in models:
        frame = aligned[model.name]
        observed[model.name] = metrics(
            frame["y_true"].to_numpy(),
            frame["y_pred"].to_numpy(),
            frame["score"].to_numpy(),
        )

    for iteration in range(args.n_bootstrap):
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        group_counts = np.bincount(sampled_groups, minlength=n_groups)
        sample_weight = group_counts[group_codes].astype(np.float64)
        for model in models:
            frame = aligned[model.name]
            distributions[model.name][iteration] = metrics(
                frame["y_true"].to_numpy(),
                frame["y_pred"].to_numpy(),
                frame["score"].to_numpy(),
                sample_weight,
            )

    metric_names = ["precision", "recall", "f1", "auprc"]
    summary_rows: list[dict[str, object]] = []
    resample_frames: list[pd.DataFrame] = []
    for model in models:
        values = distributions[model.name]
        row: dict[str, object] = {
            "method": model.name,
            "class": CLASS_NAME,
            "images": len(reference),
            "similarity_groups": n_groups,
            "positive_images": int(reference["y_true"].sum()),
            "bootstrap_resamples": args.n_bootstrap,
            "seed": args.seed,
        }
        for column, metric_name in enumerate(metric_names):
            low, high = np.quantile(values[:, column], [0.025, 0.975])
            row[metric_name] = observed[model.name][column]
            row[f"{metric_name}_ci_low"] = float(low)
            row[f"{metric_name}_ci_high"] = float(high)
        summary_rows.append(row)

        resample = pd.DataFrame(values, columns=metric_names)
        resample.insert(0, "bootstrap_iteration", np.arange(1, args.n_bootstrap + 1))
        resample.insert(0, "method", model.name)
        resample_frames.append(resample)

    summary = pd.DataFrame(summary_rows)
    resamples = pd.concat(resample_frames, ignore_index=True)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_resamples.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary, index=False)
    resamples.to_csv(args.output_resamples, index=False)

    metadata = {
        "analysis": "similarity-group bootstrap confidence intervals for ischaemia-only metrics",
        "class": CLASS_NAME,
        "images": len(reference),
        "positive_images": int(reference["y_true"].sum()),
        "similarity_groups": n_groups,
        "bootstrap_resamples": args.n_bootstrap,
        "seed": args.seed,
        "interval": "2.5th and 97.5th percentiles",
        "resampling_unit": "reconstructed similarity group",
        "paired_draws_across_models": True,
        "inputs": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            **{
                model.name: {"path": str(model.path), "sha256": sha256(model.path)}
                for model in models
            },
        },
    }
    args.output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
