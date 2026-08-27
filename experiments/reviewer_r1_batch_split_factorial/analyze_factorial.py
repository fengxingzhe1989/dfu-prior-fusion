#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


EXPERIMENT_VERSION = "reviewer_r1_batch_split_factorial_v1"
CLASS_NAMES = ("none", "infection", "ischaemia", "both")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and analyse the Reviewer 1 model x split x batch-size factorial "
            "using paired similarity-group bootstrap resampling."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / EXPERIMENT_VERSION,
    )
    parser.add_argument(
        "--grouped-manifest",
        type=Path,
        default=Path(
            "data_processed/dfuc2021_cls_v1/similarity_grouped_splits/"
            "dfuc2021_cls_cv5_similarity_grouped_manifest.csv"
        ),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def method_key(model: str, split: str, batch_size: int) -> str:
    return f"{model}_{split}_bs{batch_size}"


def macro_f1_from_confusion(confusions: np.ndarray) -> np.ndarray:
    """Return Macro-F1 for an array ending in a square confusion matrix."""
    true_positives = np.diagonal(confusions, axis1=-2, axis2=-1)
    false_positives = confusions.sum(axis=-2) - true_positives
    false_negatives = confusions.sum(axis=-1) - true_positives
    denominator = 2.0 * true_positives + false_positives + false_negatives
    per_class = np.divide(
        2.0 * true_positives,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    )
    return per_class.mean(axis=-1)


def load_cells(
    output_root: Path,
    root: Path,
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config_paths = sorted((output_root / "resolved_configs").glob("*.yaml"))
    if len(config_paths) != 8:
        raise ValueError(f"Expected eight resolved configs, found {len(config_paths)}")

    cells: dict[str, dict[str, Any]] = {}
    fold_rows: list[dict[str, Any]] = []
    reference: pd.DataFrame | None = None
    prediction_hashes: list[dict[str, Any]] = []

    for config_path in config_paths:
        config = load_yaml(config_path)
        design = config["reviewer_ablation"]
        model = str(design["model"])
        split = str(design["split"])
        batch_size = int(design["batch_size"])
        key = method_key(model, split, batch_size)
        name = str(config["experiment_name"])
        log_dir = resolve(config["outputs"]["log_dir"], root)
        prediction_dir = resolve(config["outputs"]["prediction_dir"], root)

        fold_predictions: list[pd.DataFrame] = []
        fold_scores: list[float] = []
        for fold in range(5):
            metric_path = log_dir / f"{name}_fold{fold}_metrics.json"
            prediction_path = prediction_dir / config["naming"]["predictions"].format(fold=fold)
            if not metric_path.exists() or not prediction_path.exists():
                raise FileNotFoundError(
                    f"Incomplete cell {key}, fold {fold}: {metric_path}, {prediction_path}"
                )
            metric = load_json(metric_path)
            score = float(metric["best_metrics"]["macro_f1"])
            fold_scores.append(score)
            fold_rows.append(
                {
                    "method": key,
                    "model": model,
                    "split": split,
                    "batch_size": batch_size,
                    "fold": fold,
                    "macro_f1": score,
                    "best_epoch": int(metric["best_epoch"]),
                }
            )
            frame = pd.read_csv(prediction_path).assign(fold=fold)
            fold_predictions.append(frame)
            prediction_hashes.append(
                {
                    "method": key,
                    "fold": fold,
                    "path": str(prediction_path.relative_to(root)),
                    "sha256": sha256(prediction_path),
                    "rows": int(len(frame)),
                }
            )

        predictions = pd.concat(fold_predictions, ignore_index=True)
        required = {"image", "true_label", "pred_label", "fold"}
        missing_columns = sorted(required.difference(predictions.columns))
        if missing_columns:
            raise ValueError(f"Cell {key} lacks columns {missing_columns}")
        if len(predictions) != 5955:
            raise ValueError(f"Cell {key} has {len(predictions)} OOF rows, expected 5955")
        if predictions["image"].duplicated().any():
            duplicate = predictions.loc[predictions["image"].duplicated(), "image"].iloc[0]
            raise ValueError(f"Cell {key} contains duplicate OOF image {duplicate}")
        if set(predictions["true_label"].astype(str)).difference(CLASS_TO_ID):
            raise ValueError(f"Cell {key} contains an unknown true label")
        if set(predictions["pred_label"].astype(str)).difference(CLASS_TO_ID):
            raise ValueError(f"Cell {key} contains an unknown predicted label")

        aligned = predictions[["image", "true_label", "pred_label", "fold"]].copy()
        aligned["image"] = aligned["image"].astype(str)
        aligned = aligned.sort_values("image").reset_index(drop=True)
        if reference is None:
            reference = aligned[["image", "true_label"]].copy()
        else:
            if not aligned["image"].equals(reference["image"]):
                raise ValueError(f"OOF image alignment mismatch in {key}")
            if not aligned["true_label"].equals(reference["true_label"]):
                raise ValueError(f"OOF label alignment mismatch in {key}")

        cells[key] = {
            "model": model,
            "split": split,
            "batch_size": batch_size,
            "predictions": aligned,
            "fold_scores": np.asarray(fold_scores, dtype=float),
        }

    assert reference is not None
    method_names = sorted(cells)
    return method_names, cells, reference, fold_rows, prediction_hashes


def build_group_confusions(
    method_names: list[str],
    cells: dict[str, dict[str, Any]],
    reference: pd.DataFrame,
    grouped_manifest_path: Path,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    manifest = pd.read_csv(grouped_manifest_path, usecols=["image", "similarity_group"])
    manifest["image"] = manifest["image"].astype(str)
    if manifest["image"].duplicated().any():
        raise ValueError("Grouped manifest contains duplicate images")
    aligned = reference.merge(manifest, on="image", how="left", validate="one_to_one")
    if aligned["similarity_group"].isna().any():
        missing = aligned.loc[aligned["similarity_group"].isna(), "image"].iloc[0]
        raise ValueError(f"Grouped manifest lacks image {missing}")

    group_codes, group_names = pd.factorize(aligned["similarity_group"].astype(str), sort=True)
    y_true = aligned["true_label"].astype(str).map(CLASS_TO_ID).to_numpy(dtype=np.int8)
    n_groups = len(group_names)
    confusions = np.zeros(
        (n_groups, len(method_names), len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int16,
    )
    for method_index, name in enumerate(method_names):
        y_pred = (
            cells[name]["predictions"]["pred_label"]
            .astype(str)
            .map(CLASS_TO_ID)
            .to_numpy(dtype=np.int8)
        )
        np.add.at(confusions[:, method_index], (group_codes, y_true, y_pred), 1)
    return confusions, n_groups, aligned


def coefficient(method_names: list[str], terms: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(method_names), dtype=float)
    for name, value in terms.items():
        result[method_names.index(name)] = value
    return result


def contrast_specifications(method_names: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    for model in ("full_lpt", "resnet50"):
        for split in ("standard", "grouped"):
            specs.append(
                {
                    "contrast": "batch_16_minus_128",
                    "model": model,
                    "split": split,
                    "batch_size": "",
                    "coefficients": coefficient(
                        method_names,
                        {
                            method_key(model, split, 16): 1.0,
                            method_key(model, split, 128): -1.0,
                        },
                    ),
                }
            )

    for model in ("full_lpt", "resnet50"):
        for batch_size in (16, 128):
            specs.append(
                {
                    "contrast": "grouped_minus_standard",
                    "model": model,
                    "split": "",
                    "batch_size": batch_size,
                    "coefficients": coefficient(
                        method_names,
                        {
                            method_key(model, "grouped", batch_size): 1.0,
                            method_key(model, "standard", batch_size): -1.0,
                        },
                    ),
                }
            )

    for split in ("standard", "grouped"):
        for batch_size in (16, 128):
            specs.append(
                {
                    "contrast": "full_lpt_minus_resnet50",
                    "model": "",
                    "split": split,
                    "batch_size": batch_size,
                    "coefficients": coefficient(
                        method_names,
                        {
                            method_key("full_lpt", split, batch_size): 1.0,
                            method_key("resnet50", split, batch_size): -1.0,
                        },
                    ),
                }
            )

    for split in ("standard", "grouped"):
        specs.append(
            {
                "contrast": "model_by_batch_interaction",
                "model": "",
                "split": split,
                "batch_size": "",
                "coefficients": coefficient(
                    method_names,
                    {
                        method_key("full_lpt", split, 16): 1.0,
                        method_key("resnet50", split, 16): -1.0,
                        method_key("full_lpt", split, 128): -1.0,
                        method_key("resnet50", split, 128): 1.0,
                    },
                ),
            }
        )

    for batch_size in (16, 128):
        specs.append(
            {
                "contrast": "model_by_split_interaction",
                "model": "",
                "split": "",
                "batch_size": batch_size,
                "coefficients": coefficient(
                    method_names,
                    {
                        method_key("full_lpt", "grouped", batch_size): 1.0,
                        method_key("resnet50", "grouped", batch_size): -1.0,
                        method_key("full_lpt", "standard", batch_size): -1.0,
                        method_key("resnet50", "standard", batch_size): 1.0,
                    },
                ),
            }
        )

    for model in ("full_lpt", "resnet50"):
        specs.append(
            {
                "contrast": "batch_by_split_interaction",
                "model": model,
                "split": "",
                "batch_size": "",
                "coefficients": coefficient(
                    method_names,
                    {
                        method_key(model, "grouped", 16): 1.0,
                        method_key(model, "standard", 16): -1.0,
                        method_key(model, "grouped", 128): -1.0,
                        method_key(model, "standard", 128): 1.0,
                    },
                ),
            }
        )

    specs.append(
        {
            "contrast": "model_by_batch_by_split_interaction",
            "model": "",
            "split": "",
            "batch_size": "",
            "coefficients": coefficient(
                method_names,
                {
                    method_key("full_lpt", "grouped", 16): 1.0,
                    method_key("resnet50", "grouped", 16): -1.0,
                    method_key("full_lpt", "grouped", 128): -1.0,
                    method_key("resnet50", "grouped", 128): 1.0,
                    method_key("full_lpt", "standard", 16): -1.0,
                    method_key("resnet50", "standard", 16): 1.0,
                    method_key("full_lpt", "standard", 128): 1.0,
                    method_key("resnet50", "standard", 128): -1.0,
                },
            ),
        }
    )
    return specs


def bootstrap_macro_f1(
    group_confusions: np.ndarray,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_groups = group_confusions.shape[0]
    n_methods = group_confusions.shape[1]
    values = np.empty((n_iters, n_methods), dtype=float)
    for iteration in range(n_iters):
        sampled_groups = rng.integers(0, n_groups, size=n_groups)
        confusion = group_confusions[sampled_groups].sum(axis=0, dtype=np.int64)
        values[iteration] = macro_f1_from_confusion(confusion)
    return values


def make_cell_table(
    method_names: list[str],
    cells: dict[str, dict[str, Any]],
    observed: np.ndarray,
    bootstrap: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(method_names):
        cell = cells[name]
        folds = cell["fold_scores"]
        rows.append(
            {
                "method": name,
                "model": cell["model"],
                "split": cell["split"],
                "batch_size": cell["batch_size"],
                "n_oof": 5955,
                "fold_macro_f1_mean": float(folds.mean()),
                "fold_macro_f1_population_sd": float(folds.std(ddof=0)),
                "pooled_oof_macro_f1": float(observed[index]),
                "pooled_oof_macro_f1_ci95_low": float(np.quantile(bootstrap[:, index], 0.025)),
                "pooled_oof_macro_f1_ci95_high": float(np.quantile(bootstrap[:, index], 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values(["batch_size", "model", "split"]).reset_index(drop=True)


def make_contrast_table(
    method_names: list[str],
    observed: np.ndarray,
    bootstrap: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in contrast_specifications(method_names):
        coefficients = spec.pop("coefficients")
        observed_delta = float(observed @ coefficients)
        differences = bootstrap @ coefficients
        p_le_zero = (np.sum(differences <= 0.0) + 1.0) / (len(differences) + 1.0)
        p_ge_zero = (np.sum(differences >= 0.0) + 1.0) / (len(differences) + 1.0)
        rows.append(
            {
                **spec,
                "observed_delta": observed_delta,
                "ci95_low": float(np.quantile(differences, 0.025)),
                "ci95_high": float(np.quantile(differences, 0.975)),
                "two_sided_p_unadjusted": float(min(1.0, 2.0 * min(p_le_zero, p_ge_zero))),
                "bootstrap_iters": int(len(differences)),
                "bootstrap_unit": "similarity_group",
            }
        )
    return pd.DataFrame(rows)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value)

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    cells: pd.DataFrame,
    contrasts: pd.DataFrame,
    n_groups: int,
    n_iters: int,
    seed: int,
) -> None:
    def row(contrast: str, **filters: Any) -> pd.Series:
        selected = contrasts[contrasts["contrast"] == contrast]
        for key, value in filters.items():
            selected = selected[selected[key].astype(str) == str(value)]
        if len(selected) != 1:
            raise ValueError(f"Expected one row for {contrast} {filters}, found {len(selected)}")
        return selected.iloc[0]

    def estimate(item: pd.Series) -> str:
        return (
            f"{item.observed_delta:+.4f} "
            f"(95% CI {item.ci95_low:+.4f} to {item.ci95_high:+.4f})"
        )

    standard_interaction = row("model_by_batch_interaction", split="standard")
    grouped_interaction = row("model_by_batch_interaction", split="grouped")
    lpt_standard_batch = row("batch_16_minus_128", model="full_lpt", split="standard")
    lpt_grouped_batch = row("batch_16_minus_128", model="full_lpt", split="grouped")
    resnet_standard_batch = row("batch_16_minus_128", model="resnet50", split="standard")
    resnet_grouped_batch = row("batch_16_minus_128", model="resnet50", split="grouped")
    grouped_lpt_16 = row("full_lpt_minus_resnet50", split="grouped", batch_size=16)
    grouped_lpt_128 = row("full_lpt_minus_resnet50", split="grouped", batch_size=128)
    standard_lpt_16 = row("full_lpt_minus_resnet50", split="standard", batch_size=16)
    standard_lpt_128 = row("full_lpt_minus_resnet50", split="standard", batch_size=128)

    lines = [
        "# Reviewer 1 factorial analysis",
        "",
        "## Design and integrity",
        "",
        "- Complete 2 x 2 x 2 factorial: model (Full LPT/ResNet50) x split "
        "(standard/similarity-grouped) x batch size (16/128).",
        "- Five folds per cell (40 fitted models); every cell contains 5,955 unique OOF predictions.",
        f"- Paired bootstrap: {n_iters:,} resamples of {n_groups:,} reconstructed similarity groups "
        f"with replacement (seed {seed}).",
        "- Intervals condition on the fitted OOF predictions and do not include optimisation-seed "
        "or retraining uncertainty.",
        "- The fixed-epoch design isolates the practical batch-size training regime, but batch size "
        "also changes the number of optimiser updates per epoch budget; it does not identify a pure "
        "gradient-noise mechanism.",
        "",
        "## Main interpretation",
        "",
        "The ranking reversal was reproduced by batch size within both split schemes. At batch size "
        f"128, Full LPT was below ResNet50 in standard ({estimate(standard_lpt_128)}) and grouped "
        f"({estimate(grouped_lpt_128)}) CV. At batch size 16, Full LPT was above ResNet50 in standard "
        f"({estimate(standard_lpt_16)}) and grouped ({estimate(grouped_lpt_16)}) CV.",
        "",
        f"The corresponding model-by-batch interaction was {estimate(standard_interaction)} in standard "
        f"CV and {estimate(grouped_interaction)} in grouped CV. Batch size 16 versus 128 changed Full "
        f"LPT by {estimate(lpt_standard_batch)} (standard) and {estimate(lpt_grouped_batch)} (grouped), "
        f"whereas it changed ResNet50 by {estimate(resnet_standard_batch)} (standard) and "
        f"{estimate(resnet_grouped_batch)} (grouped).",
        "",
        "Grouping lowered pooled OOF Macro-F1 for both models at both batch sizes. The controlled "
        "factorial therefore does not support similarity grouping as the cause of the ranking reversal; "
        "the practical batch-size training regime is the dominant measured explanation in these runs.",
        "",
        "## Cell metrics",
        "",
        frame_to_markdown(cells),
        "",
        "## Contrast metrics",
        "",
        frame_to_markdown(contrasts),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_iters <= 0:
        raise ValueError("--bootstrap-iters must be positive")
    root = repo_root()
    output_root = resolve(args.output_root, root)
    grouped_manifest_path = resolve(args.grouped_manifest, root)
    run_status = load_json(output_root / "run_status.json")
    if run_status.get("state") != "completed":
        raise ValueError(f"Factorial run is not complete: {run_status.get('state')}")

    method_names, cells, reference, fold_rows, prediction_hashes = load_cells(output_root, root)
    group_confusions, n_groups, aligned_manifest = build_group_confusions(
        method_names,
        cells,
        reference,
        grouped_manifest_path,
    )
    observed = macro_f1_from_confusion(group_confusions.sum(axis=0, dtype=np.int64))
    bootstrap = bootstrap_macro_f1(group_confusions, args.bootstrap_iters, args.seed)
    cell_table = make_cell_table(method_names, cells, observed, bootstrap)
    contrast_table = make_contrast_table(method_names, observed, bootstrap)

    cell_path = output_root / "factorial_cell_metrics_verified.csv"
    fold_path = output_root / "factorial_fold_metrics_verified.csv"
    contrast_path = output_root / "factorial_group_bootstrap_contrasts.csv"
    report_path = output_root / "factorial_analysis_report.md"
    integrity_path = output_root / "factorial_integrity_checks.json"
    cell_table.to_csv(cell_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(fold_rows).sort_values(["method", "fold"]).to_csv(
        fold_path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    contrast_table.to_csv(contrast_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_report(
        report_path,
        cell_table,
        contrast_table,
        n_groups,
        args.bootstrap_iters,
        args.seed,
    )
    integrity = {
        "experiment_version": EXPERIMENT_VERSION,
        "run_state": run_status.get("state"),
        "resolved_config_count": 8,
        "completed_fold_count": len(fold_rows),
        "prediction_file_count": len(prediction_hashes),
        "methods": method_names,
        "oof_images_per_method": 5955,
        "aligned_unique_images": int(reference["image"].nunique()),
        "similarity_groups": int(n_groups),
        "manifest_rows": int(len(aligned_manifest)),
        "bootstrap_iters": int(args.bootstrap_iters),
        "bootstrap_seed": int(args.seed),
        "prediction_files": prediction_hashes,
        "analysis_outputs": {
            str(path.relative_to(root)): sha256(path)
            for path in (cell_path, fold_path, contrast_path, report_path)
        },
    }
    with integrity_path.open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2)

    print(cell_table.to_string(index=False))
    print("\nKey interactions")
    print(
        contrast_table[
            contrast_table["contrast"].str.contains("interaction")
        ].to_string(index=False)
    )
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
