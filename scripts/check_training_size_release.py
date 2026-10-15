#!/usr/bin/env python3
"""Data-free integrity and arithmetic checks of the v3 training-size release.

This checks the released aggregate counts and the saved bootstrap draws. It does
not regenerate draws from raw OOF predictions, verify image/group membership,
retrain models, or independently certify the original execution markers.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "paper_assets/training_size_sensitivity_v3"
SEEDS = tuple(range(20260925, 20260930))
SIZES = (25, 50, 75, 100)
MODELS = ("lpt_full", "lpt_neutral")
FOLDS = tuple(range(5))
LABELS = ("none", "infection", "ischaemia", "both")
SUPPORT = dict(zip(LABELS, (2552, 2555, 227, 621)))
METRICS = ("macro_f1", "ischaemia_precision", "ischaemia_recall", "ischaemia_f1")
SEED_KEY = ["nominal_percent", "model", "seed"]
FOLD_KEY = SEED_KEY + ["outer"]
COUNT_COLUMNS = ["images", "groups", *LABELS]
INTERVAL_TYPE = "exploratory pointwise paired crossed seed x similarity-group percentile"
SOURCE_NAMES = ("PROTOCOL.md", "FIGURE_CONTRACT.md", "common.py", "run.py",
                "summarize.py", "render_curves.py", "test_design.py")
SUMMARY_NAMES = ("README.md", "seed_metrics.csv", "seed_class_metrics.csv",
                 "fold_seed_metrics.csv", "fold_seed_class_metrics.csv",
                 "subset_counts.csv", "cell_summary.csv", "paired_contrasts.csv",
                 "bootstrap_replicates.csv")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context: str) -> None:
    # CSVs store 12 significant digits; tolerate only rounding-level differences.
    require(np.allclose(actual, expected, rtol=1e-8, atol=2e-10, equal_nan=True),
            f"Numeric mismatch: {context}")


def coverage(frame: pd.DataFrame, keys: list[str], expected, context: str) -> None:
    require(not frame[keys].isna().any().any(), f"Missing identity: {context}")
    require(not frame.duplicated(keys).any(), f"Duplicate identities: {context}")
    require(set(frame[keys].itertuples(index=False, name=None)) == set(expected),
            f"Incomplete or unexpected coverage: {context}")


def nonnegative_integers(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    values = frame[columns].to_numpy(dtype=float)
    require(np.isfinite(values).all() and (values >= 0).all()
            and (values == np.floor(values)).all(), f"Invalid counts: {context}")


def verify_registry() -> list[dict[str, str]]:
    registry = ROOT / "paper_assets/release_audit/training_size_asset_registry.csv"
    with registry.open(newline="") as stream:
        reader = csv.DictReader(stream)
        require({"path", "sha256", "original_sha256", "handling"} <= set(reader.fieldnames or []),
                "Missing registry columns")
        records = list(reader)
    require(len({r["path"] for r in records}) == len(records), "Duplicate registry paths")
    required = {f"experiments/training_size_sensitivity_v3/{n}" for n in SOURCE_NAMES}
    required.add("experiments/dfuc2021_cls_baseline/summarize_strong_results.py")
    required |= {f"paper_assets/training_size_sensitivity_v3/summary/{n}" for n in SUMMARY_NAMES}
    require(required <= {r["path"] for r in records}, "Registry omits required source/summary assets")
    for row in records:
        relative = Path(row["path"])
        path = (ROOT / relative).resolve()
        require(not relative.is_absolute() and ".." not in relative.parts
                and path.is_relative_to(ROOT.resolve()), f"Unsafe registry path: {relative}")
        require(path.is_file(), f"Missing released asset: {relative}")
        require(sha(path) == row["sha256"], f"Hash mismatch: {relative}")
        require(bool(row["handling"]), f"Missing handling statement: {relative}")
        if row["path"] in required:
            require(row["handling"] == "byte_identical", f"Frozen asset changed: {relative}")
        if row["handling"] == "byte_identical":
            require(row["sha256"] == row["original_sha256"], f"Original hash mismatch: {relative}")
        if row["path"] == "paper_assets/training_size_sensitivity_v3/provenance_summary.json":
            projection = json.loads(path.read_text())
            require(row["handling"] == "public_provenance_projection_not_execution_seal",
                    "Registry mislabels the public provenance projection")
            require(row["original_sha256"] == projection["original_seal_sha256"],
                    "Registry and public projection disagree on the original seal hash")
        if row["path"] in required and path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(relative))
    return records


def verify_counts(counts: pd.DataFrame) -> pd.DataFrame:
    keys = ["nominal_percent", "seed", "outer", "role"]
    coverage(counts, keys, itertools.product(SIZES, SEEDS, FOLDS, ("fit", "stop", "refit", "test")),
             "subset counts")
    nonnegative_integers(counts, COUNT_COLUMNS, "subset counts")
    require((counts.groups > 0).all() and (counts.groups <= counts.images).all(), "Invalid group counts")
    close(counts[list(LABELS)].sum(axis=1), counts.images, "class counts sum to images")
    require((counts.loc[counts.role.isin(["fit", "stop"]), list(LABELS)] > 0).all().all(),
            "Missing class in fit/stopping subset")
    indexed = counts.set_index(keys).sort_index()
    for size, seed, outer in itertools.product(SIZES, SEEDS, FOLDS):
        rows = indexed.loc[(size, seed, outer)]
        close(rows.loc["fit", COUNT_COLUMNS] + rows.loc["stop", COUNT_COLUMNS],
              rows.loc["refit", COUNT_COLUMNS], f"fit + stop = refit: {size}/{seed}/{outer}")
        full = indexed.loc[(100, seed, outer, "refit")]
        close(rows.loc["refit", "fraction_of_original_outer_training_images"],
              rows.loc["refit", "images"] / full.images, "actual budget fraction")
        require((rows.loc["refit", COUNT_COLUMNS] <= full[COUNT_COLUMNS]).all(),
                "Subset exceeds full budget")
    require(counts.loc[~counts.role.eq("refit"), "fraction_of_original_outer_training_images"].isna().all(),
            "Unexpected fraction outside refit count rows")
    tests = counts.loc[counts.role.eq("test")]
    require(tests.groupby("outer")[COUNT_COLUMNS].nunique().eq(1).all().all(),
            "Outer evaluation counts vary across budgets or trajectories")
    test_by_fold = tests.drop_duplicates("outer").set_index("outer").sort_index()
    close(test_by_fold.images.sum(), 5955, "pooled evaluation images")
    close(test_by_fold.groups.sum(), 5670, "pooled evaluation groups")
    for label, n in SUPPORT.items():
        close(test_by_fold[label].sum(), n, f"pooled support: {label}")
    for seed, outer in itertools.product(SEEDS, FOLDS):
        refits = counts.loc[counts.seed.eq(seed) & counts.outer.eq(outer) & counts.role.eq("refit")]
        refits = refits.sort_values("nominal_percent")
        require((np.diff(refits[COUNT_COLUMNS].to_numpy(), axis=0) >= 0).all(),
                "Non-monotone nested-budget aggregate counts")
        full = indexed.loc[(100, seed, outer, "refit")]
        close(full.images + test_by_fold.loc[outer, "images"], 5955, "full train + test images")
        close(full.groups + test_by_fold.loc[outer, "groups"], 5670, "full train + test groups")
        for label, n in SUPPORT.items():
            close(full[label] + test_by_fold.loc[outer, label], n, "full train + test class count")
    return test_by_fold


def verify_class_metrics(classes: pd.DataFrame, metrics: pd.DataFrame, keys: list[str], expected) -> None:
    identities = list(expected)
    coverage(metrics, keys, identities, "classification cells")
    coverage(classes, keys + ["label"], (k + (label,) for k in identities for label in LABELS),
             "class-specific cells")
    nonnegative_integers(classes, ["support", "tp", "fp", "fn"], "class indicators")
    nonnegative_integers(metrics, ["n"], "evaluation n")
    close(classes.tp + classes.fn, classes.support, "TP + FN = support")
    tp = classes.tp.to_numpy(dtype=float)
    fp = classes.fp.to_numpy(dtype=float)
    fn = classes.fn.to_numpy(dtype=float)
    for name, numerator, denominator, fallback in [
        ("precision", tp, tp + fp, np.nan),
        ("recall", tp, tp + fn, np.nan),
        ("f1", 2 * tp, 2 * tp + fp + fn, 0.0),
    ]:
        expected_values = np.divide(numerator, denominator, out=np.full_like(tp, fallback), where=denominator > 0)
        close(classes[name], expected_values, f"class {name}")
    m = metrics.set_index(keys).sort_index()
    grouped = classes.groupby(keys)
    close(grouped.support.sum().reindex(m.index), m.n, "class supports and cell n")
    close(grouped.fp.sum().reindex(m.index), grouped.fn.sum().reindex(m.index), "multiclass FP = FN")
    close(grouped.tp.sum().reindex(m.index) / m.n, m.accuracy, "accuracy from TP")
    close(grouped.f1.mean().reindex(m.index), m.macro_f1, "Macro-F1 from four class F1 values")
    minority = classes.loc[classes.label.eq("ischaemia")].set_index(keys).reindex(m.index)
    for metric in ("precision", "recall", "f1"):
        close(minority[metric], m[f"ischaemia_{metric}"], f"ischaemia {metric}")


def interval(values: np.ndarray, row, context: str) -> None:
    finite = values[np.isfinite(values)]
    require(int(row.estimable_replicates) == len(finite)
            and int(row.undefined_replicates) == len(values) - len(finite),
            f"Incorrect estimable/undefined counts: {context}")
    require(row.interval_type == INTERVAL_TYPE, f"Changed interval type: {context}")
    bounds = np.percentile(finite, [2.5, 97.5]) if len(finite) else [np.nan, np.nan]
    close([row.ci_low, row.ci_high], bounds, f"stored-draw percentile interval: {context}")


def verify_tables(summary: Path) -> dict:
    tables = {name.removesuffix(".csv"): pd.read_csv(summary / name)
              for name in SUMMARY_NAMES if name.endswith(".csv")}
    test_by_fold = verify_counts(tables["subset_counts"])
    seeds, folds = tables["seed_metrics"], tables["fold_seed_metrics"]
    seed_classes, fold_classes = tables["seed_class_metrics"], tables["fold_seed_class_metrics"]
    verify_class_metrics(seed_classes, seeds, SEED_KEY, itertools.product(SIZES, MODELS, SEEDS))
    verify_class_metrics(fold_classes, folds, FOLD_KEY, itertools.product(SIZES, MODELS, SEEDS, FOLDS))
    require(seeds.n.eq(5955).all(), "Pooled evaluation n is not 5955")
    close(folds.n, folds.outer.map(test_by_fold.images), "fixed fold evaluation n")
    for label, support in SUPPORT.items():
        require(seed_classes.loc[seed_classes.label.eq(label), "support"].eq(support).all(),
                f"Changed pooled class support: {label}")
        rows = fold_classes.loc[fold_classes.label.eq(label)]
        close(rows.support, rows.outer.map(test_by_fold[label]), f"fold class support: {label}")
    class_key = SEED_KEY + ["label"]
    pooled = fold_classes.groupby(class_key)[["support", "tp", "fp", "fn"]].sum().sort_index()
    observed = seed_classes.set_index(class_key).reindex(pooled.index)
    close(pooled, observed[pooled.columns], "fold indicators sum to pooled seed indicators")
    close(folds.groupby(SEED_KEY).n.sum().sort_index(), seeds.set_index(SEED_KEY).sort_index().n,
          "fold evaluation n sums to pooled seed n")

    cells, contrasts, draws = tables["cell_summary"], tables["paired_contrasts"], tables["bootstrap_replicates"]
    coverage(cells, ["nominal_percent", "model", "metric"], itertools.product(SIZES, MODELS, METRICS),
             "cell summaries")
    coverage(draws, ["nominal_percent", "model", "replicate"], itertools.product(SIZES, MODELS, range(5000)),
             "stored bootstrap draws")
    require(not np.isinf(draws[list(METRICS)].to_numpy()).any(), "Infinite stored bootstrap metric")
    require(((draws[list(METRICS)] >= 0) & (draws[list(METRICS)] <= 1)
             | draws[list(METRICS)].isna()).all().all(), "Stored score outside [0, 1]")
    seed_cells = {(size, model): g.set_index("seed").reindex(SEEDS)
                  for (size, model), g in seeds.groupby(["nominal_percent", "model"])}
    draw_cells = {(size, model): g.set_index("replicate").reindex(range(5000))
                  for (size, model), g in draws.groupby(["nominal_percent", "model"])}
    for row in cells.itertuples():
        key = (row.nominal_percent, row.model)
        values = seed_cells[key][row.metric].to_numpy()
        close(np.mean(values), row.estimate, f"five-trajectory mean: {key}/{row.metric}")
        close(np.std(values, ddof=1), row.sample_sd_across_five_trajectories, f"sample SD: {key}/{row.metric}")
        require(row.estimable_trajectories == np.isfinite(values).sum(), "Incorrect estimable trajectory count")
        interval(draw_cells[key][row.metric].to_numpy(), row, f"{key}/{row.metric}")
    pairs = {f"{model}_{hi}_minus_{lo}": ((hi, model), (lo, model))
             for model in MODELS for hi, lo in ((50, 25), (75, 50), (100, 75), (100, 50), (100, 25))}
    pairs.update({f"full_minus_neutral_{size}": ((size, "lpt_full"), (size, "lpt_neutral")) for size in SIZES})
    coverage(contrasts, ["contrast", "metric"], itertools.product(pairs, METRICS), "paired contrasts")
    for row in contrasts.itertuples():
        a, b = pairs[row.contrast]
        close(np.mean(seed_cells[a][row.metric].to_numpy() - seed_cells[b][row.metric].to_numpy()),
              row.estimate, f"paired trajectory contrast: {row.contrast}/{row.metric}")
        delta = draw_cells[a][row.metric].to_numpy() - draw_cells[b][row.metric].to_numpy()
        interval(delta, row, f"{row.contrast}/{row.metric}")
    return dict(seed_cells=len(seeds), fold_cells=len(folds), subset_count_rows=len(tables["subset_counts"]),
                class_seed_rows=len(seed_classes), class_fold_rows=len(fold_classes),
                cell_summary_rows=len(cells), paired_contrast_rows=len(contrasts),
                bootstrap_replicates_per_cell=5000, stored_bootstrap_rows=len(draws))


def verify_metadata() -> None:
    marker = json.loads((ASSET / "summary/summary_complete.json").read_text())
    require(marker["status"] == "complete", "Summary marker incomplete")
    for key, expected in {"seed_cells": 40, "fold_cells": 200, "images_per_seed_model_size": 5955,
                          "groups": 5670, "bootstrap_replicates": 5000}.items():
        require(marker[key] == expected, f"Summary marker mismatch: {key}")
    for key, expected in {"seeds": SEEDS, "models": MODELS, "nominal_sizes": SIZES}.items():
        require(set(marker[key]) == set(expected), f"Summary marker mismatch: {key}")
    for name in SUMMARY_NAMES:
        original_path = f"outputs/training_size_sensitivity_v3/summary/{name}"
        require(marker["outputs"][original_path] == sha(ASSET / "summary" / name),
                f"Original completion marker disagrees with released output: {name}")
    analysis = json.loads((ASSET / "analysis_complete.json").read_text())
    preflight = json.loads((ASSET / "preflight_complete.json").read_text())
    smoke = json.loads((ASSET / "smoke_complete.json").read_text())
    for value in (analysis, preflight, smoke):
        require(value["status"] == "complete", "Original execution marker incomplete")
        require(value.get("seal_sha256", value.get("sealed_inputs_sha256")) == marker["seal_sha256"],
                "Original execution markers reference different seals")
    for value in (analysis, preflight):
        require(value["new_fits"] == 400 and value["reused_fits"] == 0, "Fresh-fit inventory mismatch")
    require(analysis["images_per_cell"] == 5955 and analysis["seed_model_size_cells"] == 40,
            "Analysis marker coverage mismatch")
    require(analysis["summary_complete_sha256"] == sha(ASSET / "summary/summary_complete.json"),
            "Analysis marker references another summary marker")
    require(preflight["budget_fold_trajectory_cells"] == 100 and preflight["subset_rows"] == 400,
            "Preflight marker coverage mismatch")
    require(smoke["fits"] == 4 and smoke["excluded_from_results"] is True, "Smoke exclusion mismatch")
    projection = json.loads((ASSET / "provenance_summary.json").read_text())
    require(projection["artifact_type"] == "public_provenance_projection_not_execution_seal",
            "Public provenance projection mislabelled as an execution seal")
    require(projection["original_seal_sha256"] == marker["seal_sha256"],
            "Public projection references another original seal")
    require(projection["original_summary_complete_sha256"] == sha(ASSET / "summary/summary_complete.json"),
            "Public projection references another original summary")
    require(projection["new_fits"] == 400 and projection["reused_fits"] == 0
            and projection["previous_attempts_excluded"] is True, "Projection fit inventory mismatch")
    for key, expected in {"seeds": SEEDS, "models": MODELS, "nominal_sizes": SIZES}.items():
        require(set(projection[key]) == set(expected), f"Projection coverage mismatch: {key}")
    copied_sources = [f"experiments/training_size_sensitivity_v3/{n}" for n in SOURCE_NAMES]
    copied_sources.append("experiments/dfuc2021_cls_baseline/summarize_strong_results.py")
    for name in copied_sources:
        require(projection["source_hashes"][name] == sha(ROOT / name),
                f"Copied source differs from original-seal projection: {name}")
    require(projection["assignment_hashes"]["data_processed/training_size_sensitivity_v3/subset_counts.csv"]
            == sha(ASSET / "summary/subset_counts.csv"), "Original subset-count hash mismatch")
    # Original marker input paths and figure hashes are not dereferenced: those
    # underlying artifacts are not included in this minimal aggregate release.


def main() -> None:
    records = verify_registry()
    checks = verify_tables(ASSET / "summary")
    verify_metadata()
    print(json.dumps(dict(
        status="passed", release_assets=len(records), frozen_source_files=8,
        models=list(MODELS), seeds=list(SEEDS), nominal_budgets=list(SIZES), **checks,
        checks="registry hashes; released count/metric arithmetic; saved-draw percentile intervals",
        raw_oof_bootstrap_regenerated=False, training_or_inference_rerun=False,
        source_group_membership_verified=False, original_execution_independently_verified=False,
    ), indent=2))


if __name__ == "__main__":
    main()
