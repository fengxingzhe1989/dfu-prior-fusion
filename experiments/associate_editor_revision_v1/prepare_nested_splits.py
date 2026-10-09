#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_folds(frame: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    assignments = pd.Series(-1, index=frame.index, dtype="int64")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_pos) in enumerate(
        splitter.split(frame, frame["label"], groups=frame["similarity_group"])
    ):
        assignments.loc[frame.index[val_pos]] = fold
    if (assignments < 0).any():
        raise RuntimeError("Fold assignment was incomplete.")
    return assignments


def assert_disjoint_groups(frame: pd.DataFrame, left: pd.Series, right: pd.Series, name: str) -> None:
    overlap = set(frame.loc[left, "similarity_group"]) & set(frame.loc[right, "similarity_group"])
    if overlap:
        raise RuntimeError(f"{name}: {len(overlap)} similarity groups cross the boundary.")


def class_counts(frame: pd.DataFrame, mask: pd.Series) -> dict[str, int]:
    counts = frame.loc[mask, "label"].value_counts().reindex(range(4), fill_value=0)
    return {str(int(label)): int(count) for label, count in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create frozen nested grouped splits for the revision experiment.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/similarity_grouped_splits/dfuc2021_cls_cv5_similarity_grouped_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data_processed/associate_editor_revision_v1")
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["training_seeds"]]
    split_seed = int(protocol["split_seed"])
    outer_folds = int(protocol["data"].get("outer_folds", 5))
    inner_folds = int(protocol["splits"]["inner"]["n_splits"])

    source = pd.read_csv(manifest_path)
    required = {"image", "image_path", "label", "label_name", "fold", "similarity_group"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")
    if source["image"].duplicated().any():
        raise ValueError("Source manifest contains duplicate image names.")

    rows: list[pd.DataFrame] = []
    report: dict[str, object] = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "protocol_sha256": sha256(protocol_path),
        "seeds": seeds,
        "outer_folds": {},
    }
    for outer in range(outer_folds):
        current = source.copy()
        current["outer_fold"] = outer
        current["outer_role"] = "train"
        current.loc[current["fold"] == outer, "outer_role"] = "test"
        train_mask = current["outer_role"].eq("train")
        test_mask = ~train_mask
        assert_disjoint_groups(current, train_mask, test_mask, f"outer={outer}")

        train = current.loc[train_mask]
        current["inner_fold"] = -1
        current.loc[train.index, "inner_fold"] = assign_folds(
            train, inner_folds, split_seed + 100 * outer
        )

        direct_split = assign_folds(train, 5, split_seed + 100 * outer + 4)
        current["direct_role"] = "outer_test"
        current.loc[train.index, "direct_role"] = "fit"
        current.loc[direct_split.index[direct_split.eq(0)], "direct_role"] = "stop"
        assert_disjoint_groups(
            current,
            current["direct_role"].eq("fit"),
            current["direct_role"].eq("stop"),
            f"outer={outer}, direct",
        )

        inner_report: dict[str, object] = {}
        for inner in range(inner_folds):
            column = f"inner{inner}_role"
            current[column] = "outer_test"
            validation_mask = train_mask & current["inner_fold"].eq(inner)
            candidate_mask = train_mask & ~validation_mask
            candidates = current.loc[candidate_mask]
            stop_split = assign_folds(candidates, 5, split_seed + 100 * outer + inner + 1)
            current.loc[validation_mask, column] = "validation"
            current.loc[candidate_mask, column] = "fit"
            current.loc[stop_split.index[stop_split.eq(0)], column] = "stop"
            fit_mask = current[column].eq("fit")
            stop_mask = current[column].eq("stop")
            assert_disjoint_groups(current, fit_mask, stop_mask, f"outer={outer}, inner={inner}, fit-stop")
            assert_disjoint_groups(
                current, fit_mask | stop_mask, validation_mask, f"outer={outer}, inner={inner}, train-validation"
            )
            counts = {
                "fit": class_counts(current, fit_mask),
                "stop": class_counts(current, stop_mask),
                "validation": class_counts(current, validation_mask),
            }
            if min(counts["fit"].values()) < 10 or min(counts["stop"].values()) < 10:
                raise RuntimeError(f"Insufficient class support in outer={outer}, inner={inner}: {counts}")
            inner_report[str(inner)] = counts

        report["outer_folds"][str(outer)] = {
            "train": class_counts(current, train_mask),
            "test": class_counts(current, test_mask),
            "direct_fit": class_counts(current, current["direct_role"].eq("fit")),
            "direct_stop": class_counts(current, current["direct_role"].eq("stop")),
            "inner": inner_report,
        }
        rows.append(current)

    assignments = pd.concat(rows, ignore_index=True)
    output_manifest = output_dir / "nested_grouped_assignments.csv"
    assignments.to_csv(output_manifest, index=False)
    report["assignment_manifest"] = str(output_manifest)
    report["assignment_manifest_sha256"] = sha256(output_manifest)
    report["rows"] = int(len(assignments))
    report["unique_images"] = int(assignments["image"].nunique())
    report_path = output_dir / "nested_grouped_split_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_manifest), "report": str(report_path), "rows": len(assignments)}, indent=2))


if __name__ == "__main__":
    main()
