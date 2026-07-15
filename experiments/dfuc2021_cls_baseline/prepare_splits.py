#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold


LABEL_COLUMNS = ("none", "infection", "ischaemia", "both")
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABEL_COLUMNS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DFUC2021 labeled stratified CV splits.")
    parser.add_argument("--train-csv", type=Path, default=Path("data_raw/2021_dfuc2021/train/train.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data_raw/2021_dfuc2021/train/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed/dfuc2021_cls_v1/splits"))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260430)
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def rel_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def row_label(row: pd.Series) -> str | None:
    values = row[list(LABEL_COLUMNS)]
    if values.isna().any():
        return None
    positives = [name for name in LABEL_COLUMNS if float(row[name]) == 1.0]
    if len(positives) != 1:
        return None
    return positives[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    train_csv = resolve_path(args.train_csv, repo_root)
    image_dir = resolve_path(args.image_dir, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    labeled_rows: list[dict[str, Any]] = []
    unlabeled_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        image_name = str(row["image"])
        image_path = image_dir / image_name
        label_name = row_label(row)
        base = {
            "image": image_name,
            "image_path": rel_to_repo(image_path, repo_root),
        }
        if label_name is None:
            unlabeled_rows.append(base)
            continue
        if not image_path.exists():
            raise FileNotFoundError(f"Missing DFUC2021 image: {image_path}")
        labeled_rows.append(
            {
                **base,
                "label_name": label_name,
                "label": LABEL_TO_INDEX[label_name],
                **{name: int(float(row[name])) for name in LABEL_COLUMNS},
            }
        )

    labels = [int(row["label"]) for row in labeled_rows]
    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold, (_, val_indices) in enumerate(splitter.split(labeled_rows, labels)):
        for index in val_indices:
            labeled_rows[index]["fold"] = fold

    labeled_rows = sorted(labeled_rows, key=lambda row: (int(row["fold"]), row["image"]))
    manifest_path = output_dir / "dfuc2021_cls_cv5_manifest.csv"
    write_csv(manifest_path, labeled_rows)

    for fold in range(args.num_folds):
        fold_rows = [row for row in labeled_rows if int(row["fold"]) == fold]
        write_csv(output_dir / f"fold_{fold}.csv", fold_rows)

    write_csv(output_dir / "dfuc2021_cls_unlabeled_manifest.csv", unlabeled_rows)

    class_counts = {
        label_name: sum(1 for row in labeled_rows if row["label_name"] == label_name)
        for label_name in LABEL_COLUMNS
    }
    fold_counts = {
        str(fold): {
            label_name: sum(
                1
                for row in labeled_rows
                if int(row["fold"]) == fold and row["label_name"] == label_name
            )
            for label_name in LABEL_COLUMNS
        }
        for fold in range(args.num_folds)
    }
    summary = {
        "source_train_csv": rel_to_repo(train_csv, repo_root),
        "image_dir": rel_to_repo(image_dir, repo_root),
        "num_rows_in_csv": int(len(df)),
        "labeled_samples": len(labeled_rows),
        "unlabeled_or_invalid_rows": len(unlabeled_rows),
        "num_folds": args.num_folds,
        "seed": args.seed,
        "label_to_index": LABEL_TO_INDEX,
        "class_counts": class_counts,
        "fold_counts": fold_counts,
        "manifest": rel_to_repo(manifest_path, repo_root),
    }
    with (output_dir / "dfuc2021_cls_cv5_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
