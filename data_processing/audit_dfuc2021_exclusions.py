#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


LABEL_COLUMNS = ("none", "infection", "ischaemia", "both")
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABEL_COLUMNS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit DFUC2021 released training-package composition and supervised-analysis "
            "eligibility for the four-class infection/ischaemia task."
        )
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data_raw/2021_dfuc2021/train/train.csv"))
    parser.add_argument("--image-dir", type=Path, default=Path("data_raw/2021_dfuc2021/train/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed/dfuc2021_cls_v1/audits"))
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(path: Path, root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else root / path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_binary_label(value: Any) -> bool:
    try:
        return float(value) in (0.0, 1.0)
    except (TypeError, ValueError):
        return False


def audit_row(row: pd.Series, image_dir: Path, duplicate_names: set[str], root: Path) -> dict[str, Any]:
    image = str(row["image"])
    image_path = image_dir / image
    label_values = row[list(LABEL_COLUMNS)]
    missing_count = int(label_values.isna().sum())
    nonmissing = [row[name] for name in LABEL_COLUMNS if not pd.isna(row[name])]
    nonbinary_labels = [name for name in LABEL_COLUMNS if not pd.isna(row[name]) and not is_binary_label(row[name])]
    positive_labels = [
        name
        for name in LABEL_COLUMNS
        if not pd.isna(row[name]) and is_binary_label(row[name]) and float(row[name]) == 1.0
    ]
    has_file = image_path.exists()
    has_duplicate_name = image in duplicate_names

    valid_four_class_label = (
        missing_count == 0
        and not nonbinary_labels
        and len(positive_labels) == 1
    )
    included = valid_four_class_label and has_file and not has_duplicate_name
    if included:
        reason = "included_valid_four_class"
    elif missing_count == len(LABEL_COLUMNS):
        reason = "official_unlabelled_subset"
    elif missing_count > 0:
        reason = "partially_missing_labels"
    elif nonbinary_labels:
        reason = "invalid_nonbinary_label"
    elif len(positive_labels) != 1:
        reason = "invalid_positive_count"
    elif not has_file:
        reason = "missing_image_file"
    elif has_duplicate_name:
        reason = "duplicate_image_name"
    else:
        reason = "excluded_other"

    label_name = positive_labels[0] if included else ""
    return {
        "row_index": int(row.name),
        "image": image,
        "image_path": rel(image_path, root),
        "image_file_exists": bool(has_file),
        "duplicate_image_name": bool(has_duplicate_name),
        "label_missing_count": missing_count,
        "label_nonmissing_count": len(nonmissing),
        "positive_label_count": len(positive_labels),
        "nonbinary_label_columns": ";".join(nonbinary_labels),
        "valid_four_class_label": bool(valid_four_class_label),
        "included": bool(included),
        "label_name": label_name,
        "label": LABEL_TO_INDEX[label_name] if label_name else "",
        "analysis_status": reason,
    }


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
    root = repo_root()
    train_csv = resolve(args.train_csv, root)
    image_dir = resolve(args.image_dir, root)
    output_dir = resolve(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(train_csv)
    required = {"image", *LABEL_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required annotation columns: {sorted(missing)}")

    image_counts = Counter(frame["image"].astype(str))
    duplicate_names = {name for name, count in image_counts.items() if count > 1}
    rows = [audit_row(row, image_dir, duplicate_names, root) for _, row in frame.iterrows()]
    write_csv(output_dir / "dfuc2021_exclusion_audit_rows.csv", rows)

    reason_counts = Counter(str(row["analysis_status"]) for row in rows)
    included_rows = [row for row in rows if row["included"]]
    class_counts = {class_name: sum(1 for row in included_rows if row["label_name"] == class_name) for class_name in LABEL_COLUMNS}
    missing_image_rows = [row["image"] for row in rows if not row["image_file_exists"]]
    duplicate_row_count = sum(1 for row in rows if row["duplicate_image_name"])
    summary = {
        "source_train_csv": rel(train_csv, root),
        "image_dir": rel(image_dir, root),
        "num_rows_in_csv": int(len(frame)),
        "included_valid_four_class_images": int(len(included_rows)),
        "not_eligible_for_supervised_analysis": int(len(rows) - len(included_rows)),
        "analysis_status_counts": dict(sorted(reason_counts.items())),
        "class_counts": class_counts,
        "annotation_image_files_present": int(sum(1 for row in rows if row["image_file_exists"])),
        "annotation_image_files_missing": int(len(missing_image_rows)),
        "duplicate_image_name_groups": int(len(duplicate_names)),
        "duplicate_image_name_rows": int(duplicate_row_count),
        "missing_image_examples": missing_image_rows[:20],
        "duplicate_image_name_examples": sorted(duplicate_names)[:20],
    }
    (output_dir / "dfuc2021_exclusion_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    s1_rows = [
        {"item": "DFUC2021 released training-package image rows", "value": summary["num_rows_in_csv"], "note": "Official package combines labelled and unlabelled subsets."},
        {"item": "Official labelled training subset", "value": summary["included_valid_four_class_images"], "note": "Eligible for supervised analysis; exactly one positive valid label and image file present."},
        {"item": "Official unlabelled training-package subset", "value": reason_counts.get("official_unlabelled_subset", 0), "note": "Not eligible for supervised analysis; blank class fields are intentional rather than accidental missingness."},
        {"item": "Other rows not eligible for supervised analysis", "value": summary["not_eligible_for_supervised_analysis"] - reason_counts.get("official_unlabelled_subset", 0), "note": "Should be zero for the released package used in this study."},
        {"item": "Rows with missing image files", "value": summary["annotation_image_files_missing"], "note": "Checked across all annotation rows."},
        {"item": "Duplicate image-name groups", "value": summary["duplicate_image_name_groups"], "note": "Checked across all annotation rows."},
        {"item": "none", "value": class_counts["none"], "note": "Included images only."},
        {"item": "infection", "value": class_counts["infection"], "note": "Included images only."},
        {"item": "ischaemia", "value": class_counts["ischaemia"], "note": "Included images only."},
        {"item": "both", "value": class_counts["both"], "note": "Included images only."},
    ]
    write_csv(output_dir / "dfuc2021_training_package_audit_s1_summary.csv", s1_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
