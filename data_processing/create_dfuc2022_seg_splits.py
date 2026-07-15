#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.model_selection import KFold


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_MASK_EXTS = (".png", ".jpg", ".jpeg")


def parse_exts(raw: str) -> tuple[str, ...]:
    exts = []
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.append(ext)
    if not exts:
        raise argparse.ArgumentTypeError("At least one extension is required.")
    return tuple(exts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the DFUC2022 segmentation manifest and deterministic CV folds."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data_raw/2022_dfuc2022/train/DFUC2022_train_images"),
        help="Directory containing DFUC2022 training images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=Path("data_raw/2022_dfuc2022/train/DFUC2022_train_masks"),
        help="Directory containing DFUC2022 binary lesion masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processed/dfuc2022_seg_v1/splits"),
        help="Directory where manifest and fold CSV files will be written.",
    )
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--image-exts", type=parse_exts, default=DEFAULT_IMAGE_EXTS)
    parser.add_argument("--mask-exts", type=parse_exts, default=DEFAULT_MASK_EXTS)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: Path, root: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return root / path


def rel_to_repo(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def collect_by_stem(directory: Path, extensions: tuple[str, ...], kind: str) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Missing {kind} directory: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"{kind} path is not a directory: {directory}")

    matches: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    allowed = {ext.lower() for ext in extensions}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        stem = path.stem
        if stem in matches:
            duplicates.setdefault(stem, [matches[stem]]).append(path)
            continue
        matches[stem] = path

    if duplicates:
        sample = "; ".join(
            f"{stem}: {', '.join(str(p.name) for p in paths)}"
            for stem, paths in sorted(duplicates.items())[:5]
        )
        raise ValueError(f"Duplicate {kind} stems found. Resolve before splitting. Examples: {sample}")
    return matches


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "image_path", "mask_path", "fold"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    image_dir: Path,
    mask_dir: Path,
    image_exts: tuple[str, ...],
    mask_exts: tuple[str, ...],
    num_folds: int,
    seed: int,
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_by_stem = collect_by_stem(image_dir, image_exts, "image")
    mask_by_stem = collect_by_stem(mask_dir, mask_exts, "mask")

    image_stems = set(image_by_stem)
    mask_stems = set(mask_by_stem)
    missing_masks = sorted(image_stems - mask_stems)
    masks_without_images = sorted(mask_stems - image_stems)
    if missing_masks:
        raise FileNotFoundError(
            f"{len(missing_masks)} images do not have matching masks. "
            f"First examples: {missing_masks[:10]}"
        )
    if masks_without_images:
        raise FileNotFoundError(
            f"{len(masks_without_images)} masks do not have matching images. "
            f"First examples: {masks_without_images[:10]}"
        )

    image_ids = sorted(image_by_stem)
    if not image_ids:
        raise ValueError(f"No DFUC2022 image/mask pairs found in {image_dir} and {mask_dir}.")
    if num_folds < 2:
        raise ValueError("--num-folds must be at least 2.")
    if num_folds > len(image_ids):
        raise ValueError(f"--num-folds={num_folds} exceeds sample count {len(image_ids)}.")

    rows = [
        {
            "image_id": image_id,
            "image_path": rel_to_repo(image_by_stem[image_id], root),
            "mask_path": rel_to_repo(mask_by_stem[image_id], root),
            "fold": -1,
        }
        for image_id in image_ids
    ]

    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (_, val_indices) in enumerate(splitter.split(rows)):
        for index in val_indices:
            rows[index]["fold"] = fold
    rows = sorted(rows, key=lambda row: (int(row["fold"]), row["image_id"]))

    fold_counts = {
        str(fold): sum(1 for row in rows if int(row["fold"]) == fold)
        for fold in range(num_folds)
    }
    summary = {
        "image_dir": rel_to_repo(image_dir, root),
        "mask_dir": rel_to_repo(mask_dir, root),
        "samples": len(rows),
        "num_folds": num_folds,
        "seed": seed,
        "image_extensions": list(image_exts),
        "mask_extensions": list(mask_exts),
        "fold_counts": fold_counts,
    }
    return rows, summary


def main() -> None:
    args = parse_args()
    root = repo_root()
    image_dir = resolve_path(args.image_dir, root)
    mask_dir = resolve_path(args.mask_dir, root)
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary = build_rows(
        image_dir=image_dir,
        mask_dir=mask_dir,
        image_exts=args.image_exts,
        mask_exts=args.mask_exts,
        num_folds=args.num_folds,
        seed=args.seed,
        root=root,
    )

    manifest_path = output_dir / f"dfuc2022_seg_cv{args.num_folds}_manifest.csv"
    write_csv(manifest_path, rows)
    for fold in range(args.num_folds):
        fold_rows = [row for row in rows if int(row["fold"]) == fold]
        write_csv(output_dir / f"fold_{fold}.csv", fold_rows)

    summary = {
        **summary,
        "manifest": rel_to_repo(manifest_path, root),
        "fold_files": [
            rel_to_repo(output_dir / f"fold_{fold}.csv", root)
            for fold in range(args.num_folds)
        ],
    }
    with (output_dir / f"dfuc2022_seg_cv{args.num_folds}_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
