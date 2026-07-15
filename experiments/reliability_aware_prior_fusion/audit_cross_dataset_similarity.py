#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-dataset visual overlap between DFUC2021 and DFUC2022 images.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--dfuc2021-manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument(
        "--dfuc2022-manifest",
        type=Path,
        default=Path("data_processed/dfuc2022_seg_v1/splits/dfuc2022_seg_cv5_manifest.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_tables/methodology_audit_v7"))
    parser.add_argument("--phash-threshold", type=int, default=4)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--strict-phash-threshold", type=int, default=2)
    parser.add_argument("--strict-dhash-threshold", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def phash_bits(path: Path) -> np.ndarray:
    image = read_gray(path, (32, 32))
    dct = cv2.dct(image)
    values = dct[:8, :8].reshape(-1)
    median = np.median(values[1:])
    return values > median


def dhash_bits(path: Path) -> np.ndarray:
    image = read_gray(path, (9, 8))
    return (image[:, 1:] > image[:, :-1]).reshape(-1)


def normalize_manifest(frame: pd.DataFrame, dataset: str, repo_root: Path) -> pd.DataFrame:
    if "image_path" not in frame.columns:
        raise ValueError(f"{dataset} manifest must include image_path.")
    if "image" in frame.columns:
        image_column = "image"
    elif "image_id" in frame.columns:
        image_column = "image_id"
    else:
        raise ValueError(f"{dataset} manifest must include image or image_id.")
    output = pd.DataFrame(
        {
            "dataset": dataset,
            "image": frame[image_column].astype(str),
            "image_path": frame["image_path"].astype(str),
        }
    )
    output["abs_image_path"] = output["image_path"].map(lambda value: resolve(Path(value), repo_root))
    missing = [str(path) for path in output["abs_image_path"] if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"{dataset} has missing image files. First examples: {missing[:5]}")
    return output


def add_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sha256"] = [sha256_file(Path(path)) for path in frame["abs_image_path"]]
    return frame


def hash_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    phash = np.vstack([phash_bits(Path(path)) for path in frame["abs_image_path"]])
    dhash = np.vstack([dhash_bits(Path(path)) for path in frame["abs_image_path"]])
    return phash, dhash


def exact_duplicate_pairs(dfuc2021: pd.DataFrame, dfuc2022: pd.DataFrame) -> pd.DataFrame:
    merged = dfuc2021.merge(dfuc2022, on="sha256", suffixes=("_dfuc2021", "_dfuc2022"))
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "image_dfuc2021",
                "path_dfuc2021",
                "image_dfuc2022",
                "path_dfuc2022",
                "sha256",
            ]
        )
    return pd.DataFrame(
        {
            "image_dfuc2021": merged["image_dfuc2021"],
            "path_dfuc2021": merged["image_path_dfuc2021"],
            "image_dfuc2022": merged["image_dfuc2022"],
            "path_dfuc2022": merged["image_path_dfuc2022"],
            "sha256": merged["sha256"],
        }
    )


def cross_dataset_near_pairs(
    dfuc2021: pd.DataFrame,
    dfuc2022: pd.DataFrame,
    phash_2021: np.ndarray,
    dhash_2021: np.ndarray,
    phash_2022: np.ndarray,
    dhash_2022: np.ndarray,
    phash_threshold: int,
    dhash_threshold: int,
    strict_phash_threshold: int,
    strict_dhash_threshold: int,
    chunk_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for start in range(0, len(dfuc2021), chunk_size):
        stop = min(start + chunk_size, len(dfuc2021))
        p_dist = np.count_nonzero(phash_2021[start:stop, None, :] != phash_2022[None, :, :], axis=2)
        d_dist = np.count_nonzero(dhash_2021[start:stop, None, :] != dhash_2022[None, :, :], axis=2)
        hit_i, hit_j = np.where((p_dist <= phash_threshold) | (d_dist <= dhash_threshold))
        for rel_i, j in zip(hit_i.tolist(), hit_j.tolist()):
            i = start + rel_i
            p_value = int(p_dist[rel_i, j])
            d_value = int(d_dist[rel_i, j])
            rows.append(
                {
                    "image_dfuc2021": dfuc2021.at[i, "image"],
                    "path_dfuc2021": dfuc2021.at[i, "image_path"],
                    "image_dfuc2022": dfuc2022.at[j, "image"],
                    "path_dfuc2022": dfuc2022.at[j, "image_path"],
                    "phash_distance": p_value,
                    "dhash_distance": d_value,
                    "phash_zero_or_dhash_zero": int(p_value == 0 or d_value == 0),
                    "strict_near_duplicate": int(p_value <= strict_phash_threshold and d_value <= strict_dhash_threshold),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "image_dfuc2021",
                "path_dfuc2021",
                "image_dfuc2022",
                "path_dfuc2022",
                "phash_distance",
                "dhash_distance",
                "phash_zero_or_dhash_zero",
                "strict_near_duplicate",
            ]
        )
    return pd.DataFrame(rows).sort_values(["phash_distance", "dhash_distance", "image_dfuc2021", "image_dfuc2022"])


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    text = f"""# DFUC2021-DFUC2022 Cross-Dataset Similarity Audit

Generated by `experiments/reliability_aware_prior_fusion/audit_cross_dataset_similarity.py`.

## Summary

| Item | Value |
|---|---:|
| DFUC2021 valid images | {summary["dfuc2021_valid_train_images"]} |
| DFUC2022 segmentation-training images | {summary["dfuc2022_seg_train_images"]} |
| SHA256 exact duplicate pairs | {summary["cross_dataset_exact_duplicate_pairs"]} |
| pHash/dHash candidate pairs | {summary["cross_dataset_near_duplicate_candidate_pairs"]} |
| Involved DFUC2021 images | {summary["dfuc2021_images_in_near_duplicate_pairs"]} |
| Involved DFUC2022 images | {summary["dfuc2022_images_in_near_duplicate_pairs"]} |
| Candidates with pHash = 0 or dHash = 0 | {summary["candidate_pairs_with_phash_zero_or_dhash_zero"]} |
| Candidates meeting pHash <= {summary["strict_phash_threshold"]} and dHash <= {summary["strict_dhash_threshold"]} | {summary["candidate_pairs_meeting_strict_threshold"]} |

Broad candidates use pHash <= {summary["phash_threshold"]} or dHash <= {summary["dhash_threshold"]}. The stricter combined rule is used only to summarize high-confidence cross-dataset near-duplicate evidence.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfuc2021 = normalize_manifest(pd.read_csv(resolve(args.dfuc2021_manifest, repo_root)), "dfuc2021", repo_root)
    dfuc2022 = normalize_manifest(pd.read_csv(resolve(args.dfuc2022_manifest, repo_root)), "dfuc2022", repo_root)
    dfuc2021 = add_hashes(dfuc2021)
    dfuc2022 = add_hashes(dfuc2022)

    exact_pairs = exact_duplicate_pairs(dfuc2021, dfuc2022)
    exact_pairs.to_csv(output_dir / "dfuc2021_dfuc2022_cross_dataset_exact_duplicates.csv", index=False)

    phash_2021, dhash_2021 = hash_matrices(dfuc2021)
    phash_2022, dhash_2022 = hash_matrices(dfuc2022)
    near_pairs = cross_dataset_near_pairs(
        dfuc2021,
        dfuc2022,
        phash_2021,
        dhash_2021,
        phash_2022,
        dhash_2022,
        args.phash_threshold,
        args.dhash_threshold,
        args.strict_phash_threshold,
        args.strict_dhash_threshold,
        args.chunk_size,
    )
    near_pairs.to_csv(output_dir / "dfuc2021_dfuc2022_cross_dataset_near_duplicate_pairs.csv", index=False)

    summary = {
        "dfuc2021_valid_train_images": int(len(dfuc2021)),
        "dfuc2022_seg_train_images": int(len(dfuc2022)),
        "phash_threshold": int(args.phash_threshold),
        "dhash_threshold": int(args.dhash_threshold),
        "strict_phash_threshold": int(args.strict_phash_threshold),
        "strict_dhash_threshold": int(args.strict_dhash_threshold),
        "cross_dataset_exact_duplicate_pairs": int(len(exact_pairs)),
        "cross_dataset_near_duplicate_candidate_pairs": int(len(near_pairs)),
        "dfuc2021_images_in_near_duplicate_pairs": int(near_pairs["image_dfuc2021"].nunique()) if not near_pairs.empty else 0,
        "dfuc2022_images_in_near_duplicate_pairs": int(near_pairs["image_dfuc2022"].nunique()) if not near_pairs.empty else 0,
        "candidate_pairs_with_phash_zero_or_dhash_zero": int(near_pairs["phash_zero_or_dhash_zero"].sum()) if not near_pairs.empty else 0,
        "candidate_pairs_meeting_strict_threshold": int(near_pairs["strict_near_duplicate"].sum()) if not near_pairs.empty else 0,
    }
    (output_dir / "dfuc2021_dfuc2022_cross_dataset_duplicate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_dir / "dfuc2021_dfuc2022_cross_dataset_duplicate_summary.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
