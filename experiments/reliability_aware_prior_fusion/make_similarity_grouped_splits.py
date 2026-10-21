#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DFUC2021 CV splits that keep near-duplicate image groups together.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument("--audit-pairs", type=Path, default=Path("outputs/paper_tables/similar_image_audit_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_processed/dfuc2021_cls_v1/similarity_grouped_splits"))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--pair-scope",
        choices=("all_near_duplicates", "cross_fold_same_label"),
        default="all_near_duplicates",
        help="Which audit pairs should define groups.",
    )
    parser.add_argument(
        "--write-exclusion-manifest",
        action="store_true",
        help="Also write a manifest excluding cross-fold same-label near-duplicate candidate images.",
    )
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    manifest = pd.read_csv(resolve(args.manifest, repo_root))
    pairs = pd.read_csv(resolve(args.audit_pairs, repo_root))
    if args.pair_scope == "cross_fold_same_label":
        group_pairs = pairs[pairs["risk_group"] == "cross_fold_same_label"].copy()
    else:
        group_pairs = pairs.copy()

    uf = UnionFind(manifest["image"].astype(str).tolist())
    image_set = set(uf.parent)
    for row in group_pairs.itertuples(index=False):
        if row.image_a in image_set and row.image_b in image_set:
            uf.union(str(row.image_a), str(row.image_b))

    manifest = manifest.copy()
    manifest["similarity_group"] = manifest["image"].map(lambda image: uf.find(str(image)))
    splitter = StratifiedGroupKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    manifest["fold"] = -1
    for fold, (_, val_idx) in enumerate(splitter.split(manifest, manifest["label"], groups=manifest["similarity_group"])):
        manifest.loc[val_idx, "fold"] = fold
    if (manifest["fold"] < 0).any():
        raise RuntimeError("Some samples were not assigned to a fold.")

    output_dir = resolve(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"dfuc2021_cls_cv{args.num_folds}_similarity_grouped_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    for fold in range(args.num_folds):
        manifest[manifest["fold"] == fold].to_csv(output_dir / f"fold_{fold}.csv", index=False)

    grouped = manifest.groupby("similarity_group").agg(size=("image", "size"), labels=("label_name", lambda s: sorted(set(s)))).reset_index()
    summary = {
        "source_manifest": str(args.manifest),
        "audit_pairs": str(args.audit_pairs),
        "pair_scope": args.pair_scope,
        "samples": int(len(manifest)),
        "num_folds": int(args.num_folds),
        "seed": int(args.seed),
        "near_duplicate_pairs_used": int(len(group_pairs)),
        "similarity_groups": int(manifest["similarity_group"].nunique()),
        "non_singleton_groups": int((grouped["size"] > 1).sum()),
        "largest_group_size": int(grouped["size"].max()),
        "manifest": str(manifest_path.relative_to(repo_root)),
        "fold_counts": {
            str(fold): manifest[manifest["fold"] == fold]["label_name"].value_counts().sort_index().to_dict()
            for fold in range(args.num_folds)
        },
    }

    if args.write_exclusion_manifest:
        risky = pairs[pairs["risk_group"] == "cross_fold_same_label"]
        excluded_images = sorted(set(risky["image_a"]).union(set(risky["image_b"])))
        excluded = manifest[~manifest["image"].isin(excluded_images)].copy()
        exclusion_path = output_dir / "dfuc2021_cls_cv5_excluding_cross_fold_same_label_near_duplicates_manifest.csv"
        excluded.to_csv(exclusion_path, index=False)
        summary["exclusion_manifest"] = str(exclusion_path.relative_to(repo_root))
        summary["excluded_images"] = int(len(excluded_images))
        summary["exclusion_samples"] = int(len(excluded))

    (output_dir / "dfuc2021_cls_similarity_grouped_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
