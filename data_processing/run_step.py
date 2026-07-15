#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessingStep:
    script: Path
    description: str


STEPS = {
    "dfuc2021-exclusion-audit": ProcessingStep(
        script=Path("data_processing/audit_dfuc2021_exclusions.py"),
        description="Audit DFUC2021 annotation inclusion/exclusion reasons and image-file availability.",
    ),
    "dfuc2021-splits": ProcessingStep(
        script=Path("experiments/dfuc2021_cls_baseline/prepare_splits.py"),
        description="Create the valid DFUC2021 four-class manifest and standard stratified CV folds.",
    ),
    "dfuc2022-seg-splits": ProcessingStep(
        script=Path("data_processing/create_dfuc2022_seg_splits.py"),
        description="Create the DFUC2022 segmentation manifest and deterministic CV folds.",
    ),
    "dfuc2021-similarity-audit": ProcessingStep(
        script=Path("experiments/reliability_aware_prior_fusion/audit_image_similarity.py"),
        description="Screen DFUC2021 folds for exact duplicates and pHash/dHash near-duplicate candidates.",
    ),
    "dfuc2021-similarity-grouped-splits": ProcessingStep(
        script=Path("experiments/reliability_aware_prior_fusion/make_similarity_grouped_splits.py"),
        description="Create similarity-grouped folds that keep near-duplicate connected components together.",
    ),
    "dfuc2021-dfuc2022-cross-dataset-audit": ProcessingStep(
        script=Path("experiments/reliability_aware_prior_fusion/audit_cross_dataset_similarity.py"),
        description="Screen DFUC2021 and DFUC2022 for cross-dataset exact duplicates and pHash/dHash near-duplicate candidates.",
    ),
    "dfuc2021-lesion-priors": ProcessingStep(
        script=Path("experiments/cross_task_prior_transfer/generate_dfuc2021_priors.py"),
        description="Generate DFUC2021 soft lesion-prior maps from a trained DFUC2022 segmenter.",
    ),
    "cross-dataset-overlap-exclusion-sensitivity": ProcessingStep(
        script=Path("experiments/reliability_aware_prior_fusion/cross_dataset_overlap_exclusion_sensitivity.py"),
        description="Recompute grouped OOF metrics after excluding DFUC2021 images involved in broad cross-dataset visual-overlap candidates.",
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one publication data-processing step.")
    parser.add_argument("step", nargs="?", choices=sorted(STEPS), help="Processing step to run.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the underlying script.")
    parser.add_argument("--list", action="store_true", help="List available processing steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    return parser.parse_args()


def list_steps() -> None:
    for name, step in sorted(STEPS.items()):
        print(f"{name}: {step.description}")
        print(f"  script: {step.script}")


def main() -> None:
    args = parse_args()
    if args.list:
        list_steps()
        return
    if args.step is None:
        raise SystemExit("Choose a step or pass --list.")

    forwarded = list(args.script_args)
    dry_run = args.dry_run
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    elif "--dry-run" in forwarded:
        dry_run = True
        forwarded = [item for item in forwarded if item != "--dry-run"]

    root = repo_root()
    script = root / STEPS[args.step].script
    command = [sys.executable, str(script), *forwarded]
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
