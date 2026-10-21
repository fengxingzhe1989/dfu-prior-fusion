#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_STEPS = [
    ("dfuc2021-exclusion-audit", []),
    ("dfuc2021-splits", []),
    ("dfuc2022-seg-splits", []),
    ("dfuc2021-similarity-audit", []),
    ("dfuc2021-similarity-grouped-splits", []),
    ("dfuc2021-dfuc2022-cross-dataset-audit", []),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-aligned data processing pipeline.")
    parser.add_argument(
        "--include-priors",
        action="store_true",
        help="Also generate DFUC2021 lesion-prior maps. Requires a trained DFUC2022 segmentation checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt"),
        help="Segmentation checkpoint used when --include-priors is set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run_step(root: Path, step: str, forwarded: list[str], dry_run: bool) -> None:
    runner = root / "data_processing/run_step.py"
    command = [sys.executable, str(runner), step, *forwarded]
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=root, check=True)


def main() -> None:
    args = parse_args()
    root = repo_root()
    for step, forwarded in BASE_STEPS:
        run_step(root, step, forwarded, args.dry_run)
    if args.include_priors:
        run_step(root, "dfuc2021-lesion-priors", ["--checkpoint", str(args.checkpoint)], args.dry_run)


if __name__ == "__main__":
    main()
