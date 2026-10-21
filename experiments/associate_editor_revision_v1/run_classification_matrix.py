#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml


BASE_MODELS = (
    "resnet50",
    "efficientnetb0",
    "lpt_no_alignment",
    "lpt_no_prior_mod",
    "lpt_no_roi",
    "lpt_stats_only",
    "convnext_tiny",
    "efficientnetv2s",
)
DIRECT_MODELS = ("lpt_full", "lpt_neutral")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], cwd: Path) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 900-fit classification matrix with resume support.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.yaml"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1/classification")
    )
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--outers", type=int, nargs="*", choices=range(5))
    parser.add_argument("--models", nargs="*", choices=BASE_MODELS + DIRECT_MODELS)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in protocol["training_seeds"]]
    outers = args.outers or list(range(5))
    selected = set(args.models or (BASE_MODELS + DIRECT_MODELS))
    assignments = repo_root / "data_processed/associate_editor_revision_v1/nested_grouped_assignments.csv"
    split_report = repo_root / "data_processed/associate_editor_revision_v1/nested_grouped_split_report.json"
    prior_checkpoint = repo_root / protocol["data"]["prior_checkpoint"]
    expected_prior_hash = str(protocol["data"]["expected_prior_checkpoint_sha256"])

    if not assignments.exists():
        run(
            [
                sys.executable,
                str(repo_root / "experiments/associate_editor_revision_v1/prepare_nested_splits.py"),
                "--repo-root",
                str(repo_root),
            ],
            repo_root,
        )
    if not split_report.exists():
        raise FileNotFoundError(split_report)
    if not prior_checkpoint.exists() or sha256(prior_checkpoint) != expected_prior_hash:
        raise RuntimeError("Frozen prior checkpoint is missing or has an unexpected SHA256 hash.")

    output_root.mkdir(parents=True, exist_ok=True)
    environment_path = output_root.parent / "environment.json"
    if not environment_path.exists():
        write_json(
            environment_path,
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "protocol": str(protocol_path),
                "protocol_sha256": sha256(protocol_path),
                "assignments": str(assignments),
                "assignments_sha256": sha256(assignments),
                "prior_checkpoint": str(prior_checkpoint),
                "prior_checkpoint_sha256": sha256(prior_checkpoint),
                "pid": os.getpid(),
            },
        )

    trainer = repo_root / "experiments/associate_editor_revision_v1/train_nested_classification.py"
    total = len(seeds) * len(outers) * (
        4 * len([model for model in BASE_MODELS if model in selected])
        + 2 * len([model for model in DIRECT_MODELS if model in selected])
    )
    completed = 0
    started = time.time()
    progress_path = output_root.parent / "classification_progress.json"

    for seed in seeds:
        for outer in outers:
            for model in BASE_MODELS:
                if model not in selected:
                    continue
                for inner in range(3):
                    run(
                        [
                            sys.executable,
                            str(trainer),
                            "--repo-root",
                            str(repo_root),
                            "--output-root",
                            str(output_root),
                            "--model",
                            model,
                            "--seed",
                            str(seed),
                            "--outer",
                            str(outer),
                            "--stage",
                            "inner",
                            "--inner",
                            str(inner),
                            "--device",
                            args.device,
                        ],
                        repo_root,
                    )
                    completed += 1
                    write_json(progress_path, {"completed_jobs": completed, "planned_jobs": total, "last": [seed, outer, model, "inner", inner]})
                run(
                    [
                        sys.executable,
                        str(trainer),
                        "--repo-root",
                        str(repo_root),
                        "--output-root",
                        str(output_root),
                        "--model",
                        model,
                        "--seed",
                        str(seed),
                        "--outer",
                        str(outer),
                        "--stage",
                        "refit",
                        "--device",
                        args.device,
                    ],
                    repo_root,
                )
                completed += 1
                write_json(progress_path, {"completed_jobs": completed, "planned_jobs": total, "last": [seed, outer, model, "refit"]})

            for model in DIRECT_MODELS:
                if model not in selected:
                    continue
                for stage in ("direct_select", "direct_refit"):
                    run(
                        [
                            sys.executable,
                            str(trainer),
                            "--repo-root",
                            str(repo_root),
                            "--output-root",
                            str(output_root),
                            "--model",
                            model,
                            "--seed",
                            str(seed),
                            "--outer",
                            str(outer),
                            "--stage",
                            stage,
                            "--device",
                            args.device,
                        ],
                        repo_root,
                    )
                    completed += 1
                    write_json(progress_path, {"completed_jobs": completed, "planned_jobs": total, "last": [seed, outer, model, stage]})

    write_json(
        output_root.parent / "classification_complete.json",
        {
            "status": "complete",
            "jobs": total,
            "seeds": seeds,
            "outer_folds": outers,
            "models": sorted(selected),
            "elapsed_seconds": time.time() - started,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )


if __name__ == "__main__":
    main()
