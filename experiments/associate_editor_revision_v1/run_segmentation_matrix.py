#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 25-fit source-domain resolution audit.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol_path = repo_root / "experiments/associate_editor_revision_v1/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in protocol["training_seeds"]]
    output_root = repo_root / "outputs/associate_editor_revision_v1/segmentation"
    output_root.mkdir(parents=True, exist_ok=True)
    trainer = repo_root / "experiments/associate_editor_revision_v1/train_segmentation_audit.py"
    progress_path = output_root.parent / "segmentation_progress.json"
    completed = 0
    started = time.time()
    for seed in seeds:
        for outer in range(5):
            command = [
                sys.executable,
                str(trainer),
                "--repo-root",
                str(repo_root),
                "--output-root",
                str(output_root),
                "--seed",
                str(seed),
                "--outer",
                str(outer),
                "--device",
                args.device,
            ]
            print("RUN " + " ".join(command), flush=True)
            subprocess.run(command, cwd=repo_root, check=True)
            completed += 1
            write_json(progress_path, {"completed_jobs": completed, "planned_jobs": 25, "last": [seed, outer]})
    write_json(
        output_root.parent / "segmentation_complete.json",
        {
            "status": "complete",
            "jobs": 25,
            "elapsed_seconds": time.time() - started,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )


if __name__ == "__main__":
    main()
