#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish the frozen revision pipeline after CNN classification fits.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_root = repo_root / "outputs/associate_editor_revision_v1"
    classification_marker = output_root / "classification_complete.json"
    while not classification_marker.exists():
        print(f"WAIT classification matrix {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        time.sleep(args.poll_seconds)

    stages = (
        ("classification_integrity", repo_root / "experiments/associate_editor_revision_v1/repair_nonfinite_classification.py"),
        ("stack", repo_root / "experiments/associate_editor_revision_v1/run_stack_matrix.py"),
        ("segmentation", repo_root / "experiments/associate_editor_revision_v1/run_segmentation_matrix.py"),
    )
    for name, script in stages:
        print(f"START stage={name}", flush=True)
        subprocess.run([sys.executable, str(script), "--repo-root", str(repo_root)], cwd=repo_root, check=True)
        print(f"COMPLETE stage={name}", flush=True)

    write_json(
        output_root / "training_pipeline_complete.json",
        {
            "status": "complete",
            "classification": str(classification_marker),
            "classification_prediction_integrity": str(output_root / "classification_prediction_integrity.json"),
            "stacks": str(output_root / "stack_complete.json"),
            "segmentation": str(output_root / "segmentation_complete.json"),
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )


if __name__ == "__main__":
    main()
