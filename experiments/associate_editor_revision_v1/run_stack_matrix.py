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
    parser = argparse.ArgumentParser(description="Run all prespecified nested stackers.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    protocol = yaml.safe_load((repo_root / "experiments/associate_editor_revision_v1/protocol.yaml").read_text(encoding="utf-8"))
    script = repo_root / "experiments/associate_editor_revision_v1/fit_nested_stacks.py"
    output_root = repo_root / "outputs/associate_editor_revision_v1/stacks"
    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.time()
    for seed in protocol["training_seeds"]:
        for outer in range(5):
            command = [sys.executable, str(script), "--repo-root", str(repo_root), "--seed", str(seed), "--outer", str(outer)]
            print("RUN " + " ".join(command), flush=True)
            subprocess.run(command, cwd=repo_root, check=True)
            completed += 1
            write_json(output_root.parent / "stack_progress.json", {"completed_jobs": completed, "planned_jobs": 25, "last": [seed, outer]})
    write_json(
        output_root.parent / "stack_complete.json",
        {"status": "complete", "jobs": 25, "elapsed_seconds": time.time() - started, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z")},
    )


if __name__ == "__main__":
    main()
