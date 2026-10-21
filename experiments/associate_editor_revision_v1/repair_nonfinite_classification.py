#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = ("prob_none", "prob_infection", "prob_ischaemia", "prob_both")


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


def scan_predictions(classification_root: Path) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for path in sorted(classification_root.glob("seed_*/outer_*/*/*/predictions.csv")):
        frame = pd.read_csv(path, usecols=["image", *PROBABILITY_COLUMNS])
        values = frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values).all(axis=1)
        if bad.any():
            issues.append(
                {
                    "path": str(path),
                    "job_dir": str(path.parent),
                    "bad_rows": int(bad.sum()),
                    "images": frame.loc[bad, "image"].astype(str).tolist(),
                    "prediction_sha256": sha256(path),
                }
            )
    return issues


def parse_job(job_dir: Path, classification_root: Path) -> tuple[int, int, str, str, int | None]:
    parts = job_dir.relative_to(classification_root).parts
    if len(parts) != 4:
        raise ValueError(f"Unexpected classification job path: {job_dir}")
    seed_part, outer_part, model, stage_part = parts
    seed = int(seed_part.removeprefix("seed_"))
    outer = int(outer_part.removeprefix("outer_"))
    if stage_part.startswith("inner_"):
        return seed, outer, model, "inner", int(stage_part.removeprefix("inner_"))
    return seed, outer, model, stage_part, None


def archive_job(job_dir: Path, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=False)
    for name in ("complete.json", "predictions.csv", "history.jsonl", "model.pt"):
        source = job_dir / name
        if source.exists():
            shutil.copy2(source, archive_dir / name)


def repair_issue(
    issue: dict[str, Any],
    repo_root: Path,
    classification_root: Path,
    archive_root: Path,
    device: str,
) -> dict[str, Any]:
    job_dir = Path(issue["job_dir"])
    seed, outer, model, stage, inner = parse_job(job_dir, classification_root)
    relative_job = job_dir.relative_to(classification_root)
    archive_dir = archive_root / relative_job
    archive_job(job_dir, archive_dir)

    trainer = repo_root / "experiments/associate_editor_revision_v1/train_nested_classification.py"
    command = [
        sys.executable,
        str(trainer),
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(classification_root),
        "--model",
        model,
        "--seed",
        str(seed),
        "--outer",
        str(outer),
        "--stage",
        stage,
        "--device",
        device,
        "--force",
    ]
    if inner is not None:
        command.extend(["--inner", str(inner)])
    print("REPAIR " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)

    prediction_path = job_dir / "predictions.csv"
    frame = pd.read_csv(prediction_path, usecols=["image", *PROBABILITY_COLUMNS])
    if not np.isfinite(frame[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError(f"Repair did not produce finite probabilities: {prediction_path}")
    repaired = dict(issue)
    repaired.update(
        {
            "seed": seed,
            "outer": outer,
            "model": model,
            "stage": stage,
            "inner": inner,
            "archive_dir": str(archive_dir),
            "repaired_prediction_sha256": sha256(prediction_path),
            "repaired_complete_sha256": sha256(job_dir / "complete.json"),
        }
    )
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and repair non-finite classification prediction rows.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = repo_root / "outputs/associate_editor_revision_v1"
    classification_root = output_root / "classification"
    issues = scan_predictions(classification_root)
    print(f"PREDICTION_INTEGRITY_SCAN issue_files={len(issues)} bad_rows={sum(item['bad_rows'] for item in issues)}", flush=True)
    if args.scan_only:
        for issue in issues:
            print(json.dumps(issue, sort_keys=True), flush=True)
        return

    integrity_path = output_root / "classification_prediction_integrity.json"
    if integrity_path.exists() and not issues:
        print(f"SKIP verified prediction integrity {integrity_path}", flush=True)
        return

    started = time.time()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    archive_root = output_root / "nonfinite_prediction_repairs" / run_id
    repairs = [
        repair_issue(issue, repo_root, classification_root, archive_root, args.device)
        for issue in issues
    ]
    remaining = scan_predictions(classification_root)
    if remaining:
        raise FloatingPointError(f"Prediction integrity repair left {len(remaining)} affected files.")
    write_json(
        integrity_path,
        {
            "status": "complete",
            "initial_issue_files": len(issues),
            "initial_bad_rows": sum(item["bad_rows"] for item in issues),
            "repairs": repairs,
            "archive_root": str(archive_root) if issues else None,
            "elapsed_seconds": time.time() - started,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )
    print(f"PREDICTION_INTEGRITY_COMPLETE repaired_files={len(repairs)}", flush=True)


if __name__ == "__main__":
    main()
