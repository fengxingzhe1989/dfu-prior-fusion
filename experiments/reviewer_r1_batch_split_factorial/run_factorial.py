#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml


EXPERIMENT_VERSION = "reviewer_r1_batch_split_factorial_v1"
BASE_SEED = 20260430
# Run the faster high-batch cells first so a complete controlled model × split
# comparison is available early, then finish the small-batch sensitivity cells.
BATCH_SIZES = (128, 16)
SPLITS = {
    "standard": "data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv",
    "grouped": (
        "data_processed/dfuc2021_cls_v1/similarity_grouped_splits/"
        "dfuc2021_cls_cv5_similarity_grouped_manifest.csv"
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    source_config: str
    runner: str
    train_script: str


MODELS = {
    "resnet50": ModelSpec(
        name="resnet50",
        source_config="experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml",
        runner="experiments/dfuc2021_cls_baseline/run_cv.py",
        train_script="experiments/dfuc2021_cls_baseline/train.py",
    ),
    "full_lpt": ModelSpec(
        name="full_lpt",
        source_config=(
            "experiments/cross_task_prior_transfer/configs/"
            "cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml"
        ),
        runner="experiments/cross_task_prior_transfer/run_lpt_cv.py",
        train_script="experiments/cross_task_prior_transfer/train_lpt.py",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reviewer 1 model × split × batch factorial.")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=list(SPLITS))
    parser.add_argument("--batch-sizes", nargs="+", type=int, choices=BATCH_SIZES, default=list(BATCH_SIZES))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / EXPERIMENT_VERSION,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume safely by skipping fold metrics already complete for the same cell.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def relative_to_root(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def cell_name(model: str, split: str, batch_size: int) -> str:
    return f"{EXPERIMENT_VERSION}_{model}_{split}_bs{batch_size}"


def resolved_config(
    source: dict[str, Any],
    model: str,
    split: str,
    batch_size: int,
    output_root: Path,
    root: Path,
) -> dict[str, Any]:
    name = cell_name(model, split, batch_size)
    config = json.loads(json.dumps(source))
    config["experiment_name"] = name
    config["seed"] = BASE_SEED
    config["data"]["split_manifest"] = SPLITS[split]
    config["training"]["epochs"] = 60
    config["training"]["batch_size"] = batch_size
    config["training"]["grad_accum_steps"] = 1
    config["training"]["num_workers"] = 4
    config["training"]["mixed_precision"] = True
    config["training"]["early_stopping_patience"] = 12

    cell_root = output_root / name
    config["outputs"] = {
        "checkpoint_dir": relative_to_root(cell_root / "checkpoints", root),
        "log_dir": relative_to_root(cell_root / "logs", root),
        "prediction_dir": relative_to_root(cell_root / "predictions", root),
    }
    config["naming"] = {
        "best_checkpoint": f"{name}_fold{{fold}}_best.pt",
        "last_checkpoint": f"{name}_fold{{fold}}_last.pt",
        "history_log": f"{name}_fold{{fold}}_history.jsonl",
        "predictions": f"{name}_fold{{fold}}_val_predictions.csv",
    }
    config["reviewer_ablation"] = {
        "version": EXPERIMENT_VERSION,
        "model": model,
        "split": split,
        "batch_size": batch_size,
        "fixed_settings": {
            "seed": BASE_SEED,
            "epochs": 60,
            "early_stopping_patience": 12,
            "mixed_precision": True,
            "num_workers": 4,
            "grad_accum_steps": 1,
        },
    }
    return config


def gpu_payload() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": properties.total_memory,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def provenance(root: Path, output_root: Path, selected_models: list[str]) -> dict[str, Any]:
    tracked_paths = {
        "factorial_runner": Path(__file__).resolve(),
        "factorial_summarizer": Path(__file__).resolve().parent / "summarize_factorial.py",
    }
    for model in selected_models:
        spec = MODELS[model]
        tracked_paths[f"{model}_source_config"] = resolve(spec.source_config, root)
        tracked_paths[f"{model}_cv_runner"] = resolve(spec.runner, root)
        tracked_paths[f"{model}_train_script"] = resolve(spec.train_script, root)
    for split, path in SPLITS.items():
        tracked_paths[f"{split}_manifest"] = resolve(path, root)

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repo_root": str(root),
        "output_root": str(output_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "gpu": gpu_payload(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "sha256": {relative_to_root(path, root): sha256(path) for path in tracked_paths.values()},
    }


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_root = resolve(args.output_root, root)
    config_dir = output_root / "resolved_configs"
    output_root.mkdir(parents=True, exist_ok=True)

    cells: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        for model in args.models:
            spec = MODELS[model]
            source_path = resolve(spec.source_config, root)
            source = load_yaml(source_path)
            for split in args.splits:
                name = cell_name(model, split, batch_size)
                config = resolved_config(source, model, split, batch_size, output_root, root)
                config_path = config_dir / f"{name}.yaml"
                dump_yaml(config_path, config)
                cells.append(
                    {
                        "name": name,
                        "model": model,
                        "split": split,
                        "batch_size": batch_size,
                        "config": relative_to_root(config_path, root),
                        "runner": spec.runner,
                    }
                )

    run_manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "folds": args.folds,
        "skip_existing": args.skip_existing,
        "cells": cells,
    }
    dump_json(output_root / "run_manifest.json", run_manifest)
    dump_json(output_root / "provenance.json", provenance(root, output_root, args.models))

    status: dict[str, Any] = {
        "experiment_version": EXPERIMENT_VERSION,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "state": "dry_run" if args.dry_run else "running",
        "cells": {},
    }
    dump_json(output_root / "run_status.json", status)

    for cell in cells:
        command = [
            sys.executable,
            str(resolve(cell["runner"], root)),
            "--config",
            str(resolve(cell["config"], root)),
            "--folds",
            *(str(fold) for fold in args.folds),
            "--device",
            "cuda",
            "--amp",
            "on",
            "--batch-size",
            str(cell["batch_size"]),
            "--num-workers",
            "4",
        ]
        if args.skip_existing:
            command.append("--skip-existing")

        print(f"[factorial] {cell['name']}", flush=True)
        print("[factorial] command:", " ".join(command), flush=True)
        if args.dry_run:
            status["cells"][cell["name"]] = {"state": "planned", "command": command}
            dump_json(output_root / "run_status.json", status)
            continue

        status["cells"][cell["name"]] = {
            "state": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": command,
        }
        dump_json(output_root / "run_status.json", status)
        try:
            subprocess.run(command, cwd=root, check=True)
        except BaseException as exc:
            status["state"] = "failed"
            status["cells"][cell["name"]].update(
                {
                    "state": "failed",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "error": repr(exc),
                }
            )
            dump_json(output_root / "run_status.json", status)
            raise
        status["cells"][cell["name"]].update(
            {
                "state": "completed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )
        dump_json(output_root / "run_status.json", status)

    status["state"] = "dry_run" if args.dry_run else "completed"
    status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump_json(output_root / "run_status.json", status)


if __name__ == "__main__":
    main()
