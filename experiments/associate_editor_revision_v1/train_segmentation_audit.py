#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.model_selection import KFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, get_worker_info

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.dfuc2022_seg_baseline import train as seg  # noqa: E402


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


def seed_worker(_: int) -> None:
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)
    info = get_worker_info()
    transform = getattr(info.dataset, "transform", None) if info is not None else None
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


def build_loader(dataset: Dataset, shuffle: bool, seed: int, device: torch.device, config: dict[str, Any]) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    transform = getattr(dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    workers = int(config["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
        worker_init_fn=seed_worker if workers > 0 else None,
    )


def split_records(
    records: Sequence[seg.SampleRecord], outer: int, split_seed: int
) -> tuple[list[seg.SampleRecord], list[seg.SampleRecord], list[seg.SampleRecord]]:
    outer_test = [record for record in records if record.fold == outer]
    candidates = [record for record in records if record.fold != outer]
    splitter = KFold(n_splits=5, shuffle=True, random_state=split_seed + outer)
    fit_indices, stop_indices = next(splitter.split(candidates))
    fit = [candidates[index] for index in fit_indices]
    stop = [candidates[index] for index in stop_indices]
    if set(record.image_id for record in fit) & set(record.image_id for record in stop + outer_test):
        raise RuntimeError("Segmentation split leakage detected.")
    if set(record.image_id for record in stop) & set(record.image_id for record in outer_test):
        raise RuntimeError("Segmentation stop/test leakage detected.")
    return fit, stop, outer_test


@torch.no_grad()
def evaluate_rows(model: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images, targets, image_ids = seg.move_batch_to_device(batch, device)
        with seg.autocast_context(device, amp):
            probabilities = torch.sigmoid(model(images))
        predictions = (probabilities >= 0.5).float().flatten(1)
        targets = (targets >= 0.5).float().flatten(1)
        intersection = (predictions * targets).sum(dim=1)
        pred_sum = predictions.sum(dim=1)
        target_sum = targets.sum(dim=1)
        dice = (2 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (intersection + 1e-6) / (pred_sum + target_sum - intersection + 1e-6)
        precision = (intersection + 1e-6) / (pred_sum + 1e-6)
        recall = (intersection + 1e-6) / (target_sum + 1e-6)
        for index, image_id in enumerate(image_ids):
            rows.append(
                {
                    "image_id": image_id,
                    "dice": float(dice[index].cpu()),
                    "iou": float(iou[index].cpu()),
                    "precision": float(precision[index].cpu()),
                    "recall": float(recall[index].cpu()),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else repo_root / args.protocol
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    job_dir = output_root / f"seed_{args.seed}" / f"outer_{args.outer}"
    complete_path = job_dir / "complete.json"
    if complete_path.exists() and not args.force:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if Path(payload["checkpoint_path"]).exists():
            print(f"SKIP completed {job_dir}", flush=True)
            return payload

    source_config = repo_root / protocol["source_configs"]["segmentation"]
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["seed"] = int(args.seed)
    job_seed = int(np.random.SeedSequence([args.seed, args.outer, 5]).generate_state(1)[0])
    random.seed(job_seed)
    np.random.seed(job_seed)
    torch.manual_seed(job_seed)
    torch.cuda.manual_seed_all(job_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    manifest = repo_root / config["data"]["split_manifest"]
    records = seg.load_records(manifest, repo_root)
    fit, stop, outer_test = split_records(records, args.outer, int(protocol["split_seed"]))
    device = seg.pick_device(args.device)
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    train_dataset = seg.DFU2022SegmentationDataset(fit, seg.build_transforms(config, train=True))
    stop_dataset = seg.DFU2022SegmentationDataset(stop, seg.build_transforms(config, train=False))
    train_loader = build_loader(train_dataset, True, job_seed, device, config)
    stop_loader = build_loader(stop_dataset, False, job_seed, device, config)

    model = seg.build_model(config).to(device)
    loss_fn = seg.DiceBCELoss(float(config["loss"]["dice_weight"]), float(config["loss"]["bce_weight"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else None
    job_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = job_dir / "model.pt"
    temporary_checkpoint = job_dir / "best.pt"
    history_path = job_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    total_epochs = int(config["training"]["epochs"] if args.max_epochs is None else args.max_epochs)
    patience = min(int(config["training"]["early_stopping_patience"]), total_epochs)
    best_epoch = 0
    best_dice = -float("inf")
    started = time.time()
    print(
        f"START segmentation seed={args.seed} outer={args.outer} fit={len(fit)} stop={len(stop)} test={len(outer_test)}",
        flush=True,
    )
    for epoch in range(total_epochs):
        lr = seg.compute_epoch_lr(
            float(config["optimizer"]["lr"]),
            float(config["scheduler"]["min_lr"]),
            epoch,
            int(config["training"]["epochs"]),
            int(config["scheduler"]["warmup_epochs"]),
        )
        seg.set_optimizer_lr(optimizer, lr)
        train_metrics = seg.train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, 1, amp, scaler, args.outer, epoch + 1, total_epochs
        )
        stop_metrics, _ = seg.evaluate(
            model, stop_loader, loss_fn, device, 0, amp, args.outer, epoch + 1, total_epochs
        )
        if float(stop_metrics["dice"]) > best_dice:
            best_dice = float(stop_metrics["dice"])
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "seed": args.seed,
                    "job_seed": job_seed,
                    "outer_fold": args.outer,
                    "best_epoch": best_epoch,
                    "best_stop_metrics": stop_metrics,
                },
                temporary_checkpoint,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"epoch": epoch + 1, "lr": lr, "train_metrics": train_metrics, "stop_metrics": stop_metrics},
                    sort_keys=True,
                )
                + "\n"
            )
        print(f"EPOCH {epoch + 1}/{total_epochs} stop_dice={stop_metrics['dice']:.6f} best={best_dice:.6f}", flush=True)
        if epoch + 1 - best_epoch >= patience:
            break

    checkpoint = torch.load(temporary_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    temporary_checkpoint.replace(checkpoint_path)

    summaries: dict[str, Any] = {}
    for width, height in protocol["segmentation_audit"]["evaluation_grids_width_height"]:
        evaluation_config = yaml.safe_load(yaml.safe_dump(config))
        evaluation_config["data"]["input_width"] = int(width)
        evaluation_config["data"]["input_height"] = int(height)
        dataset = seg.DFU2022SegmentationDataset(outer_test, seg.build_transforms(evaluation_config, train=False))
        loader = build_loader(dataset, False, job_seed, device, evaluation_config)
        rows = evaluate_rows(model, loader, device, amp)
        for row in rows:
            row.update({"seed": args.seed, "outer_fold": args.outer, "width": width, "height": height})
        prediction_path = job_dir / f"outer_metrics_{width}x{height}.csv"
        write_rows(prediction_path, rows)
        summaries[f"{width}x{height}"] = {
            "n": len(rows),
            "dice": float(np.mean([row["dice"] for row in rows])),
            "iou": float(np.mean([row["iou"] for row in rows])),
            "precision": float(np.mean([row["precision"] for row in rows])),
            "recall": float(np.mean([row["recall"] for row in rows])),
            "path": str(prediction_path),
            "sha256": sha256(prediction_path),
        }

    payload = {
        "status": "complete",
        "seed": args.seed,
        "job_seed": job_seed,
        "outer_fold": args.outer,
        "fit_samples": len(fit),
        "stop_samples": len(stop),
        "outer_test_samples": len(outer_test),
        "best_epoch": best_epoch,
        "best_stop_dice": best_dice,
        "outer_metrics": summaries,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "source_config": str(source_config),
        "source_config_sha256": sha256(source_config),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "elapsed_seconds": time.time() - started,
    }
    write_json(complete_path, payload)
    print(f"COMPLETE {job_dir}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen nested segmentation audit job.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1/segmentation"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_job(args)


if __name__ == "__main__":
    main()
