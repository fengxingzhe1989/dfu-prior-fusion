#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    image_id: str
    image_path: Path
    mask_path: Path
    fold: int


@dataclass
class FoldResult:
    fold: int
    train_samples: int
    val_samples: int
    best_epoch: int
    best_metrics: dict[str, float]
    best_checkpoint: str
    last_checkpoint: str
    history_path: str
    metrics_path: str


class DFU2022SegmentationDataset(Dataset):
    def __init__(self, records: list[SampleRecord], transform: A.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {record.mask_path}")
        # The annotation is semantically binary, but the stored PNG has anti-aliased edge values.
        # Threshold once at load time so the model sees a true binary target mask.
        mask = (mask >= 128).astype(np.float32)

        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"].float()
        mask_tensor = transformed["mask"]
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": record.image_id,
        }


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float, bce_weight: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets, dim=(1, 2, 3))
        union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + self.eps) / (union + self.eps))
        return self.dice_weight * dice_loss.mean() + self.bce_weight * self.bce(logits, targets)


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


class MetricTracker:
    def __init__(self) -> None:
        self.storage = {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
        }

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        target_bin = (targets >= 0.5).float()

        preds_flat = preds.flatten(1)
        targets_flat = target_bin.flatten(1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        pred_sum = preds_flat.sum(dim=1)
        target_sum = targets_flat.sum(dim=1)

        precision = (intersection + 1e-6) / (pred_sum + 1e-6)
        recall = (intersection + 1e-6) / (target_sum + 1e-6)
        dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
        iou = (intersection + 1e-6) / (pred_sum + target_sum - intersection + 1e-6)

        self.storage["dice"].extend(dice.detach().cpu().tolist())
        self.storage["iou"].extend(iou.detach().cpu().tolist())
        self.storage["precision"].extend(precision.detach().cpu().tolist())
        self.storage["recall"].extend(recall.detach().cpu().tolist())

    def compute(self) -> dict[str, float]:
        return {
            name: float(np.mean(values)) if values else 0.0
            for name, values in self.storage.items()
        }


def format_mps_memory(device: torch.device) -> str:
    if device.type != "mps":
        return "-"
    try:
        current = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver = torch.mps.driver_allocated_memory() / (1024 ** 3)
        return f"{current:.2f}/{driver:.2f}G"
    except Exception:
        return "n/a"


def format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "-"
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        return f"{allocated:.2f}/{reserved:.2f}/{total:.2f}G"
    except Exception:
        return "n/a"


def format_device_memory(device: torch.device) -> str:
    if device.type == "cuda":
        return format_cuda_memory(device)
    if device.type == "mps":
        return format_mps_memory(device)
    return "-"


def autocast_context(device: torch.device, amp_enabled: bool):
    if not amp_enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DFUC2022 segmentation baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/dfuc2022_seg_deeplabv3p_cv5_v1.yaml",
    )
    parser.add_argument("--fold", type=int, default=None, help="Single fold to train.")
    parser.add_argument("--all-folds", action="store_true", help="Train every fold in the manifest.")
    parser.add_argument("--epochs", type=int, default=None, help="Override the config epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override the config batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Override gradient accumulation.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--amp",
        choices=("auto", "on", "off"),
        default="auto",
        help="Automatic mixed precision. `auto` enables AMP only on CUDA.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Debug limit for train samples.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Debug limit for val samples.")
    parser.add_argument("--save-val-figures", type=int, default=4, help="How many validation examples to render on best epoch.")
    parser.add_argument("--disable-pretrained", action="store_true", help="Disable ImageNet encoder weights for debugging.")
    return parser.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if args.fold is not None:
        config["data"]["current_fold"] = args.fold
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.grad_accum_steps is not None:
        config["training"]["grad_accum_steps"] = args.grad_accum_steps
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["seed"] = args.seed
    if args.disable_pretrained:
        config["model"]["pretrained"] = False
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | Path, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_root / path


def load_records(manifest_path: Path, repo_root: Path) -> list[SampleRecord]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    records: list[SampleRecord] = []
    for row in rows:
        records.append(
            SampleRecord(
                image_id=row["image_id"],
                image_path=resolve_path(row["image_path"], repo_root),
                mask_path=resolve_path(row["mask_path"], repo_root),
                fold=int(row["fold"]),
            )
        )
    return records


def split_records(
    records: list[SampleRecord],
    fold: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    train_records = [record for record in records if record.fold != fold]
    val_records = [record for record in records if record.fold == fold]

    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]
    return train_records, val_records


def build_transforms(config: dict[str, Any], train: bool) -> A.Compose:
    input_width = int(config["data"]["input_width"])
    input_height = int(config["data"]["input_height"])
    aug = config["augmentation"]

    transforms: list[Any] = []
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=float(aug["horizontal_flip_p"])),
                A.VerticalFlip(p=float(aug["vertical_flip_p"])),
                A.Affine(
                    scale=(float(aug["scale_min"]), float(aug["scale_max"])),
                    rotate=(-float(aug["rotation_deg"]), float(aug["rotation_deg"])),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    fill=0,
                    fill_mask=0,
                    p=0.6,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=float(aug["brightness_contrast"]),
                    contrast_limit=float(aug["brightness_contrast"]),
                    p=0.5,
                ),
            ]
        )

    transforms.extend(
        [
            A.Resize(height=input_height, width=input_width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config["model"]["architecture"]).lower()
    if architecture != "deeplabv3plus":
        raise ValueError(f"Unsupported architecture: {architecture}")

    encoder_weights = "imagenet" if bool(config["model"]["pretrained"]) else None
    return smp.DeepLabV3Plus(
        encoder_name=str(config["model"]["backbone"]),
        encoder_weights=encoder_weights,
        in_channels=int(config["model"]["in_channels"]),
        classes=int(config["model"]["num_classes"]),
    )


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS, but torch.backends.mps.is_available() is False.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_enabled(requested: str, device: torch.device) -> bool:
    if requested == "off":
        return False
    if requested == "on":
        if device.type != "cuda":
            raise RuntimeError("AMP is only enabled in this script for CUDA devices.")
        return True
    return device.type == "cuda"


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
    )


def compute_epoch_lr(base_lr: float, min_lr: float, epoch_index: int, total_epochs: int, warmup_epochs: int) -> float:
    if total_epochs <= 1:
        return base_lr

    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)

    cosine_steps = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / cosine_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    masks = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    image_ids = list(batch["image_id"])
    return images, masks, image_ids


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_accum_steps: int,
    amp_enabled: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    fold: int,
    epoch_index: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train()
    loss_meter = RunningAverage()
    metric_tracker = MetricTracker()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"fold {fold} train {epoch_index:03d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(progress, start=1):
        images, masks, _ = move_batch_to_device(batch, device)
        context = autocast_context(device, amp_enabled)
        with context:
            logits = model(images)
            loss = loss_fn(logits, masks)
            scaled_loss = loss / grad_accum_steps

        if scaler is not None and amp_enabled:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if scaler is not None and amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item(), n=images.size(0))
        metric_tracker.update(logits.detach(), masks.detach())

        if step == 1 or step % 10 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{loss_meter.average:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                device_mem=format_device_memory(device),
            )

    progress.close()

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.average
    return metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_examples: int,
    amp_enabled: bool,
    fold: int,
    epoch_index: int,
    total_epochs: int,
) -> tuple[dict[str, float], list[dict[str, torch.Tensor | str]]]:
    model.eval()
    loss_meter = RunningAverage()
    metric_tracker = MetricTracker()
    examples: list[dict[str, torch.Tensor | str]] = []
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"fold {fold} val   {epoch_index:03d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for step, batch in enumerate(progress, start=1):
            images, masks, image_ids = move_batch_to_device(batch, device)
            context = autocast_context(device, amp_enabled)
            with context:
                logits = model(images)
                loss = loss_fn(logits, masks)
            loss_meter.update(loss.item(), n=images.size(0))
            metric_tracker.update(logits, masks)

            if len(examples) < max_examples:
                probs = torch.sigmoid(logits)
                remaining = max_examples - len(examples)
                for idx in range(min(remaining, images.size(0))):
                    examples.append(
                        {
                            "image": images[idx].detach().cpu(),
                            "target": masks[idx].detach().cpu(),
                            "prediction": probs[idx].detach().cpu(),
                            "image_id": image_ids[idx],
                        }
                    )

            if step == 1 or step % 10 == 0 or step == len(loader):
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{loss_meter.average:.4f}",
                    device_mem=format_device_memory(device),
                )

    progress.close()

    metrics = metric_tracker.compute()
    metrics["loss"] = loss_meter.average
    return metrics, examples


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.numpy().transpose(1, 2, 0)
    image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    image = np.clip(image, 0.0, 1.0)
    return image


def render_validation_examples(examples: list[dict[str, torch.Tensor | str]], output_path: Path) -> None:
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, example in enumerate(examples):
        image = denormalize_image(example["image"])  # type: ignore[arg-type]
        target = example["target"].numpy().squeeze()  # type: ignore[union-attr]
        prediction = example["prediction"].numpy().squeeze()  # type: ignore[union-attr]
        pred_bin = (prediction >= 0.5).astype(np.float32)

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{example['image_id']} | image")
        axes[row, 1].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title("target")
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(pred_bin, cmap="jet", alpha=0.35, vmin=0.0, vmax=1.0)
        axes[row, 2].set_title("prediction")

        for col in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def checkpoint_paths(config: dict[str, Any], fold: int, repo_root: Path) -> dict[str, Path]:
    checkpoint_dir = resolve_path(config["outputs"]["checkpoint_dir"], repo_root)
    log_dir = resolve_path(config["outputs"]["log_dir"], repo_root)
    figure_dir = resolve_path(config["outputs"]["figure_dir"], repo_root)
    experiment_name = str(config["experiment_name"])

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    best_name = str(config["naming"]["best_checkpoint"]).format(fold=fold)
    last_name = str(config["naming"]["last_checkpoint"]).format(fold=fold)

    return {
        "best_checkpoint": checkpoint_dir / best_name,
        "last_checkpoint": checkpoint_dir / last_name,
        "history": log_dir / f"{experiment_name}_fold{fold}_history.jsonl",
        "metrics": log_dir / f"{experiment_name}_fold{fold}_metrics.json",
        "summary": log_dir / f"{experiment_name}_cv_summary.json",
        "csv_summary": log_dir / f"{experiment_name}_cv_summary.csv",
        "tensorboard": log_dir / experiment_name / f"fold_{fold}",
        "best_figure": figure_dir / f"{experiment_name}_fold{fold}_best_examples.png",
        "last_figure": figure_dir / f"{experiment_name}_fold{fold}_last_examples.png",
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
    fold: int,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "fold": fold,
    }
    torch.save(payload, path)


def append_history(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_run_overview(
    config: dict[str, Any],
    fold: int,
    train_count: int,
    val_count: int,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": config["experiment_name"],
        "fold": fold,
        "train_samples": train_count,
        "val_samples": val_count,
        "seed": config["seed"],
        "device": str(device),
        "amp_enabled": amp_enabled,
        "batch_size": config["training"]["batch_size"],
        "grad_accum_steps": config["training"]["grad_accum_steps"],
        "num_workers": config["training"]["num_workers"],
        "epochs": config["training"]["epochs"],
        "model": config["model"],
        "loss": config["loss"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
    }


def train_fold(
    config: dict[str, Any],
    fold: int,
    args: argparse.Namespace,
    records: list[SampleRecord],
) -> FoldResult:
    device = pick_device(args.device)
    configure_runtime(device)
    set_seed(int(config["seed"]) + fold)

    train_records, val_records = split_records(records, fold, args.max_train_samples, args.max_val_samples)
    train_dataset = DFU2022SegmentationDataset(train_records, build_transforms(config, train=True))
    val_dataset = DFU2022SegmentationDataset(val_records, build_transforms(config, train=False))

    train_loader = build_dataloader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=True,
        seed=int(config["seed"]) + fold,
        device=device,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["num_workers"]),
        shuffle=False,
        seed=int(config["seed"]) + fold,
        device=device,
    )

    model = build_model(config).to(device)
    loss_fn = DiceBCELoss(
        dice_weight=float(config["loss"]["dice_weight"]),
        bce_weight=float(config["loss"]["bce_weight"]),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    amp_enabled = resolve_amp_enabled(args.amp, device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    repo_root = resolve_repo_root()
    paths = checkpoint_paths(config, fold, repo_root)
    writer = SummaryWriter(log_dir=str(paths["tensorboard"]))
    append_history(paths["history"], {"event": "run_start", **make_run_overview(config, fold, len(train_dataset), len(val_dataset), device, amp_enabled)})

    best_epoch = -1
    best_metrics: dict[str, float] = {"dice": -1.0}
    epochs_without_improvement = 0
    epochs_ran = 0
    total_epochs = int(config["training"]["epochs"])
    grad_accum_steps = int(config["training"]["grad_accum_steps"])
    patience = int(config["training"]["early_stopping_patience"])
    base_lr = float(config["optimizer"]["lr"])
    min_lr = float(config["scheduler"]["min_lr"])
    warmup_epochs = int(config["scheduler"]["warmup_epochs"])

    for epoch in range(total_epochs):
        epoch_index = epoch + 1
        epochs_ran = epoch_index
        current_lr = compute_epoch_lr(base_lr, min_lr, epoch, total_epochs, warmup_epochs)
        set_optimizer_lr(optimizer, current_lr)

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_accum_steps=grad_accum_steps,
            amp_enabled=amp_enabled,
            scaler=scaler,
            fold=fold,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )
        val_metrics, val_examples = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            max_examples=max(0, int(args.save_val_figures)),
            amp_enabled=amp_enabled,
            fold=fold,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )

        history_row = {
            "event": "epoch_end",
            "epoch": epoch_index,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        append_history(paths["history"], history_row)

        writer.add_scalar("lr", current_lr, epoch_index)
        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for metric_name, value in metrics.items():
                writer.add_scalar(f"{split_name}/{metric_name}", value, epoch_index)

        improved = val_metrics["dice"] > best_metrics["dice"]
        if improved:
            best_epoch = epoch_index
            best_metrics = dict(val_metrics)
            save_checkpoint(paths["best_checkpoint"], model, optimizer, epoch_index, val_metrics, config, fold)
            render_validation_examples(val_examples, paths["best_figure"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"[fold {fold}] epoch {epoch_index:03d}/{total_epochs} "
            f"lr={current_lr:.6g} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"best_dice={best_metrics['dice']:.4f}",
            flush=True,
        )

        if epochs_without_improvement >= patience:
            print(f"[fold {fold}] early stopping at epoch {epoch_index} after {patience} stagnant epochs", flush=True)
            break

    last_val_metrics, last_examples = evaluate(
        model=model,
        loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        max_examples=max(0, int(args.save_val_figures)),
        amp_enabled=amp_enabled,
        fold=fold,
        epoch_index=epochs_ran if epochs_ran > 0 else 0,
        total_epochs=total_epochs,
    )
    save_checkpoint(paths["last_checkpoint"], model, optimizer, epochs_ran, last_val_metrics, config, fold)
    render_validation_examples(last_examples, paths["last_figure"])

    final_payload = {
        "fold": fold,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "last_metrics": last_val_metrics,
        "best_checkpoint": str(paths["best_checkpoint"]),
        "last_checkpoint": str(paths["last_checkpoint"]),
        "history_path": str(paths["history"]),
        "best_figure": str(paths["best_figure"]),
        "last_figure": str(paths["last_figure"]),
    }
    with paths["metrics"].open("w", encoding="utf-8") as handle:
        json.dump(final_payload, handle, ensure_ascii=False, indent=2)

    append_history(paths["history"], {"event": "run_end", **final_payload})
    writer.close()

    return FoldResult(
        fold=fold,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        best_epoch=best_epoch,
        best_metrics=best_metrics,
        best_checkpoint=str(paths["best_checkpoint"]),
        last_checkpoint=str(paths["last_checkpoint"]),
        history_path=str(paths["history"]),
        metrics_path=str(paths["metrics"]),
    )


def save_cv_summary(results: list[FoldResult], config: dict[str, Any], repo_root: Path) -> None:
    if not results:
        return

    paths = checkpoint_paths(config, results[0].fold, repo_root)
    metric_names = ("dice", "iou", "precision", "recall")

    summary_payload = {
        "experiment_name": config["experiment_name"],
        "fold_results": [
            {
                "fold": result.fold,
                "train_samples": result.train_samples,
                "val_samples": result.val_samples,
                "best_epoch": result.best_epoch,
                "best_metrics": result.best_metrics,
                "best_checkpoint": result.best_checkpoint,
                "last_checkpoint": result.last_checkpoint,
                "history_path": result.history_path,
                "metrics_path": result.metrics_path,
            }
            for result in results
        ],
        "aggregate": {
            metric: {
                "mean": float(np.mean([result.best_metrics[metric] for result in results])),
                "std": float(np.std([result.best_metrics[metric] for result in results])),
            }
            for metric in metric_names
        },
    }

    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    with paths["csv_summary"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fold", "train_samples", "val_samples", "best_epoch", *metric_names, "best_checkpoint"])
        for result in results:
            writer.writerow(
                [
                    result.fold,
                    result.train_samples,
                    result.val_samples,
                    result.best_epoch,
                    *(f"{result.best_metrics[name]:.6f}" for name in metric_names),
                    result.best_checkpoint,
                ]
            )
        writer.writerow(
            [
                "mean",
                "",
                "",
                "",
                *(f"{summary_payload['aggregate'][name]['mean']:.6f}" for name in metric_names),
                "",
            ]
        )
        writer.writerow(
            [
                "std",
                "",
                "",
                "",
                *(f"{summary_payload['aggregate'][name]['std']:.6f}" for name in metric_names),
                "",
            ]
        )


def main() -> None:
    args = parse_args()

    config = load_config(args.config, args)
    repo_root = resolve_repo_root()
    manifest_path = resolve_path(config["data"]["split_manifest"], repo_root)
    records = load_records(manifest_path, repo_root)

    if args.all_folds:
        folds = list(range(int(config["data"]["num_folds"])))
    else:
        folds = [int(config["data"]["current_fold"])]

    results = []
    for fold in folds:
        print(f"Starting fold {fold} with config {args.config}", flush=True)
        results.append(train_fold(config=config, fold=fold, args=args, records=records))

    save_cv_summary(results, config, repo_root)
    print("Completed folds:", ", ".join(str(result.fold) for result in results), flush=True)
    if len(results) == 1:
        metrics = results[0].best_metrics
        print(
            f"Fold {results[0].fold} best metrics | "
            f"dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
