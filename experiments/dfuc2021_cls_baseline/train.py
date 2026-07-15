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
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm.auto import tqdm


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
METRIC_NAMES = ("macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "accuracy")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEGACY_REPO_ROOTS: tuple[Path, ...] = ()


@dataclass(frozen=True)
class SampleRecord:
    image: str
    image_path: Path
    label: int
    label_name: str
    fold: int


class DFUC2021ClassificationDataset(Dataset):
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
        transformed = self.transform(image=image)
        return {
            "image": transformed["image"].float(),
            "label": torch.tensor(record.label, dtype=torch.long),
            "image_name": record.image,
        }


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFUC2021 classification baseline fold.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/dfuc2021_cls_resnet50_cv5_v1.yaml",
    )
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--disable-pretrained", action="store_true")
    return parser.parse_args()


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def remap_legacy_repo_path(path: Path, repo_root: Path) -> Path:
    for legacy_root in LEGACY_REPO_ROOTS:
        try:
            relative_path = path.relative_to(legacy_root)
        except ValueError:
            continue
        return repo_root / relative_path
    return path


def resolve_path(path_str: str | Path, repo_root: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        if path.exists():
            return path
        return remap_legacy_repo_path(path, repo_root)
    return repo_root / path


def load_config(config_path: Path, args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    config_path = resolve_path(config_path, repo_root)
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


def autocast_context(device: torch.device, amp_enabled: bool):
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()


def format_cuda_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return "-"
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    return f"{allocated:.2f}/{reserved:.2f}/{total:.2f}G"


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_records(manifest_path: Path, repo_root: Path) -> list[SampleRecord]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records: list[SampleRecord] = []
    for row in rows:
        records.append(
            SampleRecord(
                image=row["image"],
                image_path=resolve_path(row["image_path"], repo_root),
                label=int(row["label"]),
                label_name=row["label_name"],
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
    input_size = int(config["data"]["input_size"])
    aug = config["augmentation"]
    transforms: list[Any] = []
    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=float(aug["horizontal_flip_p"])),
                A.Affine(
                    scale=(float(aug["scale_min"]), float(aug["scale_max"])),
                    rotate=(-float(aug["rotation_deg"]), float(aug["rotation_deg"])),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    fill=0,
                    p=0.5,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=float(aug["brightness_contrast"]),
                    contrast_limit=float(aug["brightness_contrast"]),
                    p=0.35,
                ),
            ]
        )
    transforms.extend(
        [
            A.Resize(height=input_size, width=input_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_name = str(config["model"]["name"]).lower().replace("-", "_")
    num_classes = int(config["model"]["num_classes"])
    pretrained = bool(config["model"]["pretrained"])

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name in {"efficientnet_b0", "efficientnetb0"}:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name in {"efficientnet_v2_s", "efficientnetv2_s"}:
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    if model_name in {"convnext_tiny", "convnexttiny"}:
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    if model_name in {"swin_t", "swint", "swin_tiny"}:
        weights = models.Swin_T_Weights.DEFAULT if pretrained else None
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported classification model: {model_name}")


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


def compute_class_weights(records: list[SampleRecord], mode: str, device: torch.device) -> torch.Tensor | None:
    mode = mode.lower()
    if mode == "none":
        return None
    counts = np.bincount([record.label for record in records], minlength=len(CLASS_NAMES)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "balanced":
        weights = counts.sum() / (len(CLASS_NAMES) * counts)
    elif mode == "inverse_sqrt":
        weights = np.sqrt(counts.sum() / counts)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_epoch_lr(base_lr: float, min_lr: float, epoch_index: int, total_epochs: int, warmup_epochs: int) -> float:
    if total_epochs <= 1:
        return base_lr
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        return base_lr * float(epoch_index + 1) / float(warmup_epochs)
    cosine_steps = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / cosine_steps))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def append_history(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        average="macro",
        zero_division=0,
    )
    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ).astype(int).tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    grad_accum_steps: int,
    epoch_index: int,
    total_epochs: int,
    lr: float,
) -> float:
    model.train()
    average_loss = RunningAverage()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch_index + 1:03d}/{total_epochs}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device=device, non_blocking=device.type == "cuda")
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)
            scaled_loss = loss / grad_accum_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        average_loss.update(float(loss.detach().cpu()), n=images.shape[0])
        if step == 1 or step % 10 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                avg=f"{average_loss.average:.4f}",
                lr=f"{lr:.2e}",
                device_mem=format_cuda_memory(device),
            )
    return average_loss.average


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    total_epochs: int,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    average_loss = RunningAverage()
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    progress = tqdm(loader, desc=f"val   {epoch_index + 1:03d}/{total_epochs}", leave=False)
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        labels = batch["label"].to(device=device, non_blocking=device.type == "cuda")
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = loss_fn(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        average_loss.update(float(loss.detach().cpu()), n=images.shape[0])
        label_list = labels.detach().cpu().tolist()
        pred_list = preds.detach().cpu().tolist()
        prob_list = probs.detach().cpu().tolist()
        y_true.extend(int(value) for value in label_list)
        y_pred.extend(int(value) for value in pred_list)
        for image_name, true_label, pred_label, prob_values in zip(batch["image_name"], label_list, pred_list, prob_list, strict=True):
            rows.append(
                {
                    "image": image_name,
                    "true_label": CLASS_NAMES[int(true_label)],
                    "pred_label": CLASS_NAMES[int(pred_label)],
                    **{f"prob_{name}": float(prob_values[index]) for index, name in enumerate(CLASS_NAMES)},
                }
            )
        if step == 1 or step % 10 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{float(loss.detach().cpu()):.4f}",
                avg=f"{average_loss.average:.4f}",
                device_mem=format_cuda_memory(device),
            )
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = average_loss.average
    return average_loss.average, metrics, rows


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    fold: int,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "fold": fold,
            "epoch": epoch,
            "class_names": CLASS_NAMES,
            "metrics": metrics,
        },
        path,
    )


def train_fold(config: dict[str, Any], args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    fold = int(config["data"]["current_fold"])
    device = pick_device(args.device)
    amp_enabled = resolve_amp_enabled(args.amp, device) and bool(config["training"]["mixed_precision"])
    configure_runtime(device)
    set_seed(int(config["seed"]) + fold)

    manifest_path = resolve_path(config["data"]["split_manifest"], repo_root)
    records = load_records(manifest_path, repo_root)
    train_records, val_records = split_records(records, fold, args.max_train_samples, args.max_val_samples)
    if not train_records or not val_records:
        raise ValueError(f"Fold {fold} has empty train or val split.")

    train_dataset = DFUC2021ClassificationDataset(train_records, build_transforms(config, train=True))
    val_dataset = DFUC2021ClassificationDataset(val_records, build_transforms(config, train=False))
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
    class_weights = compute_class_weights(train_records, str(config["loss"]["class_weights"]), device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    checkpoint_dir = resolve_path(config["outputs"]["checkpoint_dir"], repo_root)
    log_dir = resolve_path(config["outputs"]["log_dir"], repo_root)
    prediction_dir = resolve_path(config["outputs"]["prediction_dir"], repo_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = checkpoint_dir / str(config["naming"]["best_checkpoint"]).format(fold=fold)
    last_checkpoint = checkpoint_dir / str(config["naming"]["last_checkpoint"]).format(fold=fold)
    history_path = log_dir / str(config["naming"]["history_log"]).format(fold=fold)
    predictions_path = prediction_dir / str(config["naming"]["predictions"]).format(fold=fold)
    metrics_path = log_dir / f"{config['experiment_name']}_fold{fold}_metrics.json"

    if history_path.exists():
        history_path.unlink()

    total_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    primary = str(config["metrics"]["primary"])
    best_epoch = 0
    best_score = -float("inf")
    best_metrics: dict[str, Any] | None = None
    best_predictions: list[dict[str, Any]] = []
    last_metrics: dict[str, Any] | None = None

    print(
        f"Starting {config['experiment_name']} fold {fold} "
        f"train={len(train_records)} val={len(val_records)} device={device} amp={amp_enabled}",
        flush=True,
    )

    for epoch_index in range(total_epochs):
        lr = compute_epoch_lr(
            base_lr=float(config["optimizer"]["lr"]),
            min_lr=float(config["scheduler"]["min_lr"]),
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            warmup_epochs=int(config["scheduler"]["warmup_epochs"]),
        )
        set_optimizer_lr(optimizer, lr)
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_accum_steps=int(config["training"]["grad_accum_steps"]),
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            lr=lr,
        )
        _, val_metrics, val_predictions = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
        )
        last_metrics = val_metrics
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "fold": fold,
            "epoch": epoch_index + 1,
            "lr": lr,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }
        append_history(history_path, row)

        current_score = float(val_metrics[primary])
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch_index + 1
            best_metrics = val_metrics
            best_predictions = val_predictions
            save_checkpoint(best_checkpoint, model, optimizer, config, fold, best_epoch, best_metrics)
            save_predictions(predictions_path, best_predictions)

        print(
            f"[{config['experiment_name']} fold {fold}] epoch {epoch_index + 1:03d}/{total_epochs} "
            f"lr={lr:.6g} train_loss={train_loss:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"best_{primary}={best_score:.4f}",
            flush=True,
        )

        if epoch_index + 1 - best_epoch >= patience:
            print(f"Early stopping fold {fold} at epoch {epoch_index + 1}; best epoch {best_epoch}.", flush=True)
            break

    if best_metrics is None or last_metrics is None:
        raise RuntimeError("Training ended without validation metrics.")

    save_checkpoint(last_checkpoint, model, optimizer, config, fold, epoch_index + 1, last_metrics)
    payload = {
        "experiment_name": config["experiment_name"],
        "model": config["model"]["name"],
        "fold": fold,
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "last_metrics": last_metrics,
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "history_path": str(history_path),
        "best_predictions": str(predictions_path),
        "class_names": CLASS_NAMES,
        "class_counts_train": {
            name: sum(1 for record in train_records if record.label_name == name)
            for name in CLASS_NAMES
        },
        "class_counts_val": {
            name: sum(1 for record in val_records if record.label_name == name)
            for name in CLASS_NAMES
        },
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"Completed {config['experiment_name']} fold {fold}: "
        f"best_epoch={best_epoch} macro_f1={best_metrics['macro_f1']:.4f}",
        flush=True,
    )
    return payload


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    config = load_config(args.config, args, repo_root)
    train_fold(config, args, repo_root)


if __name__ == "__main__":
    main()
