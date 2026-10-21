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
import torch.nn.functional as F
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
STATS_FIELDS = ("prior_mean", "prior_max", "prior_area_ratio", "prior_confidence")


@dataclass(frozen=True)
class PriorSampleRecord:
    image: str
    image_path: Path
    prior_path: Path
    label: int
    label_name: str
    infection_attr: int
    ischaemia_attr: int
    fold: int
    prior_stats: tuple[float, ...]


class DFUC2021LPTDataset(Dataset):
    def __init__(
        self,
        records: list[PriorSampleRecord],
        transform: A.Compose,
        roi_threshold: float,
        fallback_max_threshold: float,
        fallback_area_threshold: float,
        bbox_expand_ratio: float,
        roi_mode: str,
        stats_mode: str,
    ) -> None:
        self.records = records
        self.transform = transform
        self.roi_threshold = roi_threshold
        self.fallback_max_threshold = fallback_max_threshold
        self.fallback_area_threshold = fallback_area_threshold
        self.bbox_expand_ratio = bbox_expand_ratio
        self.roi_mode = roi_mode.lower()
        self.stats_mode = stats_mode.lower()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        prior_u8 = cv2.imread(str(record.prior_path), cv2.IMREAD_GRAYSCALE)
        if prior_u8 is None:
            raise FileNotFoundError(f"Failed to read prior: {record.prior_path}")
        prior = prior_u8.astype(np.float32) / 255.0
        if prior.shape[:2] != image.shape[:2]:
            prior = cv2.resize(prior, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.roi_mode == "prior_crop":
            roi = build_roi_image(
                image=image,
                prior=prior,
                threshold=self.roi_threshold,
                fallback_max_threshold=self.fallback_max_threshold,
                fallback_area_threshold=self.fallback_area_threshold,
                bbox_expand_ratio=self.bbox_expand_ratio,
            )
        elif self.roi_mode == "image":
            roi = image.copy()
        elif self.roi_mode == "prior_enhanced":
            roi = build_prior_enhanced_image(image, prior)
        else:
            raise ValueError(f"Unsupported ROI mode: {self.roi_mode}")
        if roi.shape[:2] != image.shape[:2]:
            roi = cv2.resize(roi, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        transformed = self.transform(image=image, roi=roi, prior=prior)
        prior_tensor = transformed["prior"]
        if prior_tensor.ndim == 2:
            prior_tensor = prior_tensor.unsqueeze(0)
        prior_tensor = prior_tensor.float()

        return {
            "image": transformed["image"].float(),
            "roi": transformed["roi"].float(),
            "prior": prior_tensor,
            "stats": self.build_stats_tensor(record),
            "label": torch.tensor(record.label, dtype=torch.long),
            "infection_attr": torch.tensor(record.infection_attr, dtype=torch.float32),
            "ischaemia_attr": torch.tensor(record.ischaemia_attr, dtype=torch.float32),
            "image_name": record.image,
        }

    def build_stats_tensor(self, record: PriorSampleRecord) -> torch.Tensor:
        stats = torch.tensor(record.prior_stats, dtype=torch.float32)
        if self.stats_mode == "learned":
            return stats
        if self.stats_mode == "zero":
            return torch.zeros_like(stats)
        raise ValueError(f"Unsupported stats mode: {self.stats_mode}")


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


class LPTNet(nn.Module):
    def __init__(self, config: dict[str, Any], stats_dim: int) -> None:
        super().__init__()
        model_config = config["model"]
        pretrained = bool(model_config["pretrained"])
        num_classes = int(model_config["num_classes"])
        self.prior_alpha = float(model_config["prior_alpha"])

        global_name = str(model_config["global_backbone"]).lower().replace("-", "_")
        if global_name != "resnet50":
            raise ValueError(f"Unsupported global backbone: {global_name}")
        global_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.global_stem = nn.Sequential(
            global_model.conv1,
            global_model.bn1,
            global_model.relu,
            global_model.maxpool,
            global_model.layer1,
            global_model.layer2,
            global_model.layer3,
            global_model.layer4,
        )
        global_dim = global_model.fc.in_features
        self.alignment_head = nn.Conv2d(global_dim, 1, kernel_size=1)

        self.use_lesion_branch = bool(model_config.get("use_lesion_branch", True))
        lesion_dim = 0
        if self.use_lesion_branch:
            lesion_name = str(model_config["lesion_backbone"]).lower().replace("-", "_")
            if lesion_name not in {"mobilenet_v3_small", "mobilenetv3_small"}:
                raise ValueError(f"Unsupported lesion backbone: {lesion_name}")
            lesion_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            lesion_dim = lesion_model.classifier[0].in_features
            lesion_model.classifier = nn.Identity()
            self.lesion_branch = lesion_model

        fusion_dim = global_dim + lesion_dim + stats_dim
        self.fusion = nn.Sequential(
            nn.Dropout(float(model_config["dropout"])),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(float(model_config["dropout"])),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Linear(128, num_classes)
        self.infection_head = nn.Linear(128, 1)
        self.ischaemia_head = nn.Linear(128, 1)

    def forward(
        self,
        image: torch.Tensor,
        roi: torch.Tensor,
        prior: torch.Tensor,
        stats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.global_stem(image)
        prior_small = F.interpolate(prior, size=features.shape[-2:], mode="bilinear", align_corners=False)
        guided_features = features * (1.0 + self.prior_alpha * prior_small)
        global_vec = F.adaptive_avg_pool2d(guided_features, output_size=1).flatten(1)
        fusion_parts = [global_vec]
        if self.use_lesion_branch:
            fusion_parts.append(self.lesion_branch(roi))
        fusion_parts.append(stats)
        fused = self.fusion(torch.cat(fusion_parts, dim=1))
        attention = torch.sigmoid(self.alignment_head(features))
        return {
            "class_logits": self.class_head(fused),
            "infection_logits": self.infection_head(fused).squeeze(1),
            "ischaemia_logits": self.ischaemia_head(fused).squeeze(1),
            "attention": attention,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFUC2021 lesion-prior transfer fold.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml",
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
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    return f"{allocated:.2f}/{reserved:.2f}/{total:.2f}G"


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_records(config: dict[str, Any], repo_root: Path) -> list[PriorSampleRecord]:
    manifest_path = resolve_path(config["data"]["split_manifest"], repo_root)
    prior_meta_path = resolve_path(config["data"]["prior_meta_csv"], repo_root)
    manifest_rows = read_csv(manifest_path)
    prior_rows = {row["image"]: row for row in read_csv(prior_meta_path)}

    records = []
    for row in manifest_rows:
        image_name = row["image"]
        if image_name not in prior_rows:
            raise KeyError(f"Missing prior metadata for image {image_name}")
        prior_row = prior_rows[image_name]
        label = int(row["label"])
        stats = tuple(float(prior_row[name]) for name in config["data"].get("stats_fields", STATS_FIELDS))
        records.append(
            PriorSampleRecord(
                image=image_name,
                image_path=resolve_path(row["image_path"], repo_root),
                prior_path=resolve_path(prior_row["prior_path"], repo_root),
                label=label,
                label_name=row["label_name"],
                infection_attr=int(row["infection"]),
                ischaemia_attr=int(row["ischaemia"]),
                fold=int(row["fold"]),
                prior_stats=stats,
            )
        )
    if bool(config.get("prior", {}).get("shuffle_prior", False)):
        seed = int(config["seed"]) + int(config["prior"].get("shuffle_seed_offset", 10000))
        records = shuffle_prior_assignments(records, seed)
    return records


def shuffle_prior_assignments(records: list[PriorSampleRecord], seed: int) -> list[PriorSampleRecord]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    return [
        PriorSampleRecord(
            image=record.image,
            image_path=record.image_path,
            prior_path=prior_record.prior_path,
            label=record.label,
            label_name=record.label_name,
            infection_attr=record.infection_attr,
            ischaemia_attr=record.ischaemia_attr,
            fold=record.fold,
            prior_stats=prior_record.prior_stats,
        )
        for record, prior_record in zip(records, shuffled, strict=True)
    ]


def split_records(
    records: list[PriorSampleRecord],
    fold: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[list[PriorSampleRecord], list[PriorSampleRecord]]:
    train_records = [record for record in records if record.fold != fold]
    val_records = [record for record in records if record.fold == fold]
    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]
    return train_records, val_records


def expanded_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    pad_x = int(round(w * expand_ratio))
    pad_y = int(round(h * expand_ratio))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(image_width, x + w + pad_x),
        min(image_height, y + h + pad_y),
    )


def build_roi_image(
    image: np.ndarray,
    prior: np.ndarray,
    threshold: float,
    fallback_max_threshold: float,
    fallback_area_threshold: float,
    bbox_expand_ratio: float,
) -> np.ndarray:
    height, width = prior.shape
    binary = (prior >= threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    prior_max = float(np.max(prior))

    if num_labels <= 1:
        largest_area_ratio = 0.0
        fallback = True
    else:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(component_areas)) + 1
        largest_area_ratio = float(stats[largest_idx, cv2.CC_STAT_AREA] / float(width * height))
        fallback = bool(prior_max < fallback_max_threshold or largest_area_ratio < fallback_area_threshold)

    if fallback:
        weight = prior[..., None].astype(np.float32)
        enhanced = image.astype(np.float32) * (0.15 + 0.85 * weight)
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    x1, y1, x2, y2 = expanded_bbox(
        int(stats[largest_idx, cv2.CC_STAT_LEFT]),
        int(stats[largest_idx, cv2.CC_STAT_TOP]),
        int(stats[largest_idx, cv2.CC_STAT_WIDTH]),
        int(stats[largest_idx, cv2.CC_STAT_HEIGHT]),
        width,
        height,
        bbox_expand_ratio,
    )
    return image[y1:y2, x1:x2].copy()


def build_prior_enhanced_image(image: np.ndarray, prior: np.ndarray) -> np.ndarray:
    weight = prior[..., None].astype(np.float32)
    enhanced = image.astype(np.float32) * (0.15 + 0.85 * weight)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


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
                    mask_interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101,
                    fill=0,
                    fill_mask=0,
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
            A.Resize(
                height=input_size,
                width=input_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_LINEAR,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms, additional_targets={"roi": "image", "prior": "mask"})


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


def compute_class_weights(records: list[PriorSampleRecord], mode: str, device: torch.device) -> torch.Tensor | None:
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


def compute_attr_pos_weights(records: list[PriorSampleRecord], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    infection_pos = max(1, sum(record.infection_attr for record in records))
    infection_neg = max(1, len(records) - infection_pos)
    ischaemia_pos = max(1, sum(record.ischaemia_attr for record in records))
    ischaemia_neg = max(1, len(records) - ischaemia_pos)
    return (
        torch.tensor(float(infection_neg / infection_pos), dtype=torch.float32, device=device),
        torch.tensor(float(ischaemia_neg / ischaemia_pos), dtype=torch.float32, device=device),
    )


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
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES)))).astype(int).tolist(),
    }


def compute_binary_metrics(y_true: list[int], y_pred: list[int], prefix: str) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        zero_division=0,
    )
    return {
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image": batch["image"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
        "roi": batch["roi"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
        "prior": batch["prior"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
        "stats": batch["stats"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
        "label": batch["label"].to(device=device, non_blocking=device.type == "cuda"),
        "infection_attr": batch["infection_attr"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
        "ischaemia_attr": batch["ischaemia_attr"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"),
    }


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    class_loss_fn: nn.Module,
    infection_loss_fn: nn.Module,
    ischaemia_loss_fn: nn.Module,
    loss_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    class_loss = class_loss_fn(outputs["class_logits"], batch["label"])
    infection_loss = infection_loss_fn(outputs["infection_logits"], batch["infection_attr"])
    ischaemia_loss = ischaemia_loss_fn(outputs["ischaemia_logits"], batch["ischaemia_attr"])
    prior_small = F.interpolate(batch["prior"], size=outputs["attention"].shape[-2:], mode="bilinear", align_corners=False)
    align_loss = F.l1_loss(outputs["attention"], prior_small)
    total = (
        float(loss_config["class_weight"]) * class_loss
        + float(loss_config["attribute_weight"]) * 0.5 * (infection_loss + ischaemia_loss)
        + float(loss_config["alignment_weight"]) * align_loss
    )
    return {
        "total": total,
        "class": class_loss.detach(),
        "infection": infection_loss.detach(),
        "ischaemia": ischaemia_loss.detach(),
        "alignment": align_loss.detach(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    infection_loss_fn: nn.Module,
    ischaemia_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    grad_accum_steps: int,
    epoch_index: int,
    total_epochs: int,
    lr: float,
    loss_config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    meters = {name: RunningAverage() for name in ("total", "class", "infection", "ischaemia", "alignment")}
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch_index + 1:03d}/{total_epochs}", leave=False)
    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp_enabled):
            outputs = model(batch["image"], batch["roi"], batch["prior"], batch["stats"])
            losses = compute_losses(outputs, batch, class_loss_fn, infection_loss_fn, ischaemia_loss_fn, loss_config)
            scaled_loss = losses["total"] / grad_accum_steps
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
        batch_size = int(batch["image"].shape[0])
        for name, meter in meters.items():
            meter.update(float(losses[name].detach().cpu()), n=batch_size)
        if step == 1 or step % 10 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{meters['total'].average:.4f}",
                cls=f"{meters['class'].average:.4f}",
                align=f"{meters['alignment'].average:.4f}",
                lr=f"{lr:.2e}",
                device_mem=format_cuda_memory(device),
            )
    return {name: meter.average for name, meter in meters.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    class_loss_fn: nn.Module,
    infection_loss_fn: nn.Module,
    ischaemia_loss_fn: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    epoch_index: int,
    total_epochs: int,
    loss_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    meters = {name: RunningAverage() for name in ("total", "class", "infection", "ischaemia", "alignment")}
    y_true: list[int] = []
    y_pred: list[int] = []
    infection_true: list[int] = []
    infection_pred: list[int] = []
    ischaemia_true: list[int] = []
    ischaemia_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    progress = tqdm(loader, desc=f"val   {epoch_index + 1:03d}/{total_epochs}", leave=False)
    for step, raw_batch in enumerate(progress, start=1):
        batch = move_batch(raw_batch, device)
        with autocast_context(device, amp_enabled):
            outputs = model(batch["image"], batch["roi"], batch["prior"], batch["stats"])
            losses = compute_losses(outputs, batch, class_loss_fn, infection_loss_fn, ischaemia_loss_fn, loss_config)
        probs = torch.softmax(outputs["class_logits"], dim=1)
        preds = torch.argmax(probs, dim=1)
        infection_probs = torch.sigmoid(outputs["infection_logits"])
        ischaemia_probs = torch.sigmoid(outputs["ischaemia_logits"])
        infection_preds = (infection_probs >= 0.5).long()
        ischaemia_preds = (ischaemia_probs >= 0.5).long()

        batch_size = int(batch["image"].shape[0])
        for name, meter in meters.items():
            meter.update(float(losses[name].detach().cpu()), n=batch_size)

        label_list = batch["label"].detach().cpu().tolist()
        pred_list = preds.detach().cpu().tolist()
        prob_list = probs.detach().cpu().tolist()
        infection_true_list = batch["infection_attr"].detach().cpu().long().tolist()
        infection_pred_list = infection_preds.detach().cpu().tolist()
        infection_prob_list = infection_probs.detach().cpu().tolist()
        ischaemia_true_list = batch["ischaemia_attr"].detach().cpu().long().tolist()
        ischaemia_pred_list = ischaemia_preds.detach().cpu().tolist()
        ischaemia_prob_list = ischaemia_probs.detach().cpu().tolist()

        y_true.extend(int(value) for value in label_list)
        y_pred.extend(int(value) for value in pred_list)
        infection_true.extend(int(value) for value in infection_true_list)
        infection_pred.extend(int(value) for value in infection_pred_list)
        ischaemia_true.extend(int(value) for value in ischaemia_true_list)
        ischaemia_pred.extend(int(value) for value in ischaemia_pred_list)

        for image_name, true_label, pred_label, prob_values, inf_prob, isc_prob in zip(
            raw_batch["image_name"],
            label_list,
            pred_list,
            prob_list,
            infection_prob_list,
            ischaemia_prob_list,
            strict=True,
        ):
            rows.append(
                {
                    "image": image_name,
                    "true_label": CLASS_NAMES[int(true_label)],
                    "pred_label": CLASS_NAMES[int(pred_label)],
                    **{f"prob_{name}": float(prob_values[index]) for index, name in enumerate(CLASS_NAMES)},
                    "prob_infection_attr": float(inf_prob),
                    "prob_ischaemia_attr": float(isc_prob),
                }
            )

        if step == 1 or step % 10 == 0 or step == len(loader):
            progress.set_postfix(
                loss=f"{meters['total'].average:.4f}",
                cls=f"{meters['class'].average:.4f}",
                align=f"{meters['alignment'].average:.4f}",
                device_mem=format_cuda_memory(device),
            )

    loss_values = {name: meter.average for name, meter in meters.items()}
    metrics = compute_metrics(y_true, y_pred)
    metrics.update(compute_binary_metrics(infection_true, infection_pred, "infection_attr"))
    metrics.update(compute_binary_metrics(ischaemia_true, ischaemia_pred, "ischaemia_attr"))
    metrics["loss"] = loss_values["total"]
    metrics["class_loss"] = loss_values["class"]
    metrics["alignment_loss"] = loss_values["alignment"]
    return loss_values, metrics, rows


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

    records = load_records(config, repo_root)
    train_records, val_records = split_records(records, fold, args.max_train_samples, args.max_val_samples)
    if not train_records or not val_records:
        raise ValueError(f"Fold {fold} has empty train or val split.")

    train_dataset = DFUC2021LPTDataset(
        train_records,
        build_transforms(config, train=True),
        roi_threshold=float(config["prior"]["roi_threshold"]),
        fallback_max_threshold=float(config["prior"]["fallback_max_threshold"]),
        fallback_area_threshold=float(config["prior"]["fallback_area_threshold"]),
        bbox_expand_ratio=float(config["prior"]["bbox_expand_ratio"]),
        roi_mode=str(config["prior"].get("roi_mode", "prior_crop")),
        stats_mode=str(config["prior"].get("stats_mode", "learned")),
    )
    val_dataset = DFUC2021LPTDataset(
        val_records,
        build_transforms(config, train=False),
        roi_threshold=float(config["prior"]["roi_threshold"]),
        fallback_max_threshold=float(config["prior"]["fallback_max_threshold"]),
        fallback_area_threshold=float(config["prior"]["fallback_area_threshold"]),
        bbox_expand_ratio=float(config["prior"]["bbox_expand_ratio"]),
        roi_mode=str(config["prior"].get("roi_mode", "prior_crop")),
        stats_mode=str(config["prior"].get("stats_mode", "learned")),
    )
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

    stats_dim = len(config["data"].get("stats_fields", STATS_FIELDS))
    model = LPTNet(config, stats_dim=stats_dim).to(device)
    class_weights = compute_class_weights(train_records, str(config["loss"]["class_weights"]), device)
    class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    infection_pos_weight, ischaemia_pos_weight = compute_attr_pos_weights(train_records, device)
    infection_loss_fn = nn.BCEWithLogitsLoss(pos_weight=infection_pos_weight)
    ischaemia_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ischaemia_pos_weight)
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
        f"train={len(train_records)} val={len(val_records)} device={device} amp={amp_enabled} "
        f"batch={config['training']['batch_size']} stats_dim={stats_dim}",
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
        train_losses = train_one_epoch(
            model=model,
            loader=train_loader,
            class_loss_fn=class_loss_fn,
            infection_loss_fn=infection_loss_fn,
            ischaemia_loss_fn=ischaemia_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            grad_accum_steps=int(config["training"]["grad_accum_steps"]),
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            lr=lr,
            loss_config=config["loss"],
        )
        _, val_metrics, val_predictions = evaluate(
            model=model,
            loader=val_loader,
            class_loss_fn=class_loss_fn,
            infection_loss_fn=infection_loss_fn,
            ischaemia_loss_fn=ischaemia_loss_fn,
            device=device,
            amp_enabled=amp_enabled,
            epoch_index=epoch_index,
            total_epochs=total_epochs,
            loss_config=config["loss"],
        )
        last_metrics = val_metrics
        append_history(
            history_path,
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fold": fold,
                "epoch": epoch_index + 1,
                "lr": lr,
                "train_losses": train_losses,
                "val_metrics": val_metrics,
            },
        )

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
            f"lr={lr:.6g} train_loss={train_losses['total']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"inf_f1={val_metrics['infection_attr_f1']:.4f} "
            f"isc_f1={val_metrics['ischaemia_attr_f1']:.4f} "
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
        "model": {
            "global_backbone": config["model"]["global_backbone"],
            "lesion_backbone": config["model"]["lesion_backbone"],
        },
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
        "stats_fields": list(config["data"].get("stats_fields", STATS_FIELDS)),
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
