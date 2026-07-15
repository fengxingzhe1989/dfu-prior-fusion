#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LEGACY_REPO_ROOTS: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PriorRecord:
    row: dict[str, str]
    image: str
    image_path: Path


class DFUC2021PriorDataset(Dataset):
    def __init__(self, records: list[PriorRecord], transform: A.Compose) -> None:
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
        original_height, original_width = image.shape[:2]
        transformed = self.transform(image=image)
        return {
            "image": transformed["image"].float(),
            "image_name": record.image,
            "image_path": str(record.image_path),
            "original_width": original_width,
            "original_height": original_height,
            "row": record.row,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DFUC2021 soft lesion priors with the DFUC2022 segmenter.")
    parser.add_argument(
        "--seg-config",
        type=Path,
        default=Path("experiments/dfuc2022_seg_baseline/configs/dfuc2022_seg_deeplabv3p_cv5_v1.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_priors"),
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_summary.json"),
    )
    parser.add_argument(
        "--preview-path",
        type=Path,
        default=Path("outputs/figures/cross_task_prior_transfer_dfuc2021_prior_preview.png"),
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--fallback-max-threshold", type=float, default=0.15)
    parser.add_argument("--fallback-area-threshold", type=float, default=0.01)
    parser.add_argument("--bbox-expand-ratio", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-preview", type=int, default=16)
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


def rel_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def load_config(path: Path, repo_root: Path) -> dict[str, Any]:
    with resolve_path(path, repo_root).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_records(manifest_path: Path, repo_root: Path, limit: int | None) -> list[PriorRecord]:
    manifest_path = resolve_path(manifest_path, repo_root)
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    records = []
    for row in rows:
        image_name = row["image"]
        image_path = resolve_path(row["image_path"], repo_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing DFUC2021 image: {image_path}")
        records.append(PriorRecord(row=row, image=image_name, image_path=image_path))
    return records


def build_transform(input_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=input_size, width=input_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_segmentation_model(config: dict[str, Any]) -> nn.Module:
    model_config = config["model"]
    architecture = str(model_config["architecture"]).lower()
    if architecture != "deeplabv3plus":
        raise ValueError(f"Unsupported segmentation architecture: {architecture}")
    return smp.DeepLabV3Plus(
        encoder_name=str(model_config["backbone"]),
        encoder_weights=None,
        in_channels=int(model_config["in_channels"]),
        classes=int(model_config["num_classes"]),
    )


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device, repo_root: Path) -> dict[str, Any]:
    checkpoint_path = resolve_path(checkpoint_path, repo_root)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("model_state")
    if state is None:
        raise KeyError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    return checkpoint


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


def configure_runtime(device: torch.device) -> None:
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


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
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + w + pad_x)
    y2 = min(image_height, y + h + pad_y)
    return x1, y1, x2, y2


def prior_stats(
    prior: np.ndarray,
    threshold: float,
    fallback_max_threshold: float,
    fallback_area_threshold: float,
    bbox_expand_ratio: float,
) -> dict[str, Any]:
    height, width = prior.shape
    binary = (prior >= threshold).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    largest_area = 0
    bbox = (0, 0, width, height)
    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(np.argmax(component_areas)) + 1
        largest_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
        bbox = expanded_bbox(
            int(stats[largest_idx, cv2.CC_STAT_LEFT]),
            int(stats[largest_idx, cv2.CC_STAT_TOP]),
            int(stats[largest_idx, cv2.CC_STAT_WIDTH]),
            int(stats[largest_idx, cv2.CC_STAT_HEIGHT]),
            width,
            height,
            bbox_expand_ratio,
        )

    prior_mean = float(np.mean(prior))
    prior_max = float(np.max(prior))
    prior_area_ratio = float(np.mean(binary))
    largest_area_ratio = float(largest_area / float(width * height))
    fallback = bool(prior_max < fallback_max_threshold or largest_area_ratio < fallback_area_threshold)
    if fallback:
        bbox = (0, 0, width, height)

    confidence = prior_max * min(1.0, largest_area_ratio / max(fallback_area_threshold, 1e-6))
    x1, y1, x2, y2 = bbox
    return {
        "prior_mean": prior_mean,
        "prior_max": prior_max,
        "prior_area_ratio": prior_area_ratio,
        "largest_component_area_ratio": largest_area_ratio,
        "bbox_x1": int(x1),
        "bbox_y1": int(y1),
        "bbox_x2": int(x2),
        "bbox_y2": int(y2),
        "bbox_area_ratio": float(((x2 - x1) * (y2 - y1)) / float(width * height)),
        "prior_confidence": float(confidence),
        "prior_fallback": int(fallback),
    }


def save_preview(previews: list[tuple[str, Path, np.ndarray]], output_path: Path) -> None:
    if not previews:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4
    rows = int(np.ceil(len(previews) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for index, (image_name, image_path, prior) in enumerate(previews):
        ax = axes[index // cols, index % cols]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (prior.shape[1], prior.shape[0]), interpolation=cv2.INTER_LINEAR)
        ax.imshow(image)
        ax.imshow(prior, cmap="jet", alpha=0.38, vmin=0.0, vmax=1.0)
        ax.set_title(image_name)
        ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    device = pick_device(args.device)
    amp_enabled = resolve_amp_enabled(args.amp, device)
    configure_runtime(device)

    seg_config = load_config(args.seg_config, repo_root)
    model = build_segmentation_model(seg_config).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device, repo_root)
    model.eval()

    records = load_records(args.manifest, repo_root, args.limit)
    dataset = DFUC2021PriorDataset(records, build_transform(args.input_size))
    loader = build_dataloader(dataset, args.batch_size, args.num_workers, device)

    output_dir = resolve_path(args.output_dir, repo_root)
    meta_csv = resolve_path(args.meta_csv, repo_root)
    summary_json = resolve_path(args.summary_json, repo_root)
    preview_path = resolve_path(args.preview_path, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: list[dict[str, Any]] = []
    previews: list[tuple[str, Path, np.ndarray]] = []

    print(
        f"Generating DFUC2021 priors records={len(records)} device={device} amp={amp_enabled} "
        f"checkpoint={resolve_path(args.checkpoint, repo_root)}",
        flush=True,
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc="dfuc2021 priors", dynamic_ncols=True):
            images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            with autocast_context(device, amp_enabled):
                logits = model(images)
                probs = torch.sigmoid(logits)
            priors = probs.detach().float().cpu().numpy()[:, 0]

            rows = batch["row"]
            image_names = list(batch["image_name"])
            image_paths = [Path(value) for value in batch["image_path"]]
            original_widths = batch["original_width"].tolist()
            original_heights = batch["original_height"].tolist()

            for index, image_name in enumerate(image_names):
                prior_path = output_dir / f"{Path(image_name).stem}.png"
                if args.skip_existing and prior_path.exists():
                    saved = cv2.imread(str(prior_path), cv2.IMREAD_GRAYSCALE)
                    if saved is None:
                        raise FileNotFoundError(f"Failed to read existing prior: {prior_path}")
                    prior = saved.astype(np.float32) / 255.0
                else:
                    prior = np.clip(priors[index], 0.0, 1.0)
                    prior_u8 = np.round(prior * 255.0).astype(np.uint8)
                    cv2.imwrite(str(prior_path), prior_u8)

                stats = prior_stats(
                    prior=prior,
                    threshold=args.threshold,
                    fallback_max_threshold=args.fallback_max_threshold,
                    fallback_area_threshold=args.fallback_area_threshold,
                    bbox_expand_ratio=args.bbox_expand_ratio,
                )
                source_row = {key: rows[key][index] for key in rows}
                meta_rows.append(
                    {
                        **source_row,
                        "prior_path": rel_to_repo(prior_path, repo_root),
                        "original_width": int(original_widths[index]),
                        "original_height": int(original_heights[index]),
                        "prior_width": int(prior.shape[1]),
                        "prior_height": int(prior.shape[0]),
                        **stats,
                    }
                )
                if len(previews) < args.save_preview:
                    previews.append((image_name, image_paths[index], prior.copy()))

    write_csv(meta_csv, meta_rows)
    save_preview(previews, preview_path)

    fallback_count = sum(int(row["prior_fallback"]) for row in meta_rows)
    summary = {
        "records": len(meta_rows),
        "source_manifest": rel_to_repo(resolve_path(args.manifest, repo_root), repo_root),
        "segmentation_checkpoint": rel_to_repo(resolve_path(args.checkpoint, repo_root), repo_root),
        "segmentation_checkpoint_fold": checkpoint.get("fold"),
        "segmentation_checkpoint_epoch": checkpoint.get("epoch"),
        "output_dir": rel_to_repo(output_dir, repo_root),
        "meta_csv": rel_to_repo(meta_csv, repo_root),
        "preview_path": rel_to_repo(preview_path, repo_root),
        "threshold": args.threshold,
        "fallback_max_threshold": args.fallback_max_threshold,
        "fallback_area_threshold": args.fallback_area_threshold,
        "fallback_count": fallback_count,
        "fallback_ratio": float(fallback_count / max(1, len(meta_rows))),
        "prior_mean_mean": float(np.mean([float(row["prior_mean"]) for row in meta_rows])) if meta_rows else 0.0,
        "prior_max_mean": float(np.mean([float(row["prior_max"]) for row in meta_rows])) if meta_rows else 0.0,
        "prior_area_ratio_mean": float(np.mean([float(row["prior_area_ratio"]) for row in meta_rows])) if meta_rows else 0.0,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
