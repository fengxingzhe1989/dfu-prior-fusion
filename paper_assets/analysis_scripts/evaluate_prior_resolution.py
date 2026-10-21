#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
REPO_ROOT = Path(__file__).resolve().parents[2]


class SegmentationDataset(Dataset):
    def __init__(self, root: Path, rows: pd.DataFrame, width: int, height: int) -> None:
        self.root = root
        self.rows = rows.reset_index(drop=True)
        self.width = width
        self.height = height

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[index]
        image = cv2.imread(str(self.root / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(row["image_path"] if image is None else row["mask_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        mask = (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
        return (
            torch.from_numpy(np.transpose(image, (2, 0, 1))).contiguous(),
            torch.from_numpy(mask[None, ...]).contiguous(),
        )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float | int]:
    storage = {"dice": [], "iou": [], "precision": [], "recall": []}
    model.eval()
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            preds = (torch.sigmoid(logits) >= 0.5).float().flatten(1)
            targets = (masks >= 0.5).float().flatten(1)
            intersection = (preds * targets).sum(dim=1)
            pred_sum = preds.sum(dim=1)
            target_sum = targets.sum(dim=1)
            storage["precision"].extend(((intersection + 1e-6) / (pred_sum + 1e-6)).cpu().tolist())
            storage["recall"].extend(((intersection + 1e-6) / (target_sum + 1e-6)).cpu().tolist())
            storage["dice"].extend(
                ((2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)).cpu().tolist()
            )
            storage["iou"].extend(
                ((intersection + 1e-6) / (pred_sum + target_sum - intersection + 1e-6)).cpu().tolist()
            )
    return {"samples": len(storage["dice"]), **{key: float(np.mean(values)) for key, values in storage.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data_processed/dfuc2022_seg_v1/splits/dfuc2022_seg_cv5_manifest.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/checkpoints/dfuc2022_seg_deeplabv3p_fold1_best.pt"),
    )
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/dfuc2022_prior_resolution_evaluation.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = pd.read_csv(manifest_path)
    rows = manifest.loc[manifest["fold"] == args.fold].copy()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "fold": args.fold,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_metrics": payload.get("metrics", {}),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "resolutions": {},
    }
    for label, width, height in (("training_grid_512x384", 512, 384), ("prior_inference_grid_224x224", 224, 224)):
        dataset = SegmentationDataset(root, rows, width=width, height=height)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
        results["resolutions"][label] = evaluate(model, loader, device)

    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
