#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ("none", "infection", "ischaemia", "both")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    config: str
    checkpoint_pattern: str
    target_layer: str


METHODS = {
    "resnet50": MethodSpec(
        name="ResNet50",
        config="experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml",
        checkpoint_pattern="outputs/checkpoints/dfuc2021_cls_resnet50_fold{fold}_best.pt",
        target_layer="layer4",
    ),
    "efficientnetv2s": MethodSpec(
        name="EfficientNetV2-S",
        config="experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_efficientnetv2s_cv5_v1.yaml",
        checkpoint_pattern="outputs/checkpoints/dfuc2021_cls_efficientnetv2s_fold{fold}_best.pt",
        target_layer="features",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM/prior overlap and occlusion analysis for DFUC2021 classifiers.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--methods", nargs="+", default=["resnet50", "efficientnetv2s"], choices=sorted(METHODS))
    parser.add_argument("--manifest", type=Path, default=Path("data_processed/dfuc2021_cls_v1/splits/dfuc2021_cls_cv5_manifest.csv"))
    parser.add_argument("--prior-meta", type=Path, default=Path("data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv"))
    parser.add_argument("--output-table-dir", type=Path, default=Path("outputs/paper_tables"))
    parser.add_argument("--output-figure-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples-per-fold", type=int, default=0, help="0 means all validation samples.")
    parser.add_argument("--prior-threshold", type=float, default=0.3)
    parser.add_argument("--cam-threshold", type=float, default=0.7)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve(path: Path, repo_root: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else repo_root / path


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_train_module(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from experiments.dfuc2021_cls_baseline import train as train_mod

    return train_mod


class ExplainDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, repo_root: Path, input_size: int, prior_threshold: float) -> None:
        self.frame = frame.reset_index(drop=True)
        self.repo_root = repo_root
        self.input_size = input_size
        self.prior_threshold = prior_threshold

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        image_path = resolve(Path(row["image_path"]), self.repo_root)
        prior_path = resolve(Path(row["prior_path"]), self.repo_root)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        image_float = rgb.astype(np.float32) / 255.0
        normalized = (image_float - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()

        prior = cv2.imread(str(prior_path), cv2.IMREAD_GRAYSCALE)
        if prior is None:
            raise FileNotFoundError(prior_path)
        prior = cv2.resize(prior, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        prior_mask = (prior >= self.prior_threshold).astype(np.float32)
        return {
            "image": tensor,
            "image_name": row["image"],
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "fold": int(row["fold"]),
            "prior": torch.from_numpy(prior).float(),
            "prior_mask": torch.from_numpy(prior_mask).float(),
            "prior_fallback": int(row["prior_fallback"]),
            "rgb": torch.from_numpy(rgb.copy()),
        }


class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.hooks = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output

    def _backward_hook(self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0]

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def __call__(self, images: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(images)
        selected = logits.gather(1, target_ids.view(-1, 1)).sum()
        selected.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        mins = flat.min(dim=1).values.view(-1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - mins) / (maxs - mins + 1e-8)
        return logits.detach(), cam.detach()


def target_layer_for(model: nn.Module, key: str) -> nn.Module:
    if key == "layer4":
        return model.layer4[-1]
    if key == "features":
        return model.features[-1]
    raise ValueError(key)


def load_model(train_mod: Any, spec: MethodSpec, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"No config in {checkpoint_path}")
    config["model"]["pretrained"] = False
    model = train_mod.build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def metric_rows(method: str, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summary_rows = []
    groups = [("all", "all", frame)]
    for label, sub in frame.groupby("label_name"):
        groups.append(("class", label, sub))
    for fallback, sub in frame.groupby("prior_fallback"):
        groups.append(("fallback", "fallback" if int(fallback) else "non_fallback", sub))
    for group_type, group, sub in groups:
        summary_rows.append(
            {
                "method": method,
                "group_type": group_type,
                "group": group,
                "samples": int(len(sub)),
                "cam_prior_mass_ratio": float(sub["cam_prior_mass_ratio"].mean()),
                "cam_prior_iou": float(sub["cam_prior_iou"].mean()),
                "pointing_game_accuracy": float(sub["pointing_hit"].mean()),
                "prior_occlusion_drop_pred": float(sub["prior_occlusion_drop_pred"].mean()),
                "random_occlusion_drop_pred": float(sub["random_occlusion_drop_pred"].mean()),
                "background_occlusion_drop_pred": float(sub["background_occlusion_drop_pred"].mean()),
                "prior_minus_background_drop": float((sub["prior_occlusion_drop_pred"] - sub["background_occlusion_drop_pred"]).mean()),
                "prior_minus_random_drop": float((sub["prior_occlusion_drop_pred"] - sub["random_occlusion_drop_pred"]).mean()),
                "prior_occlusion_drop_true": float(sub["prior_occlusion_drop_true"].mean()),
                "random_occlusion_drop_true": float(sub["random_occlusion_drop_true"].mean()),
                "background_occlusion_drop_true": float(sub["background_occlusion_drop_true"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def make_translated_prior_masks(masks: torch.Tensor, seed_offset: int = 0) -> torch.Tensor:
    """Create shape/area-matched random-location controls from binary prior masks."""
    masks_cpu = masks.detach().cpu().numpy()
    random_masks = np.zeros_like(masks_cpu, dtype=np.float32)
    height, width = masks_cpu.shape[-2:]
    rng = np.random.default_rng(20260501 + int(seed_offset))
    for idx in range(masks_cpu.shape[0]):
        mask = masks_cpu[idx, 0] > 0.5
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        patch = mask[y1:y2, x1:x2].astype(np.float32)
        patch_h, patch_w = patch.shape
        if patch_h >= height or patch_w >= width:
            flat = np.zeros(height * width, dtype=np.float32)
            count = int(mask.sum())
            if count > 0:
                indices = rng.choice(height * width, size=min(count, height * width), replace=False)
                flat[indices] = 1.0
            random_masks[idx, 0] = flat.reshape(height, width)
            continue
        best_y = int(rng.integers(0, height - patch_h + 1))
        best_x = int(rng.integers(0, width - patch_w + 1))
        random_masks[idx, 0, best_y : best_y + patch_h, best_x : best_x + patch_w] = patch
    return torch.from_numpy(random_masks).to(device=masks.device, dtype=masks.dtype)


def compute_batch_metrics(
    method: str,
    batch: dict[str, Any],
    logits: torch.Tensor,
    cams: torch.Tensor,
    prior_logits: torch.Tensor,
    random_logits: torch.Tensor,
    bg_logits: torch.Tensor,
    cam_threshold: float,
) -> list[dict[str, Any]]:
    probs = torch.softmax(logits, dim=1).cpu()
    prior_probs = torch.softmax(prior_logits, dim=1).cpu()
    random_probs = torch.softmax(random_logits, dim=1).cpu()
    bg_probs = torch.softmax(bg_logits, dim=1).cpu()
    preds = probs.argmax(dim=1)
    labels = batch["label"].cpu()
    cams_cpu = cams.cpu()
    prior_masks = batch["prior_mask"].cpu()
    rows = []
    for i in range(cams_cpu.shape[0]):
        cam = cams_cpu[i].numpy().astype(np.float32)
        mask = prior_masks[i].numpy().astype(bool)
        cam_sum = float(cam.sum()) + 1e-8
        mask_sum = int(mask.sum())
        inside_mass = float(cam[mask].sum() / cam_sum) if mask_sum else 0.0
        cam_mask = cam >= cam_threshold
        inter = np.logical_and(cam_mask, mask).sum()
        union = np.logical_or(cam_mask, mask).sum()
        iou = float(inter / union) if union else 0.0
        max_y, max_x = np.unravel_index(int(cam.argmax()), cam.shape)
        pointing_hit = bool(mask[max_y, max_x]) if mask_sum else False
        pred_id = int(preds[i])
        true_id = int(labels[i])
        rows.append(
            {
                "method": method,
                "image": str(batch["image_name"][i]),
                "fold": int(batch["fold"][i]),
                "label": true_id,
                "label_name": CLASS_NAMES[true_id],
                "pred_label": CLASS_NAMES[pred_id],
                "correct": pred_id == true_id,
                "prior_fallback": int(batch["prior_fallback"][i]),
                "prior_area_ratio": float(mask.mean()),
                "cam_prior_mass_ratio": inside_mass,
                "cam_prior_iou": iou,
                "pointing_hit": pointing_hit,
                "original_prob_pred": float(probs[i, pred_id]),
                "prior_occluded_prob_pred": float(prior_probs[i, pred_id]),
                "random_occluded_prob_pred": float(random_probs[i, pred_id]),
                "background_occluded_prob_pred": float(bg_probs[i, pred_id]),
                "prior_occlusion_drop_pred": float(probs[i, pred_id] - prior_probs[i, pred_id]),
                "random_occlusion_drop_pred": float(probs[i, pred_id] - random_probs[i, pred_id]),
                "background_occlusion_drop_pred": float(probs[i, pred_id] - bg_probs[i, pred_id]),
                "original_prob_true": float(probs[i, true_id]),
                "prior_occlusion_drop_true": float(probs[i, true_id] - prior_probs[i, true_id]),
                "random_occlusion_drop_true": float(probs[i, true_id] - random_probs[i, true_id]),
                "background_occlusion_drop_true": float(probs[i, true_id] - bg_probs[i, true_id]),
            }
        )
    return rows


def render_examples(examples: list[dict[str, Any]], figure_dir: Path, dpi: int) -> None:
    if not examples:
        return
    selected = sorted(examples, key=lambda item: item["score"], reverse=True)[:8]
    fig, axes = plt.subplots(len(selected), 4, figsize=(9.6, 2.25 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, row in enumerate(selected):
        rgb = row["rgb"]
        prior = row["prior"]
        cam = row["cam"]
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"{row['image']}\ntrue={row['label_name']} pred={row['pred_label']}", fontsize=8)
        axes[row_idx, 1].imshow(rgb)
        axes[row_idx, 1].imshow(prior, cmap="magma", alpha=0.42)
        axes[row_idx, 1].set_title("Lesion prior", fontsize=8)
        axes[row_idx, 2].imshow(rgb)
        axes[row_idx, 2].imshow(cam, cmap="jet", alpha=0.42)
        axes[row_idx, 2].set_title("Grad-CAM", fontsize=8)
        axes[row_idx, 3].axis("off")
        axes[row_idx, 3].text(
            0.0,
            0.82,
            f"CAM-prior mass: {row['cam_prior_mass_ratio']:.3f}\n"
            f"Pointing hit: {int(row['pointing_hit'])}\n"
            f"Prior drop: {row['prior_occlusion_drop_pred']:.3f}\n"
            f"Random drop: {row['random_occlusion_drop_pred']:.3f}\n"
            f"Background drop: {row['background_occlusion_drop_pred']:.3f}",
            fontsize=8.5,
        )
        for col in range(3):
            axes[row_idx, col].axis("off")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "fig_cam_prior_occlusion_examples.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figure_dir / "fig_cam_prior_occlusion_examples.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    table_dir = resolve(args.output_table_dir, repo_root)
    figure_dir = resolve(args.output_figure_dir, repo_root)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    train_mod = import_train_module(repo_root)

    manifest = pd.read_csv(resolve(args.manifest, repo_root))
    prior_meta = pd.read_csv(resolve(args.prior_meta, repo_root))
    frame = manifest.merge(
        prior_meta[["image", "prior_path", "prior_fallback"]],
        on="image",
        how="left",
        validate="one_to_one",
    )
    all_rows = []
    all_summaries = []
    example_items: list[dict[str, Any]] = []
    for method_key in args.methods:
        spec = METHODS[method_key]
        method_rows = []
        for fold in range(5):
            fold_frame = frame[frame["fold"] == fold].copy()
            if args.max_samples_per_fold and args.max_samples_per_fold > 0:
                fold_frame = fold_frame.groupby("label_name", group_keys=False).head(max(1, args.max_samples_per_fold // 4))
                fold_frame = fold_frame.head(args.max_samples_per_fold)
            checkpoint_path = resolve(Path(spec.checkpoint_pattern.format(fold=fold)), repo_root)
            model = load_model(train_mod, spec, checkpoint_path, device)
            cam_runner = GradCam(model, target_layer_for(model, spec.target_layer))
            dataset = ExplainDataset(fold_frame, repo_root, input_size=224, prior_threshold=args.prior_threshold)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
            for batch in loader:
                images = batch["image"].to(device)
                masks = batch["prior_mask"].to(device).unsqueeze(1)
                with torch.no_grad():
                    base_logits = model(images)
                    targets = base_logits.argmax(dim=1)
                logits, cams = cam_runner(images, targets)
                with torch.no_grad():
                    random_masks = make_translated_prior_masks(masks, seed_offset=fold)
                    prior_occluded = images * (1.0 - masks)
                    random_occluded = images * (1.0 - random_masks)
                    background_occluded = images * masks
                    prior_logits = model(prior_occluded)
                    random_logits = model(random_occluded)
                    bg_logits = model(background_occluded)
                batch_rows = compute_batch_metrics(spec.name, batch, logits, cams, prior_logits, random_logits, bg_logits, args.cam_threshold)
                method_rows.extend(batch_rows)
                if method_key == args.methods[0]:
                    cams_cpu = cams.detach().cpu().numpy()
                    priors_cpu = batch["prior"].cpu().numpy()
                    rgbs_cpu = batch["rgb"].cpu().numpy()
                    for row_idx, row in enumerate(batch_rows):
                        score = row["cam_prior_mass_ratio"] + max(0.0, row["prior_occlusion_drop_pred"])
                        example_items.append(
                            {
                                **row,
                                "score": float(score),
                                "cam": cams_cpu[row_idx],
                                "prior": priors_cpu[row_idx],
                                "rgb": rgbs_cpu[row_idx].astype("uint8"),
                            }
                        )
                    example_items = sorted(example_items, key=lambda item: item["score"], reverse=True)[:16]
            cam_runner.close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        method_frame = pd.DataFrame(method_rows)
        all_rows.append(method_frame)
        all_summaries.append(metric_rows(spec.name, method_rows))

    detail = pd.concat(all_rows, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    detail.to_csv(table_dir / "cam_prior_occlusion_detail.csv", index=False)
    summary.to_csv(table_dir / "cam_prior_occlusion_summary.csv", index=False)
    render_examples(example_items, figure_dir, args.dpi)
    payload = {
        "methods": [METHODS[key].name for key in args.methods],
        "samples": int(len(detail)),
        "max_samples_per_fold": int(args.max_samples_per_fold),
        "detail_csv": "outputs/paper_tables/cam_prior_occlusion_detail.csv",
        "summary_csv": "outputs/paper_tables/cam_prior_occlusion_summary.csv",
    }
    (table_dir / "cam_prior_occlusion_notes.md").write_text(
        "# CAM-Prior and Occlusion Analysis\n\n"
        "Generated by `experiments/reliability_aware_prior_fusion/explainability_analysis.py`.\n\n"
        "This analysis compares Grad-CAM maps against DFUC2022-derived lesion priors and measures prediction sensitivity after occluding prior vs background regions. Interpret these results as explanatory evidence, not as ground-truth lesion localization validation.\n"
        "The occlusion analysis also includes a shape- and area-matched random-location control generated by translating each prior mask to a random image location. This controls for the amount of image content removed, while keeping the result distinct from true external validation.\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
