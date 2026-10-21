#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, get_worker_info

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cross_task_prior_transfer import train_lpt as lpt  # noqa: E402
from experiments.dfuc2021_cls_baseline import train as rgb  # noqa: E402


MODEL_CONFIGS = {
    "resnet50": "experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml",
    "efficientnetb0": "experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_efficientnetb0_cv5_v1.yaml",
    "convnext_tiny": "experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_convnext_tiny_cv5_v1.yaml",
    "efficientnetv2s": "experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_efficientnetv2s_cv5_v1.yaml",
    "lpt_full": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml",
    "lpt_neutral": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml",
    "lpt_no_alignment": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_no_alignment_cv5_v1.yaml",
    "lpt_no_prior_mod": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_no_prior_mod_cv5_v1.yaml",
    "lpt_no_roi": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_no_roi_cv5_v1.yaml",
    "lpt_stats_only": "experiments/cross_task_prior_transfer/configs/cross_task_lpt_stats_only_cv5_v1.yaml",
}
RGB_MODELS = {"resnet50", "efficientnetb0", "convnext_tiny", "efficientnetv2s"}
DIRECT_MODELS = {"lpt_full", "lpt_neutral"}
CLASS_NAMES = ("none", "infection", "ischaemia", "both")


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


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(repo_root: Path, model_key: str, job_seed: int) -> tuple[dict[str, Any], Path]:
    source_path = repo_root / MODEL_CONFIGS[model_key]
    config = load_yaml(source_path)
    config["seed"] = int(job_seed)
    config["data"]["split_manifest"] = (
        "data_processed/dfuc2021_cls_v1/similarity_grouped_splits/"
        "dfuc2021_cls_cv5_similarity_grouped_manifest.csv"
    )
    config["training"].update(
        {
            "epochs": 60,
            "batch_size": 16,
            "grad_accum_steps": 1,
            "num_workers": 4,
            "mixed_precision": True,
            "early_stopping_patience": 12,
        }
    )
    if model_key == "lpt_neutral":
        config["prior"]["roi_mode"] = "image"
        config["prior"]["stats_mode"] = "learned"
    return config, source_path


def set_deterministic_runtime(seed: int) -> None:
    rgb.set_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(_: int) -> None:
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    info = get_worker_info()
    transform = getattr(info.dataset, "transform", None) if info is not None else None
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(worker_seed)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    transform = getattr(dataset, "transform", None)
    if transform is not None and hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )


def derive_job_seed(base_seed: int, outer: int, stage: str, inner: int | None) -> int:
    stage_codes = {"inner": 1, "refit": 2, "direct_select": 3, "direct_refit": 4}
    inner_component = inner + 1 if inner is not None else (4 if stage.startswith("direct") else 0)
    sequence = np.random.SeedSequence([base_seed, outer, inner_component, stage_codes[stage]])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class NeutralPriorDataset(Dataset):
    """LPT input with no access to real prior pixels or prior-derived statistics."""

    def __init__(self, records: Sequence[lpt.PriorSampleRecord], transform: A.Compose) -> None:
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {record.image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        neutral = np.full(image.shape[:2], 0.5, dtype=np.float32)
        transformed = self.transform(image=image, roi=image.copy(), prior=neutral)
        prior_tensor = transformed["prior"]
        if prior_tensor.ndim == 2:
            prior_tensor = prior_tensor.unsqueeze(0)
        prior_tensor = torch.full_like(prior_tensor.float(), 0.5)
        return {
            "image": transformed["image"].float(),
            "roi": transformed["roi"].float(),
            "prior": prior_tensor,
            "stats": torch.tensor([0.5, 0.5, 1.0, 0.5], dtype=torch.float32),
            "label": torch.tensor(record.label, dtype=torch.long),
            "infection_attr": torch.tensor(record.infection_attr, dtype=torch.float32),
            "ischaemia_attr": torch.tensor(record.ischaemia_attr, dtype=torch.float32),
            "image_name": record.image,
        }


def lpt_dataset(
    records: Sequence[lpt.PriorSampleRecord], config: dict[str, Any], train: bool, neutral: bool
) -> Dataset:
    transform = lpt.build_transforms(config, train=train)
    if neutral:
        return NeutralPriorDataset(records, transform)
    return lpt.DFUC2021LPTDataset(
        list(records),
        transform,
        roi_threshold=float(config["prior"]["roi_threshold"]),
        fallback_max_threshold=float(config["prior"]["fallback_max_threshold"]),
        fallback_area_threshold=float(config["prior"]["fallback_area_threshold"]),
        bbox_expand_ratio=float(config["prior"]["bbox_expand_ratio"]),
        roi_mode=str(config["prior"].get("roi_mode", "prior_crop")),
        stats_mode=str(config["prior"].get("stats_mode", "learned")),
    )


def load_assignment_frame(path: Path, outer: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["outer_fold"].eq(outer)].copy()
    if frame.empty or frame["image"].duplicated().any():
        raise ValueError(f"Invalid assignment rows for outer fold {outer}.")
    return frame.set_index("image", drop=False)


def split_names(frame: pd.DataFrame, stage: str, inner: int | None) -> tuple[list[str], list[str], list[str]]:
    if stage == "inner":
        if inner is None:
            raise ValueError("--inner is required for an inner job.")
        role = frame[f"inner{inner}_role"]
        return (
            frame.loc[role.eq("fit"), "image"].tolist(),
            frame.loc[role.eq("stop"), "image"].tolist(),
            frame.loc[role.eq("validation"), "image"].tolist(),
        )
    if stage == "direct_select":
        role = frame["direct_role"]
        return (
            frame.loc[role.eq("fit"), "image"].tolist(),
            frame.loc[role.eq("stop"), "image"].tolist(),
            [],
        )
    if stage in {"refit", "direct_refit"}:
        return (
            frame.loc[frame["outer_role"].eq("train"), "image"].tolist(),
            [],
            frame.loc[frame["outer_role"].eq("test"), "image"].tolist(),
        )
    raise ValueError(stage)


def assert_name_boundaries(frame: pd.DataFrame, train: list[str], stop: list[str], predict: list[str]) -> None:
    name_sets = [set(train), set(stop), set(predict)]
    if any(name_sets[i] & name_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Image leakage across train/stop/prediction roles.")
    group_sets = [set(frame.loc[list(names), "similarity_group"]) if names else set() for names in name_sets]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Similarity-group leakage across train/stop/prediction roles.")


def select_records(records: Sequence[Any], names: list[str]) -> list[Any]:
    by_name = {record.image: record for record in records}
    missing = set(names) - set(by_name)
    if missing:
        raise KeyError(f"Records missing {len(missing)} requested images.")
    return [by_name[name] for name in names]


def checkpoint_payload(
    model: nn.Module,
    config: dict[str, Any],
    model_key: str,
    base_seed: int,
    job_seed: int,
    outer: int,
    stage: str,
    epochs: int,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "config": config,
        "model_key": model_key,
        "base_seed": base_seed,
        "job_seed": job_seed,
        "outer_fold": outer,
        "stage": stage,
        "epochs": epochs,
        "class_names": CLASS_NAMES,
    }


@torch.no_grad()
def predict_rgb(model: nn.Module, loader: Any, device: torch.device, amp: bool) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        with rgb.autocast_context(device, amp):
            probabilities = torch.softmax(model(images), dim=1)
        if not torch.isfinite(probabilities).all():
            print("NONFINITE_PREDICTION_FALLBACK model=rgb precision=float32", flush=True)
            probabilities = torch.softmax(model(images), dim=1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite RGB probabilities remained after float32 prediction fallback.")
        predictions = probabilities.argmax(dim=1).cpu().tolist()
        probability_rows = probabilities.cpu().tolist()
        labels = batch["label"].tolist()
        for image, label, prediction, values in zip(
            batch["image_name"], labels, predictions, probability_rows, strict=True
        ):
            rows.append(
                {
                    "image": image,
                    "true_label": CLASS_NAMES[int(label)],
                    "pred_label": CLASS_NAMES[int(prediction)],
                    **{f"prob_{name}": float(values[index]) for index, name in enumerate(CLASS_NAMES)},
                }
            )
    return rows


@torch.no_grad()
def predict_lpt(model: nn.Module, loader: Any, device: torch.device, amp: bool) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for raw_batch in loader:
        batch = lpt.move_batch(raw_batch, device)
        with lpt.autocast_context(device, amp):
            outputs = model(batch["image"], batch["roi"], batch["prior"], batch["stats"])
            probabilities = torch.softmax(outputs["class_logits"], dim=1)
            infection = torch.sigmoid(outputs["infection_logits"])
            ischaemia = torch.sigmoid(outputs["ischaemia_logits"])
        if not all(torch.isfinite(values).all() for values in (probabilities, infection, ischaemia)):
            print("NONFINITE_PREDICTION_FALLBACK model=lpt precision=float32", flush=True)
            outputs = model(batch["image"], batch["roi"], batch["prior"], batch["stats"])
            probabilities = torch.softmax(outputs["class_logits"], dim=1)
            infection = torch.sigmoid(outputs["infection_logits"])
            ischaemia = torch.sigmoid(outputs["ischaemia_logits"])
        if not all(torch.isfinite(values).all() for values in (probabilities, infection, ischaemia)):
            raise FloatingPointError("Non-finite LPT probabilities remained after float32 prediction fallback.")
        predictions = probabilities.argmax(dim=1).cpu().tolist()
        probability_rows = probabilities.cpu().tolist()
        labels = batch["label"].cpu().tolist()
        for image, label, prediction, values, inf, isc in zip(
            raw_batch["image_name"],
            labels,
            predictions,
            probability_rows,
            infection.cpu().tolist(),
            ischaemia.cpu().tolist(),
            strict=True,
        ):
            rows.append(
                {
                    "image": image,
                    "true_label": CLASS_NAMES[int(label)],
                    "pred_label": CLASS_NAMES[int(prediction)],
                    **{f"prob_{name}": float(values[index]) for index, name in enumerate(CLASS_NAMES)},
                    "prob_infection_attr": float(inf),
                    "prob_ischaemia_attr": float(isc),
                }
            )
    return rows


def completed_ok(path: Path, keep_checkpoint: bool) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    prediction = payload.get("prediction_path")
    checkpoint = payload.get("checkpoint_path")
    return (not prediction or Path(prediction).exists()) and (
        not keep_checkpoint or (checkpoint and Path(checkpoint).exists())
    )


def run_job(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    assignment_path = args.assignments if args.assignments.is_absolute() else repo_root / args.assignments
    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    job_dir = output_root / f"seed_{args.seed}" / f"outer_{args.outer}" / args.model / (
        f"inner_{args.inner}" if args.stage == "inner" else args.stage
    )
    completion_path = job_dir / "complete.json"
    keep_checkpoint = args.stage in {"refit", "direct_refit"}
    if not args.force and completed_ok(completion_path, keep_checkpoint):
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        print(f"SKIP completed {job_dir}", flush=True)
        return payload

    if args.stage in {"inner", "refit"} and args.model in DIRECT_MODELS:
        raise ValueError("Direct models use direct_select/direct_refit stages.")
    if args.stage in {"direct_select", "direct_refit"} and args.model not in DIRECT_MODELS:
        raise ValueError("Only Full and neutral LPT use direct stages.")

    frame = load_assignment_frame(assignment_path, args.outer)
    train_names, stop_names, prediction_names = split_names(frame, args.stage, args.inner)
    assert_name_boundaries(frame, train_names, stop_names, prediction_names)
    job_seed = derive_job_seed(args.seed, args.outer, args.stage, args.inner)
    set_deterministic_runtime(job_seed)
    config, source_config = load_config(repo_root, args.model, job_seed)
    if args.max_epochs is not None:
        config["training"]["epochs"] = int(args.max_epochs)
        config["training"]["early_stopping_patience"] = min(
            int(config["training"]["early_stopping_patience"]), int(args.max_epochs)
        )
    device = rgb.pick_device(args.device)
    amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    neutral = args.model == "lpt_neutral"
    is_rgb = args.model in RGB_MODELS

    if is_rgb:
        all_records = rgb.load_records(repo_root / config["data"]["split_manifest"], repo_root)
    else:
        all_records = lpt.load_records(config, repo_root)
    train_records = select_records(all_records, train_names)
    stop_records = select_records(all_records, stop_names)
    prediction_records = select_records(all_records, prediction_names)

    if is_rgb:
        train_dataset = rgb.DFUC2021ClassificationDataset(train_records, rgb.build_transforms(config, train=True))
        stop_dataset = (
            rgb.DFUC2021ClassificationDataset(stop_records, rgb.build_transforms(config, train=False))
            if stop_records
            else None
        )
        prediction_dataset = (
            rgb.DFUC2021ClassificationDataset(prediction_records, rgb.build_transforms(config, train=False))
            if prediction_records
            else None
        )
        model = rgb.build_model(config).to(device)
        class_weights = rgb.compute_class_weights(train_records, str(config["loss"]["class_weights"]), device)
        loss_objects: tuple[Any, ...] = (nn.CrossEntropyLoss(weight=class_weights),)
    else:
        train_dataset = lpt_dataset(train_records, config, train=True, neutral=neutral)
        stop_dataset = lpt_dataset(stop_records, config, train=False, neutral=neutral) if stop_records else None
        prediction_dataset = (
            lpt_dataset(prediction_records, config, train=False, neutral=neutral) if prediction_records else None
        )
        model = lpt.LPTNet(config, stats_dim=len(config["data"].get("stats_fields", lpt.STATS_FIELDS))).to(device)
        class_weights = lpt.compute_class_weights(train_records, str(config["loss"]["class_weights"]), device)
        infection_weight, ischaemia_weight = lpt.compute_attr_pos_weights(train_records, device)
        loss_objects = (
            nn.CrossEntropyLoss(weight=class_weights),
            nn.BCEWithLogitsLoss(pos_weight=infection_weight),
            nn.BCEWithLogitsLoss(pos_weight=ischaemia_weight),
        )

    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "device": device,
    }
    train_loader = build_loader(train_dataset, shuffle=True, seed=job_seed, **loader_args)
    stop_loader = (
        build_loader(stop_dataset, shuffle=False, seed=job_seed, **loader_args) if stop_dataset is not None else None
    )
    prediction_loader = (
        build_loader(prediction_dataset, shuffle=False, seed=job_seed, **loader_args)
        if prediction_dataset is not None
        else None
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["optimizer"]["lr"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else None
    job_dir.mkdir(parents=True, exist_ok=True)
    history_path = job_dir / "history.jsonl"
    temporary_checkpoint = job_dir / "best_selection.pt"
    final_checkpoint = job_dir / "model.pt"
    predictions_path = job_dir / "predictions.csv"

    fixed_epochs: int | None = None
    if args.stage == "refit":
        best_epochs = []
        for inner in range(3):
            path = output_root / f"seed_{args.seed}" / f"outer_{args.outer}" / args.model / f"inner_{inner}" / "complete.json"
            if not path.exists():
                raise FileNotFoundError(f"Required inner result is missing: {path}")
            best_epochs.append(int(json.loads(path.read_text(encoding="utf-8"))["best_epoch"]))
        fixed_epochs = int(statistics.median(best_epochs))
    elif args.stage == "direct_refit":
        path = output_root / f"seed_{args.seed}" / f"outer_{args.outer}" / args.model / "direct_select" / "complete.json"
        if not path.exists():
            raise FileNotFoundError(f"Required direct selection result is missing: {path}")
        fixed_epochs = int(json.loads(path.read_text(encoding="utf-8"))["best_epoch"])

    total_epochs = fixed_epochs or int(config["training"]["epochs"])
    scheduler_horizon = 60
    patience = int(config["training"]["early_stopping_patience"])
    best_epoch = 0
    best_score = -float("inf")
    started = time.time()
    if history_path.exists():
        history_path.unlink()
    print(
        f"START model={args.model} seed={args.seed} outer={args.outer} stage={args.stage} inner={args.inner} "
        f"train={len(train_records)} stop={len(stop_records)} predict={len(prediction_records)} epochs={total_epochs}",
        flush=True,
    )

    for epoch_index in range(total_epochs):
        learning_rate = rgb.compute_epoch_lr(
            float(config["optimizer"]["lr"]),
            float(config["scheduler"]["min_lr"]),
            epoch_index,
            scheduler_horizon,
            int(config["scheduler"]["warmup_epochs"]),
        )
        rgb.set_optimizer_lr(optimizer, learning_rate)
        if is_rgb:
            train_loss = rgb.train_one_epoch(
                model,
                train_loader,
                loss_objects[0],
                optimizer,
                scaler,
                device,
                amp,
                int(config["training"]["grad_accum_steps"]),
                epoch_index,
                total_epochs,
                learning_rate,
            )
            train_log: Any = float(train_loss)
        else:
            train_log = lpt.train_one_epoch(
                model,
                train_loader,
                loss_objects[0],
                loss_objects[1],
                loss_objects[2],
                optimizer,
                scaler,
                device,
                amp,
                int(config["training"]["grad_accum_steps"]),
                epoch_index,
                total_epochs,
                learning_rate,
                config["loss"],
            )

        validation_metrics: dict[str, Any] | None = None
        if stop_loader is not None:
            if is_rgb:
                _, validation_metrics, _ = rgb.evaluate(
                    model, stop_loader, loss_objects[0], device, amp, epoch_index, total_epochs
                )
            else:
                _, validation_metrics, _ = lpt.evaluate(
                    model,
                    stop_loader,
                    loss_objects[0],
                    loss_objects[1],
                    loss_objects[2],
                    device,
                    amp,
                    epoch_index,
                    total_epochs,
                    config["loss"],
                )
            score = float(validation_metrics["macro_f1"])
            if score > best_score:
                best_score = score
                best_epoch = epoch_index + 1
                torch.save(
                    checkpoint_payload(
                        model, config, args.model, args.seed, job_seed, args.outer, args.stage, best_epoch
                    ),
                    temporary_checkpoint,
                )
        else:
            best_epoch = epoch_index + 1

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch_index + 1,
                        "lr": learning_rate,
                        "train": train_log,
                        "stop_metrics": validation_metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(
            f"EPOCH {epoch_index + 1}/{total_epochs} train={train_log} "
            f"stop_macro_f1={None if validation_metrics is None else validation_metrics['macro_f1']} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if stop_loader is not None and epoch_index + 1 - best_epoch >= patience:
            break

    if stop_loader is not None:
        if not temporary_checkpoint.exists():
            raise RuntimeError("No best selection checkpoint was created.")
        selected = torch.load(temporary_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state"])
    else:
        torch.save(
            checkpoint_payload(model, config, args.model, args.seed, job_seed, args.outer, args.stage, best_epoch),
            final_checkpoint,
        )

    prediction_sha = None
    if prediction_loader is not None:
        rows = predict_rgb(model, prediction_loader, device, amp) if is_rgb else predict_lpt(model, prediction_loader, device, amp)
        for row in rows:
            row.update(
                {
                    "base_seed": args.seed,
                    "outer_fold": args.outer,
                    "inner_fold": -1 if args.inner is None else args.inner,
                    "model": args.model,
                    "stage": args.stage,
                }
            )
        write_predictions(predictions_path, rows)
        prediction_sha = sha256(predictions_path)

    checkpoint_sha = None
    checkpoint_path: str | None = None
    if keep_checkpoint:
        checkpoint_sha = sha256(final_checkpoint)
        checkpoint_path = str(final_checkpoint)
    elif temporary_checkpoint.exists():
        checkpoint_sha = sha256(temporary_checkpoint)
        temporary_checkpoint.unlink()

    payload = {
        "status": "complete",
        "model": args.model,
        "base_seed": args.seed,
        "job_seed": job_seed,
        "outer_fold": args.outer,
        "inner_fold": args.inner,
        "stage": args.stage,
        "train_samples": len(train_records),
        "stop_samples": len(stop_records),
        "prediction_samples": len(prediction_records),
        "best_epoch": best_epoch,
        "best_stop_macro_f1": None if best_score == -float("inf") else best_score,
        "source_config": str(source_config),
        "source_config_sha256": sha256(source_config),
        "assignment_manifest": str(assignment_path),
        "assignment_manifest_sha256": sha256(assignment_path),
        "prediction_path": str(predictions_path) if prediction_loader is not None else None,
        "prediction_sha256": prediction_sha,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "history_path": str(history_path),
        "elapsed_seconds": time.time() - started,
        "neutral_prior_contract": neutral,
    }
    write_json(completion_path, payload)
    print(f"COMPLETE {job_dir}", flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen nested classification training job.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--assignments",
        type=Path,
        default=Path("data_processed/associate_editor_revision_v1/nested_grouped_assignments.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/associate_editor_revision_v1/classification")
    )
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--outer", type=int, choices=range(5), required=True)
    parser.add_argument("--stage", choices=("inner", "refit", "direct_select", "direct_refit"), required=True)
    parser.add_argument("--inner", type=int, choices=range(3))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_job(parse_args())
