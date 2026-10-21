#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.profiler import ProfilerActivity, profile
import torchvision.models as models
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.cross_task_prior_transfer.train_lpt import LPTNet


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def estimated_flops(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    activities = [ProfilerActivity.CPU]
    if inputs[0].device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    model.eval()
    with torch.inference_mode(), profile(activities=activities, with_flops=True, record_shapes=True) as prof:
        model(*inputs)
        if inputs[0].device.type == "cuda":
            torch.cuda.synchronize()
    return int(sum(event.flops for event in prof.key_averages() if event.flops is not None))


def benchmark_latency(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    warmup: int = 50,
    repetitions: int = 200,
) -> dict[str, float | int]:
    model.eval()
    device = inputs[0].device
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(*inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "warmup_iterations": warmup,
        "timed_iterations": repetitions,
        "latency_ms_mean": float(statistics.mean(timings)),
        "latency_ms_sd": float(statistics.stdev(timings)),
        "latency_ms_median": float(statistics.median(timings)),
        "latency_ms_p95": float(sorted(timings)[int(0.95 * (len(timings) - 1))]),
    }


def prepare_models(
    device: torch.device,
    repo_root: Path,
) -> list[tuple[str, torch.nn.Module, tuple[torch.Tensor, ...], str]]:
    image = torch.randn(1, 3, 224, 224, device=device)
    prior = torch.rand(1, 1, 224, 224, device=device)
    stats = torch.rand(1, 4, device=device)
    classifiers = [
        ("ResNet50", models.resnet50(weights=None), "RGB classifier; 224 x 224"),
        ("EfficientNet-B0", models.efficientnet_b0(weights=None), "RGB classifier; 224 x 224"),
        ("ConvNeXt-Tiny", models.convnext_tiny(weights=None), "RGB classifier; 224 x 224"),
        ("EfficientNetV2-S", models.efficientnet_v2_s(weights=None), "RGB classifier; 224 x 224"),
    ]
    prepared: list[tuple[str, torch.nn.Module, tuple[torch.Tensor, ...], str]] = []
    for name, model, note in classifiers:
        if hasattr(model, "fc"):
            model.fc = torch.nn.Linear(model.fc.in_features, 4)
        elif name.startswith("EfficientNet"):
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
        elif name == "ConvNeXt-Tiny":
            model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 4)
        prepared.append((name, model.to(device), (image,), note))

    lpt_config = yaml.safe_load(
        (
            repo_root
            / "experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml"
        ).read_text()
    )
    lpt_config = copy.deepcopy(lpt_config)
    lpt_config["model"]["pretrained"] = False
    lpt = LPTNet(lpt_config, stats_dim=4).to(device)
    prepared.append(
        (
            "Full LPT",
            lpt,
            (image, image.clone(), prior, stats),
            "Classifier forward after prior generation; global and ROI inputs 224 x 224",
        )
    )

    segmenter = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)
    prepared.append(
        (
            "DeepLabV3+ prior generator",
            segmenter,
            (image,),
            "Actual DFUC2021 prior-inference grid; 224 x 224",
        )
    )
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark manuscript model complexity and batch-1 latency.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/benchmarks/dfu_model_complexity_benchmark.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    results = {
        "benchmark_scope": "Architecture-only FP32 forward pass; excludes image decoding, preprocessing, file I/O, stacking, and fusion search.",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "batch_size": 1,
        "models": [],
    }
    for name, model, inputs, note in prepare_models(device, args.repo_root.resolve()):
        total, trainable = parameter_counts(model)
        flops = estimated_flops(model, inputs)
        latency = benchmark_latency(
            model,
            inputs,
            warmup=args.warmup,
            repetitions=args.repetitions,
        )
        results["models"].append(
            {
                "model": name,
                "parameters_total": total,
                "parameters_trainable": trainable,
                "profiler_estimated_flops": flops,
                "profiler_estimated_gflops": flops / 1e9,
                **latency,
                "note": note,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
