"""Render the frozen, prediction-blind qualitative interpretability sample.

Figure contract
---------------
Conclusion: the fixed cases show the distribution of model localisation patterns,
including classification errors; they do not validate anatomy or clinical use.
Evidence chain: identical RGB patch -> fixed transferred soft prior -> two
global-branch Grad-CAM displays.  Cases are selected upstream without using
predictions, and all 20 cases and all five training seeds must be retained.
Archetype: exploratory image plates, not quantitative validation panels.
Backend: Python/matplotlib only.  No generated or retouched clinical images.
Export: 183 x 225 mm, 7 pt labels, 300 dpi PNG, PDF and editable-text SVG;
per-seed multipage PDFs, source hashes and explicit image-integrity notes.

This module only reads archived case arrays.  It never runs models, selects cases,
changes predictions or applies selective brightness/contrast adjustments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize


SEEDS = (20260925, 20260926, 20260927, 20260928, 20260929)
MODELS = ("resnet50", "lpt_full")
LABELS = {0: "Neither", 1: "Infection", 2: "Ischaemia", 3: "Both"}
SLUGS = {0: "neither", 1: "infection", 2: "ischaemia", 3: "both"}
WIDTH_MM, HEIGHT_MM = 183, 225
OVERLAY_ALPHA = 0.40


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(array: np.ndarray, name: str) -> Any:
    value = np.asarray(array)
    if value.size != 1:
        raise ValueError(f"{name}: expected a scalar, got {value.shape}")
    return value.reshape(()).item()


def read_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    required = {"case_rank", "image", "label", "similarity_group", "fold"}
    if not cases or required - cases[0].keys():
        raise ValueError(f"Case manifest requires columns {sorted(required)}")
    for row in cases:
        for key in ("case_rank", "label", "fold"):
            row[key] = int(row[key])
        if row["label"] not in LABELS or row["fold"] not in range(5):
            raise ValueError("Invalid label or outer fold in case manifest")
    if len(cases) != 20 or any(sum(c["label"] == x for c in cases) != 5 for x in LABELS):
        raise ValueError("Exactly five frozen cases per class are required")
    for key in ("case_rank", "image", "similarity_group"):
        if len({row[key] for row in cases}) != 20:
            raise ValueError(f"Case manifest contains duplicate {key}")
    if {row["case_rank"] for row in cases} != set(range(1, 21)):
        raise ValueError("Frozen case ranks must be exactly 1 through 20")
    stems = [Path(row["image"]).stem for row in cases]
    if len(set(stems)) != len(stems):
        raise ValueError("Image stems collide in archived NPZ naming")
    return sorted(cases, key=lambda row: row["case_rank"])


def read_archive(path: Path, case: dict[str, Any], seed: int) -> dict[str, Any]:
    required = {"rgb", "prior", "cam", "label", "pred_label", "fp32_pred_label",
                "cam_valid", "image", "seed", "outer_fold"}
    with np.load(path, allow_pickle=False) as archive:
        if required - set(archive.files):
            raise ValueError(f"Missing archive fields: {sorted(required - set(archive.files))}")
        data = {key: archive[key].copy() for key in required}
    for key in required - {"rgb", "prior", "cam"}:
        data[key] = scalar(data[key], key)
    if (str(data["image"]) != case["image"] or int(data["label"]) != case["label"]
            or int(data["seed"]) != seed or int(data["outer_fold"]) != case["fold"]):
        raise ValueError(f"Archive metadata does not match the frozen manifest: {path.name}")
    if data["rgb"].shape != (224, 224, 3) or data["rgb"].dtype != np.uint8:
        raise ValueError("RGB archive must be uint8 with shape (224,224,3)")
    for key in ("prior", "cam"):
        array = data[key]
        if array.shape != (224, 224) or not np.isfinite(array).all():
            raise ValueError(f"{key} must be a finite (224,224) array")
        if np.min(array) < 0 or np.max(array) > 1:
            raise ValueError(f"{key} values fall outside the documented [0,1] scale")
    if int(data["cam_valid"]) not in (0, 1):
        raise ValueError("cam_valid must be 0 or 1")
    if bool(data["cam_valid"]) and not np.any(data["cam"] > 0):
        raise ValueError("A zero CAM must be flagged unavailable, not valid")
    for key in ("pred_label", "fp32_pred_label"):
        if int(data[key]) not in LABELS:
            raise ValueError(f"{key} is not a valid four-class prediction")
    return data


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
        "savefig.facecolor": "white",
    })


def plate(seed: int, label: int, cases: list[dict[str, Any]],
          archives: dict[tuple[int, str, str], dict[str, Any]]) -> plt.Figure:
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    fig.text(0.5, 0.983, f"Fixed random cases: {LABELS[label]} | training seed {seed}",
             ha="center", va="top", size=9, weight="bold")
    fig.text(0.5, 0.962, "Similarity-grouped outer-held-out models; five cases from five distinct groups",
             ha="center", va="top", size=7)
    centres = np.linspace(0.145, 0.855, 4)
    titles = ("RGB input", "Fixed soft prior", "ResNet50 Grad-CAM", "Full LPT global Grad-CAM")
    for centre, title in zip(centres, titles):
        fig.text(centre, 0.938, title, ha="center", va="bottom", size=7, weight="bold")

    side_mm = 30.0
    width = side_mm / WIDTH_MM
    height = side_mm / HEIGHT_MM
    row_stride = 37.8 / HEIGHT_MM
    for row_index, case in enumerate(cases):
        top = 0.930 - row_index * row_stride
        pair = [archives[(seed, model, case["image"])] for model in MODELS]
        for column, centre in enumerate(centres):
            ax = fig.add_axes((centre - width / 2, top - height, width, height))
            ax.set_axis_off()
            if column == 1:
                ax.imshow(pair[0]["prior"], cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
            else:
                data = pair[max(column - 2, 0)]
                ax.imshow(data["rgb"], interpolation="nearest")
                if column >= 2 and data["cam_valid"]:
                    ax.imshow(data["cam"], cmap="magma", vmin=0, vmax=1,
                              alpha=OVERLAY_ALPHA, interpolation="nearest")
                elif column >= 2:
                    ax.text(0.5, 0.5, "CAM unavailable\n(zero / invalid map)",
                            transform=ax.transAxes, ha="center", va="center", size=7,
                            bbox={"facecolor": "white", "alpha": .90, "edgecolor": "none"})
            if column == 0:
                text = f"{Path(case['image']).stem} | outer fold {case['fold']}\nTruth: {LABELS[label]}"
            elif column == 1:
                text = "Transferred probability\nNot a target-domain mask"
            else:
                data = pair[column - 2]
                prediction = int(data["pred_label"])
                error = "ERROR" if prediction != label else "correct"
                text = f"Pred: {LABELS[prediction]} ({error})"
                if int(data["fp32_pred_label"]) != prediction:
                    text += f"\nFP32: {LABELS[int(data['fp32_pred_label'])]}"
                else:
                    text += "\nFP32 agrees"
            fig.text(centre, top - height - .004, text, ha="center", va="top",
                     size=7, linespacing=1.15)

    for left, cmap, title in ((0.155, "gray", "Prior probability"),
                              (0.615, "magma", "Grad-CAM (per-image normalised)")):
        colour_ax = fig.add_axes((left, .067, .22, .009))
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                           cax=colour_ax, orientation="horizontal", ticks=[0, .5, 1])
        bar.ax.tick_params(labelsize=6, length=2, pad=1)
        bar.outline.set_linewidth(.4)
        fig.text(left + .11, .081, title, size=6.5, ha="center", va="bottom")
    fig.text(.5, .033, "CAM target: archived OOF predicted class; identical magma overlay, opacity 0.40. No score-based case filtering.",
             ha="center", va="center", size=6.5)
    fig.text(.5, .017, "Exploratory localisation only; colours are not comparable attribution magnitudes across images or models.",
             ha="center", va="center", size=6.5)
    return fig


def render(repo_root: Path, output_root: Path, dpi: int) -> dict[str, Any]:
    analysis = output_root / "interpretability"
    case_path = analysis / "qualitative_cases.csv"
    cases = read_cases(case_path)
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    marker = figures / "figures_complete.json"
    if marker.exists():
        raise FileExistsError("A figures_complete.json already exists; preserve it or use a separate output root")
    sources = [{"kind": "frozen_case_manifest", "path": str(case_path.relative_to(output_root)),
                "sha256": sha256(case_path), "size_bytes": case_path.stat().st_size}]
    archives: dict[tuple[int, str, str], dict[str, Any]] = {}
    invalid = 0
    discrepant_predictions = 0
    for seed in SEEDS:
        for case in cases:
            pair = []
            for model in MODELS:
                path = analysis / "jobs" / f"seed_{seed}" / f"outer_{case['fold']}" / model / "cases" / f"{Path(case['image']).stem}.npz"
                data = read_archive(path, case, seed)
                archives[(seed, model, case["image"])] = data
                pair.append(data)
                invalid += not bool(data["cam_valid"])
                discrepant_predictions += int(data["pred_label"]) != int(data["fp32_pred_label"])
                sources.append({"kind": "case_array", "path": str(path.relative_to(output_root)),
                                "sha256": sha256(path), "size_bytes": path.stat().st_size})
            if not np.array_equal(pair[0]["rgb"], pair[1]["rgb"]):
                raise ValueError("Models have nonidentical RGB input archives")
            if not np.array_equal(pair[0]["prior"], pair[1]["prior"]):
                raise ValueError("Models have nonidentical fixed prior archives")
    # Check the frozen RGB/prior are identical across seeds, not just within a seed.
    for case in cases:
        reference = archives[(SEEDS[0], MODELS[0], case["image"])]
        for seed in SEEDS[1:]:
            other = archives[(seed, MODELS[0], case["image"])]
            for key in ("rgb", "prior"):
                if not np.array_equal(reference[key], other[key]):
                    raise ValueError(f"Frozen {key} changed across training seeds")
    script = Path(__file__).resolve()
    sources.append({"kind": "renderer", "path": str(script.relative_to(repo_root)),
                    "sha256": sha256(script), "size_bytes": script.stat().st_size})
    configure_style()
    products: list[Path] = []
    for seed in SEEDS:
        combined = figures / f"qualitative_seed_{seed}_all_classes.pdf"
        with PdfPages(combined) as pdf:
            for label in LABELS:
                selected = [case for case in cases if case["label"] == label]
                fig = plate(seed, label, selected, archives)
                stem = figures / f"qualitative_seed_{seed}_{SLUGS[label]}"
                for extension in ("png", "pdf", "svg"):
                    path = stem.with_suffix(f".{extension}")
                    fig.savefig(path, dpi=dpi)
                    products.append(path)
                pdf.savefig(fig)
                plt.close(fig)
        products.append(combined)

    captions = figures / "captions_and_methods.txt"
    captions.write_text(
        "Exploratory similarity-grouped outer-held-out Grad-CAM displays.\n\n"
        "The prediction-blind, class-stratified manifest contains 20 images from 20 distinct similarity "
        "groups (five images per class). Every case is shown for every one of five training seeds; "
        "neither prediction confidence nor correctness was used by this renderer to select or omit a case. "
        "Each row shows the unmodified archived 224 x 224 RGB evaluation input, the fixed transferred "
        "soft lesion prior, ResNet50 Grad-CAM, and Full LPT global-branch Grad-CAM. "
        "Grad-CAM targets the archived OOF predicted class, fixed even if the FP32 prediction differs. "
        "Model outputs are from each image's held-out outer fold; "
        "Pred denotes the archived prediction, and FP32 denotes the prediction during gradient evaluation. "
        "Any disagreement is explicitly displayed. Truth denotes the dataset label; Neither means neither "
        "infection nor ischaemia, and Both means both labels. Full LPT maps describe the global branch only and "
        "do not visualise the full contribution of its ROI, statistics, or prior pathways.\n\n"
        "Image integrity: RGB inputs and prior arrays are identical across compared models and seeds. "
        "No additional crop, contrast, brightness, gamma adjustment, retouching or image generation is "
        "performed. The displayed images use the existing 224 x 224 evaluation resize; no physical "
        "scale calibration is available. The grayscale prior spans 0 to 1. Grad-CAM is normalised "
        "within each image/model and displayed with the same magma colour scale, 0 to 1, and constant "
        "0.40 opacity; map intensities are therefore not comparable absolute attribution magnitudes "
        "across images or models. Invalid or zero maps are marked unavailable, without synthetic heat. "
        "The source arrays remain unchanged. The transferred prior is not a target-domain reference "
        "mask. These maps are exploratory localisation displays, not anatomical, causal or clinical "
        "validation, and no inferential statistic is derived from the visual sample.\n\n"
        "Export: 183 x 225 mm; 7 pt case labels; PNG composite at the requested dpi; "
        "PDF and SVG with editable text. Original evaluation rasters are 224 x 224 pixels; increasing "
        "export dpi does not increase their underlying spatial information.\n",
        encoding="utf-8",
    )
    products.append(captions)
    registry = figures / "source_registry.csv"
    with registry.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["kind", "path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(sources)
    products.append(registry)
    complete = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases), "seed_count": len(SEEDS), "model_count": len(MODELS),
        "plate_count": 20, "source_npz_count": 200,
        "invalid_cam_count": int(invalid),
        "archived_fp32_prediction_disagreements": int(discrepant_predictions),
        "cam_target": "archived_oof_predicted_class", "dpi": dpi, "figure_size_mm": [WIDTH_MM, HEIGHT_MM],
        "visual_qa": "required_after_export",
        "assets": [{"path": str(path.relative_to(output_root)), "sha256": sha256(path),
                    "size_bytes": path.stat().st_size} for path in products],
    }
    marker.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 150:
        parser.error("--dpi must be at least 150")
    repo = args.repo_root.resolve()
    output = (args.output_root or repo / "outputs" / "associate_editor_revision_v1").resolve()
    result = render(repo, output, args.dpi)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}, indent=2))


if __name__ == "__main__":
    main()
