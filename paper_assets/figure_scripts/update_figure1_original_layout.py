#!/usr/bin/env python3
"""Update Figure 1 reporting language while preserving the original artwork."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKGROUND = REPO_ROOT / "paper_assets" / "figure_inputs" / "Figure1_background.png"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "figures" / "Figure1.svg"
DEFAULT_DFUC2021_IMAGE_DIR = REPO_ROOT / "data_raw" / "2021_dfuc2021" / "train" / "images"
DEFAULT_DFUC2022_IMAGE_DIR = (
    REPO_ROOT / "data_raw" / "2022_dfuc2022" / "train" / "DFUC2022_train_images"
)
DEFAULT_DFUC2022_MASK_DIR = (
    REPO_ROOT / "data_raw" / "2022_dfuc2022" / "train" / "DFUC2022_train_masks"
)


def multiline_text(
    lines: list[str],
    *,
    x: float,
    y: float,
    font_size: float,
    line_height: float,
    color: str,
    weight: int = 400,
    italic: bool = False,
    anchor: str = "middle",
) -> str:
    style = "font-style:italic;" if italic else ""
    spans = []
    for index, line in enumerate(lines):
        dy = "0" if index == 0 else str(line_height)
        spans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{font_size}" '
        f'font-weight="{weight}" fill="{color}" style="{style}">'
        + "".join(spans)
        + "</text>"
    )


def build_svg(
    background: Path,
    dfuc2021_image_dir: Path,
    dfuc2022_image_dir: Path,
    dfuc2022_mask_dir: Path,
) -> str:
    encoded = base64.b64encode(background.read_bytes()).decode("ascii")
    dfuc2021_examples = {
        "none": dfuc2021_image_dir / "301684.jpg",
        "infection": dfuc2021_image_dir / "302107.jpg",
        "ischaemia": dfuc2021_image_dir / "301209.jpg",
        "both": dfuc2021_image_dir / "301806.jpg",
    }
    dfuc2022_rgb = dfuc2022_image_dir / "101657.jpg"
    dfuc2022_mask = dfuc2022_mask_dir / "101657.png"
    dfuc2022_dataset_examples = {
        "101749": (dfuc2022_image_dir / "101749.jpg", dfuc2022_mask_dir / "101749.png"),
        "100766": (dfuc2022_image_dir / "100766.jpg", dfuc2022_mask_dir / "100766.png"),
    }
    placements = {
        "none": (31, 209, 117, 126),
        "infection": (165, 209, 117, 126),
        "ischaemia": (31, 386, 117, 132),
        "both": (165, 386, 117, 132),
    }
    required_sources = [
        *dfuc2021_examples.values(),
        dfuc2022_rgb,
        dfuc2022_mask,
        *(path for pair in dfuc2022_dataset_examples.values() for path in pair),
    ]
    missing = [str(path) for path in required_sources if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Figure 1 source assets: " + ", ".join(missing))

    clip_paths = []
    source_image_layers = []
    for label, source_path in dfuc2021_examples.items():
        x, y, width, height = placements[label]
        clip_id = f"clip-dfuc2021-{label}"
        source_encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
        clip_paths.append(
            f'<clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{width}" '
            f'height="{height}" rx="7"/></clipPath>'
        )
        source_image_layers.extend(
            [
                f'<rect x="{x - 2}" y="{y - 2}" width="{width + 4}" '
                f'height="{height + 4}" rx="8" fill="#ffffff"/>',
                f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
                f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})" '
                f'href="data:image/jpeg;base64,{source_encoded}" '
                f'xlink:href="data:image/jpeg;base64,{source_encoded}"/>',
                f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="7" '
                f'fill="none" stroke="#c9d0da" stroke-width="1.1"/>',
            ]
        )

    # Show two real DFUC2022 RGB-mask training pairs, while reserving image
    # the original image-mask example for the prior-generation panel.
    dfuc2022_dataset_placements = {
        "101749": (734, 92),
        "100766": (834, 92),
    }
    dfuc2022_dataset_layers = []
    for sample_id, (y, height) in dfuc2022_dataset_placements.items():
        rgb_path, mask_path = dfuc2022_dataset_examples[sample_id]
        for modality, x, width, source_path, media_type in (
            ("rgb", 42, 92, rgb_path, "image/jpeg"),
            ("mask", 178, 92, mask_path, "image/png"),
        ):
            clip_id = f"clip-dfuc2022-dataset-{sample_id}-{modality}"
            source_encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
            clip_paths.append(
                f'<clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{width}" '
                f'height="{height}" rx="5"/></clipPath>'
            )
            dfuc2022_dataset_layers.extend(
                [
                    f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
                    f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})" '
                    f'href="data:{media_type};base64,{source_encoded}" '
                    f'xlink:href="data:{media_type};base64,{source_encoded}"/>',
                    f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="5" '
                    f'fill="none" stroke="#c9d0da" stroke-width="1.0"/>',
                ]
            )
        dfuc2022_dataset_layers.append(
            multiline_text(
                ["+"],
                x=156,
                y=y + 53,
                font_size=16.0,
                line_height=15,
                color="#172033",
                weight=700,
            )
        )

    overlays = [
        *source_image_layers,
        # Four-panel canvas: recenter the title after removing Main interpretation.
        '<rect x="0" y="0" width="1280" height="60" fill="#ffffff"/>',
        multiline_text(
            ["Study design and cross-dataset lesion-prior framework"],
            x=640,
            y=44,
            font_size=26.0,
            line_height=28,
            color="#102a58",
            weight=700,
        ),
        # Two real DFUC2022 image-mask pairs define the segmentation dataset
        # without reusing the pair shown in the prior-generation panel.
        '<rect x="25" y="712" width="266" height="320" fill="#ffffff"/>',
        multiline_text(
            ["Image"],
            x=88,
            y=726,
            font_size=10.5,
            line_height=13,
            color="#087d78",
            weight=700,
        ),
        multiline_text(
            ["Mask"],
            x=224,
            y=726,
            font_size=10.5,
            line_height=13,
            color="#087d78",
            weight=700,
        ),
        *dfuc2022_dataset_layers,
        multiline_text(
            ["2,000 images + 2,000 masks"],
            x=156,
            y=951,
            font_size=11.0,
            line_height=14,
            color="#087d78",
            weight=700,
        ),
        multiline_text(
            ["•  Segmentation masks"],
            x=45,
            y=981,
            font_size=12.2,
            line_height=16,
            color="#172033",
            anchor="start",
        ),
        multiline_text(
            ["•  Used only to train the", "   prior generator"],
            x=45,
            y=1009,
            font_size=12.2,
            line_height=21,
            color="#172033",
            anchor="start",
        ),
        # Widen Classification strategies by 42 units while preserving height.
        '<g transform="translate(620 0) scale(1.1105263158 1) translate(-620 0)">',
        # Keep the baseline family list aligned with Methods Section 2.3.
        '<rect x="635" y="273" width="336" height="34" fill="#ffffff"/>',
        multiline_text(
            [
                "ResNet50 / EfficientNet-B0 / ConvNeXt-Tiny /",
                "EfficientNetV2-S",
            ],
            x=807.5,
            y=286,
            font_size=10.2,
            line_height=15,
            color="#172033",
            italic=True,
        ),
        # Restore the manuscript-defined three-source fusion path. The panel
        # retains explicit post hoc/rank language so it cannot be read as the
        # primary matched comparison or an unbiased validation estimate.
        '<rect x="636" y="807" width="335" height="234" fill="#ffffff"/>',
        multiline_text(
            ["D. Post hoc prior-informed full-OOF"],
            x=642,
            y=827,
            font_size=18.0,
            line_height=18,
            color="#102a58",
            weight=700,
            anchor="start",
        ),
        '<rect x="638" y="837" width="95" height="68" rx="7" fill="#edf8f6" '
        'stroke="#087d78" stroke-width="1.1"/>',
        multiline_text(
            ["LPT-output", "0.400"],
            x=685.5,
            y=860,
            font_size=12.5,
            line_height=19,
            color="#087d78",
            weight=700,
        ),
        multiline_text(
            ["+"],
            x=739.5,
            y=875,
            font_size=16.0,
            line_height=16,
            color="#172033",
            weight=700,
        ),
        '<rect x="746" y="837" width="95" height="68" rx="7" fill="#eef4fa" '
        'stroke="#315a8c" stroke-width="1.1"/>',
        multiline_text(
            ["ConvNeXt-Tiny", "0.350"],
            x=793.5,
            y=860,
            font_size=12.0,
            line_height=19,
            color="#315a8c",
            weight=700,
        ),
        multiline_text(
            ["+"],
            x=847.5,
            y=875,
            font_size=16.0,
            line_height=16,
            color="#172033",
            weight=700,
        ),
        '<rect x="854" y="837" width="95" height="68" rx="7" fill="#eef4fa" '
        'stroke="#315a8c" stroke-width="1.1"/>',
        multiline_text(
            ["EffNetV2-S", "0.250"],
            x=901.5,
            y=860,
            font_size=12.0,
            line_height=19,
            color="#315a8c",
            weight=700,
        ),
        '<line x1="685.5" y1="905" x2="685.5" y2="916" stroke="#687386" '
        'stroke-width="1.2"/>',
        '<line x1="793.5" y1="905" x2="793.5" y2="916" stroke="#687386" '
        'stroke-width="1.2"/>',
        '<line x1="901.5" y1="905" x2="901.5" y2="916" stroke="#687386" '
        'stroke-width="1.2"/>',
        '<line x1="685.5" y1="916" x2="901.5" y2="916" stroke="#687386" '
        'stroke-width="1.2"/>',
        '<line x1="685.5" y1="916" x2="685.5" y2="918" stroke="#687386" '
        'stroke-width="1.4"/>',
        '<polygon points="681.5,917 689.5,917 685.5,922" fill="#687386"/>',
        '<rect x="638" y="922" width="95" height="64" rx="7" fill="#f7f9fb" '
        'stroke="#687386" stroke-width="1.0"/>',
        multiline_text(
            ["Weighted sum", "p_mix"],
            x=685.5,
            y=944,
            font_size=11.7,
            line_height=19,
            color="#172033",
            weight=700,
        ),
        '<line x1="737" y1="954" x2="742" y2="954" stroke="#687386" '
        'stroke-width="1.4"/>',
        '<polygon points="740,950 740,958 746,954" fill="#687386"/>',
        '<rect x="746" y="922" width="95" height="64" rx="7" fill="#f4effb" '
        'stroke="#6b4aa0" stroke-width="1.0"/>',
        multiline_text(
            ["Class-logit bias", "adjustment"],
            x=793.5,
            y=944,
            font_size=11.0,
            line_height=19,
            color="#6b4aa0",
            weight=700,
        ),
        '<line x1="845" y1="954" x2="850" y2="954" stroke="#687386" '
        'stroke-width="1.4"/>',
        '<polygon points="848,950 848,958 854,954" fill="#687386"/>',
        '<rect x="854" y="922" width="95" height="64" rx="7" fill="#ffffff" '
        'stroke="#9ca3af" stroke-width="1.0"/>',
        multiline_text(
            ["Four-class", "candidate"],
            x=901.5,
            y=944,
            font_size=11.7,
            line_height=19,
            color="#172033",
            weight=700,
        ),
        '<rect x="637" y="999" width="335" height="42" fill="#ffffff"/>',
        '<rect x="642" y="1004" width="330" height="34" rx="6" fill="#ffffff" '
        'stroke="#087d78" stroke-width="1.2" stroke-dasharray="6 4"/>',
        multiline_text(
            ["post hoc descriptive full-OOF candidate; rank 13"],
            x=807,
            y=1026,
            font_size=13.0,
            line_height=14,
            color="#087d78",
        ),
        '</g>',
        # Shift Validation and robustness to the right of the widened Panel 3.
        '<g transform="translate(42 0)">',
        # Keep validation terminology aligned with the reporting hierarchy.
        '<rect x="1022" y="270" width="180" height="135" fill="#ffffff"/>',
        multiline_text(
            [
                "• standard 5-fold CV",
                "• similarity-grouped OOF",
                "• group-level bootstrap CI",
                "• secondary fusion analysis",
            ],
            x=1032,
            y=287,
            font_size=13.0,
            line_height=24,
            color="#172033",
            weight=500,
            anchor="start",
        ),
        '</g>',
        # Remove the former arrow into Panel 5 and restore Panel 4's right border.
        '<rect x="1256" y="523" width="24" height="35" fill="#ffffff"/>',
        '<line x1="1263" y1="523" x2="1263" y2="558" stroke="#102a58" '
        'stroke-width="1.3"/>',
    ]
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="6.47in" height="5.49in" viewBox="0 0 1280 1086">
  <title>Study design and cross-dataset lesion-prior framework</title>
  <desc>Four-panel publication workflow with explicit internal-validation boundaries.</desc>
  <defs>
    {''.join(clip_paths)}
    <image id="background-art" x="0" y="0" width="1448" height="1086"
           preserveAspectRatio="none" href="data:image/png;base64,{encoded}"
           xlink:href="data:image/png;base64,{encoded}"/>
  </defs>
  <svg x="0" y="0" width="620" height="1086" viewBox="0 0 620 1086"
       preserveAspectRatio="none">
    <use href="#background-art" xlink:href="#background-art"/>
  </svg>
  <svg x="620" y="0" width="422" height="1086" viewBox="620 0 380 1086"
       preserveAspectRatio="none">
    <use href="#background-art" xlink:href="#background-art"/>
  </svg>
  <svg x="1042" y="0" width="238" height="1086" viewBox="1000 0 238 1086"
       preserveAspectRatio="none">
    <use href="#background-art" xlink:href="#background-art"/>
  </svg>
  {''.join(overlays)}
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dfuc2021-image-dir", type=Path, default=DEFAULT_DFUC2021_IMAGE_DIR)
    parser.add_argument("--dfuc2022-image-dir", type=Path, default=DEFAULT_DFUC2022_IMAGE_DIR)
    parser.add_argument("--dfuc2022-mask-dir", type=Path, default=DEFAULT_DFUC2022_MASK_DIR)
    args = parser.parse_args()
    if not args.background.exists():
        raise FileNotFoundError(
            f"Missing Figure 1 background: {args.background}. Pass --background explicitly."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_svg(
            args.background,
            args.dfuc2021_image_dir,
            args.dfuc2022_image_dir,
            args.dfuc2022_mask_dir,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
