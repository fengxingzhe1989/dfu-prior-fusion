# DFUC2021 RGB Classification Baselines

This experiment directory trains pure-RGB DFUC2021 four-class classifiers used as the no-prior references in the manuscript.

## Dataset Contract

The expected raw dataset layout is:

```text
data_raw/2021_dfuc2021/train/train.csv
data_raw/2021_dfuc2021/train/images/
```

The manuscript analysis uses 5,955 valid four-class images from the original 9,949-row annotation table. Excluded records are handled by the data-preparation scripts and documented in the supplementary workbook. Raw images and labels are not redistributed in this code release.

Class order is fixed throughout the project:

```text
0 none
1 infection
2 ischaemia
3 both
```

## Split Generation

Audit the annotation inclusion/exclusion counts used for Supplementary Table S1 with:

```bash
python data_processing/run_step.py dfuc2021-exclusion-audit
```

Create the standard five-fold classification manifest with:

```bash
python experiments/dfuc2021_cls_baseline/prepare_splits.py \
  --train-csv data_raw/2021_dfuc2021/train/train.csv \
  --image-dir data_raw/2021_dfuc2021/train/images \
  --output-dir data_processed/dfuc2021_cls_v1/splits
```

Similarity-grouped folds are created by the reliability-aware prior-fusion utilities after the image-similarity audit.

## Training

The standard ResNet50 baseline is:

```bash
python experiments/dfuc2021_cls_baseline/run_cv.py \
  --config experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml \
  --device cuda \
  --amp auto \
  --batch-size 64 \
  --num-workers 4 \
  --skip-existing
```

Additional backbone configs are provided for EfficientNet-B0, ConvNeXt-Tiny, and EfficientNetV2-S, including similarity-grouped variants.

## Outputs

Generated outputs are written under `outputs/` and are intentionally not committed:

```text
outputs/logs/*_cv_summary.csv
outputs/logs/*_fold{0..4}_metrics.json
outputs/predictions/*_fold{0..4}_val_predictions.csv
outputs/checkpoints/*.pt
```

The primary metric is Macro-F1. Balanced accuracy, macro precision, macro recall, accuracy, confusion matrices, and per-class metrics are recorded for auditability.

## Manuscript Alignment

These RGB classifiers provide:

- the moderate ResNet50 reference used in the direct LPT comparison;
- stronger pure-vision backbones used in the standard-CV support analyses;
- grouped RGB baselines used in the final similarity-grouped comparisons;
- probability inputs for RA-LPT stacking and prior-informed probability-fusion candidates.
