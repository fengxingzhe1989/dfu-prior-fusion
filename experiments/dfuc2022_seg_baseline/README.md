# DFUC2022 Segmentation Baseline

This experiment trains the DFUC2022 lesion segmentation model used to generate cross-dataset soft lesion-prior maps for DFUC2021 classification.

## Purpose

The segmentation model is a support component, not a standalone clinical segmentation endpoint. In the manuscript, the DFUC2022 model provides uncertain auxiliary lesion-prior maps. These priors are then used by LPT, RA-LPT, and probability-fusion analyses on DFUC2021.

## Dataset Contract

The expected raw dataset layout is:

```text
data_raw/2022_dfuc2022/train/DFUC2022_train_images/
data_raw/2022_dfuc2022/train/DFUC2022_train_masks/
```

The manuscript audit uses 2,000 segmentation-training images. Raw images and masks are not redistributed in this code release.

## Split Generation

Create fixed five-fold segmentation splits with:

```bash
python data_processing/create_dfuc2022_seg_splits.py
```

The active config is:

```text
experiments/dfuc2022_seg_baseline/configs/dfuc2022_seg_deeplabv3p_cv5_v1.yaml
```

## Model and Training

The baseline uses DeepLabV3+ with a ResNet50 encoder, ImageNet initialization, 512 x 384 training inputs, Dice plus binary cross-entropy loss, AdamW, cosine learning-rate decay, and early stopping.

Run the five-fold training workflow with:

```bash
python experiments/dfuc2022_seg_baseline/run_cv.py \
  --device cuda \
  --amp auto \
  --batch-size 8 \
  --num-workers 4 \
  --skip-existing
```

Generated outputs are written under `outputs/` and are intentionally not committed:

```text
outputs/logs/dfuc2022_seg_deeplabv3p_cv5_v1_*.json
outputs/logs/dfuc2022_seg_deeplabv3p_cv5_v1_cv_summary.csv
outputs/checkpoints/dfuc2022_seg_deeplabv3p_*.pt
```

## Manuscript Alignment

The manuscript reports the DFUC2022 segmentation validation metrics as five-fold mean +/- SD. These values document that the prior generator is functional; they are not DFUC2021 segmentation estimates.

During downstream prior generation, DeepLabV3+ is applied to 224 x 224 DFUC2021 inputs so that prior maps align with the classification input grid. The generated priors are treated only as auxiliary uncertainty-aware signals.
