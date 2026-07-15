# Cross-Task Lesion-Prior Transfer

This directory contains the structured lesion-prior transfer (LPT) experiments that apply DFUC2022-derived soft lesion priors to DFUC2021 four-class classification.

## Prior Generation

Generate 224 x 224 DFUC2021 soft lesion-prior maps from the trained DFUC2022 segmentation model:

```bash
python experiments/cross_task_prior_transfer/generate_dfuc2021_priors.py \
  --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt \
  --device cuda \
  --amp auto \
  --batch-size 256 \
  --num-workers 4 \
  --skip-existing
```

Generated files are written under:

```text
data_processed/cross_task_lesion_prior_v1/dfuc2021_train_priors/
data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv
data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_summary.json
```

The prior metadata include `prior_mean`, `prior_max`, `prior_area_ratio`, connected-component area, bounding-box area, `prior_confidence`, and fallback status. Fallback priors are assigned when `prior_max < 0.15` or `largest_component_area_ratio < 0.01`.

## Full LPT Model

The full LPT classifier combines:

- a global ResNet50 RGB branch;
- soft prior modulation, `F_guided = F_global * (1 + alpha * M)`;
- a MobileNetV3-small ROI branch based on prior-derived crops;
- prior statistics;
- an attention-alignment head;
- auxiliary binary heads for infection and ischaemia.

Non-fallback samples use prior-derived ROI crops. Fallback samples use full-image prior-enhanced inputs to avoid meaningless crops.

Run standard five-fold LPT training with:

```bash
python experiments/cross_task_prior_transfer/run_lpt_cv.py \
  --config experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml \
  --device cuda \
  --amp auto \
  --batch-size 128 \
  --num-workers 4 \
  --skip-existing
```

Similarity-grouped LPT configs are also provided and should be used for grouped retraining analyses.

## Ablations

The following configs isolate the contribution of individual prior pathways:

| Config | Purpose |
| --- | --- |
| `cross_task_lpt_no_alignment_cv5_v1.yaml` | Remove attention-prior alignment loss |
| `cross_task_lpt_no_prior_mod_cv5_v1.yaml` | Remove soft prior modulation |
| `cross_task_lpt_no_roi_cv5_v1.yaml` | Remove the prior-derived ROI branch |
| `cross_task_lpt_stats_only_cv5_v1.yaml` | Keep prior statistics but remove spatial prior pathways |
| `cross_task_lpt_random_prior_cv5_v1.yaml` | Shuffle real priors and prior statistics across samples |

## Manuscript Alignment

These experiments support the direct LPT and ablation results. The manuscript reports that direct structured prior injection caused negative transfer under standard five-fold validation, whereas later RA-LPT stacking and selected prior-informed probability fusion treated prior information as uncertain auxiliary evidence.
