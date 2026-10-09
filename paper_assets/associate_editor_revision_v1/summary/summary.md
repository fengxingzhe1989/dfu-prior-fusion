# Frozen associate-editor revision experiment: final summary

All results below were computed after the classification, stacking, and segmentation completion markers were present. No training artifact was modified by the summary step.

## Three prespecified incremental comparisons

| contrast | estimate_mean_seed_difference | sd_seed_difference | ci_low | ci_high | approximate_two_sided_p | holm_adjusted_p |
| --- | --- | --- | --- | --- | --- | --- |
| full_minus_neutral | 0.0144 | 0.0037 | 0.0048 | 0.0244 | 0.0044 | 0.0132 |
| ra_minus_output_only | -0.0026 | 0.0017 | -0.0066 | 0.0012 | 0.1540 | 0.3079 |
| rgb_prior_minus_rgb | -0.0016 | 0.0031 | -0.0057 | 0.0025 | 0.4223 | 0.4223 |

The estimate is the mean of the five seed-specific pooled-OOF Macro-F1 differences. Confidence intervals use 5,000 paired crossed seed x similarity-group bootstrap replicates unless a non-default replicate count is explicitly recorded in the completion marker. Approximate p values use plus-one two-sided bootstrap tails; Holm adjustment is across only these three prespecified new contrasts. Confidence intervals are not simultaneous.

## Classification systems

| model | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd |
| --- | --- | --- | --- | --- |
| convnext_tiny | 0.9175 | 0.0067 | 0.9150 | 0.0041 |
| efficientnetb0 | 0.8960 | 0.0056 | 0.8977 | 0.0036 |
| efficientnetv2s | 0.9036 | 0.0141 | 0.9071 | 0.0099 |
| full_lpt | 0.8921 | 0.0059 | 0.8916 | 0.0083 |
| lpt_no_alignment | 0.8726 | 0.0167 | 0.8729 | 0.0133 |
| lpt_no_prior_mod | 0.8783 | 0.0215 | 0.8847 | 0.0108 |
| lpt_no_roi | 0.8659 | 0.0137 | 0.8759 | 0.0097 |
| lpt_stats_only | 0.8724 | 0.0110 | 0.8799 | 0.0094 |
| neutral_full_lpt | 0.8777 | 0.0085 | 0.8839 | 0.0074 |
| output_only_lpt_stack | 0.9206 | 0.0028 | 0.9258 | 0.0021 |
| ra_lpt_stack | 0.9181 | 0.0038 | 0.9260 | 0.0027 |
| resnet50 | 0.8559 | 0.0107 | 0.8696 | 0.0090 |
| strong_rgb_plus_prior | 0.9354 | 0.0073 | 0.9363 | 0.0037 |
| strong_rgb_stack | 0.9370 | 0.0050 | 0.9370 | 0.0023 |

## Descriptive complete-system reference contrasts

| contrast | difference | difference_sd_across_seeds |
| --- | --- | --- |
| full_minus_resnet50 | 0.0362 | 0.0141 |
| ra_minus_resnet50 | 0.0622 | 0.0123 |

These reference contrasts compare complete systems and do not isolate the effect of the spatial prior from architecture or system composition.

## Source-domain segmentation audit

| grid | dice_mean | dice_sd | iou_mean | iou_sd | precision_mean | precision_sd | recall_mean | recall_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 224x224 | 0.4884 | 0.0256 | 0.4148 | 0.0207 | 0.8815 | 0.0172 | 0.4927 | 0.0302 |
| 512x384 | 0.7652 | 0.0054 | 0.6638 | 0.0063 | 0.7919 | 0.0098 | 0.7999 | 0.0061 |

## Reporting boundaries

- These are internal model-development estimates using image-similarity groups, not patient- or case-independent validation.
- The classification experiment keeps the original transferred prior generator fixed and therefore does not include prior-generator retraining uncertainty.
- Source-domain segmentation accuracy does not establish target-domain DFUC2021 prior accuracy because target-domain lesion masks were unavailable.
- The bootstrap resamples existing similarity groups and training seeds; it does not retrain the complete pipeline within each replicate and does not capture historical model-development uncertainty.
- Six inner-OOF EfficientNetV2-S prediction files containing 16 non-finite rows were rerun under the same frozen configuration and seed before stack fitting. The original files were archived and the repair is recorded in classification_prediction_integrity.json.
- Non-significance is not evidence of equivalence. These estimates should not be presented as external validation or clinical readiness.
