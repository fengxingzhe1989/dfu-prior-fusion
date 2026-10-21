# Frozen grouped interpretability rerun: audited summary

All 50 job markers and their listed assets passed SHA256 verification. The complete matrix contains 59,550 image × seed × model rows: 5,955 images, 5,670 similarity groups, five training seeds and two corresponding outer-fold CNN models.

## Primary descriptive score

Positive values mean a larger decrease in the original predicted-class probability after prior-region occlusion than after the mean of the same shape/area translated controls. Probabilities are not logits. Each fitted model's original predicted class is held fixed for its own perturbations; the two models may therefore have different predicted-class targets. The true-label score is reported separately.

| entity | mean_across_seed_image_means | sample_sd_across_seed_image_means | ci95_low | ci95_high | n_valid_images_min_per_seed | n_valid_images_max_per_seed |
| --- | --- | --- | --- | --- | --- | --- |
| resnet50 | 0.08443 | 0.00501 | 0.07791 | 0.09093 | 4929 | 4929 |
| lpt_full | 0.10378 | 0.00452 | 0.09654 | 0.11116 | 4929 | 4929 |
| lpt_full_minus_resnet50 | 0.01935 | 0.00614 | 0.01166 | 0.02651 | 4929 | 4929 |

## Denominators and numerical agreement

| model | seed | n_images | n_groups | n_mask_empty | n_mask_immovable | n_mask_movable | n_invalid_cam | n_fp32_prediction_disagreements | max_fp32_archived_probability_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lpt_full | 20260925 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 6 | 0.08209 |
| lpt_full | 20260926 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 15 | 0.15507 |
| lpt_full | 20260927 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 10 | 0.09027 |
| lpt_full | 20260928 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 10 | 0.12735 |
| lpt_full | 20260929 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 10 | 0.13874 |
| resnet50 | 20260925 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 5 | 0.01801 |
| resnet50 | 20260926 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 3 | 0.01330 |
| resnet50 | 20260927 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 2 | 0.01859 |
| resnet50 | 20260928 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 0 | 0.01772 |
| resnet50 | 20260929 | 5955 | 5670 | 1024 | 2 | 4929 | 1024 | 3 | 0.01553 |

## Reporting boundaries

- Image means are computed separately within each seed, then summarized across all five seeds using sample SD. Image × seed rows are not treated as independent observations.
- 5,000 paired crossed seed × similarity-group bootstrap replicates share the same group and seed draws across models, targets and classes. Sampling groups retains all their images with multiplicity. Each seed's denominator is its valid sampled image count, not its number of groups or mask placements.
- The model difference uses the intersection of valid paired image outcomes within each seed. Empty and immovable masks are reported separately and have no prior-minus-random estimate. Invalid CAM metrics are NA. Other missing metrics are excluded only from their own denominators.
- Undefined bootstrap replicates are reported and are not replaced. Percentile 95% intervals are descriptive, not simultaneous confidence intervals. No p values or confirmatory hypothesis tests are reported.
- CAM concentration and occlusion scores describe model behaviour relative to generated priors, not the anatomical accuracy of a mask, causal clinical features, a validated diagnosis, or an explanation of the entire RA-LPT stack.
- Intervals reflect the available fitted seeds and image-similarity-group resampling, not patient independence, external validation, complete bootstrap retraining or prior-generator uncertainty.
- Qualitative cases were frozen independently of prediction quality. Their 20-image subset is not the quantitative sample. This program creates summaries only; it does not edit manuscripts or select favourable cases.
