# reliability_aware_prior_fusion

Experiment code for similarity auditing, grouped validation, RA-LPT stacking, prior-informed probability fusion, and sensitivity analysis.

## Required Inputs

Run the DFUC2021 classification, DFUC2022 segmentation, and cross-task prior-transfer experiments first. The main inputs are generated under:

```text
data_processed/dfuc2021_cls_v1/
data_processed/cross_task_lesion_prior_v1/
outputs/predictions/
outputs/logs/
```

Raw datasets, generated manifests, prediction CSV files, and checkpoints are intentionally not committed.

## Core Entry Points

```bash
python experiments/reliability_aware_prior_fusion/audit_image_similarity.py
python experiments/reliability_aware_prior_fusion/audit_cross_dataset_similarity.py
python experiments/reliability_aware_prior_fusion/make_similarity_grouped_splits.py
python experiments/reliability_aware_prior_fusion/oof_stack.py
python experiments/reliability_aware_prior_fusion/search_publication_fusion.py
python experiments/reliability_aware_prior_fusion/build_publication_final_assets.py
python experiments/reliability_aware_prior_fusion/similarity_sensitivity_metrics.py
python experiments/reliability_aware_prior_fusion/cross_dataset_overlap_exclusion_sensitivity.py
python experiments/reliability_aware_prior_fusion/derive_grouped_clinical_metrics.py --help
```

Additional analysis helpers in this directory summarize grouped retraining outputs, build seed-level OOF predictions, run explainability checks, and reconstruct publication-facing result tables from fold-strict OOF predictions. Manuscript-formatting scripts and generated manuscript artifacts are not included in this experiment-code release.

## Manuscript Alignment

`oof_stack.py` implements RA-LPT as fold-strict logistic-regression stacking with optional prior reliability statistics and interaction features. `search_publication_fusion.py` enumerates prior-informed probability-fusion candidates, and `build_publication_final_assets.py` reconstructs the selected prior-informed probability-fusion candidate:

```text
p_mix = 0.400 p_LPT-output + 0.350 p_ConvNeXt-Tiny + 0.250 p_EfficientNetV2-S
```

The post hoc candidate applies the manuscript class-logit bias vector `[-0.35, -0.025, -0.25, 0.0]`. It does not directly include raw prior reliability statistics or reliability-statistic interaction terms; those explicit reliability features are reserved for the RA-LPT stacking analyses. The unrestricted full-OOF winner was RGB-only, whereas this highest-ranked candidate containing strong LPT-output evidence ranked 13th. It is therefore retained as a descriptive development analysis rather than presented as the final model.

The archived manuscript search used larger pools than the script's general-purpose defaults. Reproduce that search with explicit arguments:

```bash
python experiments/reliability_aware_prior_fusion/search_publication_fusion.py \
  --top-methods 18 \
  --top-triples 12 \
  --top-bias-candidates 60 \
  --output-prefix publication_baseline5
```

Supplementary Tables S12a and S12b are derived from the fixed grouped OOF decisions and probabilities without model retraining or threshold selection:

```bash
python experiments/reliability_aware_prior_fusion/derive_grouped_clinical_metrics.py \
  --manifest paper_assets/source_tables/reviewer_revision_v2/dfuc2021_similarity_grouped_manifest.csv \
  --prediction "ResNet50=paper_assets/source_tables/reviewer_revision_v2/resnet50_grouped_oof_predictions.csv" \
  --prediction "Full LPT=paper_assets/source_tables/reviewer_revision_v2/principal_full_lpt_grouped_oof_predictions.csv" \
  --prediction "Moderate output-only LPT stack=paper_assets/source_tables/reviewer_revision_v2/lpt_ablation_output_grouped_oof_predictions.csv" \
  --prediction "RA-LPT=paper_assets/source_tables/reviewer_revision_v2/ra_lpt_grouped_oof_predictions.csv" \
  --prediction "Strong RGB stack=paper_assets/source_tables/reviewer_revision_v2/strong_cls_grouped_oof_predictions.csv" \
  --prediction "Strong RGB + explicit prior features=paper_assets/source_tables/reviewer_revision_v2/strong_cls_prior_grouped_oof_predictions.csv" \
  --bootstrap-resamples 5000 \
  --seed 20260713 \
  --output-dir paper_assets/source_tables/reviewer_revision_v7
```
