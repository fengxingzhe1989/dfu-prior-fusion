# Reviewer Revision V7 Clinical Metrics

These tables were derived from six fixed similarity-grouped out-of-fold prediction files covering the same 5,955 DFUC2021 images and 5,670 reconstructed image-similarity groups.

- `main_reported_grouped_per_class_metrics_with_ci.csv`: one-vs-rest precision, recall, F1, AUROC, AUPRC, confusion counts, and 95% similarity-group bootstrap intervals for each class and model.
- `main_reported_grouped_binary_endpoint_metrics_with_ci.csv`: infection and ischaemia endpoint metrics and 95% similarity-group bootstrap intervals.
- `main_reported_grouped_clinical_metrics_metadata.json`: input hashes, seed, resampling unit, score definitions, and threshold-selection boundary.

No model was retrained and no binary threshold was selected. Binary decisions retain the saved four-class OOF decision: infection is positive for `infection` or `both`, and ischaemia is positive for `ischaemia` or `both`. AUROC and AUPRC use the corresponding summed class probabilities.

The exact reproduction command is documented in `experiments/reliability_aware_prior_fusion/README.md`.
