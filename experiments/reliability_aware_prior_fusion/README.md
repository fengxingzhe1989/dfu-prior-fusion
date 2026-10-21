# Shared stack features and image-similarity audits

`oof_stack.py` supplies frozen feature definitions and logistic-regression utilities to `experiments/associate_editor_revision_v1/fit_nested_stacks.py`. Its legacy CLI is not the current nested evaluator. Use the new runner to maintain the required separation between feature-generation training, stack selection and outer evaluation.

The retained `audit_image_similarity.py`, `make_similarity_grouped_splits.py` and `audit_cross_dataset_similarity.py` support cohort/grouping/overlap audits. Similarity groups do not identify patients.

Pre-revision fusion search, explanation, metric recombination and figure-data builders have been removed from the current release. Their complete earlier code is recoverable from the previous Git revision listed in [RELEASE_SCOPE.md](../../RELEASE_SCOPE.md).
