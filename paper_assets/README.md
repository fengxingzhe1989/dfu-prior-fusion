# Non-image research assets

## Current five-seed evidence

`associate_editor_revision_v1/summary/` contains complete classification, stack and segmentation aggregate outputs, bootstrap replicates, a relative-path input registry and archived completion evidence.

`associate_editor_revision_v1/interpretability/` contains the frozen inference contract, 20 prediction-blind case identifiers, coverage/seed/aggregate summaries, bootstrap replicates and audit records. It contains no image pixels or case arrays. In the frozen coverage tables, `n_invalid_cam` denotes unavailable overlap due to empty prior support, not failure to generate Grad-CAM; see `qa_acceptance.json`.

The interpretation `summary_complete.json` refers to the original input registry hash. The released `input_registry_relative.csv` removes machine-specific prefixes and is a separately hashed derivative. Other copied aggregate result values are unchanged. Archived completion records can refer to assets not publicly distributed and do not imply that this repository includes every checkpoint or prediction.

## Supporting and historical assets

- `split_metadata/`: exact original standard/grouped manifests and summaries, with image IDs/relative paths only.
- `source_tables/methodology_audit_v7/`: grouping and cross-dataset image-similarity audits retained for data provenance.
- `source_tables/reviewer_revision_v3/derived/similarity_threshold_topology_sensitivity.csv`: grouping-topology audit, not new model training.
- `source_tables/frontiers_revision_v5/`: historical original-generator resolution audit and architecture-only complexity benchmark.
- `analysis_scripts/`: the two corresponding historical audit scripts.
- `historical_factorial/`: verified cell/fold/contrast summaries from the one-seed batch/split experiment; not the new primary analysis.
- `release_audit/`: copied-asset hashes and exact removed-file inventory.

Superseded full-OOF search, old model-result tables and old figure builders are recoverable in Git history, not mixed into the current analysis directory.
