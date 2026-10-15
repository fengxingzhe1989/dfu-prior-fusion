# Non-image research assets

## Current five-seed evidence

`associate_editor_revision_v1/summary/` contains complete classification, stack and segmentation aggregate outputs, bootstrap replicates, a relative-path input registry and archived completion evidence.

`associate_editor_revision_v1/interpretability/` contains the frozen inference contract, 20 prediction-blind case identifiers, coverage/seed/aggregate summaries, bootstrap replicates and audit records. It contains no image pixels or case arrays. In the frozen coverage tables, `n_invalid_cam` denotes unavailable overlap due to empty prior support, not failure to generate Grad-CAM; see `qa_acceptance.json`.

The interpretation `summary_complete.json` refers to the original input registry hash. The released `input_registry_relative.csv` removes machine-specific prefixes and is a separately hashed derivative. Other copied aggregate result values are unchanged. Archived completion records can refer to assets not publicly distributed and do not imply that this repository includes every checkpoint or prediction.

## Later exploratory extensions

`interpretability_rgb_extension_v1/` contains the combined five-model interpretation summaries and bootstrap replicates (125 inference jobs, 148,875 image-model-seed records). It extends, rather than overwrites, the initial two-model record.

`training_size_sensitivity_v3/summary/` contains the full budget-first four-budget analysis: 40 pooled trajectory/model/budget cells, 200 fold-level cells, 400 fit/stop/refit/test count rows, class-level confusion counts, 32 cell-metric summaries, 56 metric contrasts and 40,000 bootstrap rows (5,000 draws × 8 budget/model cells). The rows are aggregates, not individual-image records. All four budgets were newly trained, totaling 400 additional CNN fits; no primary endpoint or v1/v2 result is reused.

The v3 result tables and computation markers are byte-identical copies. `training_size_sensitivity_v3/provenance_summary.json` is an explicitly derived public metadata projection with the original seal hash, not an executable seal or evidence that every registered private file is present. Its original summary marker records 1,400 undistributed fit/prediction/provenance inputs. The public checker audits the released values and saved bootstrap draws, not those private inputs. Historical completion flags are preserved rather than rewritten as current manuscript status. See the file registry and [reproduction limits](../REPRODUCIBILITY.md#6-later-four-budget-training-size-sensitivity).

## Supporting and historical assets

- `split_metadata/`: exact original standard/grouped manifests and summaries, with image IDs/relative paths only.
- `source_tables/methodology_audit_v7/`: grouping and cross-dataset image-similarity audits retained for data provenance.
- `source_tables/reviewer_revision_v3/derived/similarity_threshold_topology_sensitivity.csv`: grouping-topology audit, not new model training.
- `source_tables/frontiers_revision_v5/`: historical original-generator resolution audit and architecture-only complexity benchmark.
- `analysis_scripts/`: the two corresponding historical audit scripts.
- `historical_factorial/`: verified cell/fold/contrast summaries from the one-seed batch/split experiment; not the new primary analysis.
- `release_audit/`: copied-asset hashes and exact removed-file inventory.

Superseded full-OOF search, old model-result tables and old figure builders are recoverable in Git history, not mixed into the current analysis directory.
