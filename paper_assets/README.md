# Paper Assets

This directory contains small non-image source tables used to audit manuscript-reported values.

Included files are CSV/JSON/Markdown summaries derived from fold-strict validation outputs, similarity audits, probability-fusion search summaries, group-level bootstrap summaries, cross-dataset overlap sensitivity summaries, and interpretability summaries.

Additional audit assets are provided in:

- `split_metadata/`: non-image DFUC2021 split manifests, including the similarity-grouped manifest used to verify fold/group assignments.
- `fusion_search/`: final probability-fusion search logs and the selected prior-informed candidate trace.
- `figure_scripts/`: scripts for regenerating source-table-based numeric figures from the public source tables.
- `analysis_scripts/`: scripts for the architecture-only complexity benchmark, prior-resolution audit, and reviewer-requested bootstrap analyses.
- `source_tables/reviewer_revision_v2/`: the similarity-group manifest and six fixed grouped OOF prediction tables used by the reviewer-revision clinical-metric analysis.
- `source_tables/reviewer_revision_v7/`: derived per-class and binary-endpoint tables with 5,000 similarity-group bootstrap intervals.
- `source_tables/reviewer_revision_v3/derived/`: matched grouped metrics, repeated-run summaries, bootstrap comparisons, and mixed-component sensitivity tables used by Main Figures 3 and 4.
- `source_tables/reviewer_revision_v4/derived/`: matched cross-dataset exclusion tables and the post hoc selected-candidate per-image prediction asset used by Main Figure 6.
- `source_tables/frontiers_revision_v5/`: architecture/latency and prior-resolution audit outputs plus fold-summary logs supporting Supplementary Tables S14-S16.

Not included:

- raw DFUC images or masks;
- resized or processed image arrays;
- trained checkpoints;
- complete prediction archives from all training runs;
- private machine logs or credentials.

The DFUC datasets must be requested from the original dataset authors under the applicable Diabetic Foot Ulcer Dataset licence. The selected grouped OOF prediction tables contain image identifiers, labels, fold/group identifiers, decisions, and class probabilities but no image pixels. They are included only to make the reported analyses reproducible when raw data redistribution is not permitted.
