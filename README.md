# DFU cross-dataset ulcer-region priors

Code and non-image aggregate results for *Incremental value of a cross-dataset ulcer-region prior for infection and ischaemia classification in diabetic foot ulcer image patches* (manuscript 1944475).

## Current analysis

The current entry point is [experiments/associate_editor_revision_v1](experiments/associate_editor_revision_v1/README.md): the frozen five-seed nested classification/stacking evaluation, architecture-matched neutral-prior control, independent source-domain segmentation audit, and exploratory interpretation analysis. All seeds 20260925–20260929 are retained; the shared classification cohort has 5,955 images in 5,670 image-similarity groups.

The main estimand is the mean of five seed-specific pooled-OOF Macro-F1 differences. Three incremental contrasts are evaluated: Full LPT versus neutral Full LPT; RA-LPT versus the corresponding output-only LPT stack; and strong RGB plus explicit prior features versus strong RGB. The Full-versus-neutral contrast is positive, but the two stack-level increments are not established. Complete-system comparisons against ResNet50 are descriptive and do not isolate a prior effect. These are internal image-grouped estimates, not patient-independent or external clinical validation.

## Start here

```bash
uv sync --frozen --python 3.13
uv run python scripts/check_release.py
uv run python -m unittest discover -s experiments/associate_editor_revision_v1 -p 'test_*.py' -v
```

These checks do not train models or require medical images. Full execution requires licensed datasets and the separately archived frozen prior checkpoint: this is **not** a self-contained data/checkpoint release. Read [DATA.md](DATA.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) first.

## Contents

- `experiments/associate_editor_revision_v1/`: current frozen training, inference, summarization and recovery code.
- Other experiment directories: shared architecture, preprocessing and feature-building dependencies; their legacy runners are not the current evaluation entry point.
- `experiments/reviewer_r1_batch_split_factorial/`: historical one-seed factorial sensitivity analysis, separate from the new five-seed evidence.
- `data_processing/`: dataset inclusion, split construction and similarity-audit helpers.
- `paper_assets/associate_editor_revision_v1/`: complete non-image aggregate results, including all seeds and bootstrap replicates.
- `paper_assets/source_tables/` and `paper_assets/historical_factorial/`: retained supporting audits, explicitly historical where applicable.
- `paper_assets/release_audit/`: frozen source/output hashes and recoverable cleanup inventory.

The cleanup removes superseded search, figure-building and result-processing scripts and unused configurations from the current tree; it does not erase the older release or research records. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
