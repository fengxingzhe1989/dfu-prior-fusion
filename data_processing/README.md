# Data Processing Code

This folder contains the publication-facing entry points for dataset preparation and derived-data generation.

Large generated files are intentionally not committed. They are written to `data_processed/` and `outputs/`, both of which are ignored by Git. Small non-image source tables used for manuscript traceability may be stored under `paper_assets/`.

## Paper-Aligned Processing Flow

1. Audit DFUC2021 annotation inclusion/exclusion reasons:

```bash
python data_processing/run_step.py dfuc2021-exclusion-audit
```

2. Build the valid DFUC2021 four-class classification manifest and standard stratified folds:

```bash
python data_processing/run_step.py dfuc2021-splits
```

3. Build the DFUC2022 segmentation image/mask manifest and deterministic folds used to train the lesion-prior segmenter:

```bash
python data_processing/run_step.py dfuc2022-seg-splits
```

4. Audit DFUC2021 split leakage risk using exact SHA256 and pHash/dHash near-duplicate screening:

```bash
python data_processing/run_step.py dfuc2021-similarity-audit
```

5. Build similarity-grouped DFUC2021 folds used for grouped internal validation:

```bash
python data_processing/run_step.py dfuc2021-similarity-grouped-splits
```

6. Audit broad visual overlap between the DFUC2021 valid classification images and DFUC2022 segmentation-training images:

```bash
python data_processing/run_step.py dfuc2021-dfuc2022-cross-dataset-audit
```

7. Generate DFUC2021 lesion-prior maps from the trained DFUC2022 segmentation model:

```bash
python data_processing/run_step.py dfuc2021-lesion-priors -- --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt
```

The lesion-prior step requires a trained DFUC2022 segmentation checkpoint. That checkpoint is not part of the code-only release.

The old post hoc overlap-exclusion scoring entry point is no longer part of the current pipeline. Cross-dataset image-similarity auditing above remains available. For the current nested analysis, use the exact released grouped manifest and verify its frozen hash before constructing inner/stop splits; see [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## One-Command Pipeline

Run the preprocessing steps that do not require trained model weights:

```bash
python data_processing/prepare_manuscript_data.py
```

Include lesion-prior generation after training the segmenter:

```bash
python data_processing/prepare_manuscript_data.py --include-priors --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt
```

Use `--dry-run` to print commands without executing them.
