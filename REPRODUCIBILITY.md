# Reproduction and verification

## 1. Verify the release without training

From the repository root:

```bash
uv sync --frozen --python 3.13
uv run python scripts/check_release.py
uv run python -m unittest discover -s experiments/associate_editor_revision_v1 -p 'test_*.py' -v
uv run python data_processing/prepare_manuscript_data.py --dry-run
```

The release check verifies copied source/output hashes, declared config and preprocessing dependencies, cohort metadata, seed/summary coverage and aggregate calculations. It is not GPU retraining or a guarantee of bitwise reproduction on another platform. The executed training environment used Python 3.13, PyTorch 2.7.1 with CUDA 12.6 and an RTX 4090. The lockfile describes the packaged dependency environment.

## 2. Prepare licensed inputs

Obtain DFUC2021 and DFUC2022 through their original access arrangements; see [DATA.md](DATA.md). Do not substitute another cohort, grouping, prior checkpoint or seed set under the same protocol name.

The released `paper_assets/split_metadata/dfuc2021_cls_cv5_similarity_grouped_manifest.csv` is byte-identical to the frozen grouped manifest (SHA256 `aec52b1c6a8fc847928676409f7efef012a99c96cd15f5dfc9de2195cb402dff`). Copy it into `data_processed/dfuc2021_cls_v1/similarity_grouped_splits/` in a fresh reproduction workspace. The standard manifest is retained for the historical analysis. Preparation helpers can reconstruct audits/manifests, but newly generated assignments must be checked against the frozen artifacts before claiming exact reproduction.

After making the licensed source images/masks available, generate the segmentation manifest and priors:

```bash
uv run python data_processing/run_step.py dfuc2022-seg-splits
uv run python data_processing/run_step.py dfuc2021-lesion-priors -- --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt
uv run python experiments/associate_editor_revision_v1/prepare_nested_splits.py
```

The classification runner requires the exact prior-checkpoint hash in `protocol.yaml`. That checkpoint is not distributed here. Independently retraining a source segmenter need not reproduce it. The new segmentation audit does **not** replace this generator. The frozen nested-assignment SHA256 is `f0d4590e47ba0e58b2d24236dbace32ea6cb62f45628d5470c2914744d166dbf`; compare the generated file before training.

## 3. Execute the frozen matrix

Run sequentially in the prepared CUDA environment:

```bash
uv run python experiments/associate_editor_revision_v1/run_classification_matrix.py
uv run python experiments/associate_editor_revision_v1/run_remaining_pipeline.py
uv run python experiments/associate_editor_revision_v1/summarize.py
```

Classification contains 900 fits across all five seeds and outer folds. The remaining-pipeline runner waits for classification, checks/repairs non-finite predictions under the unchanged configuration, then fits four stacks (100 outer seed/fold/stack fits) and runs the 25-fit segmentation audit. **It does not automatically run the final summary or interpretation stage.** Outputs go to `outputs/associate_editor_revision_v1/`.

Outer evaluation images are not used for checkpoint or stack selection. Inner held-out predictions, stopping subsets and final refits have distinct roles; see frozen `METHODS.md`, `protocol.yaml` and implementation. The protocol status field is a historical freeze-time value, not a live progress monitor, and remains unchanged.

Do not select favorable seeds, drop valid completed jobs, tune on outer predictions or use pre-revision OOF probabilities. Partial-run CLI options are for operational recovery, not selective reporting. Require completion markers and hashes before summarizing. Do not routinely use `--force`; preserve failed logs and artifacts when diagnosing a failure.

## 4. Separate interpretation stage

```bash
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode prepare
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode smoke
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode run
uv run python experiments/associate_editor_revision_v1/resume_interpretability_finalization.py --verify-only
```

This requires all frozen outer-refit checkpoints and predictions: 50 model/seed/fold jobs, 59,550 quantitative image-model-seed records, and 20 prediction-blind cases shown across all seeds. The included geometry/finalization tests use synthetic data. Medical-image case plates are not republished in this repository.

If inference is complete but summary/export is incomplete, run the finalization helper without `--verify-only`; it validates provenance and resumes only missing stages. Do not reseal changed inputs or replace cases. Grad-CAM and occlusion measure generated-prior-relative model sensitivity, not anatomical or clinical validity.

## 5. Outputs and history

Current aggregates are in `paper_assets/associate_editor_revision_v1/`. Archived markers refer to some non-distributed checkpoints, per-image predictions and case arrays. `input_registry_relative.csv` is an explicitly path-redacted interpretation registry copy: its original hash is recorded and its new file hash intentionally differs. Result values are unchanged.

The original generator-resolution audit and architecture-only benchmark are retained under `paper_assets/source_tables/frontiers_revision_v5/`, with scripts under `paper_assets/analysis_scripts/`. They are not the new independent segmentation audit or end-to-end pipeline benchmarks. The historical batch/split factorial remains in its own experiment directory; do not pool its one-seed, checkpoint-selected estimates with current results.
