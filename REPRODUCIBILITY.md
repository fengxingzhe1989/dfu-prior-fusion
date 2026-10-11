# Reproduction and verification

## 1. Verify the release without training

From the repository root:

```bash
uv sync --frozen --python 3.13
uv run python scripts/check_release.py
uv run python scripts/check_rgb_extension_release.py
uv run python -m unittest discover -s experiments/associate_editor_revision_v1 -p 'test_*.py' -v
uv run python -m unittest experiments.interpretability_rgb_extension_v1.test_extension -v
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

## 4. Initial two-model interpretation stage

```bash
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode prepare
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode smoke
uv run python experiments/associate_editor_revision_v1/run_interpretability.py --mode run
uv run python experiments/associate_editor_revision_v1/resume_interpretability_finalization.py --verify-only
```

This initial stage evaluates ResNet50 and Full LPT using their frozen outer-refit checkpoints and predictions: 50 model/seed/fold jobs, 59,550 quantitative image-model-seed records, and 20 prediction-blind cases shown across all seeds. It is retained unchanged. The current five-model manuscript also uses the separate extension in Section 5 below. The included geometry/finalization tests use synthetic data. Medical-image case plates are not republished in this repository.

If inference is complete but summary/export is incomplete, run the finalization helper without `--verify-only`; it validates provenance and resumes only missing stages. Do not reseal changed inputs or replace cases. Grad-CAM and occlusion measure generated-prior-relative model sensitivity, not anatomical or clinical validity.

## 5. Later three-RGB-model exploratory extension

The separately frozen [extension protocol](experiments/interpretability_rgb_extension_v1/PROTOCOL.md) adds EfficientNet-B0, ConvNeXt-Tiny and EfficientNetV2-S. It was specified after the initial two-model analysis was available and is **additional exploratory work**, not an original prespecified primary analysis. All five training seeds (20260925–20260929), all five outer folds, all 5,955 images and the same 20 prediction-blind cases and occlusion placements are retained. No models are retrained, seeds discarded or cases reselected.

Run this only after Sections 3 and 4 are complete, with the archived checkpoints, predictions, licensed images and priors available in the same reproduction workspace:

```bash
uv run python experiments/interpretability_rgb_extension_v1/run.py --mode prepare
uv run python experiments/interpretability_rgb_extension_v1/run.py --mode smoke
uv run python experiments/interpretability_rgb_extension_v1/run.py --mode run
```

The default source is `outputs/associate_editor_revision_v1/`, which is read-only to the extension. New outputs are written separately to `outputs/interpretability_rgb_extension_v1/interpretability/`. The 75 added inference jobs contribute 89,325 image-model-seed records. Together with the initial 50 jobs, the combined analysis has **125 jobs and 148,875 records across five models**. These are inference/interpretability jobs, not 125 additional training fits. `finalize.py` audits both sets and creates the combined summaries and fixed-case displays.

If inference has completed but a summary or figure stage remains missing, use:

```bash
uv run python experiments/interpretability_rgb_extension_v1/run.py --mode finalize
```

This verifies sealed provenance, skips valid completed stages and refuses to overwrite partial output directories. Preserve incomplete outputs and diagnose the actual error before recovery; do not delete/reseal registries to bypass a failed check. `extension_complete.json` confirms computation/export, not visual review or manuscript integration. The original two-model code and outputs remain unchanged.

Combined non-image summaries and all 5,000 bootstrap replicates are in `paper_assets/interpretability_rgb_extension_v1/`. `scripts/check_rgb_extension_release.py` verifies the release hashes, all-model/all-seed coverage, case-list identity, seed means/SDs and bootstrap intervals without medical images or GPU execution. It does not re-audit undistributed checkpoints or reproduce inference. The archived summary marker links the 75 added and 50 original job audits to their registry hashes; the source code files are byte-identical to the sealed execution versions.

Reported intervals are descriptive pointwise crossed seed × similarity-group bootstrap intervals, not multiplicity-adjusted confirmatory tests. Predicted-class targets can differ between models; common true-label-target contrasts are separate. CAM target layers differ across architectures; overlap with an uncertain generated prior is not anatomical accuracy, and full-system contrasts do not isolate a causal prior effect. See `REPORTING_BOUNDARIES.md` in the combined summary directory.

### Display-only manuscript update

The sealed renderers preserve their original `magma`, opacity 0.40 exports. The final manuscript uses the same fixed cases and archived, per-image-normalized CAM arrays with a uniform `YlOrRd` colour map and opacity 0.35. This later presentation change alters neither CAM values nor quantitative results. Medical-image plates and manuscript-specific layout sources are not redistributed here; the submission package records the final display hashes. Reproducing the sealed analysis and reproducing the final presentation are distinct tasks.

## 6. Outputs and history

Classification, stacking, segmentation and initial two-model aggregates are in `paper_assets/associate_editor_revision_v1/`; combined five-model interpretation aggregates are in `paper_assets/interpretability_rgb_extension_v1/`. Do not substitute the former two-model interpretation summary for the current five-model summary. Archived markers refer to some non-distributed checkpoints, per-image predictions and case arrays. Each `input_registry_relative.csv` is an explicitly path-redacted interpretation registry copy: its original hash is recorded and its new file hash intentionally differs. Result values are unchanged.

The original generator-resolution audit and architecture-only benchmark are retained under `paper_assets/source_tables/frontiers_revision_v5/`, with scripts under `paper_assets/analysis_scripts/`. They are not the new independent segmentation audit or end-to-end pipeline benchmarks. The historical batch/split factorial remains in its own experiment directory; do not pool its one-seed, checkpoint-selected estimates with current results.
