# Reproducibility

## Environment

```bash
uv sync --python 3.13
source .venv/bin/activate
```

Alternative:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Pipeline

```bash
python data_processing/prepare_manuscript_data.py --dry-run
python data_processing/run_step.py dfuc2021-exclusion-audit
python data_processing/run_step.py dfuc2021-splits
python data_processing/run_step.py dfuc2022-seg-splits
python experiments/dfuc2022_seg_baseline/run_cv.py --device cuda --amp auto --batch-size 8 --num-workers 4 --skip-existing
python experiments/dfuc2021_cls_baseline/run_cv.py --config experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml --device cuda --amp auto --batch-size 64 --num-workers 4 --skip-existing
python data_processing/run_step.py dfuc2021-lesion-priors -- --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt --batch-size 256 --num-workers 4 --skip-existing
python experiments/cross_task_prior_transfer/run_lpt_cv.py --config experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml --device cuda --amp auto --batch-size 128 --num-workers 4 --skip-existing
python data_processing/run_step.py dfuc2021-similarity-audit
python data_processing/run_step.py dfuc2021-similarity-grouped-splits
python data_processing/run_step.py dfuc2021-dfuc2022-cross-dataset-audit
python experiments/reliability_aware_prior_fusion/oof_stack.py
python experiments/reliability_aware_prior_fusion/search_publication_fusion.py --top-methods 18 --top-triples 12 --top-bias-candidates 60 --pair-step 0.025 --triple-step 0.05 --seed 20260701
python experiments/reliability_aware_prior_fusion/build_publication_final_assets.py --audited-metrics-table paper_assets/source_tables/publication_final_metrics.csv
python data_processing/run_step.py cross-dataset-overlap-exclusion-sensitivity
```

The commands above show the publication-facing sequence. The manuscript also uses additional backbone and grouped-retraining configurations under:

```text
experiments/dfuc2021_cls_baseline/configs/
experiments/cross_task_prior_transfer/configs/
```

Run those configs with the corresponding `run_cv.py` or `run_lpt_cv.py` entry points before running the RA-LPT stacking and final probability-fusion scripts.

## Paper-Aligned Evidence Chain

- `data_processing/audit_dfuc2021_exclusions.py`: audits DFUC2021 annotation inclusion/exclusion reasons, image-file availability, and duplicate image names for Supplementary Table S1.
- `dfuc2022_seg_baseline`: trains the 512 x 384 DeepLabV3+ prior generator reported as DFUC2022 Dice/IoU mean +/- SD.
- `dfuc2021_cls_baseline`: trains RGB-only standard and similarity-grouped classifiers used as no-prior comparators.
- `cross_task_prior_transfer`: generates 224 x 224 DFUC2021 soft lesion-prior maps and trains direct LPT variants.
- `reliability_aware_prior_fusion/audit_image_similarity.py`: performs SHA256 and pHash/dHash similarity auditing.
- `reliability_aware_prior_fusion/audit_cross_dataset_similarity.py`: performs DFUC2021-DFUC2022 exact-duplicate and near-duplicate auditing.
- `reliability_aware_prior_fusion/make_similarity_grouped_splits.py`: reconstructs similarity-grouped folds.
- `reliability_aware_prior_fusion/oof_stack.py`: implements RA-LPT logistic-regression stacking with optional prior reliability statistics and interaction features.
- `reliability_aware_prior_fusion/search_publication_fusion.py`: enumerates prior-informed probability-fusion candidates on fold-strict grouped OOF predictions.
- `reliability_aware_prior_fusion/build_publication_final_assets.py`: reconstructs the selected prior-informed probability-fusion candidate and group-level bootstrap sensitivity outputs used by the manuscript.
- `reliability_aware_prior_fusion/cross_dataset_overlap_exclusion_sensitivity.py`: recomputes grouped OOF metrics after excluding DFUC2021 images involved in broad DFUC2021-DFUC2022 pHash/dHash candidate pairs.

Full checkpoints, raw logs, raw images, segmentation masks, and full OOF prediction files under `outputs/` are intentionally not part of this repository. Audited non-image manuscript assets used for value traceability are provided under `paper_assets/` when permitted by dataset-use restrictions.

## Public Audit Assets

The repository includes a publication-facing subset of derived assets:

- `paper_assets/source_tables/`: source values for manuscript and supplementary tables.
- `paper_assets/split_metadata/`: DFUC2021 fold assignments and similarity-group identifiers.
- `paper_assets/fusion_search/`: final probability-fusion search log, selected-candidate trace, and search summary.
- `paper_assets/figure_scripts/`: scripts that regenerate manuscript-style figures from source tables.

Regenerate source-table figures with:

```bash
python paper_assets/figure_scripts/build_source_table_figures.py --output-dir outputs/publication_source_table_figures
```

Regenerate the final quantitative figure layouts with:

```bash
python paper_assets/figure_scripts/build_figure3_performance_revision.py
python paper_assets/figure_scripts/build_figure4_robustness_revision.py
python paper_assets/figure_scripts/build_reviewer_revision_v4_figures.py
python paper_assets/figure_scripts/build_supplementary_main_style_figures.py
```

`build_reviewer_revision_v4_figures.py` always regenerates the quantitative reliability and similarity-sensitivity panels. It skips its qualitative main-figure panel when licensed DFUC-derived image inputs are absent; use `--require-image-assets` to make those missing inputs an error.

The final submitted Figure 2 and Figure 4 were visually adjusted in editable PowerPoint after the source-table render. Their data and panel logic are reproduced by the scripts, but pixel-identical PowerPoint typography is not claimed. Figure 1, Figure 5A, and Figure 7 contain licensed DFUC-derived images and require locally regenerated image assets that are not redistributed.

### Submitted-figure mapping

| Submitted item | Public regeneration source | Reproducibility boundary |
|---|---|---|
| Figure 1 | `update_figure1_original_layout.py` plus the editable manuscript source | Requires licensed representative DFUC images; the canonical final layout is manually edited. |
| Figure 2 | Editable PowerPoint source retained with the manuscript package | Architecture schematic; it is not a numeric plot and is not claimed to be pixel-identical from Python. |
| Figure 3 | `build_figure3_performance_revision.py` | Quantitative panels regenerate from public source tables. |
| Figure 4 | `build_figure4_robustness_revision.py` plus editable PowerPoint | Values and panel logic regenerate; final typography was adjusted in PowerPoint. |
| Figure 5 | `build_reviewer_revision_v4_figures.py` | Panels B-D regenerate from public tables; Panel A requires licensed DFUC image/prior assets. |
| Figure 6 | `build_main_composite_figures.py` (`figure3_selected_candidate_classification_summary`) | Quantitative post hoc full-OOF summary regenerates from public tables; the internal function name predates final renumbering. |
| Figure 7 | `build_main_composite_figures.py` (`figure6_gradcam_prior_examples`) | Requires licensed DFUC images and the corresponding locally generated Grad-CAM/prior plate. |
| Supplementary Figures S1, S3, S4, S6-S8 | `build_supplementary_main_style_figures.py` | Public source tables reproduce the submitted quantitative panels; output stems retain pre-renumbering identifiers. |
| Supplementary Figures S2 and S5 | `build_reviewer_revision_v4_figures.py` | Public source tables reproduce both quantitative panels; output stems retain pre-renumbering identifiers. |

The pre-renumbering filenames are retained to preserve their audit trail. The table above, rather than the numeric suffix embedded in an intermediate output stem, defines the submitted-figure mapping.

Rendered PNG dimensions, font rasterisation, and antialiasing can vary with the local Matplotlib and font stack. The public scripts are intended to reproduce the source values and panel logic; the canonical manuscript exports and retained editable PowerPoint files remain the reference for exact submission typography.

The supplementary architecture/latency and prior-resolution audits can be rerun with:

```bash
python paper_assets/analysis_scripts/benchmark_model_complexity.py --device cuda
python paper_assets/analysis_scripts/evaluate_prior_resolution.py
```

The selected prior-informed probability-fusion candidate is traceable in `paper_assets/fusion_search/publication_selected_prior_informed_candidate_trace.csv`, and its per-image grouped OOF output is provided in `paper_assets/source_tables/reviewer_revision_v4/derived/post_hoc_prior_informed_full_oof_candidate_predictions.csv`. The full fusion search log contains unconstrained candidates that may rank higher numerically; the manuscript-selected candidate is the top candidate within the prior-informed candidate family that includes `strong_lpt_full_output_ensemble`, matching the Methods description.

`paper_assets/source_tables/reviewer_revision_v4/derived/prediction_and_checkpoint_asset_registry.csv` preserves the original manuscript-package provenance path in `relative_source` and gives the current repository location in `public_release_path`. The registered checkpoint hash is retained for provenance, but the checkpoint itself is not redistributed.

## Data Location

The code uses repository-relative paths. No machine-specific absolute data path is required by the public scripts.
