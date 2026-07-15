# GitHub Upload Checklist

Run the checks from the repository root.

Before uploading, confirm:

- `paper_assets/` is included; it contains only small non-image source tables and audit summaries.
- `data_raw/`, `data_processed/`, and `outputs/` are not uploaded except for their README placeholders.
- Raw DFUC images and masks are not committed.
- Checkpoints and complete training-run prediction archives are not committed. The six selected grouped OOF tables used to reproduce Supplementary Tables S12a and S12b are intentionally included under `paper_assets/`.
- `python data_processing/run_step.py --list` shows the DFUC2021 exclusion audit, cross-dataset audit, grouped split, prior generation, and sensitivity-analysis steps.
- `python data_processing/audit_dfuc2021_exclusions.py --help` works.
- `python experiments/reliability_aware_prior_fusion/search_publication_fusion.py --help` works.
- `python experiments/reliability_aware_prior_fusion/build_publication_final_assets.py --help` works.
- `python experiments/reliability_aware_prior_fusion/derive_grouped_clinical_metrics.py --help` works.
- `python paper_assets/figure_scripts/build_figure3_performance_revision.py --help` works.
- `python paper_assets/figure_scripts/build_reviewer_revision_v4_figures.py --help` works; without licensed image inputs it should generate the quantitative panels and record the qualitative panel as skipped.
- No `.venv/`, `__pycache__/`, `.pyc`, `.DS_Store`, raw image, checkpoint, or manuscript file is staged.

Suggested upload sequence:

```bash
cd dfu-prior-fusion
git status --short
git add .
git commit -m "Prepare publication reproducibility code"
git push -u origin main
```
