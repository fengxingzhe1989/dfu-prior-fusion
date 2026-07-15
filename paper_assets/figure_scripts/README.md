# Figure Scripts

This directory contains public, source-table-based figure generation scripts.

`build_source_table_figures.py` regenerates numeric audit figures from files in `paper_assets/source_tables/` and writes PNG/PDF outputs to a user-specified directory. It is intended to make manuscript values traceable without redistributing raw DFUC images or model checkpoints.

Final manuscript mapping:

- `build_figure3_performance_revision.py`: Main Figure 3 quantitative panels.
- `build_figure4_robustness_revision.py`: source-table reconstruction for Main Figure 4. The submitted layout uses the author-edited PowerPoint version, so this script reproduces values and panel logic rather than pixel-identical typography.
- `build_reviewer_revision_v4_figures.py`: Main Figure 5 plus selected supplementary panels; its image panel requires licensed DFUC-derived inputs.
- `build_main_composite_figures.py`: quantitative composite figures and the post hoc candidate summary; the Grad-CAM example plate is skipped unless licensed image assets are supplied.
- `build_supplementary_main_style_figures.py`: the quantitative supplementary figure series. The current manuscript numbering is shifted because the architecture schematic was promoted to Main Figure 2: generated S02-S09 correspond to submitted Supplementary Figures S1-S8.

Qualitative Grad-CAM example panels require raw images, generated priors, and trained checkpoints. Those are regenerated through `experiments/reliability_aware_prior_fusion/explainability_analysis.py` after users obtain the DFUC datasets under the applicable licence.

The final Figure 2 architecture schematic and final Figure 4 layout were manually adjusted in editable PowerPoint. Those presentation files are maintained in the submission package, not in this code-only repository.
