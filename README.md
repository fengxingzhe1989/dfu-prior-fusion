# dfu-prior-fusion

Experiment-code release for the DFU prior-informed probability-fusion study. It contains the code needed to prepare manifests, train the models, run lesion-prior transfer, audit similarity grouping, and perform probability-fusion experiments.

## Included

- `experiments/`: DFUC2022 segmentation, DFUC2021 classification, lesion-prior transfer, similarity grouping, RA-LPT stacking, fusion search, and sensitivity-analysis code.
- `data_processing/`: paper-aligned data preparation entry points for manifests, split audits, grouped folds, and lesion-prior generation.
- `paper_assets/`: non-image manuscript audit assets, including source tables, selected matched grouped OOF prediction tables, split/group metadata, fusion-search logs, analysis scripts, and figure-generation scripts used to verify reported values.
- Python dependency files: `pyproject.toml`, `requirements.txt`, `uv.lock`, `.python-version`.

## Not Included

No raw images, masks, processed images, checkpoints, paper figures, supplementary workbooks, manuscript files, local render artifacts, or manuscript-formatting scripts are included in this package.

The repository includes audited non-image CSV/JSON assets under `paper_assets/` when redistribution is compatible with DFUC dataset-use restrictions. These assets are intended for manuscript-value traceability and include derived result tables, fold/group identifiers, fusion-search logs, source-table figure scripts, the six fixed grouped OOF prediction tables required to reproduce Supplementary Tables S12a and S12b, and the selected post hoc candidate prediction table used by the descriptive Main Figure 6 analysis. Complete training-run prediction archives are not included. These assets are not a substitute for rerunning the experiments from the licensed datasets and do not redistribute DFUC images, masks, or checkpoints.

DFUC images must be requested from the original dataset authors under the applicable Diabetic Foot Ulcer Dataset licence; they are not available from this repository.

## Repository Name

Repository name: `dfu-prior-fusion`.
