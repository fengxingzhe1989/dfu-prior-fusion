# Data Access

This release does not redistribute DFUC images, masks, processed image data, or checkpoints.

Audited non-image manuscript assets are included under `paper_assets/` to make reported values traceable. These include source tables, public split/group metadata, selected fusion-search logs, source-table figure-generation scripts, the matched grouped OOF prediction files needed to reproduce the reviewer-revision bootstrap comparisons, and the per-image selected-candidate prediction table used for descriptive full-OOF summaries. They do not contain raw DFUC images, segmentation masks, or checkpoints.

Access to the DFUC image datasets must be obtained from the original dataset authors under the applicable **Diabetic Foot Ulcer Dataset: Licence to Use** terms. Users should request the data/licence from the dataset authors or official dataset access point; this repository provides code and non-image derived audit assets only.

After obtaining the dataset licence and files, place them under:

```text
data_raw/
  2021_dfuc2021/
  2022_dfuc2022/
  pre_challenge/
```

Generated intermediate files should be written to `data_processed/`; generated training and analysis outputs should be written to `outputs/`. Both directories are ignored by Git in this release.

The bundled public audit assets are organized as:

- `paper_assets/source_tables/`: manuscript source tables and supplementary table inputs.
- `paper_assets/split_metadata/`: fold assignments and similarity-group identifiers for DFUC2021 image names.
- `paper_assets/fusion_search/`: final probability-fusion search logs and selected-candidate trace tables.
- `paper_assets/figure_scripts/`: scripts that regenerate manuscript-style figures from source tables, without requiring raw images.

The code that creates these generated files is in `data_processing/`. The main entry points are:

```bash
python data_processing/run_step.py --list
python data_processing/prepare_manuscript_data.py --dry-run
```
