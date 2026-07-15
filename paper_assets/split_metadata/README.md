# Split Metadata

This directory contains non-image split metadata used to audit the manuscript validation design.

Files:

- `dfuc2021_cls_cv5_manifest.csv`: standard five-fold DFUC2021 four-class manifest.
- `dfuc2021_cls_cv5_summary.json`: summary for the standard split.
- `dfuc2021_cls_cv5_similarity_grouped_manifest.csv`: similarity-grouped manifest used for grouped internal validation. It includes `similarity_group`, and no similarity group crosses folds.
- `dfuc2021_cls_similarity_grouped_summary.json`: grouped-split summary.
- `dfuc2021_cls_cv5_excluding_cross_fold_same_label_near_duplicates_manifest.csv`: earlier sensitivity manifest excluding cross-fold same-label near-duplicate candidates.

These files contain image filenames and relative paths but do not include raw images, masks, checkpoints, or pixel data. Raw DFUC data must be requested from the original dataset providers under the applicable licence.
