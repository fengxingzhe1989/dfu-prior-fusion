# Frozen non-image split metadata

The standard five-fold manifest/summary supports the historical factorial. The grouped manifest/summary defines the outer image-similarity folds retained in the current evaluation.

`dfuc2021_cls_cv5_similarity_grouped_manifest.csv` contains `similarity_group`, 5,955 unique images and 5,670 groups; no group crosses outer folds. Its SHA256 is `aec52b1c6a8fc847928676409f7efef012a99c96cd15f5dfc9de2195cb402dff`, matching the frozen current protocol. Copy this manifest to the protocol's data_processed location in a fresh reproduction workspace before constructing nested assignments.

These metadata contain dataset filenames and relative paths, not image/mask pixels or patient identifiers. Obtain the underlying datasets separately under their applicable access terms.
