# Data and artifact boundaries

Obtain DFUC2021 and DFUC2022 from the original providers under their applicable access/licence terms. This repository grants no dataset access and redistributes no raw photographs, masks, trained checkpoints or case-plate pixels.

- `data_raw/2021_dfuc2021/` and `data_raw/2022_dfuc2022/`: immutable licensed source datasets.
- `data_processed/`: derived manifests, nested assignments and lesion priors.
- `outputs/`: checkpoints, predictions, completion markers, summaries and figures.

Inspect preprocessing CLI `--help` for exact annotation filenames and subdirectories. Do not move an existing research dataset to fit a new checkout; use a separate reproduction workspace or appropriate links.

Small non-image split metadata, aggregate seed/fold/class metrics, bootstrap replicates and hashes are versioned under `paper_assets/`. Image IDs are dataset filenames, not patient identifiers. Similarity groups cannot replace unavailable patient/case linkage.

Exact replay of the fixed-generator classification analysis additionally requires the archived prior checkpoint with SHA256 `7ffbb2de0e6a9ef6c177af658587e1b94429b27a9e97b4f1a6dd2df3925d3337`, generated priors and downstream checkpoints for interpretation. They are not included. Access, if permissible, must be arranged separately with the authors; this repository does not guarantee availability. A freshly trained generator is a different realization unless its artifacts match.

Do not commit individual-level predictions, medical images or heavyweight models as part of routine execution. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [RELEASE_SCOPE.md](RELEASE_SCOPE.md).
