# Historical Reviewer 1 batch-size × split factorial

This retained sensitivity analysis uses two models (Full LPT and ResNet50), two split schemes (standard and image-similarity-grouped), two batch sizes (16 and 128), and five folds: 40 fits with one base seed, 20260430. It is **not** the current five-seed nested primary analysis.

The epoch-based training regime allows up to 60 epochs with patience 12; other settings are inherited/frozen by the runner. Changing batch size also changes optimizer-update counts, so this is not an update-matched experiment and does not identify a specific gradient-noise mechanism. Checkpoint-selected OOF estimates and group-bootstrap intervals are conditional on these fitted models and omit training-seed uncertainty.

```bash
python experiments/reviewer_r1_batch_split_factorial/run_factorial.py --dry-run
python experiments/reviewer_r1_batch_split_factorial/run_factorial.py
python experiments/reviewer_r1_batch_split_factorial/summarize_factorial.py
python experiments/reviewer_r1_batch_split_factorial/analyze_factorial.py --bootstrap-iters 5000 --seed 20260826
```

The dry run requires the data manifests/configs; it is not a data-free test. All original 40 fits completed; each cell contains 5,955 unique OOF images. Verified cell, fold and group-bootstrap tables are retained under `paper_assets/historical_factorial/`. Within this limited matrix, the model-by-batch interaction supported differential sensitivity to the batch-size training regime, whereas model-by-split intervals included zero. This does not establish a universal or causal batch-size explanation.

For the original public code snapshot, use commit `6d4208639c802ed38442729dae8dacd8a8c76cdf`. The current tree preserves this runner but updates shared modules to the frozen associate-editor execution versions; exact historical replay requires checking the original run's source/config provenance, not assuming all current dependencies are the historical versions.
