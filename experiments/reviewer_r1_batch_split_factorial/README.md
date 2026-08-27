# Reviewer 1 batch-size × split factorial

This experiment isolates the training-batch and validation-split explanations for
the apparent Full LPT versus ResNet50 ranking reversal reported in the manuscript.

## Design

The matrix contains eight five-fold CV conditions (40 fold fits):

- model: Full LPT or ResNet50;
- split: standard stratified CV or similarity-grouped CV;
- batch size: 16 or 128.

The following settings are frozen across all eight conditions:

- seed: `20260430` (fold seed is the base seed plus the fold index);
- input size: 224 × 224;
- epochs: 60;
- early-stopping patience: 12;
- mixed precision: enabled;
- data-loader workers: 4;
- pretrained initialization, augmentation, optimizer, scheduler, and
  model-specific loss settings: inherited unchanged from the original standard-CV
  source config for the corresponding model.

Only the model, split manifest, and batch size vary by design. ResNet50 is included
because the reviewer's concern is about a reversal in the *relative* ordering of
Full LPT and ResNet50, not only about the absolute Full LPT score.

## Isolation and provenance

All resolved configs, checkpoints, histories, predictions, metrics, run status,
and provenance are written below:

`outputs/reviewer_r1_batch_split_factorial_v1/`

The original manuscript outputs are neither overwritten nor treated as completed
cells. The provenance record contains hashes for the training code, source configs,
and split manifests, plus the Python/PyTorch/CUDA/GPU environment.

## Run

Validate and show the commands without training:

```bash
.venv/bin/python experiments/reviewer_r1_batch_split_factorial/run_factorial.py --dry-run
```

Run or safely resume the full matrix:

```bash
.venv/bin/python experiments/reviewer_r1_batch_split_factorial/run_factorial.py
```

Summarize completed cells:

```bash
.venv/bin/python experiments/reviewer_r1_batch_split_factorial/summarize_factorial.py
```

Verify all prediction assets and run the paired similarity-group bootstrap
analysis:

```bash
.venv/bin/python experiments/reviewer_r1_batch_split_factorial/analyze_factorial.py \
  --bootstrap-iters 5000 \
  --seed 20260826
```

The summary reports both the mean and population SD of the five fold-level
Macro-F1 values and the pooled OOF Macro-F1. This prevents the fold-mean versus
pooled-OOF aggregation difference from being mistaken for a biological or model
effect.

## Completed result

All 40 fold fits completed on the remote RTX 4090 environment. Every factorial
cell contains 5,955 unique OOF predictions. The paired analysis resampled the
5,670 reconstructed similarity groups 5,000 times.

The model-by-batch interaction was +0.0228 (95% CI, +0.0103 to +0.0355) in
standard CV and +0.0277 (95% CI, +0.0100 to +0.0458) in grouped CV. At batch
size 128, Full LPT was below ResNet50 in both standard and grouped CV; at batch
size 16, Full LPT was above ResNet50 in both split schemes. Grouping lowered
Macro-F1 for both models at both batch sizes, and the model-by-split interaction
intervals included zero. The controlled results therefore support the practical
batch-size training regime, rather than similarity grouping, as the dominant
measured explanation for the ranking reversal.

This fixed-epoch design does not isolate a pure gradient-noise mechanism because
batch size also determines the number of optimizer updates for a fixed number of
data passes. The bootstrap intervals are conditional on the fitted OOF
predictions and do not include retraining-seed uncertainty.

Primary analysis outputs:

- `outputs/reviewer_r1_batch_split_factorial_v1/factorial_cell_metrics_verified.csv`
- `outputs/reviewer_r1_batch_split_factorial_v1/factorial_group_bootstrap_contrasts.csv`
- `outputs/reviewer_r1_batch_split_factorial_v1/factorial_analysis_report.md`
- `outputs/reviewer_r1_batch_split_factorial_v1/factorial_integrity_checks.json`
