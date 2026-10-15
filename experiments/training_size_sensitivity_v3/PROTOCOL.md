# Unified fresh training-size sensitivity v3

## Authorised restart and frozen boundary

On 12 October 2026 the author requested a fresh restart because the previous
implementation did not fully match the supplied budget-first design. The v1 and
v2 runners were stopped, with code, seals, logs, completed and incomplete outputs
preserved. No performance-based selection motivated this restart. No v1/v2
predictions or historical full-data endpoints enter the v3 learning curve.
The original main analysis and its three Holm-adjusted comparisons stay intact.

The previous implementation sampled separately inside existing fit/stop roles.
v3 instead first selects each outer-training budget, then partitions that budget
internally. Because internal stopping assignments change, the 100% endpoint is
also freshly fitted. This is a unified later exploratory extension, not a
retroactively prespecified main analysis.

## Population and nested budgets

Keep the original five similarity-grouped outer folds, 5,955 evaluation images,
two models (Full LPT and architecture-matched neutral-prior Full LPT), and all five
training seeds 20260925--20260929. For each seed and outer fold, operate only on
its outer-training rows. Group complete similarity components by their four-class
image-count vectors, including mixed-label components. Sort components and
permute each stratum with SHA256-derived NumPy RNG:
`[20261012, seed, outer, "budget", class_count_vector]`; use its first four bytes
as a big-endian integer. Retain the first ceil(fraction * stratum group count)
at 25%, 50%, 75%, 100%. This makes total budget sets strictly nested while
preserving entire groups. Percentages are nominal; actual counts are reported.

Only after selecting each budget, use StratifiedGroupKFold with five splits,
shuffle=True and SHA256-derived seed from
`[20261012, seed, outer, size, "stopping"]`. Its first held-out split is the
internal stopping set and the remainder is the fitting set. Both model arms
share these assignments. Old fit/stop membership is not used. Stopping sets need
not themselves be nested across sizes. All classes must occur in fit and stop;
otherwise halt before sealing rather than silently redraw a subset or seed.
No outer-test labels enter subset or internal-partition selection.

The budget includes both fit and stop labels. Epoch selection uses only fit and
stop; the chosen duration is then refitted on their union and evaluated on the
unchanged outer fold. Report exact fit/stop/refit image, group and class counts.
The five trajectories vary subset membership, internal partitioning and model
stochastic realisation; their SD must not be labelled pure training-seed SD.

## Fixed training and workload

Use the unchanged original direct-stage trainer and its frozen model, losses,
224 input preprocessing, augmentations, batch 16, maximum 60 epochs, patience 12,
FP16/non-finite fallback, optimiser, schedule horizon and worker settings. Each
model independently selects its duration under the same rule. Start each
selection and refit from the same named ImageNet weight sources, never from a
DFUC2021-trained checkpoint. Keep the original segmentation prior generator and
stored prior maps unchanged. Verify source, input, ImageNet and environment
registries and neutral-prior contract before production. No hyperparameter tuning.

Workload: 4 budgets x 5 outer folds x 5 trajectories x 2 models x 2 stages =
400 newly fitted CNN runs, producing 200 outer prediction files and 40 pooled
OOF model/budget/trajectory cells. No endpoint is reused. Tiny smoke fits are
separate, excluded from these counts and every scientific result. Abandoned
v1/v2 computations remain separately logged and excluded. The original 900
classification fits are a separate main-analysis inventory, not overwritten.

## Complete reporting and analysis

Pool the five outer-fold predictions per trajectory, then compute four-class
Macro-F1 and all class precision/recall/F1, with special attention to ischaemia-only
precision, recall and F1. Report all five trajectories, means and sample SDs,
as well as fold-level counts and metrics. Do not pool predictions across seeds
before computing performance or treat 25 fold/trajectory cells as independent.
Undefined precision is NA, with estimable counts, not replaced with zero.

Retain the existing exploratory descriptive analysis definitions: 5,000 paired
crossed trajectory-by-similarity-group bootstrap draws (RNG 20261013), shared
across all arms and budgets. Within each draw compute each sampled trajectory's
metric before averaging them. Report pointwise percentile intervals, not new
confirmatory p values. Full-minus-neutral contrasts at all four budgets; within
model contrasts 50-25, 75-50, 100-75, 100-50, 100-25. The original three-comparison
Holm family remains unchanged. Intervals condition on fitted models/subsets and
observed groups; they do not fully resample the data-generating training process.

After every job passes provenance/coverage validation, generate the four-panel
descriptive learning-curve draft and source tables, retaining every trajectory.
All 400 fits are required irrespective of observed performance. No interpretation
of partial outcomes, monotonic smoothing, favourable-seed/case selection or
replacement based on scores is permitted. This tests DFUC2021 labelled-budget
sensitivity under fixed external priors/pretraining and an epoch-based learning
procedure, not a pure update-matched sample-count effect or total system data need.

Small 75-to-100 changes do not establish adequate sample size. The same 227
ischaemia-only evaluation images, absent patient IDs, lack of target-domain masks
and external validation remain limitations. Proportional downsampling of all
classes does not isolate the effect of adding only ischaemia cases. This cannot
retroactively supply an a priori sample-size calculation or clinical validation.

## Execution and final delivery

`run.py --mode prepare`, `smoke`, `run`, `status`, `finalize` use the existing
server virtual environment. A single runner holds the execution lock and resumes
only its own hash-verified complete jobs. Failed/incomplete attempts are preserved
before minimal repairs and reruns. Protocol changes require a new documented seal.
Do not restart the superseded v1/v2 runners or any cancelled automation.

Computation/figure export is distinct from visual QA and manuscript completion.
After complete coverage and hash verification, inspect actual curves, retrieve
small outputs and synchronise Methods, Results, supplementary N7, inventories,
limitations and editor Response 8 in new clean/tracked copies. Do not describe
this analysis as completed in the manuscript before verification.
