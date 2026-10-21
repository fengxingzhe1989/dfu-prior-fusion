# RGB interpretability extension — authorized 2026-10-10

This is an additional exploratory analysis requested after inspecting the original
two-model results. It is not retroactively part of the original frozen analysis.
The original protocol, results, code and manuscripts must remain unchanged.

## Fixed scope before new inference

- Add EfficientNet-B0 (`efficientnetb0`), ConvNeXt-Tiny (`convnext_tiny`) and
  EfficientNetV2-S (`efficientnetv2s`); do not add other models implicitly.
- Reuse the completed outer-refit checkpoints, all five seeds 20260925–20260929,
  all five grouped outer folds, and all 5,955 quantitative images. No training.
- Copy the sealed original input manifest, 20 qualitative cases and all native
  mask placements byte-for-byte. No case, seed or threshold selection.
- Reuse the archived predicted-class targets, fixed prior mask threshold 0.3,
  up to 30 shape/area-matched translated controls, native-coordinate mean-fill
  occlusion, FP16 replay and FP32 CAM/perturbations, CAM threshold 0.7, and all
  eligibility/denominator rules. No-op tolerance 1e-5, replay tolerance 1e-6
  with zero replay label disagreements. FP32/FP16 discrepancies are retained.
- For each new torchvision RGB model use `features[-1]`, its final spatial
  feature stage before pooling, as the Grad-CAM target layer. Average gradients
  over spatial dimensions; positive weighted activation; bilinear upsample to
  224 x 224 and per-image min-max normalization. Different architectures do not
  imply interchangeable feature semantics or comparable attribution magnitudes.
- Invalid/zero CAMs remain unavailable, never replaced by better-looking maps.
- 75 added full jobs, 89,325 added image x seed x model records; all five models
  together provide 148,875 records. Original 50 jobs are reused read-only.

## Audits and statistical boundary

Before full execution: verify sealed provenance and checkpoint metadata/hashes;
replay the complete first outer fold for each new model; smoke-test preprocessing,
gradient hooks and perturbations. Each later job also replays its entire fold.
Archive incomplete logs/artifacts and resume only hash-verified complete jobs.

After all 75 jobs finish, audit exact image/seed/fold coverage and all asset hashes.
Use the existing descriptive summaries and the same 5,000 crossed seed x
similarity-group bootstrap draws. Report individual-model occlusion scores and
Full LPT-minus-each-RGB comparisons for predicted and true-label targets, overall
and by class. These are additional descriptive, pointwise exploratory intervals,
not multiplicity-adjusted confirmatory findings or estimates of a causal prior
effect. CAM comparisons are descriptive means/sample SD, with all five seed
points and metric-specific denominators. No p-values or new primary claim.

## Figure contract

Purpose: compare model sensitivity on identical images, preserving both agreement
and disagreement; no prediction of which architecture should look better.
Python/matplotlib image-plate plus quantitative evidence. Preserve RGB/prior/CAM
arrays, magma color map, 0–1 per-image normalization and opacity 0.40; no retouching.
Main examples remain ranks 1/6/11/16 under first-listed seed 20260925. The same
20 cases under that seed form compact class-specific supplementary plates;
other-seed arrays/plates remain archived. Export PDF/SVG/PNG; final publication
layout must be visually checked after all computations complete. These displays
are not anatomical accuracy, external/clinical validation or RA-LPT explanations.

## Execution

Use the original server Python environment. Run `run.py --mode prepare`, then
`--mode smoke`, then `--mode run`; full mode runs finalization only after all jobs
finish. `--mode finalize` can resume summary/export without rerunning inference.
The output root defaults to `outputs/interpretability_rgb_extension_v1` and never
overwrites `outputs/associate_editor_revision_v1/interpretability`.
