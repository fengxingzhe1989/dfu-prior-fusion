# Editor Comments 10 and 12: frozen interpretability execution

This is inference on the **new revision outer-refit checkpoints**, not new training.
It implements Section D of `METHODS.md` after the main analysis moved to the new
nested evaluation. All five training seeds (20260925–20260929), both models
(`resnet50`, `lpt_full`), and all 5,955 images are retained.

## Execution

Run from the repository root on the CUDA training server:

```bash
.venv/bin/python -m unittest experiments.associate_editor_revision_v1.test_interpretability_geometry -v
.venv/bin/python -u experiments/associate_editor_revision_v1/run_interpretability.py --mode prepare
.venv/bin/python -u experiments/associate_editor_revision_v1/run_interpretability.py --mode smoke
.venv/bin/python -u experiments/associate_editor_revision_v1/run_interpretability.py --mode run
```

Preparation seals `outputs/associate_editor_revision_v1/interpretability/`:

- `qualitative_cases.csv`: 20 prediction-blind cases, five per class from distinct groups;
- `placements.json`: fixed native-grid, shape/area-preserving translations shared by both models;
- `execution_contract.json`: precision, mask, target, occlusion and uncertainty rules;
- `input_manifest.csv` and `source_registry.json`: input, model, prediction and code hashes.

The smoke check reproduces each model's complete first outer-fold archived OOF
predictions before testing representative preprocessing and perturbations.
Full execution repeats the replay check for every checkpoint, produces 50
auditable job bundles, then runs the quantitative summary and fixed-case renderer.

## Status and recovery

The first full run was started on 2026-10-01 after both smoke checks passed.
Its server log is `outputs/associate_editor_revision_v1/interpretability/runner_20261001.log`.
This note does **not** certify completion; use actual processes, `progress.json`,
all 50 job `complete.json` files and final summary/figure markers.

Completed jobs are resumed only after hash verification. Preserve incomplete
outputs and failed logs. Do not modify sealed code or inputs during execution.
The independent `resume_interpretability_finalization.py` helper may be used
after all 50 jobs finish if summary/figure finalization needs recovery; it does
not rerun inference or change the frozen analysis methods.

## Reporting boundary

Grad-CAM explains the selected CNN global branch, not the whole RA-LPT system.
The transferred prior is not a target-domain ground-truth mask. Occlusion results
are exploratory model-sensitivity measurements, not causal, anatomical or
clinical validation. Empty/immovable masks and unavailable CAMs are reported
explicitly. Random placements are not independent bootstrap observations.
All five seeds and all 20 fixed cases remain in the deliverables, including
misclassifications. Exported figures require visual QA before manuscript use.
