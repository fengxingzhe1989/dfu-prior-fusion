# Frozen associate-editor revision pipeline

This is the **executed** five-seed pipeline, not the superseded pre-execution handoff. Read [the reproduction guide](../../REPRODUCIBILITY.md) for restricted inputs, verification and execution order.

| Entry point | Role |
| --- | --- |
| `prepare_nested_splits.py` | Deterministic grouped outer/inner/stop assignments and isolation assertions |
| `run_classification_matrix.py` / `train_nested_classification.py` | 900 CNN fits, including architecture-matched Full/neutral controls |
| `repair_nonfinite_classification.py` | Integrity scan and same-configuration recovery preserving failed artifacts |
| `run_stack_matrix.py` / `fit_nested_stacks.py` | Four nested stacks; 100 outer seed/fold fits |
| `run_segmentation_matrix.py` / `train_segmentation_audit.py` | 25 source-domain fits, held-out evaluation at both grids |
| `run_remaining_pipeline.py` | Integrity, stacking and segmentation orchestration only |
| `summarize.py` | All-seed summaries, three contrasts, crossed seed/group bootstrap and Holm adjustment |
| `run_interpretability.py` / `interpretability_geometry.py` | Sealed preparation, replay/smoke validation and 50 inference jobs |
| `summarize_interpretability.py` / `render_interpretability.py` | Sensitivity summaries and all fixed-case plates |
| `resume_interpretability_finalization.py` | Verified missing-stage recovery, no retraining |
| `test_*.py` | Synthetic geometry and finalization regression tests |

Scientific source, shared dependencies, `protocol.yaml` and `METHODS.md` are byte-identical research-workspace copies. The protocol status is a freeze-time value, not live progress. `METHODS.md` is the original Chinese-language protocol. `INTERPRETABILITY_RUN.md` records execution/recovery rules rather than certifying completion. Aggregate completion evidence is released under `paper_assets/associate_editor_revision_v1/`.

Retain all seeds 20260925–20260929, five outer folds and comparators. Classification: 5,955 images / 5,670 similarity groups. Interpretation: both models, all seeds and 20 prediction-blind cases, including errors. No performance-based seed/case selection. The source-domain segmentation audit does not replace the fixed classification-prior generator.
