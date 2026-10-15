# Four-budget training-size sensitivity: public release

This README is publication documentation, not part of the original training seal. The seven other protocol/source/test files in this directory are byte-identical to the frozen execution versions. Their original hashes are retained in `paper_assets/release_audit/training_size_asset_registry.csv` and the public provenance projection.

The procedure selects each group-preserving training budget **before** its internal fit/stopping partition. Full LPT and its architecture-matched neutral-prior control share those subsets and fixed outer evaluation folds. Four budgets × five trajectories × five outer folds × two models × two training stages produce **400 fresh fits**, including a freshly fitted 100% endpoint. The original main analysis, and the excluded v1/v2 attempts, are separate records.

From the repository root, run the data-free checks:

```bash
uv run python scripts/check_training_size_release.py
uv run python experiments/training_size_sensitivity_v3/test_design.py
```

All aggregate results are in `paper_assets/training_size_sensitivity_v3/summary/`. Checks cover complete trajectory/fold/class counts, means, sample SDs, paired differences, and percentile intervals reconstructed from saved bootstrap draws. They neither rerun the bootstrap from undistributed OOF arrays nor retrain models.

Read [REPRODUCIBILITY.md, Section 6](../../REPRODUCIBILITY.md#6-later-four-budget-training-size-sensitivity) before using `run.py`. Its `prepare`, `smoke`, `run`, `status`, and `finalize` modes belong to the original archived execution workspace. Exact prior/assignment inputs, ImageNet checkpoints, frozen environment records, complete fit artifacts and registered non-executable sidecars are not all distributed here. The public provenance projection must never be substituted for `sealed_inputs.json`.

`render_curves.py` preserves the original all-trajectory display. The final manuscript later changed only presentation to mean ± sample SD without thin trajectory lines, using y limits 0.40–1.00. Public values still retain all five trajectories at every budget. The exploratory pointwise intervals are not new confirmatory tests and do not alter the primary three-comparison Holm family.
