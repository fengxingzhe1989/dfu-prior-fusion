# Complete training-size sensitivity

All 5 seeds, 5 fixed outer folds, both models and all 4 budgets are included. All four budgets were freshly fitted under the budget-first internal-partition procedure; no endpoint was reused. The full 5955-image OOF evaluation set is unchanged. Smoke fits are excluded.

This later exploratory analysis describes sensitivity to the available labelled-training budget under the frozen procedure. It is not external, patient-level or clinical validation, and does not establish that 227 ischaemia-only images suffice. Pointwise intervals condition on the observed groups and fitted models/subsets. No confirmatory p values were added. Five-trajectory variation combines subset, internal-partition and optimisation realisations. Actual sample counts are in subset_counts.csv. Undefined precision/interval counts are reported explicitly.
