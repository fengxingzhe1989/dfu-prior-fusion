PYTHON ?= uv run python

.PHONY: help env check test prepare-dry-run ae-classification ae-remaining ae-summary ae-interpret-prepare ae-interpret-smoke ae-interpret ae-interpret-verify legacy-factorial-dry-run

help:
	@echo "Start with make check and make test. Read REPRODUCIBILITY.md before training."

env:
	uv sync --frozen --python 3.13

check:
	$(PYTHON) scripts/check_release.py

test:
	$(PYTHON) -m unittest discover -s experiments/associate_editor_revision_v1 -p 'test_*.py' -v

prepare-dry-run:
	$(PYTHON) data_processing/prepare_manuscript_data.py --dry-run

ae-classification:
	$(PYTHON) experiments/associate_editor_revision_v1/run_classification_matrix.py

ae-remaining:
	$(PYTHON) experiments/associate_editor_revision_v1/run_remaining_pipeline.py

ae-summary:
	$(PYTHON) experiments/associate_editor_revision_v1/summarize.py

ae-interpret-prepare:
	$(PYTHON) experiments/associate_editor_revision_v1/run_interpretability.py --mode prepare

ae-interpret-smoke:
	$(PYTHON) experiments/associate_editor_revision_v1/run_interpretability.py --mode smoke

ae-interpret:
	$(PYTHON) experiments/associate_editor_revision_v1/run_interpretability.py --mode run

ae-interpret-verify:
	$(PYTHON) experiments/associate_editor_revision_v1/resume_interpretability_finalization.py --verify-only

legacy-factorial-dry-run:
	$(PYTHON) experiments/reviewer_r1_batch_split_factorial/run_factorial.py --dry-run
