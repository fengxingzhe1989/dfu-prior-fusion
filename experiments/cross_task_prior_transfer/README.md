# Shared lesion-prior transfer implementation

`train_lpt.py` supplies architecture, dataset and prior-processing components to the current [nested evaluation](../associate_editor_revision_v1/README.md). Its five retained configs are frozen inputs for Full LPT and the four direct variants; the architecture-matched neutral condition is constructed in the current nested trainer.

`generate_dfuc2021_priors.py` requires the separately archived source-segmenter checkpoint. The new segmentation audit does not replace it. `run_lpt_cv.py` is retained for the historical batch/split factorial, not as the new primary evaluation entry point. See [reproduction instructions](../../REPRODUCIBILITY.md).
