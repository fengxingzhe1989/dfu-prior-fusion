# Shared source-domain segmentation implementation

`train.py` and the retained DeepLabV3+ config supply the current [independent source-domain audit](../associate_editor_revision_v1/README.md). `run_cv.py` is retained to document the original prior-generator training workflow.

The new audit uses separate fit/stop/evaluation roles, all five seeds and both evaluation grids. It does not replace the original classification-prior checkpoint or establish target-domain mask accuracy. The earlier selected-checkpoint resolution audit is retained only as historical supporting material.
