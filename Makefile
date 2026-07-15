PYTHON := uv run python

.PHONY: env tensorboard notebook exp1-smoke exp1-fold0 exp1-cv exp2-splits exp2-resnet50-cv exp2-efficientnetb0-cv exp3-priors exp3-lpt-cv

env:
	uv sync --python 3.13

tensorboard:
	uv run tensorboard --logdir outputs/logs --port 6006

notebook:
	uv sync --extra notebook
	uv run jupyter lab

exp1-smoke:
	uv run python experiments/dfuc2022_seg_baseline/train.py --fold 0 --epochs 1 --batch-size 2 --num-workers 0 --max-train-samples 8 --max-val-samples 4 --disable-pretrained

exp1-fold0:
	uv run python experiments/dfuc2022_seg_baseline/train.py --fold 0

exp1-cv:
	uv run python experiments/dfuc2022_seg_baseline/run_cv.py --batch-size 8 --num-workers 0 --skip-existing

exp2-splits:
	uv run python experiments/dfuc2021_cls_baseline/prepare_splits.py

exp2-resnet50-cv:
	uv run python experiments/dfuc2021_cls_baseline/run_cv.py --config experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_resnet50_cv5_v1.yaml --batch-size 64 --num-workers 4 --skip-existing

exp2-efficientnetb0-cv:
	uv run python experiments/dfuc2021_cls_baseline/run_cv.py --config experiments/dfuc2021_cls_baseline/configs/dfuc2021_cls_efficientnetb0_cv5_v1.yaml --batch-size 96 --num-workers 4 --skip-existing

exp3-priors:
	uv run python experiments/cross_task_prior_transfer/generate_dfuc2021_priors.py --checkpoint outputs/checkpoints/dfuc2022_seg_deeplabv3p_cv5_v1_best_overall.pt --batch-size 256 --num-workers 4 --skip-existing

exp3-lpt-cv:
	uv run python experiments/cross_task_prior_transfer/run_lpt_cv.py --config experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml --batch-size 128 --num-workers 4 --skip-existing
