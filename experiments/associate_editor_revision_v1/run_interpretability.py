#!/usr/bin/env python3
"""Frozen, checkpoint-aligned editor 10/12 inference; never trains a model.

Prepare seals prediction-blind cases, native-grid control placements, input and
checkpoint hashes before interpretation. Smoke and full outputs stay separate.
All five training seeds are retained; no best-seed/case selection is performed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from experiments.associate_editor_revision_v1 import train_nested_classification as nested
from experiments.associate_editor_revision_v1.interpretability_geometry import (
    mask_bbox, mask_status, select_cases, translate_mask, translations,
)
from experiments.cross_task_prior_transfer import train_lpt as lpt
from experiments.dfuc2021_cls_baseline import train as rgb

SEEDS = tuple(range(20260925, 20260930))
MODELS = ("resnet50", "lpt_full")
STAGES = {"resnet50": "refit", "lpt_full": "direct_refit"}
LABELS = nested.CLASS_NAMES
MEAN_FILL = np.rint(np.asarray(rgb.IMAGENET_MEAN) * 255).astype(np.uint8)
CODE_NAMES = (
    "run_interpretability.py", "interpretability_geometry.py",
    "summarize_interpretability.py", "render_interpretability.py",
)


def sha(path: Path) -> str:
    return nested.sha256(path)


def save_json(path: Path, value) -> None:
    nested.write_json(path, value)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def contract() -> dict:
    return {
        "status": "frozen_before_interpretation", "version": "editor10_12_interpretability_v1",
        "seeds": list(SEEDS), "models": list(MODELS), "outer_folds": list(range(5)),
        "checkpoint_source": "new revision outer refits; all five seeds; no performance selection",
        "interpretability_seed": 20261002, "bootstrap_seed": 20261001,
        "mask_grid": "native RGB dimensions; float prior resized linearly to native before threshold",
        "prior_threshold": 0.3, "placements": "up to 30 unique legal nonidentity translations without replacement",
        "projection_for_CAM": "nearest-neighbour binary native mask to 224; zero projected support reported separately",
        "CAM": "final class logit, last global ResNet block; positive Grad-CAM; per-image min-max normalization",
        "CAM_threshold": 0.7, "CAM_target": "archived unoccluded OOF predicted class, fixed under perturbation",
        "LPT_CAM_scope": "global branch only, not a full explanation of ROI/statistics/RA-LPT",
        "occlusion_fill_RGB_u8": MEAN_FILL.tolist(),
        "occlusion_coordinate_system": "source RGB; global and ROI rebuilt from same perturbed source",
        "prior_ROI_stats": "original prior, original ROI coordinates/fallback and stats fixed",
        "fallback": "original prior-enhancement retained, including attenuation of rounded mean fill",
        "precision": "archived baseline replay AMP FP16 batch16 with original runtime; CAM/perturbations FP32 TF32 off",
        "replay_tolerance": 1e-6, "replay_label_disagreements_allowed": 0,
        "occlusion_batch_size": 16, "FP32_no_op_tolerance": 1e-5,
        "drop": "FP32 unoccluded probability minus FP32 occluded probability; never mixed with FP16 baseline",
        "primary_sensitivity_endpoint": "prior-mask drop minus mean random-placement drop for archived predicted class",
        "secondary": "true-label drops, background drops, CAM overlap; all exploratory",
        "empty_immovable": "separate status; random-controlled contrasts NA, not zero",
        "case_selection": "20 prediction-blind class-stratified cases; 5/class; globally unique groups; all seed plates",
        "interpretation": "model sensitivity to an uncertain image-derived prior; not anatomical or clinical validation",
        "bootstrap": "5000 paired crossed seed x similarity-group; placements not independent samples; no p values",
    }


def load_protocol(repo: Path) -> dict:
    return yaml.safe_load((repo / 'experiments/associate_editor_revision_v1/protocol.yaml').read_text())


def source_job(out: Path, seed: int, fold: int, model: str) -> Path:
    return out / 'classification' / f'seed_{seed}' / f'outer_{fold}' / model / STAGES[model]


def read_inputs(record):
    bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    prior_u8 = cv2.imread(str(record.prior_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or prior_u8 is None:
        raise FileNotFoundError(record.image)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    prior = prior_u8.astype(np.float32) / 255.0
    if prior.shape != image.shape[:2]:
        prior = cv2.resize(prior, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return image, prior


def roi_geometry(prior: np.ndarray, cfg: dict) -> dict:
    p = cfg['prior']
    height, width = prior.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats((prior >= p['roi_threshold']).astype(np.uint8), connectivity=8)
    area = 0.0
    component = 0
    if count > 1:
        component = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        area = float(stats[component, cv2.CC_STAT_AREA] / (height * width))
    fallback = count <= 1 or float(prior.max()) < p['fallback_max_threshold'] or area < p['fallback_area_threshold']
    bbox = (0, 0, width, height) if fallback else lpt.expanded_bbox(
        int(stats[component, cv2.CC_STAT_LEFT]), int(stats[component, cv2.CC_STAT_TOP]),
        int(stats[component, cv2.CC_STAT_WIDTH]), int(stats[component, cv2.CC_STAT_HEIGHT]),
        width, height, p['bbox_expand_ratio'],
    )
    return {"fallback": bool(fallback), "bbox": list(bbox), "largest_component_area_ratio": area}


def records_and_config(repo: Path):
    cfg, _ = nested.load_config(repo, 'lpt_full', SEEDS[0])
    return lpt.load_records(cfg, repo), cfg


def prepare(repo: Path, out: Path) -> None:
    dst = out / 'interpretability'
    marker = dst / 'source_registry.json'
    if marker.exists():
        verify_registry(repo, out, verify_inputs=True)
        print('PREPARE verified existing frozen registry', flush=True)
        return
    dst.mkdir(parents=True, exist_ok=True)
    protocol_path = repo / 'experiments/associate_editor_revision_v1/protocol.yaml'
    protocol = load_protocol(repo)
    manifest_path = repo / protocol['data']['manifest']
    assert sha(manifest_path) == protocol['data']['server_manifest_sha256']
    frame = pd.read_csv(manifest_path, dtype={'image':str, 'similarity_group':str})
    assert len(frame) == 5955 and frame.image.nunique() == 5955
    assert frame.similarity_group.nunique() == 5670
    assert frame.groupby('similarity_group')['fold'].nunique().max() == 1
    # This occurs before opening any prediction table or inspecting CAMs.
    cases = select_cases(frame, seed=20261002, per_class=5)
    cases_path = dst / 'qualitative_cases.csv'
    if cases_path.exists():
        old = pd.read_csv(cases_path, dtype={'image':str,'similarity_group':str})
        pd.testing.assert_frame_equal(old, cases, check_dtype=False)
    else:
        save_csv(cases_path, cases)
    save_json(dst / 'execution_contract.json', contract())
    print('PREPARE sealed 20 prediction-blind cases', flush=True)
    records, cfg = records_and_config(repo)
    group_lookup = frame.set_index('image').similarity_group.to_dict()
    placements = {}
    inputs = []
    for index, record in enumerate(records):
        image, prior = read_inputs(record)
        mask = prior >= 0.3
        coords = translations(mask, 30, 20261002, record.image)
        state = mask_status(mask)
        assert (state == 'movable') == bool(coords)
        placements[record.image] = {
            'height':image.shape[0], 'width':image.shape[1], 'mask_status':state,
            'mask_area':int(mask.sum()), 'bbox':mask_bbox(mask), 'placements':coords,
            'roi':roi_geometry(prior, cfg),
        }
        inputs.append({
            'image':record.image, 'label':record.label, 'fold':record.fold,
            'similarity_group':group_lookup[record.image],
            'image_path':str(record.image_path.relative_to(repo)), 'image_sha256':sha(record.image_path),
            'prior_path':str(record.prior_path.relative_to(repo)), 'prior_sha256':sha(record.prior_path),
        })
        if (index + 1) % 1000 == 0:
            print(f'PREPARE images={index+1}/5955', flush=True)
    save_json(dst / 'placements.json', placements)
    save_csv(dst / 'input_manifest.csv', pd.DataFrame(inputs))
    sources = {}
    for seed in SEEDS:
        for fold in range(5):
            for model in MODELS:
                job = source_job(out, seed, fold, model)
                done = json.loads((job / 'complete.json').read_text())
                assert done['status'] == 'complete'
                assert (done['base_seed'],done['outer_fold'],done['model'],done['stage']) == (seed,fold,model,STAGES[model])
                source_config = repo / nested.MODEL_CONFIGS[model]
                assert sha(source_config) == done['source_config_sha256'], str(source_config)
                sources[str(source_config.relative_to(repo))] = sha(source_config)
                for name, key in [('model.pt','checkpoint_sha256'),('predictions.csv','prediction_sha256')]:
                    actual = sha(job / name)
                    assert actual == done[key], str(job / name)
                    sources[str((job / name).relative_to(repo))] = actual
                sources[str((job / 'complete.json').relative_to(repo))] = sha(job / 'complete.json')
    code = {}
    for name in CODE_NAMES:
        path = repo / 'experiments/associate_editor_revision_v1' / name
        code[str(path.relative_to(repo))] = sha(path)
    for path in [protocol_path, repo / 'experiments/associate_editor_revision_v1/METHODS.md',
                 repo / 'experiments/associate_editor_revision_v1/train_nested_classification.py',
                 repo / 'experiments/cross_task_prior_transfer/train_lpt.py',
                 repo / 'experiments/dfuc2021_cls_baseline/train.py']:
        code[str(path.relative_to(repo))] = sha(path)
    for path in [manifest_path, repo / protocol['data']['prior_meta'], repo / protocol['data']['nested_assignment_manifest']]:
        sources[str(path.relative_to(repo))] = sha(path)
    assert sources[protocol['data']['nested_assignment_manifest']] == protocol['data']['nested_assignment_manifest_sha256']
    prior_ckpt = repo / protocol['data']['prior_checkpoint']
    assert sha(prior_ckpt) == protocol['data']['expected_prior_checkpoint_sha256']
    sources[str(prior_ckpt.relative_to(repo))] = sha(prior_ckpt)
    assets = {p.name:sha(p) for p in [cases_path, dst/'execution_contract.json', dst/'placements.json',dst/'input_manifest.csv']}
    save_json(marker, {
        'status':'frozen', 'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'seeds':list(SEEDS),'models':list(MODELS),'n_images':5955,'n_groups':5670,
        'assets':assets,'source_assets':sources,'code_assets':code,
        'qualitative_cases_sha256':sha(cases_path),
        'environment':{'torch':torch.__version__,'cuda':torch.version.cuda,'opencv':cv2.__version__,
                       'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    })
    print('PREPARE complete source registry', sha(marker), flush=True)


def verify_registry(repo: Path, out: Path, verify_inputs=False):
    dst = out / 'interpretability'
    reg = json.loads((dst / 'source_registry.json').read_text())
    for name, value in reg['assets'].items():
        assert sha(dst / name) == value, name
    for name, value in reg['code_assets'].items():
        assert sha(repo / name) == value, f'Frozen code changed: {name}'
    if verify_inputs:
        for name, value in reg['source_assets'].items():
            assert sha(repo / name) == value, name
        for row in pd.read_csv(dst/'input_manifest.csv').itertuples():
            assert sha(repo/row.image_path) == row.image_sha256, row.image
            assert sha(repo/row.prior_path) == row.prior_sha256, row.image
    return reg


def model_and_dataset(repo: Path, out: Path, seed: int, fold: int, name: str, device):
    job = source_job(out, seed, fold, name)
    done = json.loads((job/'complete.json').read_text())
    for fn, key in [('model.pt','checkpoint_sha256'),('predictions.csv','prediction_sha256')]:
        assert sha(job/fn) == done[key], str(job/fn)
    payload = torch.load(job/'model.pt', map_location='cpu', weights_only=False)
    assert (payload['base_seed'],payload['outer_fold'],payload['model_key'],payload['stage']) == (seed,fold,name,STAGES[name])
    assert tuple(payload['class_names']) == LABELS
    cfg = copy.deepcopy(payload['config'])
    cfg['model']['pretrained'] = False
    pred = pd.read_csv(job/'predictions.csv', dtype={'image':str})
    assert pred.image.is_unique
    assign = nested.load_assignment_frame(repo / load_protocol(repo)['data']['nested_assignment_manifest'], fold)
    names = assign.loc[assign.outer_role.eq('test'), 'image'].tolist()
    # Check exact role using saved predictions if role naming ever differs; never infer a new split.
    if not names:
        raise ValueError(f'No test assignments for fold {fold}; roles={assign.outer_role.unique()}')
    assert set(pred.image) == set(names)
    assert not set(assign.loc[assign.outer_role.eq('train'),'similarity_group']) & set(assign.loc[assign.outer_role.eq('test'),'similarity_group'])
    if name == 'resnet50':
        records = rgb.load_records(repo / cfg['data']['split_manifest'], repo)
        by_name = {r.image:r for r in records}
        ds = rgb.DFUC2021ClassificationDataset([by_name[x] for x in pred.image], rgb.build_transforms(cfg,False))
        model = rgb.build_model(cfg)
    else:
        by_name = {r.image:r for r in lpt.load_records(cfg, repo)}
        ds = nested.lpt_dataset([by_name[x] for x in pred.image], cfg, False, False)
        model = lpt.LPTNet(cfg, stats_dim=4)
    for row in pred.itertuples():
        assert by_name[row.image].fold == fold and LABELS[by_name[row.image].label] == row.true_label
    model.load_state_dict(payload['model_state'], strict=True)
    del payload
    return model.to(device).eval(), ds, cfg, pred, done


def forward(model, tensors: dict, name: str):
    if name == 'resnet50':
        return model(tensors['image'])
    return model(tensors['image'],tensors['roi'],tensors['prior'],tensors['stats'])['class_logits']


def tensorize(image, prior, stats, geometry, transform, name):
    if name == 'resnet50':
        return {'image':transform(image=image)['image'].float()}
    if geometry['fallback']:
        roi = lpt.build_prior_enhanced_image(image, prior)
    else:
        x1,y1,x2,y2 = geometry['bbox']
        roi = image[y1:y2,x1:x2].copy()
    if roi.shape[:2] != image.shape[:2]:
        roi = cv2.resize(roi,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_LINEAR)
    val = transform(image=image,roi=roi,prior=prior)
    pt = val['prior'].float()
    return {'image':val['image'].float(),'roi':val['roi'].float(),
            'prior':pt.unsqueeze(0) if pt.ndim == 2 else pt,
            'stats':torch.tensor(stats,dtype=torch.float32)}


def batch_to_device(items, device):
    return {key:torch.stack([x[key] for x in items]).to(device) for key in items[0]}


def gradcam(model, item, name, target, device):
    layer = model.layer4[-1] if name == 'resnet50' else model.global_stem[-1][-1]
    captured = []
    hook = layer.register_forward_hook(lambda _m,_i,o:captured.append(o))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = forward(model,batch_to_device([item],device),name)
            activation = captured[0]
            gradients = torch.autograd.grad(logits[0,target],activation)[0]
            if not torch.isfinite(gradients).all():
                raise FloatingPointError('Nonfinite CAM gradients')
            coarse = torch.relu((gradients.mean((2,3),keepdim=True) * activation).sum(1,keepdim=True))
            cam = F.interpolate(coarse,size=(224,224),mode='bilinear',align_corners=False)[0,0].detach().cpu().numpy()
            valid = bool(np.isfinite(cam).all() and float(cam.max()-cam.min()) > 1e-12)
            cam = (cam-cam.min()) / (cam.max()-cam.min()) if valid else np.zeros_like(cam)
        return cam,valid,torch.softmax(logits.detach(),1)[0].cpu().numpy()
    finally:
        hook.remove()


def replay(model, ds, pred, name, device, seed):
    nested.set_deterministic_runtime(seed)
    loader = DataLoader(ds,batch_size=16,shuffle=False,num_workers=4,pin_memory=True)
    rows = nested.predict_rgb(model,loader,device,True) if name == 'resnet50' else nested.predict_lpt(model,loader,device,True)
    got = pd.DataFrame(rows)
    assert list(got.image) == list(pred.image)
    columns = [c for c in got if c.startswith('prob_')]
    delta = np.abs(got[columns].to_numpy()-pred[columns].to_numpy())
    maxdiff = float(delta.max())
    disagreements = int((got.pred_label != pred.pred_label).sum())
    if maxdiff > 1e-6 or disagreements:
        raise RuntimeError(f'Archived OOF replay failed {name}: maxdiff={maxdiff}, label_disagreements={disagreements}')
    return {'samples':len(got),'max_probability_difference':maxdiff,'prediction_disagreements':disagreements,
            'precision':'original FP16+fallback','batch_size':16}


def run_job(repo: Path, out: Path, seed: int, fold: int, name: str, smoke=False):
    dst = out/'interpretability'
    registry_hash = sha(dst/'source_registry.json')
    job = dst / ('smoke' if smoke else 'jobs') / f'seed_{seed}' / f'outer_{fold}' / name
    marker = job/'complete.json'
    if marker.exists():
        complete = json.loads(marker.read_text())
        assert complete['source_registry_sha256'] == registry_hash
        for asset, digest in complete['assets'].items():
            assert sha(job/asset) == digest, asset
        print(f'SKIP {job}',flush=True)
        return
    job.mkdir(parents=True,exist_ok=True)
    for stale in ['metrics.partial.csv','placements.partial.csv','replay.json']:
        path = job/stale
        if path.exists():
            path.rename(path.with_name(path.name+'.incomplete_'+time.strftime('%Y%m%d_%H%M%S')))
    device = torch.device('cuda')
    started = time.monotonic()
    model,official_ds,cfg,pred,source_done = model_and_dataset(repo,out,seed,fold,name,device)
    audit = replay(model,official_ds,pred,name,device,seed)
    save_json(job/'replay.json',audit)
    print(f'REPLAY_OK seed={seed} fold={fold} model={name} n={len(pred)} maxdiff={audit["max_probability_difference"]}',flush=True)
    # Gradients and perturbations use a separately declared, consistent precision.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    placement_data = json.loads((dst/'placements.json').read_text())
    inputs = pd.read_csv(dst/'input_manifest.csv',dtype={'image':str,'similarity_group':str}).set_index('image')
    cases = set(pd.read_csv(dst/'qualitative_cases.csv',dtype={'image':str}).image)
    lpt_records,lpt_cfg = records_and_config(repo)
    if name == 'lpt_full':
        assert lpt_cfg['prior'] == cfg['prior'], 'LPT prior configuration drift'
        assert lpt_cfg['data'] == cfg['data'], 'LPT data configuration drift'
    records = {r.image:r for r in lpt_records}
    transform = rgb.build_transforms(cfg,False) if name == 'resnet50' else lpt.build_transforms(cfg,False)
    indices = list(range(len(pred)))
    if smoke:
        chosen = set(indices[:2])
        for fallback in [False,True]:
            chosen.update([i for i in indices if placement_data[pred.iloc[i].image]['roi']['fallback'] == fallback][:2])
        for status in ['movable','immovable','empty']:
            chosen.update([i for i in indices if placement_data[pred.iloc[i].image]['mask_status'] == status][:1])
        indices = sorted(chosen)
    metrics_path = job/'metrics.partial.csv'
    placement_path = job/'placements.partial.csv'
    case_assets = []
    noop_max = 0.0
    with metrics_path.open('w',newline='') as metric_file, placement_path.open('w',newline='') as pos_file:
        writer = None
        pos_writer = csv.DictWriter(pos_file,fieldnames=['image','seed','outer_fold','model','placement','dy','dx','drop_pred','drop_true'])
        pos_writer.writeheader()
        for step,index in enumerate(indices):
            prow = pred.iloc[index]
            image_name = prow.image
            record = records[image_name]
            src,prior = read_inputs(record)
            frozen = placement_data[image_name]
            assert sha(record.image_path) == inputs.loc[image_name,'image_sha256']
            assert sha(record.prior_path) == inputs.loc[image_name,'prior_sha256']
            mask = prior >= 0.3
            assert int(mask.sum()) == frozen['mask_area'] and list(src.shape[:2]) == [frozen['height'],frozen['width']]
            geometry = roi_geometry(prior,lpt_cfg)
            assert geometry == frozen['roi']
            make = lambda im:tensorize(im,prior,record.prior_stats,geometry,transform,name)
            base_item = make(src)
            official = official_ds[index]
            for key in base_item:
                assert torch.equal(base_item[key],official[key]), f'No-op preprocessing mismatch {image_name}:{key}'
            target = LABELS.index(prow.pred_label)
            truth = record.label
            cam,valid,cam_probs = gradcam(model,base_item,name,target,device)
            coords = [tuple(x) for x in frozen['placements']]
            assert coords == translations(mask,30,20261002,image_name)
            # Condition0 baseline,1 prior,2 background, then all fixed placements.
            tensors = [base_item]
            masks = [mask,~mask] + [translate_mask(mask,*xy) for xy in coords]
            for m in masks:
                if m is not mask and m is not masks[1]:
                    assert int(m.sum()) == int(mask.sum())
                perturbed = src.copy()
                perturbed[m.astype(bool)] = MEAN_FILL
                item = make(perturbed)
                if name == 'lpt_full':
                    assert torch.equal(item['prior'],base_item['prior']) and torch.equal(item['stats'],base_item['stats'])
                tensors.append(item)
            probabilities = []
            with torch.no_grad():
                for first in range(0,len(tensors),16):
                    probabilities.append(torch.softmax(forward(model,batch_to_device(tensors[first:first+16],device),name),1).cpu().numpy())
            probs = np.concatenate(probabilities)
            if not np.isfinite(probs).all():
                raise FloatingPointError(f'Nonfinite FP32 probabilities {image_name}')
            noop_error = float(np.abs(probs[0]-cam_probs).max())
            noop_max = max(noop_max,noop_error)
            assert noop_error <= 1e-5, f'FP32 no-op probability mismatch: {noop_error}'
            drops = probs[0,None,:]-probs
            random_pred = float(drops[3:,target].mean()) if coords else float('nan')
            random_true = float(drops[3:,truth].mean()) if coords else float('nan')
            projected = cv2.resize(mask.astype(np.uint8),(224,224),interpolation=cv2.INTER_NEAREST).astype(bool)
            cam_support_valid = bool(valid and projected.any())
            if cam_support_valid:
                cam_mask = cam >= .7
                mass = float(cam[projected].sum()/cam.sum())
                iou = float((cam_mask & projected).sum()/(cam_mask | projected).sum())
                pointing = float(projected[np.unravel_index(cam.argmax(),cam.shape)])
            else:
                mass = iou = pointing = float('nan')
            archived = np.array([prow['prob_'+label] for label in LABELS])
            row = {
                'image':image_name,'label':truth,'similarity_group':inputs.loc[image_name,'similarity_group'],
                'outer_fold':fold,'seed':seed,'model':name,'mask_status':frozen['mask_status'],
                'mask_area_native':int(mask.sum()),'mask_fraction_native':float(mask.mean()),
                'n_placements':len(coords),'prior_fallback':int(geometry['fallback']),
                'cam_valid':int(cam_support_valid),'cam_signal_valid':int(valid),
                'cam_support_empty_224':int(not projected.any()),
                'cam_prior_mass_ratio':mass,'cam_prior_iou':iou,'pointing_hit':pointing,
                'archived_pred_label':target,'fp32_pred_label':int(probs[0].argmax()),
                'baseline_prob_pred':float(probs[0,target]),
                'prior_drop_pred':float(drops[1,target]),'random_drop_pred':random_pred,
                'prior_minus_random_pred':float(drops[1,target])-random_pred,
                'prior_drop_true':float(drops[1,truth]),'random_drop_true':random_true,
                'prior_minus_random_true':float(drops[1,truth])-random_true,
                'background_drop_pred':float(drops[2,target]),
                'baseline_fp32_vs_archived_maxdiff':float(np.abs(probs[0]-archived).max()),
                'baseline_fp32_pred_disagree':int(probs[0].argmax() != target),
                'fp32_noop_probability_maxdiff':noop_error,
            }
            if writer is None:
                writer = csv.DictWriter(metric_file,fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            for position,(dy,dx) in enumerate(coords):
                pos_writer.writerow({'image':image_name,'seed':seed,'outer_fold':fold,'model':name,
                                     'placement':position,'dy':dy,'dx':dx,
                                     'drop_pred':float(drops[position+3,target]),'drop_true':float(drops[position+3,truth])})
            if image_name in cases or smoke:
                case_path = job/'cases'/f'{Path(image_name).stem}.npz'
                case_path.parent.mkdir(exist_ok=True)
                np.savez_compressed(case_path,rgb=cv2.resize(src,(224,224),interpolation=cv2.INTER_LINEAR),
                                    prior=cv2.resize(prior,(224,224),interpolation=cv2.INTER_LINEAR),cam=cam,
                                    label=truth,pred_label=target,fp32_pred_label=int(probs[0].argmax()),
                                    cam_valid=int(valid),image=image_name,seed=seed,outer_fold=fold)
                case_assets.append(case_path)
            metric_file.flush()
            pos_file.flush()
            if (step+1)%50 == 0 or step+1 == len(indices):
                progress = {'status':'running','seed':seed,'outer_fold':fold,'model':name,
                            'images_done_in_job':step+1,'images_total_in_job':len(indices),
                            'elapsed_seconds':time.monotonic()-started,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                save_json(dst/('smoke_progress.json' if smoke else 'progress.json'),progress)
                print('PROGRESS',json.dumps(progress),flush=True)
    metrics_path.replace(job/'metrics.csv')
    placement_path.replace(job/'placements.csv')
    audit['fp32_no_op_maxdiff'] = noop_max
    save_json(job/'replay.json',audit)
    assets = {str(p.relative_to(job)):sha(p) for p in [job/'metrics.csv',job/'placements.csv',job/'replay.json',*case_assets]}
    save_json(marker,{'status':'complete','source_registry_sha256':registry_hash,
                      'seed':seed,'outer_fold':fold,'model':name,'images':len(indices),'smoke':smoke,
                      'checkpoint_sha256':source_done['checkpoint_sha256'],'assets':assets,
                      'elapsed_seconds':time.monotonic()-started})
    del model,official_ds
    torch.cuda.empty_cache()
    print(f'COMPLETE seed={seed} fold={fold} model={name} smoke={smoke}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/associate_editor_revision_v1'))
    parser.add_argument('--mode',choices=['prepare','smoke','run'],required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    out = args.output_root if args.output_root.is_absolute() else repo/args.output_root
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA execution environment required; no silent CPU fallback')
    if args.mode == 'prepare':
        prepare(repo,out)
        return
    verify_registry(repo,out,verify_inputs=True)
    if args.mode == 'smoke':
        for name in MODELS:
            run_job(repo,out,SEEDS[0],0,name,smoke=True)
        save_json(out/'interpretability/smoke_complete.json',{
            'status':'complete','source_registry_sha256':sha(out/'interpretability/source_registry.json'),
            'models':list(MODELS),'scope':'full-fold archived replay; deterministic cropped/fallback/empty/movable probe images',
        })
        return
    smoke = json.loads((out/'interpretability/smoke_complete.json').read_text())
    assert smoke['status'] == 'complete' and smoke['source_registry_sha256'] == sha(out/'interpretability/source_registry.json')
    for seed in SEEDS:
        for fold in range(5):
            for name in MODELS:
                run_job(repo,out,seed,fold,name)
    for script in ['summarize_interpretability.py','render_interpretability.py']:
        subprocess.run([sys.executable,str(repo/'experiments/associate_editor_revision_v1'/script),
                        '--repo-root',str(repo),'--output-root',str(out)],cwd=repo,check=True)
    save_json(out/'interpretability/interpretability_complete.json',{
        'status':'complete','jobs':50,'images_per_model_per_seed':5955,
        'source_registry_sha256':sha(out/'interpretability/source_registry.json'),
        'summary_complete_sha256':sha(out/'interpretability/summary/summary_complete.json'),
        'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    })


if __name__ == '__main__':
    main()
