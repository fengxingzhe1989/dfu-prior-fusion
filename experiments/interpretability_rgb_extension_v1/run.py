"""Add three RGB models without changing the original sealed analysis."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from experiments.associate_editor_revision_v1 import run_interpretability as base

MODELS = ('efficientnetb0', 'convnext_tiny', 'efficientnetv2s')
SEEDS = base.SEEDS
NAMES = {'resnet50': 'ResNet50', 'lpt_full': 'Full LPT',
         'efficientnetb0': 'EfficientNet-B0', 'convnext_tiny': 'ConvNeXt-Tiny',
         'efficientnetv2s': 'EfficientNetV2-S'}
sha, save_json = base.sha, base.save_json


def source_job(source, seed, fold, model):
    return source/'classification'/f'seed_{seed}'/f'outer_{fold}'/model/'refit'


def validate_done(job, seed, fold, model):
    done = json.loads((job/'complete.json').read_text())
    assert (done['status'], done['base_seed'], done['outer_fold'], done['model'], done['stage']) == (
        'complete', seed, fold, model, 'refit')
    for name, field in [('model.pt', 'checkpoint_sha256'), ('predictions.csv', 'prediction_sha256')]:
        assert sha(job/name) == done[field], str(job/name)
    return done


def verify(repo, source, out, full=False):
    dst = out/'interpretability'
    reg = json.loads((dst/'source_registry.json').read_text())
    assert tuple(reg['models']) == MODELS and tuple(reg['seeds']) == SEEDS
    assert Path(reg['source_root']).resolve() == source.resolve()
    for name, digest in reg['assets'].items():
        assert sha(dst/name) == digest, name
    for name, digest in reg['code_assets'].items():
        assert sha(repo/name) == digest, name
    for name, digest in reg['source_assets'].items():
        if full or name.endswith('source_registry.json'):
            assert sha(repo/name) == digest, name
    if full:
        for row in pd.read_csv(dst/'input_manifest.csv').itertuples():
            assert sha(repo/row.image_path) == row.image_sha256, row.image
            assert sha(repo/row.prior_path) == row.prior_sha256, row.image
    return reg


def prepare(repo, source, out):
    dst = out/'interpretability'
    if (dst/'source_registry.json').exists():
        verify(repo, source, out, True)
        return
    original = source/'interpretability'
    frozen = base.verify_registry(repo, source, verify_inputs=True)
    assert json.loads((original/'interpretability_complete.json').read_text())['status'] == 'complete'
    dst.mkdir(parents=True, exist_ok=True)
    for name in ['qualitative_cases.csv', 'placements.json', 'input_manifest.csv']:
        assert sha(original/name) == frozen['assets'][name]
        if (dst/name).exists():
            assert sha(dst/name) == sha(original/name)
        else:
            shutil.copy2(original/name, dst/name)
    contract = base.contract()
    contract.update(version='rgb_extension_v1', models=list(MODELS),
                    status='additional_exploratory_scope_frozen_before_new_inference',
                    CAM='features[-1] final spatial stage; positive Grad-CAM; per-image min-max',
                    original_analysis_read_only=True, full_jobs=75,
                    display='same four main and twenty supplementary cases; no reselection')
    save_json(dst/'execution_contract.json', contract)
    sources = {str((original/'source_registry.json').relative_to(repo)): sha(original/'source_registry.json'),
               str((original/'interpretability_complete.json').relative_to(repo)): sha(original/'interpretability_complete.json')}
    for seed in SEEDS:
        for fold in range(5):
            for model in MODELS:
                job = source_job(source, seed, fold, model)
                done = validate_done(job, seed, fold, model)
                config = repo/base.nested.MODEL_CONFIGS[model]
                assert sha(config) == done['source_config_sha256']
                for p in [config, job/'model.pt', job/'predictions.csv', job/'complete.json']:
                    sources[str(p.relative_to(repo))] = sha(p)
    sources.update({p: v for p, v in frozen['source_assets'].items() if '/classification/' not in p})
    code = dict(frozen['code_assets'])
    for p in Path(__file__).parent.iterdir():
        if p.suffix in {'.py', '.md'}:
            code[str(p.relative_to(repo))] = sha(p)
    assets = {name: sha(dst/name) for name in ['qualitative_cases.csv', 'placements.json', 'input_manifest.csv', 'execution_contract.json']}
    save_json(dst/'source_registry.json', {
        'status': 'frozen', 'seeds': list(SEEDS), 'models': list(MODELS),
        'n_images': 5955, 'n_groups': 5670, 'source_root': str(source),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'assets': assets, 'source_assets': sources, 'code_assets': code,
        'qualitative_cases_sha256': assets['qualitative_cases.csv'],
        'environment': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(0)},
    })
    print('PREPARE_OK 75 existing checkpoints; original cases and placements reused', flush=True)


def load_model(repo, source, seed, fold, name, device):
    job = source_job(source, seed, fold, name)
    done = validate_done(job, seed, fold, name)
    payload = torch.load(job/'model.pt', map_location='cpu', weights_only=False)
    assert (payload['base_seed'], payload['outer_fold'], payload['model_key'], payload['stage']) == (seed, fold, name, 'refit')
    assert tuple(payload['class_names']) == base.LABELS
    cfg = copy.deepcopy(payload['config']); cfg['model']['pretrained'] = False
    pred = pd.read_csv(job/'predictions.csv', dtype={'image': str})
    assign = base.nested.load_assignment_frame(repo/base.load_protocol(repo)['data']['nested_assignment_manifest'], fold)
    test = assign.loc[assign.outer_role.eq('test')]
    assert pred.image.is_unique and set(pred.image) == set(test.image)
    assert not set(test.similarity_group) & set(assign.loc[assign.outer_role.eq('train'), 'similarity_group'])
    records = base.rgb.load_records(repo/cfg['data']['split_manifest'], repo)
    by_name = {r.image: r for r in records}
    for row in pred.itertuples():
        assert by_name[row.image].fold == fold and base.LABELS[by_name[row.image].label] == row.true_label
    ds = base.rgb.DFUC2021ClassificationDataset([by_name[x] for x in pred.image], base.rgb.build_transforms(cfg, False))
    model = base.rgb.build_model(cfg)
    model.load_state_dict(payload['model_state'], strict=True)
    return model.to(device).eval(), ds, cfg, pred, done


def gradcam(model, item, target, device):
    captured = []
    hook = model.features[-1].register_forward_hook(lambda _m, _i, o: captured.append(o))
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(base.batch_to_device([item], device)['image'])
            activation = captured[0]
            assert activation.ndim == 4 and min(activation.shape[2:]) > 1
            gradients = torch.autograd.grad(logits[0, target], activation)[0]
            assert torch.isfinite(gradients).all()
            coarse = torch.relu((gradients.mean((2, 3), keepdim=True)*activation).sum(1, keepdim=True))
            cam = F.interpolate(coarse, size=(224, 224), mode='bilinear', align_corners=False)[0, 0].detach().cpu().numpy()
            valid = bool(np.isfinite(cam).all() and float(cam.max()-cam.min()) > 1e-12)
            cam = (cam-cam.min())/(cam.max()-cam.min()) if valid else np.zeros_like(cam)
        return cam, valid, torch.softmax(logits.detach(), 1)[0].cpu().numpy(), list(activation.shape)
    finally:
        hook.remove()


def check_marker(job, registry_hash):
    marker = json.loads((job/'complete.json').read_text())
    assert marker['status'] == 'complete' and marker['source_registry_sha256'] == registry_hash
    for name, digest in marker['assets'].items():
        assert sha(job/name) == digest, str(job/name)
    return marker


def run_job(repo, source, out, seed, fold, name, smoke=False):
    dst = out/'interpretability'; registry_hash = sha(dst/'source_registry.json')
    job = dst/('smoke' if smoke else 'jobs')/f'seed_{seed}'/f'outer_{fold}'/name
    if (job/'complete.json').exists():
        check_marker(job, registry_hash)
        print('SKIP', job, flush=True); return
    job.mkdir(parents=True, exist_ok=True)
    for filename in ['metrics.partial.csv', 'placements.partial.csv', 'replay.json']:
        p = job/filename
        if p.exists():
            p.rename(p.with_name(p.name+'.incomplete_'+str(time.time_ns())))
    start = time.monotonic(); device = torch.device('cuda')
    model, ds, cfg, pred, done = load_model(repo, source, seed, fold, name, device)
    audit = base.replay(model, ds, pred, 'resnet50', device, seed)
    save_json(job/'replay.json', audit)
    print(f'REPLAY_OK {seed} {fold} {name} n={len(pred)} maxdiff={audit["max_probability_difference"]}', flush=True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    placements = json.loads((dst/'placements.json').read_text())
    inputs = pd.read_csv(dst/'input_manifest.csv', dtype={'image': str, 'similarity_group': str}).set_index('image')
    cases = set(pd.read_csv(dst/'qualitative_cases.csv').image)
    all_records, _ = base.records_and_config(repo); records = {r.image: r for r in all_records}
    transform = base.rgb.build_transforms(cfg, False)
    indices = list(range(len(pred)))
    if smoke:
        selected = set(indices[:2])
        for status in ['empty', 'immovable', 'movable']:
            selected.update([i for i in indices if placements[pred.iloc[i].image]['mask_status'] == status][:1])
        for flag in [False, True]:
            selected.update([i for i in indices if placements[pred.iloc[i].image]['roi']['fallback'] == flag][:2])
        indices = sorted(selected)
    archives = []; noop_max = 0.0
    with (job/'metrics.partial.csv').open('w', newline='') as mf, (job/'placements.partial.csv').open('w', newline='') as pf:
        writer = None
        pw = csv.DictWriter(pf, fieldnames=['image', 'seed', 'outer_fold', 'model', 'placement', 'dy', 'dx', 'drop_pred', 'drop_true']); pw.writeheader()
        for step, index in enumerate(indices):
            prow = pred.iloc[index]; image = prow.image; record = records[image]
            src, prior = base.read_inputs(record); frozen = placements[image]; mask = prior >= 0.3
            assert sha(record.image_path) == inputs.loc[image, 'image_sha256']
            assert sha(record.prior_path) == inputs.loc[image, 'prior_sha256']
            assert int(mask.sum()) == frozen['mask_area'] and list(src.shape[:2]) == [frozen['height'], frozen['width']]
            make = lambda im: {'image': transform(image=im)['image'].float()}
            item = make(src); assert torch.equal(item['image'], ds[index]['image'])
            target = base.LABELS.index(prow.pred_label); truth = record.label
            cam, valid, cam_probs, shape = gradcam(model, item, target, device)
            coords = [tuple(x) for x in frozen['placements']]
            assert coords == base.translations(mask, 30, 20261002, image)
            tensors = [item]
            for m in [mask, ~mask] + [base.translate_mask(mask, *xy) for xy in coords]:
                changed = src.copy(); changed[m.astype(bool)] = base.MEAN_FILL
                tensors.append(make(changed))
            with torch.no_grad():
                probs = np.concatenate([torch.softmax(model(base.batch_to_device(tensors[first:first+16], device)['image']), 1).cpu().numpy()
                                        for first in range(0, len(tensors), 16)])
            assert np.isfinite(probs).all()
            error = float(np.abs(probs[0]-cam_probs).max()); noop_max = max(noop_max, error)
            assert error <= 1e-5, (image, error)
            drops = probs[0, None, :]-probs
            random_pred = float(drops[3:, target].mean()) if coords else float('nan')
            random_true = float(drops[3:, truth].mean()) if coords else float('nan')
            projected = cv2.resize(mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
            support = bool(valid and projected.any())
            if support:
                cm = cam >= .7
                mass = float(cam[projected].sum()/cam.sum()); iou = float((cm & projected).sum()/(cm | projected).sum())
                hit = float(projected[np.unravel_index(cam.argmax(), cam.shape)])
            else:
                mass = iou = hit = float('nan')
            archived = np.array([prow['prob_'+label] for label in base.LABELS])
            row = dict(image=image, label=truth, similarity_group=inputs.loc[image, 'similarity_group'],
                       outer_fold=fold, seed=seed, model=name, mask_status=frozen['mask_status'],
                       mask_area_native=int(mask.sum()), mask_fraction_native=float(mask.mean()),
                       n_placements=len(coords), prior_fallback=int(frozen['roi']['fallback']),
                       cam_valid=int(support), cam_signal_valid=int(valid), cam_support_empty_224=int(not projected.any()),
                       cam_prior_mass_ratio=mass, cam_prior_iou=iou, pointing_hit=hit,
                       archived_pred_label=target, fp32_pred_label=int(probs[0].argmax()), baseline_prob_pred=float(probs[0, target]),
                       prior_drop_pred=float(drops[1, target]), random_drop_pred=random_pred,
                       prior_minus_random_pred=float(drops[1, target])-random_pred,
                       prior_drop_true=float(drops[1, truth]), random_drop_true=random_true,
                       prior_minus_random_true=float(drops[1, truth])-random_true,
                       background_drop_pred=float(drops[2, target]), baseline_fp32_vs_archived_maxdiff=float(np.abs(probs[0]-archived).max()),
                       baseline_fp32_pred_disagree=int(probs[0].argmax() != target), fp32_noop_probability_maxdiff=error)
            if writer is None:
                writer = csv.DictWriter(mf, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row)
            for position, (dy, dx) in enumerate(coords):
                pw.writerow(dict(image=image, seed=seed, outer_fold=fold, model=name, placement=position,
                                 dy=dy, dx=dx, drop_pred=float(drops[position+3, target]), drop_true=float(drops[position+3, truth])))
            if image in cases or smoke:
                p = job/'cases'/f'{Path(image).stem}.npz'; p.parent.mkdir(exist_ok=True)
                np.savez_compressed(p, rgb=cv2.resize(src, (224, 224), interpolation=cv2.INTER_LINEAR),
                                    prior=cv2.resize(prior, (224, 224), interpolation=cv2.INTER_LINEAR), cam=cam,
                                    label=truth, pred_label=target, fp32_pred_label=int(probs[0].argmax()),
                                    cam_valid=int(valid), image=image, seed=seed, outer_fold=fold)
                archives.append(p)
            mf.flush(); pf.flush()
            if (step+1) % 50 == 0 or step+1 == len(indices):
                progress = dict(status='running', seed=seed, outer_fold=fold, model=name,
                                images_done_in_job=step+1, images_total_in_job=len(indices),
                                elapsed_seconds=time.monotonic()-start, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                save_json(dst/('smoke_progress.json' if smoke else 'progress.json'), progress)
                print('PROGRESS', json.dumps(progress), flush=True)
    for name_csv in ['metrics', 'placements']:
        (job/(name_csv+'.partial.csv')).replace(job/(name_csv+'.csv'))
    audit.update(fp32_no_op_maxdiff=noop_max, target_layer='features[-1]', activation_shape=shape)
    save_json(job/'replay.json', audit)
    assets = {str(p.relative_to(job)): sha(p) for p in [job/'metrics.csv', job/'placements.csv', job/'replay.json', *archives]}
    save_json(job/'complete.json', dict(status='complete', source_registry_sha256=registry_hash,
              seed=seed, outer_fold=fold, model=name, images=len(indices), smoke=smoke,
              checkpoint_sha256=done['checkpoint_sha256'], assets=assets, elapsed_seconds=time.monotonic()-start))
    del model, ds; torch.cuda.empty_cache()
    print(f'COMPLETE {seed} {fold} {name} smoke={smoke}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=ROOT)
    p.add_argument('--source-root', type=Path, default=Path('outputs/associate_editor_revision_v1'))
    p.add_argument('--output-root', type=Path, default=Path('outputs/interpretability_rgb_extension_v1'))
    p.add_argument('--mode', choices=['prepare', 'smoke', 'run', 'finalize'], required=True)
    args = p.parse_args(); repo = args.repo_root.resolve()
    source = (repo/args.source_root).resolve(); out = (repo/args.output_root).resolve()
    assert source != out and not out.is_relative_to(source), 'Extension must not write original outputs'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cv2.setNumThreads(1); torch.set_num_threads(2)
        if args.mode == 'prepare':
            prepare(repo, source, out); return
        verify(repo, source, out, full=True)
        registry_hash = sha(out/'interpretability/source_registry.json')
        if args.mode == 'smoke':
            for model in MODELS:
                run_job(repo, source, out, SEEDS[0], 0, model, True)
            save_json(out/'interpretability/smoke_complete.json', dict(status='complete', source_registry_sha256=registry_hash, models=list(MODELS)))
            return
        if args.mode == 'run':
            smoke = json.loads((out/'interpretability/smoke_complete.json').read_text())
            assert smoke['status'] == 'complete' and smoke['source_registry_sha256'] == registry_hash
            for seed in SEEDS:
                for fold in range(5):
                    for model in MODELS:
                        run_job(repo, source, out, seed, fold, model)
        subprocess.run([sys.executable, str(Path(__file__).with_name('finalize.py')), '--repo-root', str(repo),
                        '--source-root', str(source), '--output-root', str(out)], check=True)


if __name__ == '__main__':
    main()
