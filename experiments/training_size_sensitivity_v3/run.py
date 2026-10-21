#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import common as base

from common import (ASSIGN, CONFIG, DATA, EXP, EXPECTED_ASSIGN, EXPECTED_TRAINER,
                    LABELS, MODELS, OLD, OUT, ROOT, SEEDS, SIZES, TRAINER, assignment,
                    jobdir, predictions, read_json, reg, sha, subset,
                    validate_assignment, verify_job, verify_registry, write_json)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def environment() -> dict:
    import torch
    import cv2
    packages = {name: importlib.metadata.version(name) for name in (
        'torch', 'torchvision', 'timm', 'albumentations', 'numpy', 'pandas',
        'scikit-learn', 'PyYAML')}
    for name in ('opencv-python', 'opencv-python-headless', 'opencv-contrib-python', 'opencv-contrib-python-headless'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    packages['cv2_imported_version'] = cv2.__version__
    packages['torch_imported_version'] = str(torch.__version__)
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                packages=packages)


def verify_seal(full: bool = False) -> dict:
    seal = read_json(OUT / 'sealed_inputs.json')
    verify_registry(seal['sources'])
    verify_registry(seal['assignments'])
    for path, digest in seal['pretrained_checkpoints'].items():
        if sha(Path(path)) != digest:
            raise RuntimeError('ImageNet initialisation source changed')
    return seal


def assert_environment(seal: dict) -> None:
    current = environment()
    for key in ('python', 'cuda', 'gpu', 'packages'):
        if current[key] != seal['environment'][key]:
            raise RuntimeError(f'Environment drift: {key}')


def cpu_preflight() -> dict:
    sys.path.insert(0, str(ROOT))
    import torch
    from torchvision import models
    from experiments.associate_editor_revision_v1 import train_nested_classification as trainer
    checkpoints = {}
    for weights in (models.ResNet50_Weights.DEFAULT, models.MobileNet_V3_Small_Weights.DEFAULT):
        p = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
        if not p.is_file():
            raise RuntimeError(f'Existing ImageNet checkpoint is unavailable: {p.name}')
        checkpoints[str(p)] = base.sha(p)
    configs, shapes = [], []
    for model_name in base.MODELS:
        job_seed = trainer.derive_job_seed(base.SEEDS[0], 0, 'direct_select', None)
        config, _ = trainer.load_config(ROOT, model_name, job_seed)
        if config['training']['batch_size'] != 16 or not config['model']['pretrained']:
            raise AssertionError('Original training/initialisation contract changed')
        model = trainer.lpt.LPTNet(config, stats_dim=len(config['data']['stats_fields']))
        shapes.append({name: list(value.shape) for name, value in model.state_dict().items()})
        configs.append(config)
        del model
    if shapes[0] != shapes[1]:
        raise AssertionError('Full and neutral architectures differ')
    frame = pd.read_csv(assignment(25, base.SEEDS[0]))
    names = frame.loc[frame.outer_fold.eq(0) & frame.direct_role.eq('fit'), 'image'].tolist()
    records = trainer.select_records(trainer.lpt.load_records(configs[1], ROOT), names)
    dataset = trainer.lpt_dataset(records[:2], configs[1], train=True, neutral=True)
    for i in range(len(dataset)):
        sample = dataset[i]
        if not torch.all(sample['prior'].eq(0.5)) or not torch.equal(sample['stats'], torch.tensor([0.5, 0.5, 1.0, 0.5])):
            raise AssertionError('Neutral prior contract failed')
        if not torch.equal(sample['image'], sample['roi']):
            raise AssertionError('Neutral ROI is not the whole RGB input')
        if sample['image'].shape != (3, 224, 224):
            raise AssertionError('Unexpected input size')
    return dict(status='complete', architecture_matched=True, starts_from_imagenet=True,
                neutral_contract_checked_after_augmentation=True, uses_no_gpu=True,
                pretrained_checkpoints=checkpoints, model_state_shapes=shapes[0])



def prepare() -> None:
    if (OUT / 'sealed_inputs.json').exists():
        verify_seal(full=True)
        print('Existing v3 seal verifies; no overwrite.', flush=True)
        return
    if sha(ASSIGN) != EXPECTED_ASSIGN or sha(TRAINER) != EXPECTED_TRAINER:
        raise RuntimeError('Original assignment/trainer differs from inspected source')
    env = environment()
    old_env = read_json(OLD / 'environment.json')
    if (env['packages']['torch_imported_version'], env['cuda'], env['python'], env['gpu']) != (
            old_env['torch'], old_env['cuda'], old_env['python'], old_env['gpu']):
        raise RuntimeError('Original training environment differs')
    prior = Path(old_env['prior_checkpoint'])
    if sha(prior) != old_env['prior_checkpoint_sha256']:
        raise RuntimeError('Fixed prior generator mismatch')
    original = pd.read_csv(ASSIGN)
    if len(original) != 5955 * 5 or original.outer_fold.nunique() != 5:
        raise RuntimeError('Unexpected original assignment coverage')
    if original.loc[original.outer_fold.eq(0), 'similarity_group'].nunique() != 5670:
        raise RuntimeError('Unexpected group coverage')
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    reports, manifests = [], []
    for seed in SEEDS:
        frames = {size: [] for size in SIZES}
        for outer in range(5):
            source = original.loc[original.outer_fold.eq(outer)].copy()
            reduced = {size: subset(source, seed, outer, size) for size in SIZES}
            sets = [set(reduced[size].loc[reduced[size].outer_role.eq('train'), 'image']) for size in SIZES]
            if not all(a < b for a, b in zip(sets[:-1], sets[1:])):
                raise AssertionError('Budgets are not strictly nested')
            if sets[-1] != set(source.loc[source.outer_role.eq('train'), 'image']):
                raise AssertionError('100% budget is not the complete outer-training population')
            for size, f in reduced.items():
                validate_assignment(f, source)
                frames[size].append(f)
                for role, mask in [('fit', f.direct_role.eq('fit')), ('stop', f.direct_role.eq('stop')),
                                   ('refit', f.outer_role.eq('train')), ('test', f.outer_role.eq('test'))]:
                    part = f.loc[mask]
                    row = dict(seed=seed, outer=outer, nominal_percent=size, role=role,
                               images=len(part), groups=part.similarity_group.nunique())
                    row.update({label: int(part.label.eq(i).sum()) for i, label in enumerate(LABELS)})
                    row['fraction_of_original_outer_training_images'] = len(part)/int(source.outer_role.eq('train').sum()) if role == 'refit' else None
                    reports.append(row)
        for size, fs in frames.items():
            path = assignment(size, seed)
            pd.concat(fs, ignore_index=True).to_csv(path, index=False)
            manifests.append(path)
    pd.DataFrame(reports).to_csv(DATA / 'subset_counts.csv', index=False)
    manifests.append(DATA / 'subset_counts.csv')
    check = cpu_preflight()
    write_json(OUT / 'cpu_preflight_complete.json', check)
    sources = [ASSIGN, CONFIG, TRAINER, prior, OLD / 'environment.json',
               ROOT / 'data_processed/dfuc2021_cls_v1/similarity_grouped_splits/dfuc2021_cls_cv5_similarity_grouped_manifest.csv',
               ROOT / 'data_processed/cross_task_lesion_prior_v1/dfuc2021_train_prior_meta.csv']
    sources += list(EXP.glob('*.py')) + [EXP / 'PROTOCOL.md', EXP / 'FIGURE_CONTRACT.md', OUT / 'cpu_preflight_complete.json']
    sources += list((ROOT / 'experiments/cross_task_prior_transfer').glob('*.py'))
    sources += list((ROOT / 'experiments/dfuc2021_cls_baseline').glob('*.py'))
    sources += [ROOT / 'experiments/associate_editor_revision_v1/protocol.yaml']
    sources += [ROOT / 'outputs' / version / 'superseded_by_v3.json'
                for version in ('training_size_sensitivity_v1', 'training_size_sensitivity_v2')]
    write_json(OUT / 'sealed_inputs.json', dict(created_at=now(), status='sealed_before_v3_training',
               sources=reg(sources), assignments=reg(manifests), environment=env,
               pretrained_checkpoints=check['pretrained_checkpoints'],
               rendering_packages={name: importlib.metadata.version(name) for name in ('matplotlib', 'Pillow')},
               models=list(MODELS), seeds=list(SEEDS), nominal_sizes=list(SIZES), new_fits=400,
               reused_fits=0, previous_attempts_excluded=True))
    counts = pd.DataFrame(reports)
    write_json(OUT / 'preflight_complete.json', dict(status='complete', completed_at=now(),
               sealed_inputs_sha256=sha(OUT / 'sealed_inputs.json'), subset_rows=len(reports),
               budget_fold_trajectory_cells=100, new_fits=400, reused_fits=0,
               minimum_class_count_in_fit_stop=int(counts.loc[counts.role.isin(['fit','stop']),list(LABELS)].min().min())))
    print(counts.loc[counts.role.eq('refit')].groupby('nominal_percent')[['images','ischaemia']].agg(['min','max']).to_string(), flush=True)
    print('Unified fresh v3 sealed; all four endpoints require new fitting.', flush=True)

def tiny_groups(rows: pd.DataFrame, per_class: int) -> set:
    counts = np.zeros(4, int)
    chosen = set()
    for group, part in rows.groupby('similarity_group', sort=True):
        contribution = part.label.value_counts().reindex(range(4), fill_value=0).to_numpy()
        if np.any((counts < per_class) & (contribution > 0)):
            chosen.add(group)
            counts += contribution
        if np.all(counts >= per_class):
            return chosen
    raise AssertionError('Cannot construct smoke groups')


def make_smoke_assignment() -> Path:
    f = pd.read_csv(assignment(25, SEEDS[0]))
    f = f.loc[f.outer_fold.eq(0)].copy()
    for role, minimum in [('fit', 8), ('stop', 8), ('test', 4)]:
        mask = f.outer_role.eq('test') if role == 'test' else f.direct_role.eq(role)
        keep = tiny_groups(f.loc[mask], minimum)
        f.loc[mask & ~f.similarity_group.isin(keep), ['outer_role', 'direct_role']] = 'excluded'
    p = OUT / 'smoke/assignments.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(p, index=False)
    return p


def execute_job(size: int, seed: int, outer: int, model: str, stage: str, smoke: bool = False) -> None:
    seal = verify_seal()
    p = OUT / 'smoke/assignments.csv' if smoke else assignment(size, seed)
    frame = pd.read_csv(p)
    frame = frame.loc[frame.outer_fold.eq(outer)]
    d = jobdir(size, seed, outer, model, stage, smoke)
    contract = dict(seal_sha256=sha(OUT / 'sealed_inputs.json'), assignment_sha256=sha(p),
                    model=model, seed=seed, outer=outer, stage=stage, size=size, smoke=smoke)
    if (d / 'complete.json').exists():
        if read_json(d / 'job_contract.json') != contract:
            raise RuntimeError(f'Existing complete job contract mismatch: {d}')
        verify_job(d, frame, seed, outer, model, stage, p)
        expected_verification = dict(status='complete', complete_sha256=sha(d / 'complete.json'),
                                     contract_sha256=sha(d / 'job_contract.json'))
        if (d / 'verified.json').exists():
            previous = read_json(d / 'verified.json')
            if any(previous.get(k) != v for k, v in expected_verification.items()):
                raise RuntimeError(f'Existing verification mismatch: {d}')
        else:
            write_json(d / 'verified.json', dict(expected_verification, checked_at=now()))
        print(f'SKIP verified {d}', flush=True)
        return
    if d.exists():
        archived = OUT / 'incomplete_attempts' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%f') / d.relative_to(OUT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(archived))
        print(f'PRESERVED incomplete attempt: {archived}', flush=True)
    d.mkdir(parents=True, exist_ok=True)
    write_json(d / 'job_contract.json', contract)
    root = OUT / ('smoke/classification' if smoke else f'fraction_{size}/classification')
    command = [sys.executable, str(TRAINER), '--repo-root', str(ROOT), '--assignments', str(p),
               '--output-root', str(root), '--model', model, '--seed', str(seed), '--outer', str(outer),
               '--stage', stage, '--device', 'cuda']
    if smoke:
        command += ['--max-epochs', '1']
    print('RUN ' + ' '.join(command), flush=True)
    started = time.time()
    with (d / 'runner.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = verify_job(d, frame, seed, outer, model, stage, p)
    if smoke and result['best_epoch'] != 1:
        raise AssertionError('Smoke job exceeded one epoch')
    write_json(d / 'verified.json', dict(status='complete', checked_at=now(), seconds=time.time() - started,
               complete_sha256=sha(d / 'complete.json'), contract_sha256=sha(d / 'job_contract.json')))
    print(f'VERIFIED size={size} seed={seed} fold={outer} model={model} stage={stage}', flush=True)


def smoke() -> None:
    seal = verify_seal()
    assert_environment(seal)
    if (OUT / 'smoke_complete.json').exists():
        marker = read_json(OUT / 'smoke_complete.json')
        if marker['seal_sha256'] != sha(OUT / 'sealed_inputs.json'):
            raise RuntimeError('Smoke seal mismatch')
        verify_registry(marker['files'])
        print('Existing smoke verified', flush=True)
        return
    make_smoke_assignment()
    for model in MODELS:
        for stage in ('direct_select', 'direct_refit'):
            execute_job(25, SEEDS[0], 0, model, stage, smoke=True)
    paths = list((OUT / 'smoke').rglob('complete.json')) + [OUT / 'smoke/assignments.csv']
    write_json(OUT / 'smoke_complete.json', dict(status='complete', completed_at=now(),
               seal_sha256=sha(OUT / 'sealed_inputs.json'), fits=4, excluded_from_results=True, files=reg(paths)))


def status() -> dict:
    complete = list(OUT.glob('fraction_*/classification/seed_*/outer_*/*/*/verified.json'))
    progress = read_json(OUT / 'progress.json') if (OUT / 'progress.json').exists() else {}
    return dict(verified_fits=len(complete), expected_fits=400, progress=progress,
                summary_complete=(OUT / 'summary/summary_complete.json').exists(),
                analysis_complete=(OUT / 'analysis_complete.json').exists())


def finalize() -> None:
    verify_seal(full=True)
    if status()['verified_fits'] != 400:
        raise RuntimeError('All 400 fits must complete before final analysis')
    subprocess.run([sys.executable, str(EXP / 'summarize.py')], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(EXP / 'render_curves.py')], cwd=ROOT, check=True)
    write_json(OUT / 'analysis_complete.json', dict(status='complete', completed_at=now(),
               new_fits=400, reused_fits=0, seed_model_size_cells=40, images_per_cell=5955,
               summary_complete_sha256=sha(OUT / 'summary/summary_complete.json'),
               seal_sha256=sha(OUT / 'sealed_inputs.json'), manuscript_updated=False, visual_qa_complete=False,
               figure_exports_sha256=sha(OUT / 'figures/exports_complete.json'),
               boundary='Exploratory internal training-size sensitivity; no clinical or patient-level validation.'))


def run() -> None:
    seal = verify_seal()
    assert_environment(seal)
    marker = read_json(OUT / 'smoke_complete.json')
    if marker['seal_sha256'] != sha(OUT / 'sealed_inputs.json') or marker['fits'] != 4:
        raise RuntimeError('Smoke not valid for current seal')
    verify_registry(marker['files'])
    write_json(OUT / 'runner_process.json', dict(pid=os.getpid(), started_at=now(), command=sys.argv,
               environment=environment(), seal_sha256=sha(OUT / 'sealed_inputs.json')))
    done = 0
    try:
        for seed in SEEDS:
            for outer in range(5):
                for size in SIZES:
                    for model in MODELS:
                        for stage in ('direct_select', 'direct_refit'):
                            current = dict(seed=seed, outer=outer, size=size, model=model, stage=stage)
                            write_json(OUT / 'progress.json', dict(status='running', verified_fits=done,
                                       expected_fits=400, current=current, updated_at=now()))
                            execute_job(size, seed, outer, model, stage)
                            done += 1
        write_json(OUT / 'progress.json', dict(status='finalizing', verified_fits=done, expected_fits=400, updated_at=now()))
        finalize()
        write_json(OUT / 'progress.json', dict(status='complete', verified_fits=done, expected_fits=400, updated_at=now()))
    except Exception as exc:
        old = read_json(OUT / 'progress.json') if (OUT / 'progress.json').exists() else {}
        write_json(OUT / 'failure.json', dict(time=now(), error=repr(exc), progress=old))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified fresh four-budget training-size sensitivity')
    parser.add_argument('--mode', choices=['prepare', 'smoke', 'run', 'finalize', 'status'], required=True)
    args = parser.parse_args()
    if args.mode == 'status':
        import json
        print(json.dumps(status(), indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        dict(prepare=prepare, smoke=smoke, run=run, finalize=finalize)[args.mode]()


if __name__ == '__main__':
    main()
