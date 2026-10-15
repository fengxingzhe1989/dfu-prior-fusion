from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / 'experiments/training_size_sensitivity_v3'
OUT = ROOT / 'outputs/training_size_sensitivity_v3'
DATA = ROOT / 'data_processed/training_size_sensitivity_v3'
OLD = ROOT / 'outputs/associate_editor_revision_v1'
ASSIGN = ROOT / 'data_processed/associate_editor_revision_v1/nested_grouped_assignments.csv'
TRAINER = ROOT / 'experiments/associate_editor_revision_v1/train_nested_classification.py'
CONFIG = ROOT / 'experiments/cross_task_prior_transfer/configs/cross_task_lpt_resnet50_mobilenetv3_cv5_v1.yaml'
SEEDS = tuple(range(20260925, 20260930))
MODELS = ('lpt_full', 'lpt_neutral')
SIZES = (25, 50, 75, 100)
LABELS = ('none', 'infection', 'ischaemia', 'both')
EXPECTED_ASSIGN = 'f0d4590e47ba0e58b2d24236dbace32ea6cb62f45628d5470c2914744d166dbf'
EXPECTED_TRAINER = 'e7527b1acb7d41c53fe372c945a77dba23e7f9f21b8036491814535f18bb84bc'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def reg(paths) -> dict:
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def verify_registry(registry: dict) -> None:
    for name, digest in registry.items():
        p = ROOT / name
        if not p.is_file() or sha(p) != digest:
            raise RuntimeError(f'Sealed artifact mismatch: {p}')


def stream_seed(*parts) -> int:
    key = json.dumps([20261012, *parts], separators=(',', ':'))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big')


def subset(frame: pd.DataFrame, seed: int, outer: int, size: int) -> pd.DataFrame:
    """First select a complete budget, then partition only that budget."""
    from sklearn.model_selection import StratifiedGroupKFold
    if size not in SIZES:
        raise ValueError(size)
    result = frame.copy()
    rows = frame.loc[frame.outer_role.eq('train')]
    counts = rows.groupby(['similarity_group', 'label']).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(4), fill_value=0)
    strata = {}
    for group, values in counts.iterrows():
        strata.setdefault(tuple(int(x) for x in values), []).append(group)
    retained = set()
    for signature, groups in sorted(strata.items()):
        rng = np.random.default_rng(stream_seed(seed, outer, 'budget', signature))
        order = rng.permutation(sorted(groups))
        retained.update(order[:(len(order) * size + 99) // 100].tolist())
    train = frame.outer_role.eq('train')
    result.loc[train, ['outer_role', 'direct_role']] = 'excluded'
    selected = train & frame.similarity_group.isin(retained)
    result.loc[selected, 'outer_role'] = 'train'
    result.loc[selected, 'direct_role'] = 'fit'
    candidates = result.loc[selected]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=stream_seed(seed, outer, size, 'stopping'))
    _, stop_indices = next(splitter.split(candidates, candidates.label,
                                          groups=candidates.similarity_group))
    result.loc[candidates.index[stop_indices], 'direct_role'] = 'stop'
    return result

def validate_assignment(frame: pd.DataFrame, original: pd.DataFrame, minimum: int = 1) -> None:
    if frame.image.duplicated().any() or frame.image.tolist() != original.image.tolist():
        raise AssertionError('Image identity/order changed')
    for role in ('fit', 'stop'):
        rows = frame.loc[frame.direct_role.eq(role)]
        if rows.groupby('label').size().reindex(range(4), fill_value=0).min() < minimum:
            raise AssertionError(f'Insufficient samples in {role}')
    if not frame.loc[frame.outer_role.eq('test')].equals(original.loc[original.outer_role.eq('test')]):
        raise AssertionError('Outer test rows changed')
    if set(frame.loc[frame.outer_role.eq('train'), 'image']) != set(frame.loc[frame.direct_role.isin(['fit', 'stop']), 'image']):
        raise AssertionError('Selection/refit populations disagree')
    roles = frame.groupby('similarity_group').agg(outer=('outer_role', 'nunique'), direct=('direct_role', 'nunique'))
    if (roles > 1).any().any():
        raise AssertionError('A similarity component was split across roles')
    for column in ('image', 'label', 'label_name', 'similarity_group', 'fold'):
        if not frame[column].equals(original[column]):
            raise AssertionError(f'Identity column changed: {column}')


def predictions(path: Path, test: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_csv(path)
    if p.image.duplicated().any() or set(p.image) != set(test.image):
        raise AssertionError(f'Prediction coverage mismatch: {path}')
    p = p.set_index('image').loc[test.image].reset_index()
    values = p[[f'prob_{x}' for x in LABELS]].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise AssertionError(f'Invalid probabilities: {path}')
    if not np.allclose(values.sum(1), 1, atol=0.002):
        raise AssertionError(f'Probabilities not normalised: {path}')
    if not np.array_equal(p.true_label.to_numpy(), test.label_name.to_numpy()):
        raise AssertionError(f'Label mismatch: {path}')
    if not np.array_equal(p.pred_label.to_numpy(), np.array(LABELS)[values.argmax(1)]):
        raise AssertionError(f'Argmax mismatch: {path}')
    return p


def jobdir(size: int, seed: int, outer: int, model: str, stage: str, smoke: bool = False) -> Path:
    base = OUT / ('smoke/classification' if smoke else f'fraction_{size}/classification')
    return base / f'seed_{seed}' / f'outer_{outer}' / model / stage


def assignment(size: int, seed: int) -> Path:
    return DATA / f'fraction_{size}_seed_{seed}.csv'


def verify_job(directory: Path, frame: pd.DataFrame, seed: int, outer: int, model: str, stage: str, assignment_path: Path) -> dict:
    p = read_json(directory / 'complete.json')
    for key, val in dict(status='complete', base_seed=seed, outer_fold=outer, model=model, stage=stage,
                         assignment_manifest_sha256=sha(assignment_path), source_config_sha256=sha(CONFIG),
                         neutral_prior_contract=(model == 'lpt_neutral')).items():
        if p.get(key) != val:
            raise AssertionError(f'{directory}: mismatch in {key}')
    code = 3 if stage == 'direct_select' else 4
    expected_seed = int(np.random.SeedSequence([seed, outer, 4, code]).generate_state(1, dtype=np.uint32)[0])
    if p.get('job_seed') != expected_seed or not 1 <= p['best_epoch'] <= 60:
        raise AssertionError(f'Invalid epoch/seed: {directory}')
    fit = frame.direct_role.eq('fit') if stage == 'direct_select' else frame.outer_role.eq('train')
    stop = frame.direct_role.eq('stop') if stage == 'direct_select' else np.zeros(len(frame), bool)
    test = frame.outer_role.eq('test') if stage == 'direct_refit' else np.zeros(len(frame), bool)
    if (p['train_samples'], p['stop_samples'], p['prediction_samples']) != (int(fit.sum()), int(stop.sum()), int(test.sum())):
        raise AssertionError(f'Wrong sample counts: {directory}')
    if stage == 'direct_refit':
        if sha(directory / 'model.pt') != p['checkpoint_sha256'] or sha(directory / 'predictions.csv') != p['prediction_sha256']:
            raise AssertionError(f'Output hash mismatch: {directory}')
        predictions(directory / 'predictions.csv', frame.loc[test])
        selected = read_json(directory.parent / 'direct_select/complete.json')
        if p['best_epoch'] != selected['best_epoch']:
            raise AssertionError(f'Refit duration mismatch: {directory}')
    return p
