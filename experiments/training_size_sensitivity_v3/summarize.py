#!/usr/bin/env python3
"""Summarise only the complete frozen sample-size matrix; no partial analysis."""
from __future__ import annotations

import datetime as dt
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ASSIGN, DATA, LABELS, MODELS, OUT, ROOT, SEEDS, SIZES,
                    assignment, jobdir, predictions, read_json, reg, sha,
                    verify_job, verify_registry, write_json)
from run import verify_seal

METRICS = ('macro_f1', 'ischaemia_precision', 'ischaemia_recall', 'ischaemia_f1')


def metric_array(matrix: np.ndarray) -> np.ndarray:
    tp = np.diagonal(matrix, axis1=-2, axis2=-1).astype(float)
    pred = matrix.sum(axis=-2)
    actual = matrix.sum(axis=-1)
    precision = np.divide(tp, pred, out=np.full_like(tp, np.nan), where=pred > 0)
    recall = np.divide(tp, actual, out=np.full_like(tp, np.nan), where=actual > 0)
    f1 = np.divide(2 * tp, pred + actual, out=np.zeros_like(tp), where=(pred + actual) > 0)
    return np.stack([f1.mean(axis=-1), precision[..., 2], recall[..., 2], f1[..., 2]], axis=-1)


def confusion(y: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.bincount(y * 4 + predicted, minlength=16).reshape(4, 4)


def metric_rows(matrix: np.ndarray, identity: dict) -> tuple[dict, list]:
    vals = metric_array(matrix)
    row = dict(identity, n=int(matrix.sum()), **dict(zip(METRICS, vals.tolist())))
    row['accuracy'] = float(np.trace(matrix) / matrix.sum())
    classes = []
    for i, label in enumerate(LABELS):
        tp = int(matrix[i, i])
        fp, fn = int(matrix[:, i].sum()) - tp, int(matrix[i].sum()) - tp
        classes.append(dict(identity, label=label, support=tp+fn, tp=tp, fp=fp, fn=fn,
                            precision=tp/(tp+fp) if tp+fp else np.nan,
                            recall=tp/(tp+fn) if tp+fn else np.nan,
                            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0))
    return row, classes


def crossed_bootstrap(groups: np.ndarray, repetitions: int = 5000, seed: int = 20261013) -> np.ndarray:
    # groups: [size/model cell, seed, similarity group, true label, predicted label]
    cells, seeds, n_groups, _, _ = groups.shape
    flat = groups.reshape(cells, seeds, n_groups, 16).astype(np.float64)
    rng = np.random.default_rng(seed)
    result = np.empty((repetitions, cells, len(METRICS)), dtype=float)
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        weights = np.empty((count, n_groups), float)
        for i in range(count):
            weights[i] = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        matrices = np.einsum('bg,csgk->bcsk', weights, flat, optimize=True).reshape(count, cells, seeds, 4, 4)
        scores = metric_array(matrices)
        sampled = rng.integers(seeds, size=(count, seeds))
        selected = scores[np.arange(count)[:, None, None], np.arange(cells)[None, :, None], sampled[:, None, :], :]
        # An undefined sampled-seed precision makes this replicate undefined.
        result[start:start+count] = selected.mean(axis=2)
    return result


def intervals(point: np.ndarray, draws: np.ndarray, identity: dict) -> list:
    rows = []
    for i, metric in enumerate(METRICS):
        values = draws[:, i]
        finite = values[np.isfinite(values)]
        low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
        rows.append(dict(identity, metric=metric, estimate=float(point[i]), ci_low=low, ci_high=high,
                         estimable_replicates=len(finite), undefined_replicates=len(values)-len(finite),
                         interval_type='exploratory pointwise paired crossed seed x similarity-group percentile'))
    return rows


def main() -> None:
    verify_seal(full=True)
    target = OUT / 'summary'
    if (target / 'summary_complete.json').exists():
        complete = read_json(target / 'summary_complete.json')
        if complete['seal_sha256'] != sha(OUT / 'sealed_inputs.json'):
            raise RuntimeError('Completed summary belongs to a different seal')
        verify_registry(complete['inputs'])
        verify_registry(complete['outputs'])
        print('Existing summary verified; no overwrite', flush=True)
        return
    if target.exists():
        raise RuntimeError('Incomplete summary directory requires explicit diagnosis; refusing overwrite')
    original = pd.read_csv(ASSIGN)
    reference = original.loc[original.outer_fold.eq(0)].sort_values('image')
    group_names = sorted(reference.similarity_group.unique())
    group_ids = reference.similarity_group.map({g: i for i, g in enumerate(group_names)}).to_numpy(int)
    y = reference.label.to_numpy(int)
    cells = [(size, model) for size in SIZES for model in MODELS]
    group_confusions = np.zeros((len(cells), len(SEEDS), len(group_names), 4, 4), dtype=np.int32)
    fold_rows, fold_classes, seed_rows, seed_classes, inputs = [], [], [], [], []
    for ci, (size, model) in enumerate(cells):
        for si, seed in enumerate(SEEDS):
            path = assignment(size, seed)
            assignments = pd.read_csv(path)
            parts = []
            for outer in range(5):
                f = assignments.loc[assignments.outer_fold.eq(outer)]
                for stage in ('direct_select', 'direct_refit'):
                    d = jobdir(size, seed, outer, model, stage)
                    verify_job(d, f, seed, outer, model, stage, path)
                    inputs += [d / 'complete.json']
                    if size in SIZES:
                        contract = read_json(d / 'job_contract.json')
                        if contract['seal_sha256'] != sha(OUT / 'sealed_inputs.json') or contract['smoke']:
                            raise RuntimeError('Production job contract mismatch')
                        verified = read_json(d / 'verified.json')
                        if verified['complete_sha256'] != sha(d / 'complete.json'):
                            raise RuntimeError('Verified marker no longer matches completion')
                        inputs += [d / 'job_contract.json', d / 'verified.json']
                test = f.loc[f.outer_role.eq('test')]
                p = predictions(d / 'predictions.csv', test)
                inputs += [d / 'predictions.csv']
                parts.append(p)
                true = p.true_label.map(dict(zip(LABELS, range(4)))).to_numpy(int)
                predicted = p.pred_label.map(dict(zip(LABELS, range(4)))).to_numpy(int)
                row, class_rows = metric_rows(confusion(true, predicted), dict(nominal_percent=size, model=model, seed=seed, outer=outer))
                fold_rows.append(row)
                fold_classes.extend(class_rows)
            p = pd.concat(parts).set_index('image').loc[reference.image]
            if len(p) != 5955 or p.index.nunique() != 5955:
                raise AssertionError('Incorrect pooled coverage')
            predicted = p.pred_label.map(dict(zip(LABELS, range(4)))).to_numpy(int)
            np.add.at(group_confusions[ci, si], (group_ids, y, predicted), 1)
            row, class_rows = metric_rows(confusion(y, predicted), dict(nominal_percent=size, model=model, seed=seed))
            seed_rows.append(row)
            seed_classes.extend(class_rows)
    if len(seed_rows) != 40 or len(fold_rows) != 200:
        raise AssertionError('Incomplete factorial coverage')
    seed_array = metric_array(group_confusions.sum(axis=2))
    point = seed_array.mean(axis=1)
    sd = seed_array.std(axis=1, ddof=1)
    print('All 400 fresh v3 fits verified; running 5000 paired bootstrap draws.', flush=True)
    draws = crossed_bootstrap(group_confusions)
    cell_rows, contrast_rows, draw_rows = [], [], []
    for ci, (size, model) in enumerate(cells):
        rows = intervals(point[ci], draws[:, ci], dict(nominal_percent=size, model=model))
        for mi, row in enumerate(rows):
            row['sample_sd_across_five_trajectories'] = float(sd[ci, mi])
            row['estimable_trajectories'] = int(np.isfinite(seed_array[ci, :, mi]).sum())
        cell_rows += rows
        for b in range(len(draws)):
            draw_rows.append(dict(replicate=b, nominal_percent=size, model=model, **dict(zip(METRICS, draws[b, ci].tolist()))))
    contrasts = []
    for model in MODELS:
        for higher, lower in ((50, 25), (75, 50), (100, 75), (100, 50), (100, 25)):
            contrasts.append((f'{model}_{higher}_minus_{lower}', cells.index((higher, model)), cells.index((lower, model))))
    for size in SIZES:
        contrasts.append((f'full_minus_neutral_{size}', cells.index((size, 'lpt_full')), cells.index((size, 'lpt_neutral'))))
    for name, a, b in contrasts:
        contrast_rows += intervals(point[a] - point[b], draws[:, a] - draws[:, b], dict(contrast=name))
    temporary = Path(tempfile.mkdtemp(prefix='summary_partial_', dir=OUT))
    try:
        tables = dict(fold_seed_metrics=fold_rows, fold_seed_class_metrics=fold_classes,
                      seed_metrics=seed_rows, seed_class_metrics=seed_classes,
                      cell_summary=cell_rows, paired_contrasts=contrast_rows, bootstrap_replicates=draw_rows)
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(temporary / f'{name}.csv', index=False, na_rep='NA', float_format='%.12g')
        shutil.copy2(DATA / 'subset_counts.csv', temporary / 'subset_counts.csv')
        (temporary / 'README.md').write_text(
            '# Complete training-size sensitivity\n\n'
            'All 5 seeds, 5 fixed outer folds, both models and all 4 budgets are included. '
            'All four budgets were freshly fitted under the budget-first internal-partition procedure; no endpoint was reused. '
            'The full 5955-image OOF evaluation set is unchanged. Smoke fits are excluded.\n\n'
            'This later exploratory analysis describes sensitivity to the available labelled-training '
            'budget under the frozen procedure. It is not external, patient-level or clinical validation, '
            'and does not establish that 227 ischaemia-only images suffice. Pointwise intervals condition '
            'on the observed groups and fitted models/subsets. No confirmatory p values were added. '
            'Five-trajectory variation combines subset, internal-partition and optimisation realisations. Actual sample counts '
            'are in subset_counts.csv. Undefined precision/interval counts are reported explicitly.\n')
        # Move only once complete; a failed partial remains preserved under its own directory.
        temporary.rename(target)
        write_json(target / 'summary_complete.json', dict(status='complete', completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   seal_sha256=sha(OUT / 'sealed_inputs.json'), seeds=list(SEEDS), models=list(MODELS),
                   nominal_sizes=list(SIZES), images_per_seed_model_size=5955, groups=5670,
                   seed_cells=40, fold_cells=200, bootstrap_replicates=5000,
                   inputs=reg(inputs), outputs=reg(list(target.iterdir()))))
    except Exception:
        print(f'Preserved incomplete summary at {temporary}', flush=True)
        raise
    print(f'ANALYSIS COMPLETE {target}', flush=True)


if __name__ == '__main__':
    main()
