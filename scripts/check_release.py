"""Read-only, data-free integrity checks for the publication code/aggregate release."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
AE = ROOT / 'experiments/associate_editor_revision_v1'
ASSETS = ROOT / 'paper_assets/associate_editor_revision_v1'
SEEDS = set(range(20260925, 20260930))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, message: str) -> None:
    require(math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=2e-9), message)


def main() -> None:
    server_hashes = json.loads((ROOT / 'paper_assets/release_audit/server_source_hashes_20261009.json').read_text())
    for relative, expected in server_hashes.items():
        require(sha256(ROOT / relative) == expected, f'Server/source snapshot mismatch: {relative}')
    registry = ROOT / 'paper_assets/release_audit/frozen_asset_registry.csv'
    with registry.open() as handle:
        rows = list(csv.DictReader(handle))
    require(len({r['path'] for r in rows}) == len(rows), 'Duplicate frozen-asset paths')
    for row in rows:
        path = ROOT / row['path']
        require(path.is_file(), f'Missing frozen asset: {path}')
        require(sha256(path) == row['sha256'], f'Frozen asset hash mismatch: {path}')

    protocol = yaml.safe_load((AE / 'protocol.yaml').read_text())
    require(set(protocol['training_seeds']) == SEEDS, 'Five-seed protocol mismatch')
    for rel in protocol['source_configs'].values():
        require((ROOT / rel).is_file(), f'Missing source configuration: {rel}')
    sources = [*ROOT.glob('experiments/**/*.py'), *ROOT.glob('data_processing/*.py'), *ROOT.glob('paper_assets/analysis_scripts/*.py')]
    for path in sources:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if value.startswith(('experiments/', 'data_processing/')) and value.endswith(('.py', '.yaml')):
                    require((ROOT / value).is_file(), f'Broken source/config reference in {path}: {value}')
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('experiments.'):
                parts = node.module.split('.')
                local = ROOT.joinpath(*parts)
                require(local.is_dir() or local.with_suffix('.py').is_file(), f'Missing local import: {node.module}')

    spec = importlib.util.spec_from_file_location('release_processing_steps', ROOT / 'data_processing/run_step.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for step in module.STEPS.values():
        require((ROOT / step.script).is_file(), f'Missing preprocessing entry point: {step.script}')

    manifest_path = ROOT / 'paper_assets/split_metadata/dfuc2021_cls_cv5_similarity_grouped_manifest.csv'
    require(sha256(manifest_path) == protocol['data']['server_manifest_sha256'], 'Grouped manifest hash mismatch')
    manifest = pd.read_csv(manifest_path)
    require(len(manifest) == manifest.image.nunique() == 5955, 'Image coverage mismatch')
    require(manifest.similarity_group.nunique() == 5670, 'Group count mismatch')
    require(manifest.groupby('similarity_group').fold.nunique().max() == 1, 'A similarity group crosses folds')

    summary = ASSETS / 'summary'
    seed = pd.read_csv(summary / 'classification_seed_metrics.csv')
    fold = pd.read_csv(summary / 'classification_fold_seed_metrics.csv')
    classes = pd.read_csv(summary / 'classification_seed_class_metrics.csv')
    fold_classes = pd.read_csv(summary / 'classification_fold_seed_class_metrics.csv')
    means = pd.read_csv(summary / 'classification_model_summary.csv').set_index('model')
    require(len(seed) == 70 and len(fold) == 350 and len(classes) == 280 and len(fold_classes) == 1400, 'Classification table coverage mismatch')
    require(not seed.duplicated(['seed', 'model']).any(), 'Duplicate seed/model results')
    require(not fold.duplicated(['seed', 'outer_fold', 'model']).any(), 'Duplicate fold results')
    for model, group in seed.groupby('model'):
        require(set(group.seed) == SEEDS and group.n.eq(5955).all(), f'Incomplete seeds/images: {model}')
        require(int(means.loc[model, 'seeds']) == 5, f'Seed summary count mismatch: {model}')
        for metric in ['macro_f1', 'accuracy']:
            close(group[metric].mean(), means.loc[model, metric + '_mean'], f'Mean mismatch: {model}/{metric}')
            close(group[metric].std(ddof=1), means.loc[model, metric + '_sd'], f'SD mismatch: {model}/{metric}')
    require(fold.groupby(['model', 'seed']).n.sum().eq(5955).all(), 'Fold image coverage mismatch')
    require(classes.groupby(['model', 'seed']).support.sum().eq(5955).all(), 'Class support mismatch')

    contrasts = pd.read_csv(summary / 'primary_contrast_bootstrap.csv')
    expected = {'full_minus_neutral', 'ra_minus_output_only', 'rgb_prior_minus_rgb'}
    require(len(contrasts) == 3 and set(contrasts.contrast) == expected, 'Primary contrast family mismatch')
    replicates = pd.read_csv(summary / 'bootstrap_replicates.csv')
    require(len(replicates) == 5000, 'Bootstrap replicate count mismatch')
    pivot = seed.pivot(index='seed', columns='model', values='macro_f1')
    for row in contrasts.itertuples():
        differences = pivot[row.model_a] - pivot[row.model_b]
        close(differences.mean(), row.estimate_mean_seed_difference, f'Contrast mean mismatch: {row.contrast}')
        close(differences.std(ddof=1), row.sd_seed_difference, f'Contrast SD mismatch: {row.contrast}')
        distribution = replicates['crossed_' + row.contrast]
        require(distribution.notna().all(), 'Nonfinite classification bootstrap')
        close(distribution.quantile(0.025), row.ci_low, 'CI lower bound mismatch')
        close(distribution.quantile(0.975), row.ci_high, 'CI upper bound mismatch')
        p_value = min(1.0, 2 * min(((distribution <= 0).sum() + 1) / 5001, ((distribution >= 0).sum() + 1) / 5001))
        close(p_value, row.approximate_two_sided_p, 'Approximate bootstrap p mismatch')
    ordered = contrasts.sort_values('approximate_two_sided_p')
    adjusted = np.minimum(1, np.maximum.accumulate(ordered.approximate_two_sided_p.to_numpy() * np.array([3, 2, 1])))
    require(np.allclose(adjusted, ordered.holm_adjusted_p, rtol=1e-7, atol=2e-9), 'Holm-family mismatch')

    seg = pd.read_csv(summary / 'segmentation_seed_metrics.csv')
    seg_fold = pd.read_csv(summary / 'segmentation_fold_seed_metrics.csv')
    seg_mean = pd.read_csv(summary / 'segmentation_grid_summary.csv').set_index('grid')
    require(len(seg) == 10 and len(seg_fold) == 50, 'Segmentation coverage mismatch')
    for grid, group in seg.groupby('grid'):
        require(set(group.seed) == SEEDS and group.n.eq(2000).all(), f'Segmentation seed/cohort mismatch: {grid}')
        for metric in ['dice', 'iou', 'precision', 'recall']:
            close(group[metric].mean(), seg_mean.loc[grid, metric + '_mean'], 'Segmentation mean mismatch')
            close(group[metric].std(ddof=1), seg_mean.loc[grid, metric + '_sd'], 'Segmentation SD mismatch')

    interpret = ASSETS / 'interpretability'
    coverage = pd.read_csv(interpret / 'summary/coverage_audit.csv')
    all_scope = coverage[coverage.scope.eq('all')]
    require(len(all_scope) == 10 and not all_scope.duplicated(['model', 'seed']).any(), 'Interpretation coverage mismatch')
    require(set(all_scope.model) == {'resnet50', 'lpt_full'}, 'Interpretation models mismatch')
    for _, group in all_scope.groupby('model'):
        require(set(group.seed) == SEEDS and group.n_images.eq(5955).all(), 'Interpretation seeds/images mismatch')
    require(all_scope.n_images.sum() == 59550, 'Interpretation total mismatch')
    cases = pd.read_csv(interpret / 'qualitative_cases.csv')
    require(len(cases) == cases.image.nunique() == cases.similarity_group.nunique() == 20, 'Fixed-case uniqueness mismatch')
    require(cases.groupby('label').size().eq(5).all() and cases.label.nunique() == 4, 'Fixed-case class count mismatch')
    require(cases.image.isin(manifest.image).all(), 'Case outside frozen cohort')

    print(json.dumps({'status': 'passed', 'frozen_assets': len(rows), 'python_sources': len(sources), 'classification_models': len(means), 'training_seeds': sorted(SEEDS), 'primary_contrasts': sorted(expected), 'interpretability_rows': 59550, 'fixed_cases': 20, 'training_rerun': False}, indent=2))


if __name__ == '__main__':
    main()
