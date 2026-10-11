"""Audit all 75 added jobs, summarize five models and export fixed-case previews."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from experiments.interpretability_rgb_extension_v1 import run as ext
from experiments.associate_editor_revision_v1 import summarize_interpretability as s
from experiments.associate_editor_revision_v1 import render_interpretability as renderer

ALL_MODELS = ('resnet50', 'efficientnetb0', 'convnext_tiny', 'efficientnetv2s', 'lpt_full')


def audit(repo, source, out):
    ext.verify(repo, source, out, full=True)
    original_models = s.MODELS
    try:
        s.MODELS = ext.MODELS
        new, ref, inputs, new_audit = s.load_and_audit(repo, out)
        s.MODELS = ('resnet50', 'lpt_full')
        old, old_ref, old_inputs, old_audit = s.load_and_audit(repo, source)
    finally:
        s.MODELS = original_models
    pd.testing.assert_frame_equal(ref, old_ref)
    assert (out/'interpretability/qualitative_cases.csv').read_bytes() == (source/'interpretability/qualitative_cases.csv').read_bytes()
    assert (out/'interpretability/placements.json').read_bytes() == (source/'interpretability/placements.json').read_bytes()
    data = pd.concat([old, new], ignore_index=True)
    assert len(new) == 89325 and len(data) == 148875
    for column in ['label', 'similarity_group', 'outer_fold', 'mask_status', 'n_placements']:
        assert data.groupby('image')[column].nunique(dropna=False).eq(1).all()
    return data, ref, pd.concat([old_inputs, inputs], ignore_index=True), dict(original=old_audit, extension=new_audit)


def bootstrap(data, reference):
    order = reference.image.tolist(); labels = reference.label.to_numpy()
    groups = sorted(reference.similarity_group.unique()); mapping = {g: i for i, g in enumerate(groups)}
    codes = reference.similarity_group.map(mapping).to_numpy()
    values, definitions, rows = [], [], []
    for metric in s.CONTRAST_METRICS:
        arrays = {model: np.stack([data.loc[data.model.eq(model) & data.seed.eq(seed)].set_index('image').loc[order, metric].to_numpy(float)
                                   for seed in ext.SEEDS]) for model in ALL_MODELS}
        arrays.update({f'lpt_full_minus_{model}': arrays['lpt_full']-arrays[model] for model in ALL_MODELS if model != 'lpt_full'})
        for entity, array in arrays.items():
            for scope, label in [('all', None), *[(name, i) for i, name in enumerate(s.CLASS_NAMES)]]:
                scoped = array.copy()
                if label is not None:
                    scoped[:, labels != label] = np.nan
                definition = dict(series_id=f'{entity}__{metric}__{scope}', entity=entity, metric=metric, scope=scope,
                                  endpoint_role='additional_exploratory_descriptive')
                definitions.append(definition); values.append(scoped)
                for index, seed in enumerate(ext.SEEDS):
                    valid = np.isfinite(scoped[index])
                    rows.append({**definition, 'seed': seed, 'image_mean': scoped[index, valid].mean() if valid.any() else np.nan,
                                 'n_valid_images': int(valid.sum()), 'n_valid_groups': len(np.unique(codes[valid]))})
    samples, design = s.crossed_group_bootstrap(np.stack(values), codes, len(groups), 5000)
    observed = pd.DataFrame(rows); intervals = []
    for i, definition in enumerate(definitions):
        seed_rows = observed.loc[observed.series_id.eq(definition['series_id'])]
        valid = samples[:, i][np.isfinite(samples[:, i])]
        low, high = np.quantile(valid, [.025, .975]) if len(valid) else [np.nan, np.nan]
        intervals.append({**definition, 'mean_across_seed_image_means': seed_rows.image_mean.mean(),
                          'sample_sd_across_seed_image_means': seed_rows.image_mean.std(ddof=1),
                          'n_seeds_estimable': seed_rows.image_mean.notna().sum(),
                          'n_valid_images_min_per_seed': seed_rows.n_valid_images.min(),
                          'n_valid_images_max_per_seed': seed_rows.n_valid_images.max(),
                          'ci95_low': low, 'ci95_high': high, 'n_replicates_estimable': len(valid),
                          'n_replicates_undefined': 5000-len(valid)})
    return observed, pd.DataFrame(intervals), pd.DataFrame(samples, columns=[d['series_id'] for d in definitions]), design


def valid_outputs(folder, marker):
    complete = json.loads((folder/marker).read_text())
    assert complete['status'] == 'complete'
    for name, digest in complete['assets'].items():
        assert ext.sha(folder/name) == digest, str(folder/name)
    return complete


def summarize(repo, source, out):
    data, ref, inputs, audit_record = audit(repo, source, out)
    folder = out/'interpretability/summary'
    if (folder/'summary_complete.json').exists():
        marker = valid_outputs(folder, 'summary_complete.json')
        assert marker['extension_registry_sha256'] == ext.sha(out/'interpretability/source_registry.json')
        return
    assert not folder.exists(), 'Keep incomplete directory for diagnosis; never overwrite'
    seed_metrics, model_summary, coverage = s.summarize_descriptives(data)
    paired, intervals, draws, design = bootstrap(data, ref)
    original_design = json.loads((source/'interpretability/summary/bootstrap_design.json').read_text())
    assert design['draw_sequence_sha256'] == original_design['draw_sequence_sha256']
    tmp = Path(tempfile.mkdtemp(prefix='.summary_pending_', dir=folder.parent))
    for name, frame in [('seed_metrics', seed_metrics), ('model_summary', model_summary), ('coverage_audit', coverage),
                        ('paired_seed_metrics', paired), ('bootstrap_intervals', intervals), ('bootstrap_replicates', draws), ('input_registry', inputs)]:
        s.csv_write(frame, tmp/(name+'.csv'))
    s.write_json(tmp/'bootstrap_design.json', design)
    (tmp/'REPORTING_BOUNDARIES.md').write_text(
        '# Additional exploratory RGB interpretability analysis\n\n'
        'Three RGB models were added after the original two-model analysis. All five training seeds and all 5,955 images were retained. '
        'The original 20 prediction-blind cases and every control placement were reused. No training or favourable selection occurred.\n\n'
        'All five-model summaries use per-seed image means, sample SD, metric-specific denominators and 5,000 crossed seed x similarity-group bootstrap draws. '
        'Intervals are descriptive pointwise intervals, not confirmatory tests, not multiplicity-adjusted and not patient-level validation. '
        'Full LPT-minus-RGB contrasts compare complete systems; they do not isolate a causal prior effect. '
        'Archived predicted-class targets may differ across models; common true-label-target results are reported separately.\n\n'
        'CAM target stages differ across architectures. Per-image normalized CAM appearance/overlap is not a common absolute attribution scale, '
        'anatomical accuracy, external clinical validity, or an explanation of the RA-LPT stack. '
        'FP32 changes and classification errors are retained. Zero/invalid CAMs are unavailable.\n')
    s.write_json(tmp/'summary_complete.json', dict(status='complete', models=list(ALL_MODELS), added_models=list(ext.MODELS),
                 seeds=list(ext.SEEDS), images_per_model_seed=5955, n_image_seed_model_rows=len(data), audited_inputs=audit_record,
                 extension_registry_sha256=ext.sha(out/'interpretability/source_registry.json'), bootstrap=design,
                 assets={p.name: ext.sha(p) for p in tmp.iterdir()}, p_values_reported=False))
    tmp.rename(folder)


def render(source, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    folder = out/'interpretability/figures'
    if (folder/'figures_complete.json').exists():
        valid_outputs(folder, 'figures_complete.json'); return
    assert not folder.exists(), 'Never overwrite partial figure exports'
    cases = renderer.read_cases(out/'interpretability/qualitative_cases.csv')
    arrays, provenance = {}, []
    for seed in ext.SEEDS:
        for model in ALL_MODELS:
            root = out if model in ext.MODELS else source
            for case in cases:
                p = root/'interpretability/jobs'/f'seed_{seed}'/f'outer_{case["fold"]}'/model/'cases'/f'{Path(case["image"]).stem}.npz'
                arrays[seed, model, case['image']] = renderer.read_archive(p, case, seed)
                provenance.append(dict(seed=seed, model=model, image=case['image'], path=str(p), sha256=ext.sha(p)))
    for seed in ext.SEEDS:
        for case in cases:
            for model in ALL_MODELS[1:]:
                for key in ['rgb', 'prior']:
                    assert np.array_equal(arrays[seed, model, case['image']][key], arrays[seed, 'resnet50', case['image']][key])
    renderer.configure_style()
    tmp = Path(tempfile.mkdtemp(prefix='.figures_pending_', dir=folder.parent))
    for seed in ext.SEEDS:
        with PdfPages(tmp/f'five_model_cases_seed_{seed}.pdf') as pdf:
            for label in range(4):
                chosen = [c for c in cases if c['label'] == label]
                fig = plt.figure(figsize=(183/25.4, 210/25.4), facecolor='white')
                fig.text(.5, .98, f'{renderer.LABELS[label]} | five fixed cases | seed {seed}', ha='center', size=9, weight='bold')
                fig.text(.5, .955, 'Exploratory five-model comparison; identical cases and archived predicted-class targets', ha='center', size=6.5)
                centres = np.linspace(.075, .925, 7)
                titles = ['RGB input', 'Soft prior', 'ResNet50', 'EfficientNet-B0', 'ConvNeXt-Tiny', 'EfficientNetV2-S', 'Full LPT']
                for x, title in zip(centres, titles):
                    fig.text(x, .923, title, ha='center', size=5.8, weight='bold')
                for index, case in enumerate(chosen):
                    top = .901-index*.161
                    for col, x in enumerate(centres):
                        data = arrays[seed, ALL_MODELS[max(0, col-2)], case['image']]
                        ax = fig.add_axes([x-.062, top-.108, .124, .108]); ax.axis('off')
                        if col == 1:
                            ax.imshow(data['prior'], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                        else:
                            ax.imshow(data['rgb'], interpolation='nearest')
                            if col >= 2 and data['cam_valid']:
                                ax.imshow(data['cam'], cmap='magma', vmin=0, vmax=1, alpha=.40, interpolation='nearest')
                        if col == 0:
                            text = f'{Path(case["image"]).stem}\nfold {case["fold"]}'
                        elif col == 1:
                            text = 'Generated\nnot reference'
                        else:
                            pred = int(data['pred_label']); wrong = pred != label
                            state = 'ERROR' if wrong else 'correct'
                            if not data['cam_valid']:
                                state += '; CAM NA'
                            if pred != int(data['fp32_pred_label']):
                                state += '; FP32 differs'
                            text = f'{renderer.LABELS[pred]}\n{state}'
                        fig.text(x, top-.113, text, ha='center', va='top', size=5.6,
                                 color='#9a381c' if col >= 2 and int(data['pred_label']) != label else '#263746')
                for x, cmap, title in [(.11, 'gray', 'Prior'), (.61, 'magma', 'CAM')]:
                    ax = fig.add_axes([x, .055, .25, .01]); ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=cmap, vmin=0, vmax=1, aspect='auto'); ax.axis('off')
                    fig.text(x-.03, .055, '0', size=6); fig.text(x+.26, .055, '1', size=6)
                    fig.text(x+.125, .076, title, ha='center', size=6)
                fig.text(.5, .029, 'Per-image normalized CAM; opacity 0.40; model targets can differ; no anatomical or clinical validation.', ha='center', size=6)
                pdf.savefig(fig)
                stem = tmp/f'five_model_seed_{seed}_class_{label}'
                fig.savefig(stem.with_suffix('.png'), dpi=300)
                if seed == ext.SEEDS[0]:
                    fig.savefig(stem.with_suffix('.svg'))
                    fig.savefig(stem.with_suffix('.pdf'))
                plt.close(fig)
    pd.DataFrame(provenance).to_csv(tmp/'source_arrays.csv', index=False)
    s.write_json(tmp/'figures_complete.json', dict(status='complete', models=list(ALL_MODELS), seeds=list(ext.SEEDS),
                 case_count=20, source_array_count=len(provenance), plate_count=20, visual_review='pending',
                 assets={p.name: ext.sha(p) for p in tmp.iterdir()}))
    tmp.rename(folder)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', type=Path, default=ROOT)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    a = p.parse_args(); repo=a.repo_root.resolve(); source=a.source_root.resolve(); out=a.output_root.resolve()
    assert source != out and not out.is_relative_to(source)
    summarize(repo, source, out); render(source, out)
    ext.save_json(out/'interpretability/extension_complete.json', dict(status='complete', added_jobs=75,
                  summary_sha256=ext.sha(out/'interpretability/summary/summary_complete.json'),
                  figures_sha256=ext.sha(out/'interpretability/figures/figures_complete.json'),
                  visual_review='pending', manuscript_modified=False))
    print('EXTENSION_COMPUTATION_COMPLETE: visual review and manuscript integration pending', flush=True)


if __name__ == '__main__':
    main()
