"""Data-free checks of the separately frozen extension and combined aggregates."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'paper_assets/interpretability_rgb_extension_v1'
SEEDS=set(range(20260925,20260930))
MODELS={'resnet50','lpt_full','efficientnetb0','convnext_tiny','efficientnetv2s'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):
    assert np.isclose(float(a),float(b),rtol=1e-7,atol=2e-9,equal_nan=True),(a,b)

def main():
    records=list(csv.DictReader((ROOT/'paper_assets/release_audit/rgb_extension_asset_registry.csv').open()))
    assert len(records)==16 and len({r['path'] for r in records})==16
    for row in records:
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
        if row['handling']=='byte_identical': assert row['sha256']==row['original_sha256']
    for p in (ROOT/'experiments/interpretability_rgb_extension_v1').glob('*.py'):
        ast.parse(p.read_text())
    summary=ASSET/'summary'; marker=json.loads((summary/'summary_complete.json').read_text())
    assert marker['status']=='complete' and marker['n_image_seed_model_rows']==148875
    assert marker['audited_inputs']['original']['audited_jobs']==50
    assert marker['audited_inputs']['extension']['audited_jobs']==75
    cases=ROOT/'paper_assets/associate_editor_revision_v1/interpretability/qualitative_cases.csv'
    for stage in ['original','extension']:
        assert marker['audited_inputs'][stage]['qualitative_cases_sha256']==sha(cases)
    for name,digest in marker['assets'].items():
        if name=='input_registry.csv':
            redacted=next(r for r in records if r['path'].endswith('input_registry_relative.csv'))
            assert redacted['original_sha256']==digest
        else: assert sha(summary/name)==digest,name
    coverage=pd.read_csv(summary/'coverage_audit.csv')
    all_scope=coverage.loc[coverage.scope.eq('all')]
    assert len(all_scope)==25 and set(all_scope.model)==MODELS
    assert all_scope.n_images.eq(5955).all() and all_scope.n_images.sum()==148875
    assert not all_scope.duplicated(['model','seed']).any()
    for _,g in all_scope.groupby('model'): assert set(g.seed)==SEEDS
    seeds=pd.read_csv(summary/'seed_metrics.csv')
    means=pd.read_csv(summary/'model_summary.csv')
    for r in means.itertuples():
        g=seeds.loc[seeds.model.eq(r.model)&seeds.scope.eq(r.scope)&seeds.metric.eq(r.metric)]
        assert set(g.seed)==SEEDS
        close(g.image_mean.mean(),r.mean_across_seed_image_means)
        close(g.image_mean.std(ddof=1),r.sample_sd_across_seed_image_means)
    paired=pd.read_csv(summary/'paired_seed_metrics.csv')
    intervals=pd.read_csv(summary/'bootstrap_intervals.csv')
    draws=pd.read_csv(summary/'bootstrap_replicates.csv')
    assert len(draws)==5000 and set(intervals.series_id)==set(draws.columns)
    for r in intervals.itertuples():
        g=paired.loc[paired.series_id.eq(r.series_id)]
        assert set(g.seed)==SEEDS
        close(g.image_mean.mean(),r.mean_across_seed_image_means)
        close(g.image_mean.std(ddof=1),r.sample_sd_across_seed_image_means)
        v=draws[r.series_id].dropna()
        assert len(v)==r.n_replicates_estimable
        close(v.quantile(.025),r.ci95_low);close(v.quantile(.975),r.ci95_high)
    original=json.loads((ROOT/'paper_assets/associate_editor_revision_v1/interpretability/summary/bootstrap_design.json').read_text())
    assert marker['bootstrap']['draw_sequence_sha256']==original['draw_sequence_sha256']
    assert marker['p_values_reported'] is False
    assert not any(ASSET.rglob('*.npz')) and not any(ASSET.rglob('*.pt'))
    print(json.dumps(dict(status='passed',sealed_code_files=4,release_assets=len(records),models=sorted(MODELS),seeds=sorted(SEEDS),
        original_jobs=50,added_jobs=75,total_jobs=125,image_model_seed_records=148875,descriptive_intervals=len(intervals),
        medical_image_inference_rerun=False),indent=2))

if __name__=='__main__': main()
