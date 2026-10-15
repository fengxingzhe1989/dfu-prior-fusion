#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from common import OUT, SIZES
import common as base
from run import verify_seal


def main() -> None:
    verify_seal()
    summary = OUT / 'summary'
    marker = base.read_json(summary / 'summary_complete.json')
    if marker['seed_cells'] != 40 or marker['fold_cells'] != 200 or marker['nominal_sizes'] != list(SIZES):
        raise AssertionError('Four-budget data are incomplete')
    base.verify_registry(marker['outputs'])
    out = OUT / 'figures'
    if (out / 'exports_complete.json').exists():
        old = base.read_json(out / 'exports_complete.json')
        if old['summary_sha256'] != base.sha(summary / 'summary_complete.json'):
            raise RuntimeError('Existing exports use a different summary')
        base.verify_registry(old['files'])
        print('Existing figure exports verified')
        return
    if out.exists():
        raise RuntimeError('Incomplete figure folder requires diagnosis, not overwrite')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                               'axes.spines.top': False, 'axes.spines.right': False,
                               'axes.linewidth': 0.7, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                               'legend.frameon': False, 'axes.labelcolor': '#20252B', 'text.color': '#20252B'})
    data = pd.read_csv(summary / 'seed_metrics.csv')
    counts = pd.read_csv(summary / 'subset_counts.csv')
    counts = counts.loc[counts.role.eq('refit')].groupby(['nominal_percent', 'seed'], as_index=False).agg(
        mean_training_images=('images', 'mean'), mean_training_groups=('groups', 'mean'),
        mean_ischaemia_training_images=('ischaemia', 'mean'))
    data = data.merge(counts, on=['nominal_percent', 'seed'], validate='many_to_one')
    if len(data) != 40:
        raise AssertionError('Missing curve source rows')
    methods = {'lpt_full': ('Full LPT', '#2878B5', 'o', '-'),
               'lpt_neutral': ('Neutral-prior Full LPT', '#747B83', 's', '--')}
    panels = [('macro_f1', 'Macro-F1'), ('ischaemia_recall', 'Ischaemia-only recall'),
              ('ischaemia_precision', 'Ischaemia-only precision'), ('ischaemia_f1', 'Ischaemia-only F1')]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))
    for letter, ax, (metric, title) in zip('ABCD', axs.flat, panels):
        axis_lower, axis_upper = 0.0, 1.025
        for model, (_, colour, shape, linestyle) in methods.items():
            part = data.loc[data.model.eq(model)].sort_values(['seed', 'nominal_percent'])
            for _, trajectory in part.groupby('seed'):
                ax.plot(trajectory.mean_training_images, trajectory[metric], color=colour,
                        lw=0.55, alpha=0.25, linestyle=linestyle)
            grouped = part.groupby('nominal_percent').agg(x=('mean_training_images', 'mean'),
                                                         value=(metric, 'mean'), sd=(metric, 'std'),
                                                         valid=(metric, 'count')).reindex(SIZES)
            # A five-trajectory mean is undefined if any trajectory's precision is undefined.
            grouped.loc[grouped.valid.ne(5), ['value', 'sd']] = np.nan
            ax.errorbar(grouped.x, grouped.value, yerr=grouped.sd, color=colour, marker=shape,
                        markersize=4, linewidth=1.3, linestyle=linestyle, capsize=2, elinewidth=0.8)
            lows = (grouped.value-grouped.sd).dropna()
            highs = (grouped.value+grouped.sd).dropna()
            if len(lows): axis_lower = min(axis_lower, float(lows.min())-0.025)
            if len(highs): axis_upper = max(axis_upper, float(highs.max())+0.025)
        ticks = counts.groupby('nominal_percent').mean(numeric_only=True).reindex(SIZES).mean_training_images
        ax.set_xticks(ticks, [f'{s}%\n{n:,.0f}' for s, n in zip(SIZES, ticks)])
        ax.set_xlabel('Nominal budget / mean training images per fold', fontsize=7)
        ax.set_ylabel(title)
        ax.set_ylim(axis_lower, axis_upper)
        ax.text(-0.17, 1.035, letter, transform=ax.transAxes, fontsize=10, fontweight='bold')
        ax.grid(axis='y', linewidth=0.4, alpha=0.2)
    handles = [Line2D([0], [0], color=c, marker=m, linestyle=l, label=n)
               for n, c, m, l in methods.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.text(0.5, 0.01, 'Thin lines: all five trajectories. Points and bars: mean +/- sample SD.', ha='center', fontsize=7)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.90, wspace=0.38, hspace=0.47)
    out.mkdir(parents=True)
    stem = out / 'training_size_curves'
    for ext in ('svg', 'pdf'):
        fig.savefig(stem.with_suffix('.'+ext), facecolor='white')
    fig.savefig(stem.with_suffix('.tiff'), dpi=600, facecolor='white', pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
    plt.close(fig)
    data.to_csv(out / 'curve_source_data.csv', index=False, na_rep='NA', float_format='%.12g')
    (out / 'caption.txt').write_text(
        'Exploratory training-size sensitivity under a fixed external prior and ImageNet pretraining. '
        '(A) Four-class Macro-F1; (B-D) ischaemia-only recall, precision and F1. Full LPT and its '
        'architecture-matched neutral-prior control used the same group-preserving nested subsets '
        'at nominal 25%, 50%, 75% and 100% training budgets. Both fit and stopping data count '
        'towards each budget. All four endpoints were freshly fitted. Thin lines show all five trajectories; points and bars show their '
        'mean and sample SD, reflecting subset, internal-partition and optimisation variation. Scores '
        'were calculated from 5955 pooled outer-held-out predictions per trajectory, not by '
        'treating fold-by-trajectory cells as independent cohorts. Horizontal coordinates show '
        'actual mean outer-training image counts across five folds; nominal fractions may differ '
        'from achieved image fractions because groups remain intact and strata are rounded. '
        'Counts of training images, groups and ischaemia-only images and paired model differences '
        'are supplied in the source tables. These curves do not establish sample-size sufficiency, '
        'patient independence or clinical utility.\n')
    base.write_json(out / 'exports_complete.json', dict(status='complete', completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    summary_sha256=base.sha(summary / 'summary_complete.json'), visual_qa_complete=False,
                    files=base.reg(list(out.iterdir()))))


if __name__ == '__main__':
    main()
