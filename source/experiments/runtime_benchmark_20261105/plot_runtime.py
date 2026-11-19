"""Export a standalone engineering runtime chart from the measured pass."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    summary = json.loads((root/'benchmark_summary.json').read_text(encoding='utf-8'))
    rows = sorted(summary['packages'], key=lambda row: row['package'])
    positions = np.arange(len(rows))
    names = [row['package'].split('_sample_')[-1] for row in rows]
    labels = [row['family'].split('_')[0] + '-' + name + f"  (n={row['samples']:,})" for row, name in zip(rows, names)]
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, (ax, per) = plt.subplots(1, 2, figsize=(14.8, 8.3), gridspec_kw={'width_ratios':[3.1, 1.25]}, sharey=True)
    left = np.zeros(len(rows))
    for field, name, color in [
        ('classificationSeconds', 'Rules / hash checks / frozen fusion', '#467da5'),
        ('groundingSeconds', 'Fresh RGB localization / matching', '#3c927b'),
        ('contextScorecardAndIndexSeconds', 'CSV context / scorecards / output', '#d49a42'),
    ]:
        values = np.array([row[field] for row in rows])
        ax.barh(positions, values, left=left, height=.67, label=name, color=color)
        left += values
    for y, value in zip(positions, left):
        ax.text(value + max(left)*.012, y, f'{value:.2f} s', va='center', fontsize=9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left)*1.19)
    ax.set_xlabel('Measured package service time across three stages (seconds)')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='#e5e8eb', linewidth=.7)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='center', bbox_to_anchor=(.54, .128), frameon=False, fontsize=9, ncol=3)
    times = [row['millisecondsPerSample'] for row in rows]
    per.barh(positions, times, color='#758195', height=.67)
    for y, value in zip(positions, times):
        per.text(value + max(times)*.025, y, f'{value:.2f}', va='center', fontsize=9)
    per.set_xlim(0, max(times)*1.25)
    per.tick_params(axis='y', left=False)
    per.set_xlabel('Amortized milliseconds per registered sample')
    per.set_axisbelow(True)
    per.grid(axis='x', color='#e5e8eb', linewidth=.7)
    fig.suptitle('Roadshow evaluation runtime: measured refresh of all 12 packages', x=.53, y=.975, fontsize=17, fontweight='medium')
    fig.text(.53, .925, f"A100 40GB / GPU1 / batch 16  |  {summary['samples']:,} records  |  {summary['imageForwards']:,} fresh image forwards", ha='center', color='#536474', fontsize=11)
    fig.text(.53, .075, f"Whole-pass wall time {summary['endToEndSeconds']:.2f} s; weighted wall-time average {summary['fullPassMillisecondsPerSample']:.2f} ms/sample.", ha='center', color='#37434b', fontsize=10)
    fig.text(.53, .044, '21 upstream expert predictions were verified and reused. Timings exclude new inference of those experts and API explanation.', ha='center', fontsize=9, color='#6b747d')
    fig.text(.53, .019, 'One batch pass; per-package values sum actual stage spans. First-use model loading is included once. OS file caches were not purged.', ha='center', fontsize=9, color='#6b747d')
    fig.subplots_adjust(left=.255, right=.965, top=.875, bottom=.23, wspace=.12)
    for extension in ['png', 'svg']:
        fig.savefig(root/('roadshow_runtime_current.'+extension), dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps({'status':'complete','png':str(root/'roadshow_runtime_current.png'),'svg':str(root/'roadshow_runtime_current.svg')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
