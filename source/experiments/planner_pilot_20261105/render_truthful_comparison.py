"""Show primary task accuracy alongside secondary mechanical-check F1.

Recompute every plotted value from saved responses. Do not change original
experiment receipts or substitute a secondary endpoint for task accuracy.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from evaluate_outputs import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source / 'run_v3'
    tasks = {r['task_id']: r for r in json.loads((source/'tasks_private.json').read_text(encoding='utf-8'))}
    normalized = evaluate(source)
    assert len(tasks) == 36 and len(normalized['rows']) == 72
    names = [('direct', '直接判定'), ('dependency', '前置证据'), ('multi_tool', '多工具联合')]
    metrics = {}
    tables = []
    for group, label in names:
        metrics[group] = {}
        for arm in ['fixed', 'agent']:
            rows = [r for r in normalized['rows'] if r['group'] == group and r['arm'] == arm]
            assert len(rows) == 12
            tp = fp = fn = tn = unresolved = 0
            for row in rows:
                for code, target in tasks[row['task_id']]['expected'].items():
                    if target not in ['pass', 'fail']:
                        continue
                    predicted = row['checks'].get(code, {}).get('status', 'undetermined')
                    tp += target == 'fail' and predicted == 'fail'
                    fp += target == 'pass' and predicted == 'fail'
                    fn += target == 'fail' and predicted != 'fail'
                    tn += target == 'pass' and predicted == 'pass'
                    unresolved += predicted not in ['pass', 'fail']
            correct = sum(r['exact_correct'] for r in rows)
            value = dict(n=12, correct=correct, accuracy=correct/12,
                         TP=tp, FP=fp, FN=fn, TN=tn, unresolved_known_slots=unresolved,
                         precision=tp/(tp+fp), recall=tp/(tp+fn),
                         F1=2*tp/(2*tp+fp+fn),
                         complete_answers=sum(bool(r['checks']) for r in rows))
            metrics[group][arm] = value
            tables.append(dict(group=group, task_type=label, arm=arm, **value))
    fixed_correct = sum(metrics[group]['fixed']['correct'] for group, _ in names)
    agent_correct = sum(metrics[group]['agent']['correct'] for group, _ in names)
    assert fixed_correct == agent_correct == 35
    multi_fixed, multi_agent = metrics['multi_tool']['fixed'], metrics['multi_tool']['agent']
    assert (multi_fixed['TP'], multi_fixed['FP'], multi_fixed['FN']) == (5, 1, 0)
    assert (multi_agent['TP'], multi_agent['FP'], multi_agent['FN']) == (5, 0, 0)
    delta = 100*(multi_agent['F1']-multi_fixed['F1'])
    with (args.output/'图表真实数据.csv').open('w', encoding='utf-8-sig', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    (args.output/'图表复算结果.json').write_text(json.dumps(dict(
        source=str(source), source_data='Saved first complete verdicts with identical wrapper normalization',
        task_metrics=metrics, overall_correct=dict(fixed=35, agent=35),
        primary_accuracy_delta_pp=0., secondary_multi_tool_F1_delta_pp=delta,
        input_files_sha256={str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [source/'tasks_private.json', *sorted((source/'runs').glob('*.json'))]}),
        ensure_ascii=False, indent=2), encoding='utf-8')

    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font_path)).get_name(),
        'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False,
        'svg.fonttype':'none', 'font.size':11, 'figure.facecolor':'white'})
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.4375))
    colors = {'fixed':'#9bb5bf', 'agent':'#159ca6'}
    x = np.arange(3)
    for ax, key, title in [(axes[0], 'accuracy', '主要指标：整项任务准确率'),
                            (axes[1], 'F1', '次要指标：机械检查F1')]:
        for offset, arm, label in [(-.18, 'fixed', '固定流程'), (.18, 'agent', 'Agent自主规划')]:
            values = [metrics[group][arm] for group, _ in names]
            ys = [100*v[key] for v in values]
            bars = ax.bar(x+offset, ys, width=.32, color=colors[arm], label=label, zorder=3)
            for bar, value in zip(bars, values):
                text = f"{100*value[key]:.2f}%"
                if key == 'accuracy':
                    text += f"\n({value['correct']}/12)"
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2.5, text,
                        ha='center', va='bottom', fontsize=11, weight='bold', color='#12384d')
        ax.set_title(title, fontsize=15, weight='bold', color='#07394e', pad=18)
        ax.set_xticks(x, [label for _, label in names], fontsize=11)
        ax.set_ylim(0, 119)
        ax.set_yticks(np.arange(0,101,20))
        ax.set_ylabel('准确率（%）' if key == 'accuracy' else 'F1（%）')
        ax.grid(axis='y', color='#e6ecee', zorder=0)
    fig.suptitle('固定流程与Agent自主规划：真实结果对照', x=.065, ha='left',
                 y=.975, fontsize=23, weight='bold', color='#07394e')
    fig.text(.065, .90, '总体任务准确率均为97.22%（35/36）；多工具组机械检查F1提高9.09个百分点',
             fontsize=14, color='#486873')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52,.865),
               ncol=2, frameon=False, fontsize=12)
    fig.text(.065,.225,'任务准确率：三组差异均为0个百分点', color='#486873', fontsize=12)
    fig.text(.555,.225, '多工具组：90.91% → 100%，ΔF1=+9.09 pp',
             color='#087f88', fontsize=12, weight='bold')
    fig.text(.065, .135, '同批36例、六类公开数据，每种任务12例；相同DeepSeek模型及工具预算上限。', fontsize=11, color='#486873')
    fig.text(.065, .092, '多工具组：固定流程TP/FP/FN=5/1/0；Agent=5/0/0。Agent另有1次接口超时，已计入任务失败。',
             fontsize=11, color='#486873')
    fig.text(.065, .047, 'F1仅统计可复算的机械检查，超时例没有阳性参考，未产生阳性FN；小样本局部提升不代表整体准确率或官方定位成绩提升。',
             fontsize=10, color='#486873')
    fig.subplots_adjust(left=.065, right=.975, top=.735, bottom=.31, wspace=.23)
    for suffix in ['png', 'svg', 'pdf']:
        fig.savefig(args.output/f'Agent对比_准确率与机械F1_真实数据.{suffix}', dpi=200)
    plt.close(fig)
    note = '''# Agent小试验：图表及汇报口径

所有数字均从已有36例、72条最终执行记录重新计算，本次没有启动模型或训练，没有修改原始回答。

| 指标 | 固定流程 | Agent自主规划 | 差异 |
|---|---:|---:|---:|
| 全部36例整项准确率 | 97.22%（35/36） | 97.22%（35/36） | 0个百分点 |
| 多工具组整项准确率 | 91.67%（11/12） | 91.67%（11/12） | 0个百分点 |
| 多工具组机械检查精确率 | 83.33% | 100% | +16.67个百分点 |
| 多工具组机械检查召回率 | 100% | 100% | 0个百分点 |
| 多工具组机械检查F1 | 90.91% | 100% | +9.09个百分点 |

多工具组的固定流程TP=5、FP=1、FN=0；Agent为TP=5、FP=0、FN=0。F1=2TP/(2TP+FP+FN)。9.09是百分点差，若按相对增幅表达则为10%，二者不能混写。

Agent有1次最终接口超时，该例的参考检查均为通过或不适用，属于整项失败；因为没有阳性参考，不增加阳性F1中的FN。固定流程另有一例EPB几何检查误报。两条流程在不同案例上各失败1例，因此整项准确率持平。不能将100%机械F1解释为所有任务都执行成功。

本图把主要指标和次要指标同时展示。真实的局部F1差异仅由5个阳性检查和1个误报的差别产生，尚不能宣称自主规划整体更优或有显著增益。Agent工具调用110次，固定流程90次，实际消耗也应一并考虑。

可用汇报词：在覆盖六类数据的36例配对小试验中，两种流程总体任务准确率均为97.22%。多工具联合组中，Agent少产生1条机械检查误报，相关F1从90.91%提升至100%，提高9.09个百分点；同时有1次接口超时，整项准确率仍持平，后续需扩大样本验证稳定性。

该结果属于文件、字段、尺寸及数值边界核验，不是八类科学缺陷及位置联合F1。原实验报告、逐例台账、执行日志和源文件哈希均保留在上级实验目录。
'''
    (args.output/'图表说明与汇报词.md').write_text(note, encoding='utf-8')
    shutil.copy2(Path(__file__), args.output/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('evaluate_outputs.py'), args.output/'evaluate_outputs.py')
    files = [p for p in args.output.iterdir() if p.is_file() and p.suffix != '.zip' and p.name != 'SHA256.json']
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = args.output/'Agent真实对比图与数据.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files+[args.output/'SHA256.json']:
            z.write(p, p.name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    print(json.dumps(dict(status='pass', paired_tasks=36, actual_trajectories=72,
                         primary_accuracy_delta_pp=0., secondary_multi_tool_F1_delta_pp=delta,
                         archive_bytes=archive.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    main()
