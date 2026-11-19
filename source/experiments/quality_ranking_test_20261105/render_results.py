"""Render a source-linked slide from full historical grouped test predictions."""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

COLORS={'AIA':'#00d7e6','HMI':'#88bdff','CME':'#c9a6ff','EPB':'#ffda58','DMSP':'#ff9f83','ULF':'#47dabb'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def ranks(values):
    result=[0.]*len(values)
    for value in set(values):
        positions=[i+1 for i,v in enumerate(sorted(values)) if v==value]
        average=sum(positions)/len(positions)
        for i,v in enumerate(values):
            if v==value:result[i]=average
    return result


def rho(xs,ys):
    a,b=ranks(xs),ranks(ys);am=sum(a)/len(a);bm=sum(b)/len(b)
    numerator=sum((x-am)*(y-bm) for x,y in zip(a,b))
    denominator=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator/denominator if denominator else None


def percent(value):return '—' if value is None else f'{100*value:.2f}%'


def draw_card(root,scope,packages,filename):
    dark='#111b27';muted='#b6c8d9';aqua='#00d7e6'
    fig=plt.figure(figsize=(18.5,4.4),facecolor=dark)
    all_names=[p['name'] for p in packages if p['tier']=='P1' and p['family'] in scope['families']]
    fig.text(.025,.91,'03  质量排序一致性｜公开测试集',fontsize=24,weight='bold',color=aqua)
    fig.text(.027,.79,'P1—P4 问题／疑点样本检出比例（同一样本去重）',fontsize=14,color=muted)
    fig.text(.70,.91,f"{' / '.join(all_names)} ｜ {scope['packages']}包 ｜ {scope['total_samples']:,}条记录",
             fontsize=11,color=muted)
    ax=fig.add_axes([.075,.24,.55,.43],facecolor=dark)
    groups=[]
    for name in all_names:
        group=sorted([p for p in packages if p['name']==name and p['tier']!='P0'],key=lambda p:p['tier'])
        groups.append(group)
        ys=[100*p['detected_sample_rate'] for p in group]
        ax.plot(range(1,5),ys,color=COLORS[name],linewidth=2.4,marker='o',markersize=5.5,label=name)
        if len(all_names)==3:
            for i,y in enumerate(ys):
                if i==0:continue  # P1 labels are too close; exact values are in the table.
                offset=(0,11) if name!='ULF' or i==1 else (0,-19)
                ax.annotate(f'{y:.2f}%',(i+1,y),textcoords='offset points',xytext=offset,
                            ha='center',fontsize=10,color=COLORS[name],weight='bold')
    ax.set_xlim(.72,4.28);ax.set_ylim(0,68)
    ax.set_xticks(range(1,5),['P1','P2','P3','P4'])
    ax.set_yticks([0,20,40,60],[f'{x}%' for x in [0,20,40,60]])
    ax.tick_params(colors=muted,labelsize=12,length=0,pad=8)
    ax.grid(axis='y',color='#294152',linewidth=.8,zorder=0)
    for name,spine in ax.spines.items():
        spine.set_visible(name in ['left','bottom']);spine.set_color('#537386')
    legend=ax.legend(loc='lower center',bbox_to_anchor=(.5,1.025),ncol=len(all_names),
                     frameon=False,fontsize=11,handlelength=1.5,columnspacing=1.5)
    for text in legend.get_texts():text.set_color(COLORS[text.get_text()])
    fig.add_artist(Line2D([.665,.665],[.20,.87],transform=fig.transFigure,color='#537386',linewidth=1))
    metric=scope['P1_P4_micro'];p0=scope['P0'];ranking=scope['detected_rate_ranking']
    cards=[('微平均 F1（八类已知项）',f"{metric['F1']:.4f}",
            f"Precision {100*metric['precision']:.2f}% / Recall {100*metric['recall']:.2f}%"),
           ('P0样本误报率',f"{100*p0['known_sample_false_positive_rate']:.3f}%",
            f"{p0['known_false_positive_samples']} / {p0['samples']:,}（已知检查范围）"),
           ('组内平均 Spearman ρ',f"{ranking['mean_within_family_spearman']:.4f}",
            f"跨类型全局ρ={ranking['global_spearman_P1_P4']:.4f}；组内次序{ranking['correct_pairs']}/{ranking['total_pairs']}")]
    for y,(label,value,description) in zip([.74,.52,.30],cards):
        fig.text(.69,y,label,fontsize=14,color='white',weight='bold')
        fig.text(.974,y-.014,value,fontsize=29,color=aqua,weight='bold',ha='right')
        fig.text(.69,y-.075,description,fontsize=10.5,color=muted)
        if y>.31:fig.add_artist(Line2D([.69,.976],[y-.11,y-.11],transform=fig.transFigure,color='#345466',linewidth=.8))
    fig.text(.027,.10,'固定权重与阈值，复算历史分组测试集；曲线包含模型候选，检出比例不等于真实污染率。',fontsize=10,color=muted)
    fig.text(.027,.037,'F1排除未知参考项，未计位置匹配；P0数字不是门控前后增益。ULF的P3、P4存在一处次序倒置。',fontsize=10,color=muted)
    for suffix in ['png','svg','pdf']:fig.savefig(root/f'{filename}.{suffix}',dpi=200,facecolor=dark)
    plt.close(fig)


def main(root):
    result=read(root/'results.json');verification=read(root/'verification.json')
    assert result['records']==18450 and verification['status']=='pass'
    with (root/'all30_packages.csv').open(encoding='utf-8-sig',newline='') as fp:
        packages=list(csv.DictReader(fp))
    integer=['samples','detected_samples','detected_known_samples','confirmed_rule_samples','model_candidate_samples',
             'reference_known_positive_samples','known_class_slots','unknown_class_slots','TP','FP','FN','TN']
    for p in packages:
        for key in list(p):
            if key in integer:p[key]=int(p[key])
            elif key not in ['family','name','tier']:p[key]=float(p[key]) if p[key] else None
    # Recompute headline counts from the delivered paired ledger, without using
    # the server's summary helper. Also check Spearman by average-rank Pearson.
    total=collections.Counter();clean=collections.Counter();seen=set()
    with gzip.open(root/'paired_test_samples.jsonl.gz','rt',encoding='utf-8') as fp:
        for line in fp:
            r=json.loads(line);assert r['id'] not in seen;seen.add(r['id'])
            if r['tier']=='P0':
                clean['N']+=1
                clean['FP']+=any(a and known for a,known in zip(r['prediction'],r['strict_valid']))
            else:
                for y,v,p in zip(r['target'],r['strict_valid'],r['prediction']):
                    if v:total['TP' if y and p else 'FN' if y else 'FP' if p else 'TN']+=1
    assert len(seen)==18450
    all_six=result['all_six'];m=all_six['P1_P4_micro'];ranking=all_six['detected_rate_ranking']
    assert all(total[k]==m[k] for k in ['TP','FP','FN','TN'])
    assert clean==collections.Counter(N=3690,FP=2)
    assert math.isclose(2*total['TP']/(2*total['TP']+total['FP']+total['FN']),m['F1'])
    for item in ranking['per_family']:
        group=sorted([p for p in packages if p['family']==item['family'] and p['tier']!='P0'],key=lambda p:p['tier'])
        assert math.isclose(rho([1,2,3,4],[p['detected_sample_rate'] for p in group]),item['spearman_P1_P4'])
    nonclean=[p for p in packages if p['tier']!='P0']
    assert math.isclose(rho([int(p['tier'][1]) for p in nonclean],[p['detected_sample_rate'] for p in nonclean]),ranking['global_spearman_P1_P4'])
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
        'axes.unicode_minus':False,'svg.fonttype':'none','pdf.fonttype':42})
    draw_card(root,all_six,packages,'01_测试集质量排序_六类全量')
    draw_card(root,result['three_for_original_slide'],packages,'02_测试集质量排序_AIA_ULF_EPB')
    table=[]
    for family in all_six['families']:
        group=[p for p in packages if p['family']==family]
        row=[group[0]['name']]
        for tier in ['P0','P1','P2','P3','P4']:
            p=next(p for p in group if p['tier']==tier)
            row.append(f"{p['detected_samples']}/{p['samples']}（{100*p['detected_sample_rate']:.3f}%）")
        item=next(r for r in ranking['per_family'] if r['family']==family)
        row += [f"{item['spearman_P1_P4']:.4f}",f"{item['correct_pairs']}/6"]
        table.append('| '+' | '.join(row)+' |')
    package_rows=[]
    for p in packages:
        package_rows.append(f"| {p['name']} {p['tier']} | {p['samples']} | {p['detected_samples']} | {100*p['detected_sample_rate']:.3f}% | {p['weighted_pollution_index']:.5f} | {percent(p['precision'])} | {percent(p['recall'])} | {percent(p['F1'])} |")
    perclass=[]
    with (root/'all30_classes.csv').open(encoding='utf-8-sig',newline='') as fp:
        class_rows=list(csv.DictReader(fp))
    for code in result['classes']:
        values=collections.Counter()
        for r in class_rows:
            if r['tier']!='P0' and r['code']==code:values.update({k:int(r[k]) for k in ['TP','FP','FN','TN']})
        tp,fp,fn=values['TP'],values['FP'],values['FN']
        perclass.append(f"| {code} | {tp} | {fp} | {fn} | {percent(tp/(tp+fp) if tp+fp else None)} | {percent(tp/(tp+fn) if tp+fn else None)} | {percent(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)} |")
    report=f'''# 测试集质量排序与识别指标：完整复算结果

已在完整六类、30包、18,450条历史分组测试记录上完成复算，未缩小子集。使用当前`fusion_v2/selected_eight`冻结预测；这次新执行的是逐样本核对、统计和排序检验，没有重跑21个专家或改变阈值。服务器复算耗时{result['seconds']:.2f}秒，其余时间用于取回、交叉复核与作图。

## PPT可用结果

| 项目 | 测试集实测 | 原图应如何改 |
|---|---:|---|
| 八类识别微平均F1，P1—P4严格已知项 | {m['F1']:.6f}（{100*m['F1']:.2f}%） | 将0.85改为0.9445，并写明“微平均F1” |
| 精确率 / 召回率 | {percent(m['precision'])} / {percent(m['recall'])} | 作为识别指标补充 |
| 数据包宏平均F1 | {all_six['P1_P4_macro_package_F1']:.6f}（{100*all_six['P1_P4_macro_package_F1']:.2f}%） | 若标题保留“平均F1”，需说明此项与微平均不同 |
| P0已知检查范围样本误报比例 | 2/3690＝0.054201% | 将“误报率（核验后）0.51%”改为“P0样本误报率0.054%” |
| 六类各自组内Spearmanρ的平均 | {ranking['mean_within_family_spearman']:.6f} | 改为“组内平均Spearmanρ＝0.9667” |
| 跨类型24包全局Spearmanρ | {ranking['global_spearman_P1_P4']:.6f} | 若声称“全局排序一致性”，应填0.8130 |
| 同类型不同档位顺序 | {ranking['correct_pairs']}/{ranking['total_pairs']}正确；{ranking['reversed_pairs']}处倒置；{ranking['tied_pairs']}处并列 | ULF的P3、P4存在倒置，不写“全部排序正确” |

微平均来自TP={m['TP']}、FP={m['FP']}、FN={m['FN']}、TN={m['TN']}，F1＝2TP/(2TP+FP+FN)。P1—P4共有14,760条记录，已知分类检查单元{sum(m[k] for k in ['TP','FP','FN','TN']):,}个；其余未知单元不计入TP/FP/FN/TN。包宏平均仅有23/24包的F1可定义：CME P1在严格掩码下没有阳性参考且没有已知阳性预测，F1分母为0，按未定义保留，没有擅自计为0或1。

这里的P0比例来自当前八类组合，不是门控前后对照结果。先前旧组合19/3690≈0.51%与当前2/3690并非只切换一个门控组件，不能据此计算门控因果增益。

## 各档问题／疑点样本检出比例

同一样本有多个问题只计一次。这里包含模型候选，和“规则已确认问题数”分开统计；候选也可能落在没有严格参考的单元，因此曲线不能当作实际污染率或严格阳性参考比例。完整CSV保留了已知范围检出、规则确认、模型候选、参考阳性和标签覆盖数量。

| 数据类型 | P0 | P1 | P2 | P3 | P4 | 组内ρ，P1—P4 | 正确次序 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

ULF P3为155/950（16.316%），P4为154/950（16.211%），差1个样本，故ULF组内ρ为0.8，其余五类均为1。六类平均为0.9667。它衡量同类P1—P4的检出比例是否随档位递增，不等价于质量等级分类准确率，也不证明缺陷比例估计准确。

## 八类已知项识别指标

| 类别 | TP | FP | FN | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(perclass)}

## 30包明细

加权污染指数＝每个样本各类过阈值概率之和，再对全部样本取平均。规则已确认项取证据权重1，模型项使用冻结的类别阈值和概率。一个样本可含多类问题，指数范围0—8；它未做科学概率校准，不应称为真实污染百分比。该指数的跨类型全局ρ为{all_six['weighted_index_ranking']['global_spearman_P1_P4']:.4f}，组内平均ρ为{all_six['weighted_index_ranking']['mean_within_family_spearman']:.4f}；主图仍按原图定义使用“样本检出比例”。

| 数据包 | 样本数 | 检出问题/疑点样本 | 检出比例 | 加权污染指数 | 精确率 | 召回率 | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(package_rows)}

## 当前模型与测试划分

当前模型沿用全六类训练的21个专家输出。八类分类器候选为当前结构特征HGB15、融合HGB7、融合HGB15，按数据类型与缺陷类别采用验证阶段已冻结的选择，也可直接选择确定性规则。当前测试没有挑选新阈值或重选最佳模型。

原始测试清单来自`/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031/test.jsonl`；同源与重复内容按`training_group_id`划分，本轮重新检查了测试组与原train、val均无交集。每条预测的ID、数据类型、档位、当前输入哈希和规则位都逐条核对，并从冻结阈值重建全部18,450条八类判定；30包TP/FP/FN/TN还与既有严格测试表独立对照一致。

参考来自P0差异及确定性字段核验，不是官方独立八类定位真值。AIA/HMI存在像素变化等情况时，不能配对的空间或数量单元已按严格掩码排除。该结果不包含定位匹配分，不宣称达到官方关键缺陷召回门槛；历史公开测试此前已有报告，因此这不是全新封存盲测。

## 图与复现材料

- `01_测试集质量排序_六类全量.png/svg/pdf`：完整六类，主结果。
- `02_测试集质量排序_AIA_ULF_EPB.png/svg/pdf`：沿用原图三类，右侧指标也只计算同三类，不混用全六类结果。该子集F1为0.9652、P0为0/1607、组内平均ρ为0.9333。
- `all30_packages.csv`、`all30_classes.csv`、`ranking_by_family.csv`：包、类别及排序明细。
- `paired_test_samples.jsonl.gz`：18,450条逐样本预测、严格参考、路径、哈希与加权负担。
- `results.json`、`verification.json`、`local_reverification.json`：统计、原始来源复核及本机独立复算。
- `code/recompute.py`、`code/render_results.py`：统计与绘图代码，数据和权重未打包。

在icarus0重新统计时使用新目录，避免覆盖已有结果：

```bash
/staff/tangjiarui/OPC/.venv-image-models/bin/python code/recompute.py --output /staff/tangjiarui/OPC/quality_ranking_test_reproduce
```

本机从已交付JSON、CSV和逐样本台账重画，不调用模型：

```powershell
uv run --with matplotlib --with numpy python code/render_results.py --input .
```

可用汇报词：在覆盖六类、30包的18,450条分组测试记录上，当前组合的八类已知项微平均F1为94.45%，精确率96.06%，召回率92.89%；P0范围误报2例。按同类污染档位检验，组内平均Spearmanρ为0.9667，36组先后关系中35组正确。跨类型全局ρ为0.8130，说明不同类型之间的比较仍有改进空间。
'''
    (root/'测试集真实结果与PPT修改说明.md').write_text(report,encoding='utf-8')
    local=dict(status='pass',paired_ledger_rows=len(seen),independently_recomputed_counts=dict(total),
               P0_samples=clean['N'],P0_known_false_positive_samples=clean['FP'],
               within_and_global_spearman_recomputed_without_scipy=True)
    (root/'local_reverification.json').write_text(json.dumps(local,ensure_ascii=False,indent=2),encoding='utf-8')
    code=root/'code';code.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):
        target=code/path.name
        if path.resolve()!=target.resolve():shutil.copy2(path,target)
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='SHA256.json' and '__pycache__' not in p.parts]
    manifest={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'SHA256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=root/'SpaceQA_测试集质量排序_全六类30包_20261105.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'SHA256.json']:z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(json.dumps(dict(status='pass',records=18450,micro_F1=m['F1'],
        within_family_rho=ranking['mean_within_family_spearman'],global_rho=ranking['global_spearman_P1_P4'],
        archive_bytes=archive.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    main(p.parse_args().input)
