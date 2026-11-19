"""Produce paired pilot tables and figures from frozen model receipts."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import math
from evaluate_outputs import evaluate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def percent(value):
    return '—' if value is None else f'{100*value:.2f}%'

def csv_file(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    current = root / 'run_v3'
    summary = read(current/'summary.json')
    assert summary['status'] == 'complete' and summary['completed_arms'] == 72
    tasks = {t['task_id']: t for t in read(current/'tasks_private.json')}
    source_audit = read(current/'independent_source_audit.json')
    assert source_audit['status'] == 'pass' and source_audit['source_tasks'] == 36
    results = [read(p) for p in (current/'runs').glob('*.json')]
    normalized=evaluate(current)
    (current/'normalized_evaluation.json').write_text(json.dumps(normalized,ensure_ascii=False,indent=2),encoding='utf-8')
    parsed={(r['task_id'],r['arm']):r for r in normalized['rows']}
    all_receipts = {}
    for phase in ['run_v1','run_v2','run_v3']:
        for path in (root/phase/'runs').glob('*.json'):
            for receipt in read(path)['calls']:
                key = receipt.get('response_id') or hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
                all_receipts[key] = receipt
    all_usage = {key:sum(r.get('usage',{}).get(key,0) for r in all_receipts.values())
                 for key in ['prompt_tokens','completion_tokens','total_tokens']}
    pairs = collections.defaultdict(dict)
    # Recompute primary accuracy independently of runner summary fields.
    for result in results:
        task = tasks[result['task_id']]
        checks = (result.get('answer') or {}).get('checks')
        correct = isinstance(checks, dict) and set(checks) == set(task['expected']) and all(isinstance(checks[k], dict) and checks[k].get('status') == v for k,v in task['expected'].items())
        assert correct == result['exact_correct']
        pairs[result['task_id']][result['arm']] = result
        for message in result['messages']:
            if message['role'] == 'user':
                for forbidden in ['"truth_valid":', '"training_group_id":', '"base_sample_id":', '"tier":', '"expected":', '"target":']:
                    assert forbidden not in message['content'], (result['task_id'], forbidden)
    assert len(pairs) == 36 and all(set(p) == {'fixed','agent'} for p in pairs.values())
    # Raw protocol metrics remain intact in run_v3/summary.json. Recompute the
    # displayed factual metrics from uniformly extracted check-status vectors.
    for group in summary['groups']:
        for arm in ['fixed','agent']:
            values=group['arms'][arm]
            rows=[r for r in normalized['rows'] if r['group']==group['group'] and r['arm']==arm]
            count=sum(r['exact_correct'] for r in rows)
            values['correct']=count
            values['accuracy']=count/12
            z=1.959963984540054
            p=count/12
            den=1+z*z/12
            center=(p+z*z/24)/den
            half=z*math.sqrt(p*(1-p)/12+z*z/(4*144))/den
            values['wilson95']=[center-half,center+half]
            slots=[(tasks[r['task_id']]['expected'][name],r['checks'].get(name,{}).get('status','undetermined'))
                   for r in rows for name in tasks[r['task_id']]['checks'] if tasks[r['task_id']]['expected'][name] in ['pass','fail']]
            tp=sum(a=='fail' and p=='fail' for a,p in slots)
            fp=sum(a=='pass' and p=='fail' for a,p in slots)
            fn=sum(a=='fail' and p!='fail' for a,p in slots)
            tn=sum(a=='pass' and p=='pass' for a,p in slots)
            values.update(TP=tp,FP=fp,FN=fn,TN=tn,precision=tp/(tp+fp) if tp+fp else None,
                          recall=tp/(tp+fn) if tp+fn else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                          raw_protocol_correct=sum(r['raw_protocol_exact_correct'] for r in rows),
                          references_resolve=sum(r['references_resolve'] for r in rows))
        group['delta_accuracy_pp']=100*(group['arms']['agent']['accuracy']-group['arms']['fixed']['accuracy'])
        ids=[key for key,task in tasks.items() if task['group']==group['group']]
        better=sum(parsed[(key,'agent')]['exact_correct'] and not parsed[(key,'fixed')]['exact_correct'] for key in ids)
        worse=sum(parsed[(key,'fixed')]['exact_correct'] and not parsed[(key,'agent')]['exact_correct'] for key in ids)
        n=better+worse
        group['paired_agent_only_correct']=better
        group['paired_fixed_only_correct']=worse
        group['paired_exact_p']=min(1.,2*sum(math.comb(n,k) for k in range(min(better,worse)+1))/2**n) if n else 1.
    (root/'displayed_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    table, detail = [], []
    for group in summary['groups']:
        fixed, agent = group['arms']['fixed'], group['arms']['agent']
        assert fixed['completed'] == agent['completed'] == group['n'] == 12
        for arm, values in [('fixed',fixed),('agent',agent)]:
            raw = [r for r in results if r['group'] == group['group'] and r['arm'] == arm]
            assert sum(parsed[(r['task_id'],arm)]['exact_correct'] for r in raw) == values['correct']
        table.append({'任务类型':group['name'],'样本数':12,
                      '固定流程正确数':fixed['correct'],'固定流程准确率':fixed['accuracy'],
                      'Agent正确数':agent['correct'],'Agent准确率':agent['accuracy'],
                      '准确率差_百分点':group['delta_accuracy_pp'],
                      '固定流程Precision':fixed['precision'],'固定流程Recall':fixed['recall'],'固定流程F1':fixed['F1'],
                      'AgentPrecision':agent['precision'],'AgentRecall':agent['recall'],'AgentF1':agent['F1'],
                      '固定流程工具调用':fixed['tool_calls'],'Agent工具调用':agent['tool_calls'],
                      '固定流程已保存模型调用':fixed['model_calls'],'Agent已保存模型调用':agent['model_calls'],
                      '固定流程HTTP尝试':fixed['http_attempts'],'AgentHTTP尝试':agent['http_attempts'],
                      '固定流程原始协议正确数':fixed['raw_protocol_correct'],'Agent原始协议正确数':agent['raw_protocol_correct']})
    for task_id, pair in sorted(pairs.items()):
        task = tasks[task_id]
        fixed, agent = pair['fixed'], pair['agent']
        detail.append({'task_id':task_id,'任务类型':task['group'],'数据类型':task['source']['family'],
            '原始档位_仅评估保存':task['source']['tier'],'source_sample_id':task['source']['id'],
            'base_sample_id':task['source']['base_sample_id'],'split_group_id':task['source']['training_group_id'],
            '参考答案':json.dumps(task['expected'],ensure_ascii=False),
            '固定流程答案':json.dumps(parsed[(task_id,'fixed')]['checks'],ensure_ascii=False),
            'Agent答案':json.dumps(parsed[(task_id,'agent')]['checks'],ensure_ascii=False),
            '固定流程正确':parsed[(task_id,'fixed')]['exact_correct'],'Agent正确':parsed[(task_id,'agent')]['exact_correct'],
            '固定流程原始协议正确':fixed['exact_correct'],'Agent原始协议正确':agent['exact_correct'],
            '固定流程累计秒':fixed['seconds'],'Agent累计秒':agent['seconds'],
            '固定流程工具数':len(fixed['tools']),'Agent工具数':len(agent['tools']),
            'label_path':task['source']['label_path'],'data_path':task['source']['data_path'],
            'label_sha256':task['source']['label_sha256'],'data_sha256':task['source']['data_sha256']})
    csv_file(root/'三类任务对比结果.csv', table)
    csv_file(root/'36例逐任务配对台账.csv', detail)
    fonts = [Path('C:/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
    for path in fonts:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'white'})
    figures = root/'figures'
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8,7.2))
    x = np.arange(3)
    for offset, arm, color, label in [(-.18,'fixed','#9bb5bf','固定流程'),(.18,'agent','#159ca6','Agent 自主规划')]:
        values = [g['arms'][arm] for g in summary['groups']]
        ys = np.array([100*v['accuracy'] for v in values])
        errors = np.array([[100*(v['accuracy']-v['wilson95'][0]) for v in values],
                           [100*(v['wilson95'][1]-v['accuracy']) for v in values]])
        bars = ax.bar(x+offset,ys,width=.32,color=color,label=label,zorder=3)
        ax.errorbar(x+offset,ys,yerr=errors,fmt='none',ecolor='#536d76',elinewidth=1.1,capsize=4,zorder=4)
        for bar, v in zip(bars,values):
            label_y = max(bar.get_height(), 100*v['wilson95'][1]) + 3
            ax.text(bar.get_x()+bar.get_width()/2,label_y,f"{100*v['accuracy']:.2f}%\n({v['correct']}/12)",ha='center',va='bottom',fontsize=11,color='#12384d',weight='bold')
    labels = [f"{g['name']}\n(n=12；Δ={g['delta_accuracy_pp']:+.2f} pp)" for g in summary['groups']]
    ax.set_xticks(x,labels)
    ax.set_ylim(0,117)
    ax.set_yticks(np.arange(0,101,20))
    ax.set_ylabel('整项核验结论准确率（%）')
    ax.grid(axis='y',color='#e6ecee',zorder=0)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2,frameon=False)
    fig.suptitle('固定流程 vs. Agent 自主规划：36例事实核验小试验',x=.08,ha='left',fontsize=19,weight='bold',color='#07394e',y=.98)
    fig.text(.08,.90,'同一批案例、同一 DeepSeek 模型、相同工具集及预算上限；覆盖全部六类公开数据',fontsize=11,color='#486873')
    fig.text(.08,.035,'误差线：Wilson 95%区间。每类仅12例，单例改变8.33个百分点。\n参考为文件/JSON/PNG/CSV可复算事实；不是路演科学缺陷准确率，也不是官方八类定位F1。',fontsize=9,color='#486873')
    fig.subplots_adjust(left=.09,right=.97,bottom=.19,top=.78)
    for suffix in ['png','svg','pdf']:
        fig.savefig(figures/f'01_固定流程与Agent准确率.{suffix}',dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12.8,5.8))
    for ax,key,title in [(axes[0],'tool_calls','累计工具调用次数'),(axes[1],'http_attempts','最终轨迹HTTP尝试次数（含有记录重试）')]:
        for offset,arm,color,label in [(-.18,'fixed','#9bb5bf','固定流程'),(.18,'agent','#159ca6','Agent 自主规划')]:
            vals=[g['arms'][arm][key] for g in summary['groups']]
            bars=ax.bar(x+offset,vals,width=.32,color=color,label=label)
            ax.bar_label(bars,padding=4)
        ax.set_xticks(x,[g['name'] for g in summary['groups']],fontsize=10)
        ax.set_title(title,fontsize=12,pad=15)
        ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    axes[0].legend(frameon=False)
    fig.suptitle('自主规划的调用代价：同样的准确率需同时比较资源消耗',fontsize=16,color='#07394e')
    fig.text(.06,.025,'两条流程均允许最多4次工具调用、5次模型决策；固定流程按预定顺序取证，模型给最终判断。',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.91))
    fig.savefig(figures/'02_工具与模型调用量.png',dpi=220)
    plt.close(fig)
    overall = {arm:sum(r['exact_correct'] for r in normalized['rows'] if r['arm']==arm) for arm in ['fixed','agent']}
    agent_better = [g for g in summary['groups'] if g['delta_accuracy_pp']>0]
    if agent_better:
        conclusion = 'Agent在'+'、'.join(g['name'] for g in agent_better)+'上取得更高的本次任务准确率；其余分组见表。'
    else:
        conclusion = '本次小试验没有证明Agent自主规划比完整固定流程更准确。保留真实结果，不沿用示意图中的+11.11个百分点。'
    errors = collections.Counter(str((c.get('error'),c.get('http_code'))) for c in all_receipts.values() if c.get('error'))
    positive_na = collections.Counter(v for t in tasks.values() for v in t['expected'].values())
    rows_md = '\n'.join(f"| {r['任务类型']} | 12 | {r['固定流程正确数']}/12（{100*r['固定流程准确率']:.2f}%） | {r['Agent正确数']}/12（{100*r['Agent准确率']:.2f}%） | {r['准确率差_百分点']:+.2f} |" for r in table)
    metrics_md = '\n'.join(f"| {r['任务类型']} | {percent(r['固定流程Precision'])} / {percent(r['固定流程Recall'])} / {percent(r['固定流程F1'])} | {percent(r['AgentPrecision'])} / {percent(r['AgentRecall'])} / {percent(r['AgentF1'])} |" for r in table)
    example_sections=[]
    examples=[]
    for group in summary['groups']:
        ids=[key for key in sorted(pairs) if tasks[key]['group']==group['group']]
        discordant=[key for key in ids if parsed[(key,'agent')]['exact_correct'] != parsed[(key,'fixed')]['exact_correct']]
        key=(discordant or ids)[0]
        task=tasks[key]
        example={'task_id':key,'group':group['name'],'family':task['source']['family'],'reference':task['expected'],
                 'fixed_checks':parsed[(key,'fixed')]['checks'],'agent_checks':parsed[(key,'agent')]['checks'],
                 'fixed_tools':[t['name'] for t in pairs[key]['fixed']['tools']],
                 'agent_tools':[t['name'] for t in pairs[key]['agent']['tools']],
                 'current_header':task['header'],'raw_measurements':task['facts'],'source':task['source']}
        examples.append(example)
        fixed_states={k:v.get('status') for k,v in example['fixed_checks'].items()}
        agent_states={k:v.get('status') for k,v in example['agent_checks'].items()}
        example_sections.append(f"### {group['name']}：{task['source']['family']}，{key}\n\n参考状态：`{json.dumps(task['expected'],ensure_ascii=False)}`。固定流程：`{json.dumps(fixed_states,ensure_ascii=False)}`；Agent：`{json.dumps(agent_states,ensure_ascii=False)}`。\n\n固定取证顺序：{' → '.join(example['fixed_tools'])}。Agent取证顺序：{' → '.join(example['agent_tools'])}。原始字段、测量值、解释和路径见`典型案例.json`及完整轨迹。")
    (root/'典型案例.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
    now = dt.datetime.now().astimezone().isoformat(timespec='seconds')
    report = f'''# 固定流程与Agent自主规划：36例小规模对照实测

生成时间：{now}。

**{conclusion}** 总体固定流程答对{overall['fixed']}/36，Agent答对{overall['agent']}/36。

## 主结果

准确率要求任务列出的全部检查状态都正确；多工具任务任一子检查错即整项错。接口最终失败或没有可读完整答案保留在分母。对两边一律取第一次完整终止检查答案，仅机械归一化JSON外层action/type写法，不改任何pass/fail状态，不依据解释替模型纠错；原始协议成功率另外保留。Δ为Agent减固定流程，单位为百分点。

| 任务类型 | 案例数 | 固定流程准确率 | Agent准确率 | Δ（百分点） |
|---|---:|---:|---:|---:|
{rows_md}

![核验准确率对比](figures/01_固定流程与Agent准确率.png)

## 精确率、召回率与F1

以下按已知pass/fail机械检查单元计数；阳性是fail。非适用参考不进入这组P/R/F1；主准确率仍要求非适用状态正确。共有{sum(positive_na.values())}个参考状态，构成为{dict(positive_na)}。不是独立科学目标定位成绩。

| 任务类型 | 固定流程 P / R / F1 | Agent P / R / F1 |
|---|---|---|
{metrics_md}

多工具组中，Agent有1例在最终回答时接口超时，因此整项准确率为11/12；该例参考全部为通过或不适用，没有阳性缺陷，未回答不会增加这组阳性F1的漏检数。所以Agent该组F1虽为100%，仍有任务失败，不能理解成整体执行完全正确。固定流程的1例错误则是EPB数值几何检查的误报，计入假阳性。主结果以包含全部任务的整项准确率为准。

## 如何抽样与运行

- 从既有公开test清单抽取36个不同同源组，覆盖AIA、HMI、CME、EPB、DMSP、ULF。三类任务各12例，每类数据各2例。按初版机械参考分层抽样；随后复核CME的隐式类别格式并修正适用判断，最终正负构成以上表为准。没有修改原始数据。
- 这是有意平衡的病例小试验，不代表真实污染率；模型输入不含P档位、clean、参考答案或训练目标。
- 样本来自项目此前已经使用过的公开test清单，本轮未据结果调节权重或分类阈值；不称为全新盲测。
- 直接判定：核验原始文件存在性。前置依赖：先读取摘要/记录或配对/测量，再核验声明。多工具联合：联合检查文件、声明、文件名及坐标/时间数值边界。
- 固定流程采用强基线：按任务预先安排充分取证顺序，再由同一个模型给最终判断。Agent通过JSON动作自主选工具、顺序和参数并给最终判断。工具本身不返回参考答案。
- 两边使用deepseek-v4-flash-ascend1，temperature=0；相同四个只读工具，最多4次工具调用、5次模型决策，每次最多900输出Token。实际调用量分别报告，不要求强基线故意浪费预算。
- 相同的是预算上限，实际模型调用次数并不匹配，因此对照反映两种完整执行方式，不能把差异全部归因于纯规划算法。
- 这次新增了隔离的小型评测驱动，复用现有DeepSeek接口和读取/配对/测量功能；没有替换前端生产规划器，也没有重跑21个专家。LLM读取的是结构化测量与标注事实，并未进行原始图像语义识别。

## 参考与可复核性

原始文件存在性、PNG尺寸、JSON摘要/明细一致性、CSV采样及记录数、数值坐标/时间边界构成参考。36例的文件哈希和72个参考状态已由另一实现重新读取原始JSON、Pillow解码及CSV复核，全部一致。独立实现不等于新增了科学缺陷人工真值。

逐案例的原文件路径、SHA256、参考状态、两条流程完整回答和取证轨迹见`36例逐任务配对台账.csv`及`run_v3/runs/`。原始数据、权重、密钥没有打包。

CME原始事件可省略逐事件类别字段，这不能直接当成类别错误。本轮修正了这项判定约定，对全部4个受影响任务的两条流程都重新执行，其他32个任务原回答原样保留。修正前的结果也保存在包内；修正不是只重试答错项。详见`run_v3/schema_amendment.json`。

## 限流与消耗

初始4并发实测遭遇HTTP429，故改为2并发、请求间隔至少8秒，429冷却60秒，单逻辑请求最多3次HTTP尝试。所有已有终止回答（包括答错）固定保留，仅接口失败时恢复原上下文，没有按准确率挑选重试。原始`run_v1`日志保留。

按响应ID去重，全部阶段（含CME修正前结果）有回执的累计Token为{all_usage['total_tokens']:,}，其中输入{all_usage['prompt_tokens']:,}、输出{all_usage['completion_tokens']:,}。最终72条配对轨迹自身记录{summary['total_usage']['total_tokens']:,} Token。全部阶段历史错误回执分类为{dict(errors)}。初始任务中止时最多4个在途请求可能没有回执，相关消耗未知，未伪造补值。

![调用量](figures/02_工具与模型调用量.png)

## 典型任务回查

{chr(10).join(example_sections)}

## PPT怎样写

可写：“在覆盖六类公开数据的36例事实核验小试验中，对固定流程与DeepSeek自主工具规划进行同样本配对比较，完整记录准确率、取证过程与调用消耗。”表中数字可直接替换示意图，样本量统一改为每类n=12。

每类12例，一例影响8.33个百分点，图附Wilson 95%区间，仅作当前小任务集的不确定性展示。逐组配对精确检验记录在`displayed_summary.json`，不以小样本点估计直接宣称显著提升。不能把此结果扩大为3867任务、路演八类定位准确率或官方关键缺陷召回。若结果没有提升，就不写“自主规划显著增强准确率”；可据实说明工具选择、解释和审计能力。
'''
    (root/'实验结果与PPT填写说明.md').write_text(report,encoding='utf-8')
    verification = {'status':'pass','paired_tasks':36,'arms':72,'distinct_source_groups':36,
                   'independent_source_audit':'pass','primary_accuracies_independently_recomputed':True,
                   'reference_fields_absent_from_user_messages':True,'overall_correct':overall,
                   'syntax_only_normalization':'Identical first-complete-check extraction; no verdict repair',
                   'all_recorded_stages_deduplicated_token_usage':all_usage,
                   'note':'Evidence-qualified accuracy in raw runner is a stricter witness-citation diagnostic; primary chart uses exact task-state accuracy.'}
    (root/'delivery_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    for source in Path(__file__).parent.glob('*.py'):
        (root/'code').mkdir(exist_ok=True)
        shutil.copy2(source,root/'code'/source.name)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix != '.zip' and p.name not in ['checksums.json'] and '__pycache__' not in p.parts]
    manifest = {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'checksums.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=root/'SpaceQA_Agent自主规划_36例配对实验_20261105.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/'checksums.json']:
            z.write(p,p.relative_to(root))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(json.dumps({'status':'pass','overall_correct':overall,'archive_bytes':archive.stat().st_size,
                      'groups':[{k:g[k] for k in ['name','delta_accuracy_pp']} for g in summary['groups']]},ensure_ascii=False))

if __name__ == '__main__':
    main()
