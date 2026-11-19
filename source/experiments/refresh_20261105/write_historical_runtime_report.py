from pathlib import Path
import csv
import json
OUT=Path('D:/OPC/outputs/runtime_benchmark_20261105')
d=json.loads((OUT/'historical_21expert_components.json').read_text(encoding='utf-8'));t=d['totals']
rows=[]
for p in d['packages']:
    row={k:p[k] for k in ['package','family','registeredSamples','modelEligibleSamples','missingInputs','rawPrepareSeconds','neural20Seconds','gbdtSeconds','expert21Seconds','sumRecordedPrepareAnd21Seconds','expert21SecondsPerRegisteredSample','expert21SecondsPerEligibleSample','prepareAnd21SecondsPerRegisteredSample','expert20ModelLoadIncluded','gbdtModelLoadSeconds','gbdtModelLoadIncluded']}
    row.update(hardware=' | '.join(p['neural20Hardware']),prepareTimerPath=p['prepareTimerSource']['path'],prepareTimerSha256=p['prepareTimerSource']['sha256'],
        gbdtTimerPath=p['gbdtTimerSource']['path'],gbdtTimerSha256=p['gbdtTimerSource']['sha256'],
        neural20TimerEvidence=json.dumps([{k:i[k] for k in ['route','seconds','hardware','modelLoadIncluded','predictionSerializationIncluded']}|dict(timerPath=i['timerSource']['path'],timerSha256=i['timerSource']['sha256']) for i in p['neural20TimerSources']],ensure_ascii=False))
    rows.append(row)
with (OUT/'historical_21expert_components.csv').open('w',encoding='utf-8-sig',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# 历史12包原始输入准备与21专家计时证据','',
'这是历史各阶段实测秒数的相加，**不是一次冷启动端到端墙钟测试**，不包含当前新增融合、定位与报告链的全流程。最终主表应使用本轮当前完整运行的实测。本文未启动GPU、未运行模型、未改服务，仅读取原计时回执和源码。','',
f"12包共{t['registeredSamples']:,}条注册样本，{t['modelEligibleSamples']:,}条模型可用输入，{t['missingInputs']}条缺数据。",'',
'| 阶段 | 历史实测相加秒数 | 分钟 |','|---|---:|---:|']
for title,key in [('原始输入准备（含解压/规则/视图/特征）','rawPrepareSeconds'),('20个神经专家（含各包模型加载）','neural20Seconds'),('GBDT逐包特征与推理（不含全局模型加载）','gbdtSeconds'),('21专家合计','expert21Seconds'),('准备＋21专家已记录阶段合计','sumRecordedPrepareAnd21Seconds')]:
    lines.append(f"| {title} | {t[key]:.3f} | {t[key]/60:.2f} |")
lines+=['',f"QLoRA一条路线合计{t['qloraSeconds']:.3f}秒（{t['qloraSeconds']/60:.2f}分钟）。按原batch32实现，原prepared输入基础的fresh20历史合计为{t['neural20Seconds']/60:.2f}分钟，不能承诺全部12包在1小时内完成。",'',
'## 各包实测阶段与摊销平均','',
'“每样本”是含模型加载的批量吞吐摊销，不是单条请求延迟。注册分母含缺数据记录；另外给出模型可用输入分母。','',
'| 包 | 注册/可用 | 准备秒 | 20模型秒 | GBDT秒 | 21专家秒 | 准备＋21秒 | 21/注册 秒/样本 | 21/可用 秒/样本 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in d['packages']:
    lines.append(f"| {p['package']} | {p['registeredSamples']}/{p['modelEligibleSamples']} | {p['rawPrepareSeconds']:.2f} | {p['neural20Seconds']:.2f} | {p['gbdtSeconds']:.2f} | {p['expert21Seconds']:.2f} | {p['sumRecordedPrepareAnd21Seconds']:.2f} | {p['expert21SecondsPerRegisteredSample']:.5f} | {p['expert21SecondsPerEligibleSample']:.5f} |")
lines+=['',f"全样本加权平均＝所有包阶段秒数之和÷对应样本数之和：21专家为**{t['expert21WeightedSecondsPerRegisteredSample']:.5f}秒/注册样本**，或**{t['expert21WeightedSecondsPerEligibleSample']:.5f}秒/可用样本**；准备＋21专家为**{t['prepareAnd21WeightedSecondsPerRegisteredSample']:.5f}秒/注册样本**。不是对12个包的平均值做简单平均。",'',
'## 计时范围、硬件与缺项','',
'20神经专家的时间从每条route开始，到CUDA同步结束，包含checkpoint/基座加载、模型构造、数据加载和前向。之后才写预测JSON，因此预测序列化没有包含在该marker秒数中。QLoRA使用Qwen3-VL-8B-Instruct基座＋已训练adapter，运行时NF4 4bit；初始推理batch32，OOM时减半。其余视觉骨干batch16、学生batch64。','',
'GBDT模型与vectorizer在包循环之前一次性加载，其耗时未单独记录；逐包marker含当前字段提取、树预测、JSON写出和校验。不能凭空补上加载时间。','',
'prepare计时含源归档校验与解压、文件及规则盘点、原始视图、标注对照图、256特征和prompt准备，不只是读manifest。','',
'20专家marker的硬件字段为`icarus1 A100; not official A10 acceptance`。准备和GBDT逐包marker未记精确CPU型号/主机；不能将A100记录当作官方A10验收。','',
'缺少的阶段：GBDT全局加载；神经预测序列化；跨阶段启动/排队间隔；当前HGB7、八类融合、最新localizer、最终报告与审计。这些不能从历史各阶段秒数恢复为一个总墙钟。','',
'## 证据','',
'JSON逐包嵌入原始prepare、20模型、GBDT计时回执，并包含各回执服务器绝对路径、SHA-256和原始字段。CSV保留同样的路径、哈希及逐模型证据列表。没有读取大模型权重或批量原始数据，不干扰当前GPU测量。','',
'主要计时源：','',
'- `/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/packages/<包>/preparation.json`','- `/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/packages/<包>/models/<route>.complete.json`','- `/staff/tangjiarui/OPC/inference_prior_20261102/gbdt/roadshow_predictions/<包>.complete.json`','',
'相关源码哈希：','']
for item in d['sourceCode']:lines.append(f"- `{item['path']}`：`{item['sha256']}`")
(OUT/'historical_21expert_components.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(str(OUT/'historical_21expert_components.md'))
