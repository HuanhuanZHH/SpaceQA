from pathlib import Path
import csv
import json
import shutil
import subprocess

SRC=Path('D:/OPC/outputs/chat_insights_20261105/fresh_12_packages')
OUT=Path('D:/OPC/outputs/chat_insights_refresh_20261105');OUT.mkdir(parents=True,exist_ok=True)
for path in SRC.iterdir():
    if path.is_file():shutil.copyfile(path,OUT/path.name)
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
audit=load('independent_validation.json');index=load('report_index.json');ground=load('grounded_complete.json');compare=load('comparison_to_previous.json')
names={'L0-01':'数据文件缺失','L2-G':'边界与位置偏差','L2-P':'类别标注错误/契约冲突','L2-T':'时段起止时刻疑点','L2-F':'元数据尺寸矛盾','L3-06':'漏标','L3-07':'虚标','L3-MIS':'覆盖不足/覆盖过度'}
local=[]
for code,name in names.items():
    r=audit['perClass'][code]
    local.append(dict(defectCode=code,defectName=name,**r,
        locationMeaning='questioned input event interval, not independent true endpoint' if code=='L2-T' else
        'questioned input object location, not independent spurious truth' if code=='L3-07' else
        'direct file/field evidence or model comparison region; confirmation is a separate state',independentVisualEndpointTruthSamples=0))
write('八类检出与定位汇总.csv',local)
(OUT/'eight_class_localization_table.json').write_text(json.dumps(local,ensure_ascii=False,indent=2)+'\n')
duration={r['package']:r['seconds'] for r in ground['packages']}
rank=[];ordered=sorted(index['reports'],key=lambda r:(r['detectedRatio'],r['name']))
for i,r in enumerate(ordered,1):
    rank.append(dict(displayRank=i,competitionRank=1+sum(o['detectedRatio']<r['detectedRatio'] for o in ordered),package=r['name'],family=r['family'],
        foundSamples=r['foundSamples'],detectedRatio=r['detectedRatio'],confirmedSamples=r['confirmedSamples'],candidateOnlySamples=r['candidateOnlySamples'],totalSamples=r['totalSamples'],
        actualFreshRgbForwards=r['inferenceCoverage']['actualFreshImageForwards'],imageStageSeconds=duration[r['name']],
        scoreMeaning='sort by detected confirmed-or-candidate sample ratio, lower first; not official true pollution proportion'))
write('十二包发现样本与排序.csv',rank)
(OUT/'twelve_package_ranking.json').write_text(json.dumps(rank,ensure_ascii=False,indent=2)+'\n')
allkeys=list(dict.fromkeys(k for row in compare['packages'] for k in row))
write('新旧版差异明细.csv',[{k:row.get(k,0) for k in allkeys} for row in compare['packages']])
(OUT/'说明.md').write_text('''# 本轮重新推理材料

- `12包重新推理与八类定位核验报告.md`：范围、八类表、每包结果、实际耗时、旧新差异与限制。
- `八类真实案例分析.md`：八类真实样例及逐例讲解；可按包名、sampleId在前端检索。
- `十二包发现样本与排序.csv`：第二个统计主项是找到的去重样本，发现占比越小越靠前；这不是有真值保证的真实污染率。
- `八类检出与定位汇总.csv`：区分规则确认、模型候选、位置指认和仍仅样本级。
- `independent_validation.json`：57,910项来源与统计一致性核验；不是TP/FP/FN真值评测。
- `comparison_to_previous.json` / `新旧版差异明细.csv`：278条新增RGB几何/目标位置键，与其余输入对象/时段/字段关联分开统计。
- `report_index.json`：服务器当前报告来源与版本，所含路径是服务器绝对路径。

本轮10,332张图片确实重新做RGB前向；八类融合和规则也重新执行。21个上游专家是已验证预测缓存复用，没有重新前向。ULF起止时段是被质疑的输入事件指认，CSV数值t轴不含独立UTC原点，因此没有得到独立真实端点。规则确认只是直接核验的文件/字段问题，不是用真值算出的TP。模型候选不进入确认技术分。官方P/R/F1仍缺独立路演真值，当前文件中保留为null。

服务器索引：`/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105/report_index.json`。旧报告与本轮v1来源均保留。
''',encoding='utf-8')
# Sync the final source corrections without replaying any completed inference.
here=Path(__file__).parent
subprocess.run(['scp',str(here/'finish_refresh.py'),'icarus0:/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105/'],check=True)
print(str(OUT))
