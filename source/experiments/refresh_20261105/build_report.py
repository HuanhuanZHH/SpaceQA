from pathlib import Path
import json

OUT=Path('D:/OPC/outputs/chat_insights_20261105/fresh_12_packages')
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
audit=load('independent_validation.json');complete=load('complete.json');index=load('report_index.json');comparison=load('comparison_to_previous.json');ground=load('grounded_complete.json')
names={'L0-01':'数据文件缺失','L2-G':'边界与位置偏差','L2-P':'类别标注错误/类别契约冲突','L2-T':'时段起止时刻疑点','L2-F':'元数据尺寸矛盾','L3-06':'漏标','L3-07':'虚标','L3-MIS':'覆盖不足/覆盖过度'}
text=['# 11月5日路演12包重新推理与定位核验','',
'12包共24,782条记录已重新执行确定性规则和冻结八类融合。AIA与EPB的10,332张现存可读图片，全部在GPU1以batch16完成新的原始RGB模型前向；未使用旧定位缓存。ULF为CSV时序，不计入像素模型前向数。21个上游分类专家的预测经过哈希校验后复用，本轮没有重跑这21个专家，也没有训练或付费API调用。','',
'合计1,690个样本有规则可核验的已确认问题，1,872个样本只有待复核疑点；两者去重后共3,562个发现样本。272条缺数据记录全部保留。路演未提供独立的八类逐样本与定位真值，因此本轮不产生官方P/R/F1。','',
'## 八类覆盖与定位层级','',
'同一样本可以属于多类，下面各行不可直接相加。候选置信度不等于准确率；有输入指认位置不等于科学上已证实缺陷。','',
'| 缺陷 | 检出样本 | 规则确认 | 待核候选 | 有位置指认 | 仍仅样本级 | 位置含义 |',
'|---|---:|---:|---:|---:|---:|---|']
meaning={'L0-01':'缺失预期文件路径','L2-G':'原图像素区域；相对独立RGB模型比较','L2-P':'具体类别字段/标注记录；规则仅确认声明契约冲突','L2-T':'被质疑的输入事件区间；独立真实端点定位仍为0','L2-F':'具体尺寸字段及PNG实测值','L3-06':'独立RGB预测的未匹配目标区域','L3-07':'被质疑的输入标注对象位置；未匹配不等于真实虚标','L3-MIS':'输入对象区域及相对RGB预测的覆盖测量'}
for code,name in names.items():
    r=audit['perClass'][code]
    text.append(f"| {name} | {r['foundSamples']:,} | {r['confirmedSamples']:,} | {r['candidateSamples']:,} | {r['locatedSamples']:,} | {r['sampleOnlyCandidateSamples']:,} | {meaning[code]} |")
text+=['','ULF CSV实际表头为`t,H,D,Z,T`，共1,479个相关样本提取了t的首末原始数值。CSV没有独立UTC原点；不能把输入事件时段当真实端点，也不能据此声称真实端点偏差已定位。',
'','## 各包结果与本轮像素阶段耗时','',
'本轮像素阶段秒数包括该包解码、校验、模型、区域比较与聚合，不是纯GPU内核耗时。ULF对应数值只是空像素任务路由与记账耗时；不代表完整时序模型运行耗时。','',
'| 包 | 样本 | 规则确认 | 仅候选 | 去重发现 | 发现占比 | 新RGB前向 | 像素阶段秒 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
by={r['package']:r for r in ground['packages']}
for r in sorted(index['reports'],key=lambda r:(r['detectedRatio'],r['name'])):
    g=by[r['name']]
    text.append(f"| {r['name']} | {r['totalSamples']:,} | {r['confirmedSamples']:,} | {r['candidateOnlySamples']:,} | {r['foundSamples']:,} | {100*r['detectedRatio']:.2f}% | {g['executed']:,} | {g['seconds']:.2f} |")
t=comparison['timing']
text+=['',f"规则与八类冻结融合：{t['classificationSeconds']:.2f}秒。新RGB全量阶段：{t['gpuStageStartedAt']}—{t['gpuStageLastLogAt']}，墙钟{t['gpuStageWallSeconds']:.2f}秒（6分52.83秒）；各包阶段之和{t['imagePackageSecondsSum']:.2f}秒。额外对象/时段指认后处理{t['postprocessingSeconds']:.2f}秒。三个活动阶段合计约{t['classificationSeconds']+t['gpuStageWallSeconds']+t['postprocessingSeconds']:.2f}秒，约7分40秒，未计人工修正、等待及后续CSV上下文整理。该耗时不包括21个基础专家重新推理，不能作为完整21专家冷启动的3小时验收。",'',
'## 相对上一份报告新增了什么','',
'以样本、缺陷类别、输入记录指针和坐标构成位置键，与保留的旧报告逐条比较：新增2,168个唯一候选位置键，原有候选位置键未删除。','',
'| 新增位置键类型 | 条数 | 正确解释 |','|---|---:|---|',
'| 新RGB几何/目标比较疑点 | 278 | 边界237、漏标41；是新的模型疑点，仍非科学确认 |',
'| 受质疑输入对象的虚标关联 | 310 | 使用新RGB未匹配证据，把疑点关联到输入对象；不是独立缺陷框 |',
'| 受质疑输入事件区间关联 | 1,531 | 补充输入事件记录上下文；不是新的真实起止端点 |',
'| 类别字段关联 | 49 | 补充具体字段指认；不是新增科学类别真值 |','',
'本轮总计保留1,308条RGB几何/目标候选实例、310条对象关联、1,531条时段关联和50条类别字段候选实例。新增位置不等于新增真实问题；新八类口径与旧粗类疑点数不能直接比较为F1提升。','',
'## 核验与交付路径','',
f"独立复核{audit['checks']:,}项全部通过：覆盖全部标签哈希、数据存在性、输出有限值、各包/各类阶段计数、坐标数值、候选禁止计分、规则分数复算。该复核证明输出与来源一致，不替代科学准确率评测。",'',
'服务器报告索引：`/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105/report_index.json`。','',
'原始数据、上一轮输出和本轮v1来源均保留；v2仅为ULF补充数值坐标上下文，并通过原子替换发布索引。']
(OUT/'12包重新推理与八类定位核验报告.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
if (OUT/'eight_class_case_examples.json').is_file():
    cases=load('eight_class_case_examples.json');lines=['# 八类真实输出案例与讲解','',
    '案例来自本轮真实输出。每例均可通过包名与sampleId在前端检索；模型候选与规则确认分开说明。','']
    explanations={
        'L0-01':'程序读取标签的文件引用并在实际文件系统核验，对应文件不存在时保留预期路径。这是直接可核验的文件缺失，未凭模型猜测补图。',
        'L2-G':'RGB模型独立看原图预测目标区域，再与输入标注进行匹配，测量IoU和中心位移。当前框标出被比较的输入对象；真实科学边界仍需独立参考确认。',
        'L2-P':'程序比较具体对象类别与本任务声明的允许类别集合。冲突可在输入JSON字段核验；它证明类别契约不一致，不自动证明图片在科学上属于哪一类。',
        'L2-T':'冻结时段分类头筛出此样本后，程序将疑点关联到输入事件的起止记录。数值t轴缺独立UTC原点，因此区间是复核入口，不是已经证实的真实错误端点。',
        'L2-F':'程序读取PNG头中的真实宽高，与JSON整图尺寸字段直接比较。这里只检查整图字段，避免将目标框宽高误当图像宽高。',
        'L3-06':'RGB模型在图像中预测出一个与输入对象没有匹配的区域，并满足候选阈值。该框来自独立图像预测，是漏标待核区域；模型检出不等于已证实真实目标。',
        'L3-07':'八类分类器提出虚标疑点，新RGB组件与输入对象匹配后仍无支持。本框来自被质疑的输入标注，不是模型预测的独立缺陷框；目标未检出也可能是模型漏检，所以不能自动扣分。',
        'L3-MIS':'把匹配对象的输入标注面积与独立RGB预测面积比较，报告相对覆盖率、面积比等。覆盖不足或过度的方向来自相对预测的测量，科学正确边界仍待核。'}
    for code in names:
        c=cases[code];f=c['finding'];loc=f['locator'];lines += [f"## {names[code]}（{code}）",'',
            f"包：`{c['package']}`；样本：`{c['sampleId']}`。",'',
            f"状态：{'规则确认、可计技术分' if f['findingStage']=='confirmed' else '模型/上下文候选、不可计确认分'}。",'',
            explanations[code],'',f"本例输出：{f['detail']}",'',
            '定位：`'+json.dumps(loc,ensure_ascii=False)+'`','',
            '测量：`'+json.dumps(f.get('measurements'),ensure_ascii=False)+'`','']
    (OUT/'八类真实案例分析.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(str(OUT/'12包重新推理与八类定位核验报告.md'))
