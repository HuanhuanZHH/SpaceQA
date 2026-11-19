import json,sys
from pathlib import Path

def render(root):
 root=Path(root);p=root/'03_评价规范.yaml';x=json.loads(p.read_text());x['documentationEdition']='20261110-repair-final';x['runtimeRegistryVersion']='DEL4-executable-20261110-v2';x['documentationScope']='Status/provenance errata for delivered specification; inference decision code and thresholds unchanged from frozen R4 image. Embedded image registry preserves its build-time snapshot.'
 if 'currentResults' in x:x['historicalResults20261109']=x.pop('currentResults')
 x['currentExecution']={'directory':'repair_20261110/runtime_candidate/full_repeat_final','singleCheckReplay':'repair_20261110/single_check_replay','formalReplay':'repair_20261110/formal_replay','sourceRunIdPolicy':'Native runId per package; historical results are explicitly separated','completion':'See actual complete.json and independent comparison, not this design document'}
 x['completion']['reason']='All 18 checks have execution entry points. Execution coverage is distinct from evidence completeness; pending scientific or contractual prerequisites remain in applicable denominators.'
 for d in x['dimensions']:
  if d.get('dimId') in ['coverageDistribution','traceability','usageConditions']:d['currentScoreReason']='Executable checks; current absence of target protocol or verified evidence means undetermined, not unimplemented.'
 for c in x['checks']:
  c['programRole']='Execute the bound function; preserve four states, real measurements, provenance and absent-evidence reasons. Historical counts are not rerun measurements.'
  for key in ['log_event','logEvent']:c[key]='started/completed/failed; actual runId+dirId+checkId, UTC timestamp and monotonic elapsed time'
  for key in ['currentConfirmedRecords','currentCandidateRecords']:
   if key in c:c['historical'+key[0].upper()+key[1:]]=c.pop(key)
  if 'weight_basis' in c:c['weight_basis']='Participant-defined equal within-dimension weight. It is not prescribed by the cited papers or official requirements.'
 x['scoring']['newDimensionsBoundsActivation']='Only report algebraic bounds when utility definitions, applicability, units and weights are frozen. Missing prerequisites keep range null; bounds are not statistical confidence intervals.'
 x['scope']['registeredCheckIds']=[c['checkId'] for c in x['checks']]
 x['scoring']['newDimensionsWeightStatus']='C41-C50 use equal check weights within each dimension; this is a declared participant policy. It does not supply the six-dimensional composite approval.'
 x['scoring']['newDimensionsProposed']['total']=x['scoring']['sixDimensionComposite']
 x['scoring']['examplesApprovedComposite']={'purpose':'Arithmetic test fixture, not actual dataset score or real approval','dimensionScores':[100,90,80,70,60,50],'weights':[1,1,1,1,1,1],'resultWhenApprovalAndEvidenceConditionsHold':75,'resultWhenAnyPrerequisiteMissing':None}
 x['linkage']['nativeExports']='04_评分卡/*.json → findings.jsonl → evidence_index.csv → actual evidence bytes; run.log.jsonl links native runId/dirId/checkId' if isinstance(x.get('linkage'),dict) else x.get('linkage')
 payload=json.dumps(x,ensure_ascii=False,indent=2)+'\n';tmp=p.with_suffix('.new');tmp.write_text(payload);tmp.replace(p)
 lines=['# 数据集质量评价规范','','文档修订：2026-11-10；执行注册表：DEL4-executable-20261110-v2。此处修订实现状态与追溯说明，不更改冻结 R4 镜像的推理算法、路由或阈值。镜像内的规范副本是构建时快照，正式交付规范以本 MD/YAML 为准。','','## A. 规则体系','','| checkId | 名称 | 维度 | 判定对象与条件 | 容差依据 |','|---|---|---|---|---|']
 def cell(v):return str(v).replace('|',' / ').replace('\n',' ')
 for c in x['checks']:lines.append('| '+' | '.join(cell(v) for v in [c['checkId'],c['name'],','.join(c['dimensionIds']),c.get('criterion',c.get('decision')),c.get('tolerance')])+' |')
 lines+=['','四状态：通过、不通过、待判定、不适用。目标、校准、参考或权限证据缺失时为待判定；只有可核验的非适用前提才能设为不适用。确认规则记录、模型候选和独立科学真值分别保留。C31 文件存在性和 C35 声明尺寸一致性是确定性计量；科学定位容差没有统一的任意常数。文献支持原则，不为本项目指定 TV、空间或时间阈值。','','## B. 计算公式','','技术分：`100 × (1 − 确认可计分的去重样本数 / 全部登记标注样本数)`。同一样本跨对象、同类或跨类缺陷总分只扣一次；候选不扣分。登记缺文件样本留在分母。','','维度分：按冻结的统计单位先求检查效用 q；适用检查全部有有效证据和已冻结判据时，`S_d = 100 × Σ(w_i q_i) / Σw_i`；否则保留 null 与待判原因。分母为零时不强算 0 或 100。当前 C31-C38 的技术维度仍按样本去重，不能与包级 C41-C50 记录直接相加。','','六维合成：`S = Σ(w_d S_d) / Σw_d`，必须提供六个正有限权重、依据、实际批准人，且六维都可计量。任一前提不足则 null。权重不是官方或文献指定；不能假造批准。算术边界例：六维 100、90、80、70、60、50，假设等权且满足前提，结果为 75；缺任一前提返回 null。','','技术排序按未舍入技术分降序，采用竞赛并列名次；跨载荷只作技术观察。关键缺陷单列，不额外重复扣分。完整六维分、技术观察、S6 限定统计严格区分。','','## C. 技术落实','','| checkId | 执行工具 | 程序与函数 | 参数来源 | 技能版本 |','|---|---|---|---|---|']
 for c in x['checks']:lines.append('| '+' | '.join(cell(v) for v in [c['checkId'],c['tool'],c['program'],c['threshold_location'],c['skill']['version']])+' |')
 lines+=['','适用性判定在 analyze_sample/measure；行为边界由规则及缺证状态控制；数值计量由程序负责；模型只给候选；分数合成由 aggregate/composite 完成；日志与证据由 fine_scan、formal_checks 和 publish_run 记录。','','C31-C38 单查适配器包含共用分析器开销，单独实测耗时不能称为纯模型耗时。正式日志 ts 是实际 timestamp 的字段映射；tokenUsage 指外部计费 API，局部模型 token 未计量。','','## D. 与其他交付物的对应','','checkId 关联 findings.jsonl、evidence_index.csv 和 run.log.jsonl；没有缺陷记录的检查仍保留执行状态及检查证据。包级资料检查证据另列 check_evidence_index.csv，不能伪造问题记录填补数量。记录通过原生 runId、dirId 和 checkId 回查。','','## 完整规范数据','','以下 JSON 与同名 YAML 是同一对象，保留逐条适用边界、单位、参考文献、缺值语义及历史来源；历史结果不冒充本轮新前向。','','```json',json.dumps(x,ensure_ascii=False,indent=2),'```','']
 p=root/'03_评价规范.md';tmp=p.with_suffix('.new');tmp.write_text('\n'.join(lines));tmp.replace(p)
 return x
if __name__=='__main__':render(sys.argv[1])
