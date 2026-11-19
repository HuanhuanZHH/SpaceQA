"""Export one independently audited native round into DEL-5 through DEL-10."""
import csv,collections,hashlib,html,json,shutil,sys
from pathlib import Path
from audit_evaluation import audit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows,fields):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main(src,out,expected_packages=12):
 src=Path(src);out=Path(out)
 if out.exists():raise ValueError('New export output required')
 roots=sorted(p.parent for p in src.glob('*/complete.json'));assert len(roots)==expected_packages
 audits=[audit(p) for p in roots];assert all(a['status']=='pass' for a in audits),audits
 for d in ['04_评分卡','05_排序表','06_问题清单','07_证据索引/evidence','08_运行日志/check_results','07_证据索引/check_evidence','11_展示材料']:(out/d).mkdir(parents=True)
 cards=[];allfind=[];index=[];logs=[];envs=[];provenance=[];check_index=[];evidence_bodies=[]
 for root in roots:
  c=json.load(open(root/'scorecard.json'));c.update(datasetId=c['family'],datasetVersion='v34',totalScore=c['observedTechnicalScore'],totalScoreMeaning='当前确认规则缺陷率派生的技术分，越高越好；不是完整六维综合分',sourceRun=str(root),sourceRunHash=sha(root/'complete.json'));c['dimensionScores']={d['id']:d.get('score') for d in c['dimensions']};delivery=Path(__file__).resolve().parents[2];c['sourceRunRelative']='../'+str(root.relative_to(delivery)) if root.is_relative_to(delivery) else None;c['keyIssues']=[]
  for d in c['dimensions']:
   d['range']=[d['score'],d['score']] if d.get('score') is not None else None
   d['dimId']=d['id'];d['grade']={'pass':'通过','fail':'不通过','undetermined':'待判定','not_applicable':'不适用'}.get(d.get('state'),'待判定');d['meaning']=d.get('reason','仅当前检查范围；缺证据不计正式分')
   for check in d.get('checkResults',[]):
    source=root/'formal_checks'/check['evidenceRef']['path'];dest=out/'07_证据索引/check_evidence'/c['dirId']/source.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
    check['evidenceRef']['path']='../07_证据索引/'+str(dest.relative_to(out/'07_证据索引'));check_index.append(dict(runId=c['runId'],dirId=c['dirId'],checkId=check['checkId'],path=str(dest.relative_to(out)),contentHash=sha(dest)))
  dest=out/'08_运行日志/check_results'/(c['dirId']+'.jsonl');shutil.copyfile(root/'samples.jsonl',dest);c['sampleCheckEvidence']={'path':'../08_运行日志/check_results/'+dest.name,'contentHash':sha(dest),'scope':'Actual sample-level C31-C38 results'}
  refs={e['evidenceId']:e for e in csv.DictReader((root/'evidence_index.csv').open())};fs=[json.loads(x) for x in (root/'findings.jsonl').read_text().splitlines()]
  for f in fs:
   f['annotationSourceVersion']=f['datasetVersion'];f['datasetVersion']='v34';f['status']='不通过' if f.get('scoreEligible') else '待判定';f['exportTool']='export_native_release-20261110'
   if f['criticality']=='critical':c['keyIssues'].append(f['findingId'])
   for eid in f['evidenceRef']:
    source=root/refs[eid]['path'];e=json.load(open(source));e.update(finding=f,exportProvenance={'sourceEvidenceSha256':sha(source),'sourceRunId':c['runId'],'tool':'export_native_release-20261110'});p=out/'07_证据索引/evidence'/(eid+'.json');write(p,e);evidence_bodies.append(e);index.append(dict(evidenceId=eid,findingId=f['findingId'],path=str(p.relative_to(out/'07_证据索引')),contentHash=sha(p),capturedBy='publish_run-20261110; export_native_release-20261110',runId=c['runId'],dirId=c['dirId'],locatorType=f.get('locator',{}).get('type'),locator=json.dumps(f.get('locator'),ensure_ascii=False)))
  c['findings']=fs;c['evidenceIds']=[eid for f in fs for eid in f['evidenceRef']];c['foundProblemSamples']=len({f['sampleId'] for f in fs});c['technicalObservations']=json.load(open(root/'six_dimensions.json'));cards.append(c);allfind+=fs
  logs += [json.loads(x) for x in (root/'run.log.jsonl').read_text().splitlines()];envs.append(json.load(open(root/'system.json')));provenance.append(dict(dirId=c['dirId'],runId=c['runId'],completeHash=sha(root/'complete.json'),source=str(root)))
 ordered=sorted(cards,key=lambda c:(-c['totalScore'],c['dirId']))
 ranking=[];road=[]
 for c in ordered:
  c['rankGlobal']=1+sum(x['totalScore']>c['totalScore'] for x in cards);c['rankInDataset']=1+sum(x['family']==c['family'] and x['totalScore']>c['totalScore'] for x in cards);c['rank']=c['rankGlobal'];c['groupId']=c['family'];c['comparisonBasis']='same frozen image, fresh round; observedTechnicalScore only; fullQualityScore evidence-gated; runId='+c['runId']
  write(out/'04_评分卡'/(c['dirId']+'.json'),c)
  rows=''.join('<tr id="'+html.escape(f['findingId'])+'"><td>'+html.escape(f['checkId'])+'</td><td>'+html.escape(f['sampleId'])+'</td><td>'+html.escape(f['status'])+'</td><td><a href="../07_证据索引/evidence/'+f['evidenceRef'][0]+'.json">'+html.escape(f['findingId'])+'</a></td></tr>' for f in c['findings'])
  doc='<!doctype html><meta charset="utf-8"><title>'+html.escape(c['dirId'])+'</title><style>body{font:16px system-ui;margin:30px;color:#16324f}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccd8e4}pre{white-space:pre-wrap}</style><h1>'+html.escape(c['dirId'])+'</h1><p>原生 runId：'+c['runId']+'</p><p>登记样本 '+str(c['sampleCount'])+'；确认问题样本 '+str(c['confirmedProblemSamples'])+'；确认和候选并集 '+str(c['foundProblemSamples'])+'；技术分 '+str(c['totalScore'])+'（越高越好）。</p><p>完整六维分：'+str(c['fullQualityScore'])+'；'+html.escape(c['fullQualityScoreReason'])+'</p><pre>'+html.escape(json.dumps(c['dimensions'],ensure_ascii=False,indent=2))+'</pre><table><tr><th>检查</th><th>样本</th><th>状态</th><th>证据</th></tr>'+rows+'</table>'
  (out/'04_评分卡'/(c['dirId']+'.html')).write_text(doc)
  ranking.append({k:c[k] for k in ['rank','dirId','datasetId','datasetVersion','totalScore','rankInDataset','rankGlobal','groupId','comparisonBasis']}|{'dimensionScores':json.dumps(c['dimensionScores'],ensure_ascii=False),'retainSubRanking':'true'})
  road.append(dict(datasetId=c['datasetId'],datasetVersion=c['datasetVersion'],dirId=c['dirId'],rankInDataset=c['rankInDataset'],rankGlobal=c['rankGlobal'],totalScore=c['totalScore'],mainDefectTypes=';'.join(sorted({f['defectType'] for f in c['findings']})),evidence='原生runId='+c['runId']+'；登记'+str(c['sampleCount'])+'；确认样本'+str(c['confirmedProblemSamples'])+'；确认和候选并集'+str(c['foundProblemSamples'])+'；100*(1-确认去重样本/登记样本)，候选不扣分；完整六维分待证据；见04_评分卡/'+c['dirId']+'.json'))
 csvout(out/'05_排序表/ranking_global.csv',ranking,['rank','dirId','datasetId','datasetVersion','totalScore','dimensionScores','rankInDataset','rankGlobal','groupId','comparisonBasis','retainSubRanking'])
 (out/'05_排序表/ranking_explanation.md').write_text('# 技术观察排序\n\n总分为确认规则缺陷率派生的技术分，非完整六维分。未舍入分值降序、竞赛并列排名。跨载荷仅作技术观察。\n\n| 名次 | dirId | 技术分 | runId |\n|---|---|---|---|\n'+'\n'.join('| '+str(c['rank'])+' | '+c['dirId']+' | '+str(c['totalScore'])+' | '+c['runId']+' |' for c in ordered))
 with (out/'06_问题清单/findings.jsonl').open('w') as f:
  for row in allfind:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 counts=collections.Counter((f['checkId'],f['status']) for f in allfind)
 (out/'06_问题清单/findings_report.md').write_text('# 本轮问题清单\n\n每条问题关联原生runId。类别词表为技术契约，模型候选不等于独立科学真值。\n\n| 检查 | 状态 | 记录数 |\n|---|---|---|\n'+'\n'.join('| '+k[0]+' | '+k[1]+' | '+str(n)+' |' for k,n in sorted(counts.items()))+'\n\n## 关键缺陷\n\n'+'\n'.join('- '+f['dirId']+' / '+f['findingId']+' / '+f['sampleId'] for f in allfind if f['criticality']=='critical'))
 csvout(out/'07_证据索引/evidence_index.csv',index,['evidenceId','findingId','locatorType','locator','contentHash','capturedBy','path','runId','dirId'])
 with (out/'07_证据索引/evidence/evidence_records.jsonl').open('w') as f:
  for row in evidence_bodies:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 csvout(out/'07_证据索引/check_evidence_index.csv',check_index,['runId','dirId','checkId','path','contentHash'])
 families={c['dirId']:c['family'] for c in cards}
 for row in logs:
  row['ts']=row['timestamp'];row['datasetId']=row.get('datasetId') or families.get(row['dirId']);row.setdefault('event','check_'+row['status']);row['tokenUsageBasis']='external_api_only; local model input tokens not instrumented'
 with (out/'08_运行日志/run.log.jsonl').open('w') as f:
  for row in logs:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 write(out/'08_运行日志/system.json',{'runs':envs,'tokenUsageDefinition':'external metered API tokens; zero means no external API. Local Qwen forward token count was not instrumented; forward sample count is in each complete.json.'});write(out/'08_运行日志/seeds.yaml',{'globalSeed':20261104,'bModuleSeed':20261109,'sampling':'none','sourceRuns':provenance})
 (out/'08_运行日志/system_env.txt').write_text((out/'08_运行日志/system.json').read_text())
 csvout(out/'11_展示材料/路演结果_LTH团队_SpaceQA智能体.csv',road,['datasetId','datasetVersion','dirId','rankInDataset','rankGlobal','totalScore','mainDefectTypes','evidence'])
 evrows=[dict(findingId=f['findingId'],checkId=f['checkId'],ruleVersion=f.get('ruleVersion','eight-class-ledger-1.0'),dirId=f['dirId'],objectId=(f.get('locator') or {}).get('objectId',''),fieldPath=(f.get('locator') or {}).get('path',''),locatorType=(f.get('locator') or {}).get('type',''),verdict=f.get('verdict','undetermined'),detail=json.dumps(dict(runId=f['runId'],defectType=f['defectType'],sampleId=f['sampleId'],stage=f.get('findingStage'),scoreEligible=f.get('scoreEligible'),locator=f.get('locator'),severity=f['severity'],evidenceRef=f['evidenceRef']),ensure_ascii=False)) for f in allfind]
 csvout(out/'11_展示材料/证据表_LTH团队_SpaceQA智能体.csv',evrows,['findingId','checkId','ruleVersion','dirId','objectId','fieldPath','locatorType','verdict','detail'])
 dimrows=[]
 for dim in ['completeness','annotationQuality','sampleCorrespondence','coverageDistribution','traceability','usageConditions']:
  selected=[(c,d) for c in cards for d in c['dimensions'] if d['id']==dim];dimrows.append('| '+dim+' | '+str(sum(d.get('confirmedSamples',0) for c,d in selected))+' | '+str(sum(d.get('score') is None for c,d in selected))+' / 12 |')
 with (out/'06_问题清单/findings_report.md').open('a') as f:f.write('\n\n## 维度汇总\n\n同一样本可涉及多个维度，下表不可相加为全局样本数。后三维是包级目标协议检查，缺材料保持待判定，完整执行证据见 check_evidence_index.csv。\n\n| 维度 | 确认问题样本数 | 分数待判定包数 |\n|---|---|---|\n'+'\n'.join(dimrows))
 summary=dict(samples=sum(c['sampleCount'] for c in cards),confirmed=sum(c['confirmedProblemSamples'] for c in cards),found=sum(c['foundProblemSamples'] for c in cards),findings=len(allfind),confirmedRecords=sum(bool(f.get('scoreEligible')) for f in allfind),candidateRecords=sum(not f.get('scoreEligible') for f in allfind),runs=provenance,audits=audits)
 write(out/'export_summary.json',summary);write(out/'cards_summary.json',[{k:c[k] for k in ['dirId','sampleCount','confirmedProblemSamples','foundProblemSamples','totalScore','rankInDataset','rankGlobal','runId']} for c in ordered]);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main(*sys.argv[1:])
