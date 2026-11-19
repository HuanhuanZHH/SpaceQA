"""Validate actual published files; field contract audit, not official JSON Schema."""
import argparse,csv,hashlib,json,math,re,subprocess
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def run(root,full=True):
 root=Path(root);errors=[];checks=[]
 def check(ok,label):
  checks.append({'check':label,'pass':bool(ok)})
  if not ok:errors.append(label)
 cards=[json.load(open(p)) for p in sorted((root/'04_评分卡').glob('*.json'))]
 findings=rows(root/'06_问题清单/findings.jsonl');index=list(csv.DictReader((root/'07_证据索引/evidence_index.csv').open(encoding='utf-8-sig')));logs=rows(root/'08_运行日志/run.log.jsonl')
 ids=[f['findingId'] for f in findings];byid={e['evidenceId']:e for e in index};check(len(ids)==len(set(ids)),'globally_unique_finding_ids');check(len(index)==len(byid),'globally_unique_evidence_ids');check(set(ids)=={e['findingId'] for e in index},'finding_evidence_coverage')
 required={'datasetId','datasetVersion','status','evidenceRef','affectedScope','criticality','basis','notApplicableReason','confirmedOrPending','severity','runId','checkId','findingId','dirId'}
 for f in findings:
  check(required<=f.keys(),'finding_fields:'+f['findingId']);check({'metric','value','unit','tolerance'}<=f['severity'].keys(),'severity_fields:'+f['findingId']);check(bool(f['evidenceRef']) and all(e in byid for e in f['evidenceRef']),'evidence_ref:'+f['findingId'])
 for e in index:
  p=(root/'07_证据索引'/e['path']).resolve();check(p.is_relative_to(root.resolve()) and p.is_file() and sha(p)==e['contentHash'],'hash:'+e['evidenceId'])
 for c in cards:
  fs=[f for f in findings if f['dirId']==c['dirId']];bad={f['sampleId'] for f in fs if f.get('scoreEligible')};expected=100*(1-len(bad)/c['sampleCount']);check(abs(c['totalScore']-expected)<1e-9,'score:'+c['dirId']);check(len(bad)==c['confirmedProblemSamples'],'confirmed_count:'+c['dirId']);check(len({f['sampleId'] for f in fs})==c['foundProblemSamples'],'found_count:'+c['dirId']);check({f['findingId'] for f in fs}=={f['findingId'] for f in c['findings']},'card_ids:'+c['dirId']);check(bool(c['runId']) and all(f['runId']==c['runId'] for f in fs),'native_run:'+c['dirId']);check(len(c['dimensions'])==6,'six_dimensions:'+c['dirId'])
  for d in c['dimensions']:check({'dimId','score','range','grade','meaning'}<=d.keys(),'dimension_fields:'+c['dirId']+':'+d['dimId'])
  check(c['rankGlobal']==1+sum(x['totalScore']>c['totalScore'] for x in cards),'rank:'+c['dirId']);check(c['rankInDataset']==1+sum(x['datasetId']==c['datasetId'] and x['totalScore']>c['totalScore'] for x in cards),'group_rank:'+c['dirId'])
  p=root/'04_评分卡'/c['sampleCheckEvidence']['path'];check(p.is_file() and sha(p)==c['sampleCheckEvidence']['contentHash'],'sample_check_hash:'+c['dirId'])
  for d in c['dimensions']:
   for e in d.get('checkResults',[]):
    p=root/'04_评分卡'/e['evidenceRef']['path'];check(p.is_file(),'formal_check_ref:'+c['dirId']+':'+e['checkId'])
  cl=[x for x in logs if x['runId']==c['runId'] and x['dirId']==c['dirId']];check({f'C{i}' for i in list(range(31,39))+list(range(41,51))}<={x['checkId'] for x in cl},'18_check_log_coverage:'+c['dirId'])
 for i,e in enumerate(logs):check({'ts','runId','datasetId','dirId','event','checkId','status','elapsedMs','resourcePeak','tokenUsage','retryCount'}<=e.keys(),'log_fields:'+str(i))
 fields=['rank','dirId','datasetId','datasetVersion','totalScore','dimensionScores','rankInDataset','rankGlobal','groupId','comparisonBasis','retainSubRanking']
 with (root/'05_排序表/ranking_global.csv').open(encoding='utf-8-sig') as f:
  reader=csv.DictReader(f);rank=list(reader);check(reader.fieldnames==fields,'ranking_columns')
 check({r['dirId'] for r in rank}=={c['dirId'] for c in cards},'ranking_packages')
 ev=next((root/'11_展示材料').glob('证据表_*.csv'));road=next((root/'11_展示材料').glob('路演结果_*.csv'))
 with ev.open(encoding='utf-8-sig') as f:
  reader=csv.DictReader(f);rr=list(reader);check(reader.fieldnames==['findingId','checkId','ruleVersion','dirId','objectId','fieldPath','locatorType','verdict','detail'],'evidence_display_columns');check({r['findingId'] for r in rr}==set(ids),'display_finding_ids')
 with road.open(encoding='utf-8-sig') as f:
  reader=csv.DictReader(f);rr=list(reader);check(reader.fieldnames==['datasetId','datasetVersion','dirId','rankInDataset','rankGlobal','totalScore','mainDefectTypes','evidence'],'roadshow_columns');check({r['dirId'] for r in rr}=={c['dirId'] for c in cards},'display_packages')
 if full:
  check(len(cards)==12 and sum(c['sampleCount'] for c in cards)==24782,'12_packages_24782_records')
  spec=json.load(open(root/'03_评价规范.yaml'));md=(root/'03_评价规范.md').read_text();embedded=json.loads(md.split('```json\n')[-1].split('\n```')[0]);check(embedded==spec,'MD_YAML_same_object');check({x['checkId'] for x in spec['checks']}=={f'C{i}' for i in list(range(31,39))+list(range(41,51))},'18_check_registry');check(all(x.get('program') and x.get('tool') and x.get('threshold_location') for x in spec['checks']),'18_implementation_bindings')
  fixed=['00_MANIFEST.yaml','SHA256SUMS.txt','01_方案规划.md','02_部署与使用说明.md','03_评价规范.md','03_评价规范.yaml','09_未完成事项说明.md','10_演示视频.mp4','05_排序表/ranking_global.csv','05_排序表/ranking_explanation.md','06_问题清单/findings.jsonl','06_问题清单/findings_report.md','07_证据索引/evidence_index.csv','08_运行日志/run.log.jsonl','08_运行日志/system_env.txt','08_运行日志/seeds.yaml','11_展示材料/运行记录.txt','image/LTH_image.tar','image/Dockerfile','image/docker-compose.yml']
  fixed += ['04_评分卡/'+c['dirId']+ext for c in cards for ext in ('.json','.html')];fixed += ['agent/'+x+'.yaml' for x in ('manifest','tools','skills','rules','memory','plugins')];fixed += [str(next((root/'11_展示材料').glob(pattern)).relative_to(root)) for pattern in ['技术方案_*.pptx','证据表_*.csv','路演结果_*.csv']]
  check(len(fixed)==53,'53_fixed_positions');check(all((root/p).is_file() for p in fixed),'all_fixed_files_exist')
  comp=json.load(open(root/'repair_20261110/runtime_candidate/repeat_comparison/comparison.json'));check(comp['status']=='pass' and comp['samplesPerRound']==24782 and len(comp['packages'])==12,'two_fresh_rounds_pass')
  probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name','-of','json',str(root/'10_演示视频.mp4')]));check(float(probe['format']['duration'])<=300 and probe['streams'][0]['codec_name']=='h264','video_H264_under_300s')
  manifest=json.load(open(root/'00_MANIFEST.yaml'));check({'deliverables','selfCheck','declarations','roadshowMaterials'}<=manifest.keys(),'manifest_fields')
  for d in manifest['deliverables']:
   for p in d['paths']:check((root/p).exists(),'manifest_path:'+p)
 return {'status':'pass' if not errors else 'fail','auditScope':'v2.3 documented fixed-file and field contract audit; official JSON Schema not supplied; scientific correctness is outside scope','packages':len(cards),'samples':sum(c['sampleCount'] for c in cards),'findings':len(findings),'checks':len(checks),'errors':errors,'passedChecks':sum(x['pass'] for x in checks)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--export-only',action='store_true');p.add_argument('--output');a=p.parse_args();result=run(a.root,not a.export_only);text=json.dumps(result,ensure_ascii=False,indent=2);print(text)
 if a.output:Path(a.output).write_text(text+'\n')
 raise SystemExit(result['status']!='pass')
