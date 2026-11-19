import json,csv,hashlib,collections,platform,subprocess,sys,importlib.metadata
from pathlib import Path
from spaceqa_agent import formal_checks

def finish(root,out,card,ledgers,run_id):
 def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 evidence=out/'evidence';evidence.mkdir();findings=[];index=[];n=len(ledgers)
 for sample in ledgers:
  for f in sample['findings']:
   eid='EV-'+hashlib.sha256(f['findingId'].encode()).hexdigest()[:24];f.update(runId=run_id,datasetId=sample['family'],datasetVersion=sample.get('sourceVersion') or 'unspecified',status='fail' if f.get('scoreEligible') else 'undetermined',evidenceRef=[eid],affectedScope={'unit':'registered_sample','numerator':1,'denominator':n},criticality='critical' if f['defectType']=='L0-01' and f.get('scoreEligible') else 'general',basis={'ruleVersion':card['version'],'source':f.get('source'),'scientificGroundTruth':False},notApplicableReason=None,confirmedOrPending='confirmed' if f.get('scoreEligible') else 'pending')
   sev=f.get('severity') or {};f['severity']=sev
   for key in ('metric','value','unit','tolerance'):sev.setdefault(key,None)
   if any(sev[k] is None for k in ('metric','value','unit','tolerance')):sev['unavailableReason']='No calibrated physical severity or tolerance; model probabilities are not degree measurements'
   p=evidence/(eid+'.json');label=Path(sample['labelPath']);write(p,{'finding':f,'labelSnapshot':json.loads(label.read_text()),'labelSha256':sample['labelSha256'],'inputMount':'/input','originalDataPath':sample['dataPath'],'capturedBy':'publish_run-20261110','runId':run_id})
   index.append(dict(evidenceId=eid,findingId=f['findingId'],path=str(p.relative_to(out)),contentHash=hashlib.sha256(p.read_bytes()).hexdigest(),capturedBy='publish_run-20261110',runId=run_id));findings.append(f)
 with (out/'findings.jsonl').open('w') as fp:
  for f in findings:fp.write(json.dumps(f,ensure_ascii=False)+'\n')
 with (out/'evidence_index.csv').open('w') as fp:
  writer=csv.DictWriter(fp,fieldnames=['evidenceId','findingId','path','contentHash','capturedBy','runId']);writer.writeheader();writer.writerows(index)
 contract_path=root/'target_protocol.json';contract=json.loads(contract_path.read_text()) if contract_path.is_file() else {}
 formal=formal_checks.run(root,out/'formal_checks',formal_checks.IDS,contract,run_id)
 with (out/'run.log.jsonl').open('a') as f:f.write((out/'formal_checks/run.log.jsonl').read_text())
 byid={r['checkId']:r for r in formal}
 groups={'coverageDistribution':['C41','C42','C43','C44'],'traceability':['C45','C46','C47'],'usageConditions':['C48','C49','C50']}
 for d in card['dimensions']:
  if d['id'] not in groups:continue
  checks=[byid[c] for c in groups[d['id']]];app=[c for c in checks if c['status']!='not_applicable'];done=[c for c in app if c['status'] in ('pass','fail')]
  d.update(checkResults=checks,checkCoverage=len(done)/len(app) if app else None,score=100*sum(c['q'] for c in done)/len(done) if app and len(done)==len(app) else None,reason='Evidence-gated formal contract checks; missing prerequisites are undetermined')
 comp=formal_checks.composite({d['id']:d.get('score') for d in card['dimensions']},contract.get('composite',{}));card.update(runId=run_id,fullQualityScore=comp['score'],fullQualityScoreReason=comp['reason'],findings=findings,evidenceIds=[r['evidenceId'] for r in index]);write(out/'scorecard.json',card)
 deps={}
 for name in ('torch','numpy','scipy','transformers','scikit-learn','Pillow','setuptools','ultralytics'):
  try:deps[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:deps[name]=None
 write(out/'system.json',{'runId':run_id,'python':sys.version,'platform':platform.platform(),'dependencies':deps,'seed':20261104,'bModuleSeed':20261109,'sampling':'none; all selected input records evaluated','offline':True})
