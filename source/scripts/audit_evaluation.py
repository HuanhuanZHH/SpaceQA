"""Independent audit of native evaluate output; standard-library only."""
import json,csv,hashlib,sys,math
from pathlib import Path

def audit(root):
 root=Path(root);card=json.loads((root/'scorecard.json').read_text());complete=json.loads((root/'complete.json').read_text());findings=[json.loads(x) for x in (root/'findings.jsonl').read_text().splitlines()];samples=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()];logs=[json.loads(x) for x in (root/'run.log.jsonl').read_text().splitlines()];index=list(csv.DictReader((root/'evidence_index.csv').open()));errors=[]
 def check(condition,message):
  if not condition:errors.append(message)
 run=complete['runId'];check(card['runId']==run,'Card runId mismatch');check(len(samples)==complete['records']==card['sampleCount'],'Denominator mismatch');ids=[f['findingId'] for f in findings];check(len(ids)==len(set(ids)),'Duplicate finding ID');check(set(ids)=={e['findingId'] for e in index},'Evidence ID coverage mismatch');check({f['findingId'] for f in card['findings']}==set(ids),'Card/findings mismatch')
 byid={e['evidenceId']:e for e in index}
 for e in index:
  p=(root/e['path']).resolve();check(p.is_relative_to(root.resolve()),'Evidence escapes output');check(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==e['contentHash'],'Evidence hash mismatch '+e['evidenceId'])
 for f in findings:
  for k in ('datasetId','datasetVersion','status','evidenceRef','affectedScope','criticality','basis','notApplicableReason','confirmedOrPending','severity','runId'):check(k in f,'Missing finding field '+k)
  check(f['runId']==run,'Finding runId mismatch');check(all(i in byid for i in f['evidenceRef']),'Dangling evidenceRef')
  for k in ('metric','value','unit','tolerance'):check(k in f['severity'],'Missing severity '+k)
 bad={f['sampleId'] for f in findings if f.get('scoreEligible') and f.get('findingStage')=='confirmed'}
 expected=100*(1-len(bad)/len(samples)) if samples else None
 check(expected is not None and abs(expected-card['observedTechnicalScore'])<1e-9,'Technical score cannot be reproduced')
 check(len(complete['experts'])==21,'Not all experts present')
 for e in complete['experts'].values():check(e['fresh_forward_samples']==len(samples)-complete['missing_data'] and not e['cache_reuse_from_training_or_demo'],'Fresh forward coverage mismatch')
 for row in logs:
  for k in ('runId','dirId','checkId','elapsedMs','resourcePeak','tokenUsage','retryCount','status'):check(k in row,'Missing log field '+k)
  check(row['runId']==run,'Log runId mismatch')
 check({d['id'] for d in card['dimensions']}=={'completeness','annotationQuality','sampleCorrespondence','coverageDistribution','traceability','usageConditions'},'Six dimensions missing')
 return dict(status='pass' if not errors else 'fail',runId=run,samples=len(samples),findings=len(findings),confirmedSamples=len(bad),score=expected,errors=errors)
if __name__=='__main__':
 result=audit(sys.argv[1]);print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(result['status']!='pass')
