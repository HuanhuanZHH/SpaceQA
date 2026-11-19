"""Current-input fine checks. Does not invent model/scientific confirmation."""
import json,time,uuid,datetime,hashlib,resource
from pathlib import Path
from . import priors,quality_ledger as q
from .engine import png_header
FAMILIES=['AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample']
CHECKS={v:k for k,v in q.CHECKS.items()}
def run(root,output,enabled,seed):
 root=Path(root).resolve();output=Path(output).resolve()
 if not root.is_dir() or output.exists() or output.is_relative_to(root):raise ValueError('Existing input and new external output required')
 family=next((f for f in FAMILIES if root.name.startswith(f)),None)
 if family is None:raise ValueError('Explicit family-prefixed package required')
 output.mkdir(parents=True);run_id='RUN-'+uuid.uuid4().hex;start=time.monotonic();results=[]
 def event(check,status,elapsed=0,reason=None):
  row=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),runId=run_id,datasetId=family,dirId=root.name,checkId=check,status=status,elapsedMs=elapsed,resourcePeak={'processMaxRssKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'whole process high-water mark, not per-check delta'},tokenUsage=0,tokenUsageBasis='deterministic checks; no API calls',retryCount=0,reason=reason,seed=seed)
  with (output/'run.log.jsonl').open('a') as fp:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
 event(None,'inventory_started')
 try:records=priors.load_package(root,family)
 except Exception as e:event(None,'failed',reason=str(e));raise
 index={}
 for check in sorted(enabled):
  started=time.monotonic();event(check,'started')
  try:
   for r in records:
    data=Path(r['data_path']);label=Path(r['label_path']);sample=dict(id=root.name+':'+r['sample_id'],package=root.name,family=family,data_path=str(data),label_path=str(label),data_exists=data.is_file(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest())
    if data.is_file() and data.suffix.lower()=='.png':
     wh=png_header(data);sample['image_size']=[wh['width'],wh['height']]
    result=q.analyze_sample(sample,r['label'],allowed_labels=None)
    result['runId']=run_id
    result['checks']={c:v for c,v in result['checks'].items() if q.CHECKS[c]==check}
    result['findings']=[f for f in result['findings'] if f['checkId']==check]
    for f in result['findings']:f['runId']=run_id
    if sample['id'] not in index:index[sample['id']]=result;results.append(result)
    else:index[sample['id']]['checks'].update(result['checks']);index[sample['id']]['findings'].extend(result['findings'])
   event(check,'completed',elapsed=(time.monotonic()-started)*1000,reason='Measured separate check invocation, includes shared analyzer overhead; missing reference remains undetermined')
  except Exception as e:
   event(check,'failed',elapsed=(time.monotonic()-started)*1000,reason=str(e));raise
 for name,rows in [('sample_results.jsonl',results),('findings.jsonl',[f for r in results for f in r['findings']])]:
  with (output/name).open('w') as fp:
   for row in rows:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
 receipt=dict(runId=run_id,checks=sorted(enabled),samples=len(results),scope='mechanical checks plus explicit pending scientific checks; no GPU/model invocation',elapsedMs=(time.monotonic()-start)*1000,seed=seed)
 (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt
