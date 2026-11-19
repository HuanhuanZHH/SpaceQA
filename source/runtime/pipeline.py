"""Current-input all-expert evaluation, localization and auditable reports."""
from pathlib import Path
import argparse,types,json,time,sys,hashlib,importlib.util,collections,html
import numpy as np
import uuid,random,datetime,resource,os
import compact_infer as ci
R=ci.R
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as fp:
  for b in iter(lambda:fp.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def jsonl(p,rows):
 with Path(p).open('w') as fp:
  for r in rows:fp.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def localize(records,ledgers,pred,score,out,batch):
 import torch
 from PIL import Image
 sys.path.insert(0,str(R/'eight_class_20261104/code'))
 import localizer as l
 import ground_roadshow as ground
 from spaceqa_agent import quality_ledger as q
 h=l.helper();ann,_=h.annotation_module(R);policy=ci.read(R/'eight_class_20261104/candidate_comparison_v1/deployment_policy.json')
 models={};count=0;started=time.monotonic()
 for family in h.FAMILIES:
  indices=[i for i,r in enumerate(records) if r['family']==family and not r['input_missing'] and pred[i,[1,5,6,7]].any()]
  if not indices:continue
  route=policy['selected'][family]
  if route not in models:
   path=R/'eight_class_20261104/localizer_v2'/(route+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
   model=l.network(saved['width']);model.load_state_dict(saved['state_dict'],strict=True);models[route]=(model.cuda().eval(),saved['size'],sha(path))
  model,size,weight_hash=models[route];th=ci.read(policy['localizer_threshold_files'][route])['thresholds'][family]
  for off in range(0,len(indices),batch):
   ix=indices[off:off+batch];items=[]
   for i in ix:
    with Image.open(records[i]['data_path']) as im:rgb=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
    items.append(dict(rgb=rgb))
   probs,masks,presence=ground.predict_batch(model,items,h.FAMILIES.index(family),th['pixel'],th['presence'],False,use_presence=route=='presence_gated')
   for k,i in enumerate(ix):
    r=records[i];ledger=ledgers[i];width,height=ledger['imageSize'];objects=ann.extract_annotation(ci.read(r['label_path']),family)
    geometry,errors=h.extract_geometry(objects['objects'],width,height)
    components=h.instances(masks[k],probs[k],min_pixels=18,max_components=64)
    signals=dict(model_candidates=[code for b,code in enumerate(ci.CLASSES) if pred[i,b] and not any(f['defectType']==code and f.get('scoreEligible') for f in ledger['findings'])],probabilities=score[i].tolist(),classes=ci.CLASSES)
    row=dict(r,package=ledger['dirId'])
    proposed,matched=ground.candidates(row,ledger,signals,components,geometry,errors,h,weight_hash)
    q.attach_candidates(ledger,proposed);count+=len(proposed)
  print(json.dumps(dict(stage='localized',family=family,screened=len(indices),candidate_regions=count)),flush=True)
 del models;torch.cuda.empty_cache()
 return dict(seconds=time.monotonic()-started,region_candidates=count,independent_scientific_truth=False)
def main(argv=None):
 ap=argparse.ArgumentParser(description='Full offline all-21-expert evaluation; no historical prediction cache')
 ap.add_argument('--input',type=Path,required=True);ap.add_argument('--family',choices=ci.FAMILIES,required=True);ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--batch',type=int,default=16);ap.add_argument('--qbatch',type=int,default=32);ap.add_argument('--limit',type=int)
 ap.add_argument('--release-profile',choices=['public-v1','roadshow-b-v1'],default='public-v1');ap.add_argument('--no-localization',action='store_true')
 args=ap.parse_args(argv);out=args.output.resolve();root=args.input.resolve()
 if out.exists() or out==root or out.is_relative_to(root):raise ValueError('Use a new output folder outside input')
 out.mkdir(parents=True);start=time.monotonic()
 run_id='LTH-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
 random.seed(20261104);np.random.seed(20261104)
 import torch
 torch.manual_seed(20261104);torch.cuda.manual_seed_all(20261104)
 def event(stage,status,**kw):
  row=dict(runId=run_id,datasetId=args.family,dirId=root.name,checkId=kw.pop('checkId',None),event=stage,status=status,timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsedMs=(time.monotonic()-start)*1000,resourcePeak=dict(processMaxRssKiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),tokenUsage=0,retryCount=0,seed=20261104,**kw)
  with (out/'run.log.jsonl').open('a') as fp:fp.write(json.dumps(row,ensure_ascii=False)+'\n')
 event('evaluate','started',family=args.family,input=str(root))
 from spaceqa_agent import quality_ledger as q,chat_scoring
 portable=module('compact_preprocess',Path('/opt/spaceqa/portable_experts.py'));lock=ci.read('/opt/spaceqa/preprocess_lock.json')
 prep_args=types.SimpleNamespace(input=root,family=args.family,selection_json=None,limit=args.limit,release_profile=args.release_profile)
 prepared=out/'prepared';prepared.mkdir();records,_=portable.prepare(prep_args,lock,prepared)
 preparation_seconds=time.monotonic()-start
 ledgers=[];broad=[]
 contracts={'AIA_AR_sample':['AR'],'HMI_AR_sample':['AR'],'CME_C2_sample':['CME'],'EPB_seg_sample':['EPB'],'DMSP_SSUSI':['PCA','TPA','unclear']}
 from spaceqa_agent.engine import png_header
 for r in records:
  if r.get('features') is None:r['features']=[0.]*256
  pure=r.get('prior_observation',{}).get('confirmed_bits',[0,0,0]);old=r.get('rule_bits',[0,0,0]);broad.append([pure[0] or old[0],pure[1] or old[1],pure[2]])
  if not r.get('label_path'):continue
  sample=dict(id=r['id'],package=root.name,family=r['family'],data_path=r['data_path'],label_path=r['label_path'],data_exists=Path(r['data_path']).is_file(),data_sha256=r.get('data_sha256'),label_sha256=r['label_sha256'],data_hash_verification='fresh SHA256')
  if sample['data_exists'] and Path(r['data_path']).suffix.lower()=='.png':
   m=png_header(Path(r['data_path']));sample['image_size']=[m['width'],m['height']]
  annotation=ci.read(r['label_path']);ledger=None
  for code,check in q.CHECKS.items():
   check_start=time.monotonic();event('check','started',checkId=check,sampleId=r['id'])
   partial=q.analyze_sample(sample,annotation,allowed_labels=contracts.get(args.family),contract_source='Frozen task-family vocabulary; technical contract, not independent scientific truth')
   event('check','completed',checkId=check,sampleId=r['id'],checkElapsedMs=(time.monotonic()-check_start)*1000,timingScope='separate adapter invocation including common analyzer overhead')
   if ledger is None:ledger=dict(partial,checks={},findings=[])
   ledger['checks'][code]=partial['checks'][code];ledger['findings'].extend(f for f in partial['findings'] if f['defectType']==code)
  ledgers.append(ledger)
 if len(ledgers)!=len(records):raise ValueError('Orphan data require separate rule-only inventory; preserve records in prepared/inventory.jsonl')
 fine=np.zeros((len(records),8),bool)
 for i,l in enumerate(ledgers):
  for finding in l['findings']:
   if finding['defectType'] in ci.CLASSES:fine[i,ci.CLASSES.index(finding['defectType'])]=True
 event('experts','started')
 pp,timings=ci.infer(records,out/'fresh_experts',args.batch,args.qbatch)
 event('experts','completed',expertTimings=timings)
 pred,scores,_=ci.classify(records,pp,np.array(broad,bool),fine)
 scores=np.where(fine,1.,scores);localization={'disabled':True}
 if not args.no_localization:localization=localize(records,ledgers,pred,scores,out,args.batch)
 if args.release_profile=='roadshow-b-v1':
  from portable_b import apply
  apply(records,ledgers,pred,out)
 for ledger in ledgers:
  ledger['runId']=run_id
  for finding in ledger['findings']:finding['runId']=run_id
 classified=[]
 for i,r in enumerate(records):
  classified.append(dict(runId=run_id,sampleId=r['id'],family=r['family'],classes=ci.CLASSES,prediction=pred[i].tolist(),probabilities=scores[i].tolist(),rule=fine[i].tolist(),
    labelPath=r['label_path'],dataPath=r['data_path'],labelSha256=r['label_sha256'],dataSha256=r.get('data_sha256'),inputSha256=r.get('input_sha256'),
    candidatePollution=[float(scores[i,b]) if pred[i,b] else 0. for b in range(8)]))
 jsonl(out/'predictions.jsonl',classified);jsonl(out/'samples.jsonl',ledgers)
 card=q.aggregate(ledgers);dimensions=chat_scoring.evaluate(ledgers,root,out/'six_dimension_checks.jsonl.gz')
 card['runId']=run_id;dimensions['runId']=run_id
 ci.write(out/'scorecard.json',card);ci.write(out/'six_dimensions.json',dimensions)
 pollution=[dict(code=code,index=float(np.where(pred[:,b],scores[:,b],0).sum()/len(records)),flagged_samples=int(pred[:,b].sum())) for b,code in enumerate(ci.CLASSES)]
 ci.write(out/'pollution.json',dict(denominator=len(records),classes=pollution,formula='sum(probability where frozen prediction is positive) / all records; rule positives have weight 1',lower_is_better=True,calibrated_physical_proportion=False))
 duration=time.monotonic()-start
 receipt=dict(runId=run_id,status='complete',version='0.6.1-repair2',family=args.family,input=str(root),records=len(records),seconds=duration,preparation_seconds=preparation_seconds,
  expert_seconds=sum(v['seconds'] for v in timings.values()),localization=localization,experts=timings,fresh_expert_count=len(timings),
  under_3h=duration<=10800,hardware='CUDA device '+__import__('torch').cuda.get_device_name(0),seed=20261104,
  model_policy_sha256=sha(R/'eight_class_20261104/fusion_v3_missing4/frozen_policy.json'),
  missing_data=sum(r['input_missing'] for r in records),preparation_failures=sum(r['status']=='preparation_failed' for r in records),
  scope='Current-input technical quality observations; localized model candidates remain distinct from confirmed rule defects',external_api_calls=0)
 from publish_run import finish
 finish(root,out,card,ledgers,run_id)
 ci.write(out/'complete.json',receipt)
 rows=''.join('<tr><td>'+html.escape(x['code'])+'</td><td>'+str(x['flagged_samples'])+'</td><td>'+f"{x['index']:.4f}"+'</td></tr>' for x in pollution)
 dims=''.join('<li>'+html.escape(d['name'])+': '+(f"{d['displayScore']:.2f}" if d['displayScore'] is not None else 'Evidence unavailable')+'</li>' for d in dimensions['dimensions'].values())
 (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SpaceQA report</title><style>body{font:17px system-ui;max-width:1100px;margin:45px auto;color:#16324f}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd8e4;padding:12px}h1{color:#123e69}</style><h1>SpaceQA · '+html.escape(root.name)+'</h1><p>Full fresh inference · 21 experts · '+str(len(records))+' records · '+f'{duration:.1f}'+' seconds</p><h2>Eight defect groups</h2><table><tr><th>Defect</th><th>Flagged samples</th><th>Pollution index</th></tr>'+rows+'</table><h2>Six scoped technical dimensions</h2><ul>'+dims+'</ul><p>Probabilistic candidates and confirmed rule findings are separate. Full provenance is in samples.jsonl, predictions.jsonl and six_dimension_checks.jsonl.gz.</p>',encoding='utf-8')
 event('evaluate','completed',records=len(records))
 print(json.dumps(receipt),flush=True);return 0
if __name__=='__main__':main()
