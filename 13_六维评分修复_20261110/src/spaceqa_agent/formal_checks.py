"""Evidence-gated DEL-4 checks; no inferred target, license, or scientific truth."""
import argparse,collections,datetime,hashlib,json,math,resource,time,uuid
from pathlib import Path
VERSION='formal-1.1-20261110-sixdim'
IDS=tuple('C'+str(i) for i in range(41,51))
DIMENSION_IDS=('completeness','annotationQuality','sampleCorrespondence','coverageDistribution','traceability','usageConditions')
GROUPS={'coverageDistribution':('C41','C42','C43','C44'),'traceability':('C45','C46','C47'),'usageConditions':('C48','C49','C50')}
def finite_number(value):
 return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def relative(root,name):
 p=(root/name).resolve()
 if not p.is_relative_to(root):raise ValueError('Evidence path escapes input root')
 return p
def measure(root,check,cfg,evidence_root=None):
 root=Path(root).resolve();evidence_root=Path(evidence_root).resolve() if evidence_root else root
 # A contract is a supplied versioned target; possession does not prove authority.
 if not cfg or not cfg.get('basis') or not cfg.get('version'):return {'status':'undetermined','reason':'Missing versioned target and basis'}
 if check=='C44' and cfg.get('splitTask') is False:return {'status':'not_applicable','reason':'Contract explicitly declares no split task'}
 path=relative(evidence_root,cfg.get('records',''))
 if not path.is_file():return {'status':'undetermined','reason':'Required evidence table unavailable'}
 rows=json.loads(path.read_text());rows=rows.get('records',rows) if isinstance(rows,dict) else rows
 if not isinstance(rows,list) or not rows:return {'status':'undetermined','reason':'Empty or invalid evidence denominator'}
 result={'inputHash':digest(path),'inputPath':str(path.relative_to(evidence_root)),'inputEvidenceRoot':'input' if evidence_root==root else 'sidecar'}
 if check in ('C41','C42'):
  targets=cfg.get('minimumCounts');field=cfg.get('field')
  if not targets or not field:return dict(result,status='undetermined',reason='Missing fixed required strata/cells and minimum counts')
  if any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in targets.values()):raise ValueError('minimumCounts must be positive integers')
  if any(field not in r or 'sampleId' not in r for r in rows):return dict(result,status='undetermined',reason='Missing sample ID or trusted stratum/cell')
  seen=collections.defaultdict(set)
  for r in rows:seen[str(r[field])].add(r['sampleId'])
  n=len(targets);ok=sum(len(seen[k])>=v for k,v in targets.items());metric={'counts':{k:len(seen[k]) for k in targets},'minimumCounts':targets}
 elif check=='C43':
  target=cfg.get('targetDistribution');threshold=cfg.get('maxTV');field=cfg.get('field')
  if not target or threshold is None or not field:return dict(result,status='undetermined',reason='Missing external distribution, frozen bins, or calibrated TV threshold')
  if not finite_number(threshold) or not 0<=threshold<=1 or any(not finite_number(v) or v<0 for v in target.values()) or abs(sum(target.values())-1)>1e-9:raise ValueError('Invalid target distribution or threshold')
  if any(field not in r for r in rows):return dict(result,status='undetermined',reason='Missing distribution measurements')
  counts=collections.Counter(str(r[field]) for r in rows);keys=set(counts)|set(target);tv=.5*sum(abs(counts[k]/len(rows)-target.get(k,0)) for k in keys)
  return dict(result,status='pass' if tv<=threshold else 'fail',q=1-tv,denominator=len(rows),metric={'TV':tv,'maxTV':threshold},reason='Frozen target comparison; descriptive utility, not scientific accuracy')
 elif check=='C44':
  if cfg.get('requireIndependentGroups') is not True:return dict(result,status='undetermined',reason='Missing independent-group requirement')
  if any(not all(k in r for k in ('sampleId','sourceGroup','split','sha256')) for r in rows):return dict(result,status='undetermined',reason='Incomplete source grouping, hashes, or split assignment')
  memberships=collections.defaultdict(set)
  for r in rows:
   for key in ('sourceGroup','sha256'):memberships[(key,r[key])].add(r['split'])
  bad={r['sampleId'] for r in rows if any(len(memberships[(k,r[k])])>1 for k in ('sourceGroup','sha256'))};n=len({r['sampleId'] for r in rows});ok=n-len(bad);metric={'overlapSampleIds':sorted(bad)}
 elif check=='C45':
  if not cfg.get('datasetId') or not cfg.get('datasetVersion'):return dict(result,status='undetermined',reason='Dataset identity/version missing')
  n=len(rows);ok=0;bad=[]
  for r in rows:
   p=relative(root,r['path'])
   if p.is_file() and (not isinstance(r.get('sha256'),str) or len(r['sha256'])!=64 or any(c not in '0123456789abcdefABCDEF' for c in r['sha256'])):return dict(result,status='undetermined',reason='Expected file fingerprint unavailable or invalid')
   valid=p.is_file() and digest(p)==r['sha256'].lower();ok+=valid
   if not valid:bad.append(r['path'])
  metric={'mismatches':bad}
 elif check in ('C46','C48','C49'):
  fields=cfg.get('requiredFields')
  if not fields:return dict(result,status='undetermined',reason='Required fields/conditions not frozen')
  # Independent verification table must identify the reviewer and hashed evidence.
  if any(not r.get('verifiedBy') or not r.get('evidencePath') or not r.get('evidenceSha256') for r in rows):return dict(result,status='undetermined',reason='Missing verification provenance; model assertions do not count')
  for r in rows:
   p=relative(evidence_root,r['evidencePath'])
   if not p.is_file() or digest(p)!=r['evidenceSha256']:return dict(result,status='undetermined',reason='Verification evidence missing or changed')
  if any(r.get('conflict') or r.get('verificationStatus') not in ('verified','rejected') for r in rows):return dict(result,status='undetermined',reason='Conflicting or unverified evidence requires independent review')
  true_fields=cfg.get('requiredTrueFields',[])
  n=len(rows);ok=sum(all(k in r and r[k] is not None and r[k]!='' for k in fields) and all(r.get(k) is True for k in true_fields) and r.get('verificationStatus')=='verified' for r in rows);metric={'requiredFields':fields,'requiredTrueFields':true_fields,'scope':'Verified document/condition completeness; no automatic legal or scientific verdict'}
 elif check=='C47':
  n=len(rows);ok=0
  for r in rows:
   if 'declared' not in r or 'measured' not in r:return dict(result,status='undetermined',reason='Declaration or measured value missing')
   if isinstance(r['declared'],(int,float)) and not isinstance(r['declared'],bool) or isinstance(r['measured'],(int,float)) and not isinstance(r['measured'],bool):
    if not finite_number(r['declared']) or not finite_number(r['measured']):return dict(result,status='undetermined',reason='Non-finite or inconsistent numeric measurements')
    if not finite_number(r.get('tolerance')) or r['tolerance']<0 or not r.get('toleranceBasis'):return dict(result,status='undetermined',reason='Finite non-negative numeric tolerance and basis required for integer and float measurements')
    valid=abs(r['declared']-r['measured'])<=r['tolerance']
   else:valid=r['declared']==r['measured']
   ok+=valid
  metric={'scope':'Comparison of supplied signed measurements and declarations'}
 elif check=='C50':
  n=len(rows);ok=0;detail=[]
  for r in rows:
   p=relative(root,r['path']);valid=False
   try:
    if r.get('format')=='json':json.loads(p.read_text());valid=True
    elif r.get('format')=='csv':
     import csv
     with p.open(newline='') as f:
      reader=csv.reader(f,strict=True);first=next(reader,None);valid=bool(first)
      for row in reader:pass
    elif r.get('format')=='image':
     from PIL import Image
     with Image.open(p) as im:im.verify();valid=True
    else:return dict(result,status='undetermined',reason='Unsupported required access test')
   except (OSError,ValueError):pass
   except Exception as exc:
    if type(exc).__module__=='_csv':valid=False
    else:raise
   ok+=valid;detail.append({'path':r['path'],'readable':valid})
  metric={'tests':detail,'scope':'Actual local format decoding; does not infer permission'}
 else:raise ValueError('Unknown check')
 return dict(result,status='pass' if ok==n else 'fail',q=ok/n,numerator=ok,denominator=n,metric=metric,reason='Contract-defined completeness, zero missing required units')
def run(root,output,checks,contract=None,run_id=None,evidence_root=None):
 root=Path(root).resolve();output=Path(output).resolve()
 if not root.is_dir() or output.exists() or output.is_relative_to(root):raise ValueError('Existing input and new external output required')
 if not set(checks)<=set(IDS):raise ValueError('Unknown formal check')
 output.mkdir(parents=True);run_id=run_id or 'RUN-'+uuid.uuid4().hex;contract=contract or {};results=[]
 def event(check,status,start,reason=None):
  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
  row=dict(runId=run_id,datasetId=contract.get('datasetId'),dirId=root.name,checkId=check,event='check_'+status,status=status,ts=timestamp,timestamp=timestamp,elapsedMs=(time.monotonic()-start)*1000,resourcePeak={'processMaxRssKiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},tokenUsage=0,retryCount=0,reason=reason)
  with (output/'run.log.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 for check in sorted(checks):
  start=time.monotonic();event(check,'started',start)
  try:result=measure(root,check,contract.get('checks',{}).get(check,{}),evidence_root);event(check,'completed',start,result['reason'])
  except Exception as exc:result={'status':'undetermined','reason':type(exc).__name__+': '+str(exc)};event(check,'failed',start,result['reason'])
  result.update(checkId=check,runId=run_id,dirId=root.name,version=VERSION)
  ep=output/(check+'.evidence.json');ep.write_text(json.dumps({'result':result,'contract':contract.get('checks',{}).get(check)},ensure_ascii=False,indent=2));result['evidenceRef']={'path':ep.name,'contentHash':digest(ep),'capturedBy':VERSION};results.append(result)
 (output/'results.json').write_text(json.dumps({'runId':run_id,'checks':results},ensure_ascii=False,indent=2));return results
def summarize_dimension(checks, expected_ids, weights=None):
 """Aggregate check-level units, with unknowns retained in the denominator."""
 byid={}
 for row in checks:
  if row['checkId'] in byid:raise ValueError('Duplicate check result: '+row['checkId'])
  byid[row['checkId']]=row
 if set(byid)-set(expected_ids):raise ValueError('Unexpected check in dimension')
 rows=[dict(byid.get(k,{'checkId':k,'status':'undetermined','reason':'Required check result missing'})) for k in expected_ids]
 weights=weights or dict.fromkeys(expected_ids,1.)
 if set(weights)!=set(expected_ids) or any(not finite_number(v) or v<=0 for v in weights.values()):raise ValueError('Positive finite per-check weights required for every expected check')
 invalid=[]
 for row in rows:
  if row.get('status') not in ('pass','fail','undetermined','not_applicable'):raise ValueError('Invalid four-state check result')
  if row['status'] in ('pass','fail') and (not finite_number(row.get('q')) or not 0<=row['q']<=1):
   invalid.append(row['checkId']);row.update(sourceStatus=row['status'],status='undetermined',reason='Determinate check lacks finite utility in [0,1]')
 applicable=[x for x in rows if x['status']!='not_applicable'];done=[x for x in applicable if x['status'] in ('pass','fail')];pending=[x for x in applicable if x['status']=='undetermined'];failed=[x for x in done if x['status']=='fail']
 denominator=sum(weights[x['checkId']] for x in applicable);completed=sum(weights[x['checkId']] for x in done)
 score=100*sum(weights[x['checkId']]*x['q'] for x in done)/denominator if applicable and not pending else None
 state='not_applicable' if not applicable else 'fail' if failed else 'undetermined' if pending else 'pass'
 return dict(score=score,state=state,range=[score,score] if score is not None else None,checkCoverage=completed/denominator if denominator else None,denominator=len(applicable),statisticalUnit='package_check',applicableCheckDenominator=len(applicable),completedChecks=len(done),notApplicableChecks=len(rows)-len(applicable),pendingChecks=[x['checkId'] for x in pending],failedChecks=[x['checkId'] for x in failed],invalidUtilityChecks=invalid,applicableWeight=denominator,completedWeight=completed,checkWeights=weights,checkResults=rows,reason='All applicable contract checks determined' if score is not None else 'All checks explicitly not applicable' if not applicable else 'Missing evidence or utility: '+', '.join(x['checkId'] for x in pending))

def apply_formal_dimensions(dimensions,results,contract):
 byid={x['checkId']:x for x in results}
 if len(byid)!=len(results):raise ValueError('Duplicate formal check result')
 for d in dimensions:
  if d['id'] not in GROUPS:continue
  ids=GROUPS[d['id']];weights={k:contract.get('checks',{}).get(k,{}).get('weight',1.) for k in ids}
  d.update(summarize_dimension([byid[k] for k in ids if k in byid],ids,weights))
 return dimensions

def composite(dimensions,policy):
 weights=policy.get('weights',{});blockers=[]
 if set(dimensions)!=set(DIMENSION_IDS):blockers.append('Expected exactly the six registered dimension IDs')
 if set(weights)!=set(DIMENSION_IDS):blockers.append('Weights must cover the six registered dimensions')
 if not isinstance(policy.get('basis'),str) or not policy['basis'].strip():blockers.append('Weight basis unavailable')
 if not isinstance(policy.get('approvedBy'),str) or not policy['approvedBy'].strip() or policy.get('approvalStatus') != 'approved':blockers.append('Actual weight approval unavailable')
 if any(not finite_number(v) or v<=0 for v in weights.values()):raise ValueError('Positive finite non-boolean weights required')
 missing=[k for k,v in dimensions.items() if v is None]
 if missing:blockers.append('Undetermined dimensions: '+', '.join(missing))
 if any(v is not None and (not finite_number(v) or not 0<=v<=100) for v in dimensions.values()):raise ValueError('Finite non-boolean dimension score in [0,100] required')
 if blockers:return {'score':None,'reason':'; '.join(blockers),'blockers':blockers,'pendingDimensions':missing}
 denominator=sum(weights.values());return {'score':sum(weights[k]*v for k,v in dimensions.items())/denominator,'reason':'Approved weighted composite; critical findings remain separately reported','blockers':[],'normalizedWeights':{k:v/denominator for k,v in weights.items()}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--checks',default=','.join(IDS));ap.add_argument('--contract',type=Path);a=ap.parse_args();print(json.dumps(run(a.input,a.output,a.checks.split(','),json.loads(a.contract.read_text()) if a.contract else None),ensure_ascii=False))
if __name__=='__main__':main()
