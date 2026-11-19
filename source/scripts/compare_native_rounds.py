"""Compare independent fresh rounds under a prespecified engineering gate."""
import json,math,sys,collections
from pathlib import Path
from audit_evaluation import audit

def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def compare(root,protocol,out):
 root=Path(root);out=Path(out);out.mkdir(exist_ok=False);policy=json.load(open(protocol));names=sorted(p.name for p in (root/'round1').iterdir() if p.is_dir());assert len(names)==12;differences=[];packages=[]
 def diff(package,sample,kind,**kw):differences.append(dict(package=package,sampleId=sample,kind=kind,**kw))
 for name in names:
  a=root/'round1'/name;b=root/'round2'/name;aa=audit(a);ab=audit(b)
  assert aa['status']==ab['status']=='pass',(aa,ab)
  pa={r['sampleId']:r for r in rows(a/'predictions.jsonl')};pb={r['sampleId']:r for r in rows(b/'predictions.jsonl')};maxprob=0.;maxcoord=0.;count0=len(differences)
  for sid in sorted(set(pa)|set(pb)):
   if sid not in pa or sid not in pb:diff(name,sid,'sample_set');continue
   x,y=pa[sid],pb[sid]
   if x['prediction']!=y['prediction']:diff(name,sid,'decision',round1=x['prediction'],round2=y['prediction'])
   for k in ['dataSha256','labelSha256','inputSha256']:
    if x.get(k)!=y.get(k):diff(name,sid,'input_fingerprint',field=k)
   delta=max(abs(v-w) for v,w in zip(x['probabilities'],y['probabilities']));assert all(math.isfinite(v) for v in x['probabilities']+y['probabilities']);maxprob=max(maxprob,delta)
   if delta>policy['probabilityAbsoluteTolerance']:diff(name,sid,'probability',maxAbs=delta)
  def groups(path):
   g=collections.defaultdict(list)
   for f in rows(path):g[(f['sampleId'],f['defectType'],f.get('findingStage'))].append(f)
   for key in g:g[key].sort(key=lambda f:json.dumps(f.get('locator'),sort_keys=True))
   return g
  ga,gb=groups(a/'findings.jsonl'),groups(b/'findings.jsonl')
  def equivalent(x,y):
   nonlocal maxcoord
   if isinstance(x,(int,float)) and not isinstance(x,bool) and isinstance(y,(int,float)) and not isinstance(y,bool):
    delta=abs(x-y);maxcoord=max(maxcoord,delta);return delta<=policy['coordinateAbsoluteTolerancePixels']
   if type(x)!=type(y):return False
   if isinstance(x,dict):return x.keys()==y.keys() and all(equivalent(x[k],y[k]) for k in x)
   if isinstance(x,list):return len(x)==len(y) and all(equivalent(v,w) for v,w in zip(x,y))
   return x==y
  for key in sorted(set(ga)|set(gb)):
   if len(ga[key])!=len(gb[key]):diff(name,key[0],'finding_count',defect=key[1],round1=len(ga[key]),round2=len(gb[key]));continue
   for x,y in zip(ga[key],gb[key]):
    if not equivalent(x.get('locator'),y.get('locator')):diff(name,key[0],'locator',defect=key[1],round1=x.get('locator'),round2=y.get('locator'))
    if x.get('scoreEligible')!=y.get('scoreEligible'):diff(name,key[0],'eligibility')
  ca=json.load(open(a/'scorecard.json'));cb=json.load(open(b/'scorecard.json'))
  if abs(ca['observedTechnicalScore']-cb['observedTechnicalScore'])>policy['scoreAbsoluteTolerance']:diff(name,None,'score')
  if [(d['id'],d.get('score')) for d in ca['dimensions']]!=[(d['id'],d.get('score')) for d in cb['dimensions']]:diff(name,None,'dimension_score')
  if ca['fullQualityScore']!=cb['fullQualityScore']:diff(name,None,'full_quality_score')
  packages.append(dict(dirId=name,samples=len(pa),round1RunId=ca['runId'],round2RunId=cb['runId'],maxProbabilityAbs=maxprob,maxCoordinateAbs=maxcoord,differences=len(differences)-count0,score=ca['observedTechnicalScore']))
 scores=[]
 for n in (1,2):
  cs=[json.load(open(root/('round'+str(n))/name/'scorecard.json')) for name in names];scores.append({c['dirId']:1+sum(x['observedTechnicalScore']>c['observedTechnicalScore'] for x in cs) for c in cs})
 if scores[0]!=scores[1]:diff(None,None,'ranking',round1=scores[0],round2=scores[1])
 with (out/'differences.jsonl').open('w') as f:
  for d in differences:f.write(json.dumps(d,ensure_ascii=False)+'\n')
 result=dict(status='pass' if not differences else 'fail',packages=packages,samplesPerRound=sum(p['samples'] for p in packages),differences=len(differences),protocol=policy,ranking=scores,scope='Actual independent new forwards. Probability/coordinate tolerance is engineering reproducibility only, not scientific validation.')
 (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));return result
if __name__=='__main__':result=compare(*sys.argv[1:]);raise SystemExit(result['status']!='pass')
