"""Finish already-running validation, stop queue parents, validate final runtime only."""
from pathlib import Path
import sys,json,time,os,signal,subprocess,datetime
R=Path('/staff/tangjiarui/OPC/LTH团队-SpaceQA智能体/repair_20261110/runtime_candidate');sys.path.insert(0,str(R));os.environ['SPACEQA_RUNNER_CONFIG']=str(R/'clean_r4_config.json');import run_container as c
sys.path.insert(0,str(R.parents[1]/'source/scripts'));from audit_evaluation import audit
jobs=[('icarus0',3986295,3986333,'AIA_AR_sample_3mogtggl50'),('icarus1',1940554,None,'AIA_AR_sample_7qn2pdeube')];done=[]
# Parent processes are STOPPED at subprocess.wait. Current containers continue.
while jobs:
 for job in jobs[:]:
  host,parent,child,name=job
  code="from pathlib import Path; import json; p=Path('/proc/%d/task/%d/children'); ids=p.read_text().split() if p.exists() else []; print(json.dumps([{'pid':int(i),'stat':Path('/proc/'+i+'/stat').read_text()} for i in ids]))"%(parent,parent)
  cmd=['python3','-c',code] if host=='icarus0' else ['ssh',host,'python3 -c '+__import__('shlex').quote(code)]
  proc=subprocess.run(cmd,capture_output=True,text=True)
  if proc.returncode:raise RuntimeError(proc.stderr)
  entries=json.loads(proc.stdout)
  if not entries:raise RuntimeError('Queue child unexpectedly absent: '+host)
  fields=[e['stat'].split(') ',1)[1].split() for e in entries]
  if not all(f[0]=='Z' for f in fields):continue
  exit_codes=[int(f[49]) for f in fields]
  if host=='icarus0':os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
  else:subprocess.run(['ssh',host,'kill -TERM '+str(parent)+'; kill -CONT '+str(parent)],check=True)
  path=R/'full_repeat_final/round1'/name
  result=audit(path) if all(x==0 for x in exit_codes) else {'status':'fail','processExitWaitStatus':exit_codes}
  row={'host':host,'dirId':name,'audit':result,'processExitWaitStatus':exit_codes,'completedUtc':datetime.datetime.now(datetime.timezone.utc).isoformat()};done.append(row);jobs.remove(job)
  (R/'acceptance/runtime_full_package_validation.json').write_text(json.dumps(done,ensure_ascii=False,indent=2));print(json.dumps(row),flush=True)
  state={'status':'runtime_validation_complete' if result['status']=='pass' else 'failed','completed':1,'total':1,'host':host,'round':1,'package':name,'scope':'User narrowed to runtime validation; subsequent full runs intentionally not started'};(R/'full_repeat_final'/('state_'+host+'.json')).write_text(json.dumps(state,ensure_ascii=False,indent=2))
  if result['status']!='pass':raise SystemExit(1)
 time.sleep(10 if jobs else 0)
for family in ['AIA_AR_sample','EPB_seg_sample','ULF_sample']:
 package=next(p.name for p in (R/'inputs').iterdir() if p.name.startswith(family));name='clean_final_'+family
 rc=c.execute(name,['evaluate','--input','/input/'+package,'--family',family,'--output','/output/'+name,'--release-profile','roadshow-b-v1'],True)
 if rc:raise SystemExit(rc)
 result=audit(R/'acceptance/output'/name);(R/'acceptance'/(name+'_audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if result['status']!='pass':raise SystemExit(1)
print('Runtime-only end-to-end validation complete; no report/PPT publication',flush=True)
