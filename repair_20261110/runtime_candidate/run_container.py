import sys,subprocess,json,time,os
from pathlib import Path
sys.path.insert(0,'/staff/tangjiarui/OPC/compact_release_20261107')
import build_image as b
R=Path(__file__).resolve().parent
HOST_CONFIG=json.loads(Path(os.environ['SPACEQA_RUNNER_CONFIG']).read_text()) if os.environ.get('SPACEQA_RUNNER_CONFIG') else {}
if HOST_CONFIG:
 b.command=lambda:HOST_CONFIG['command']
 original_env=b.env
 b.env=lambda:dict(original_env(),**HOST_CONFIG['environment'])
TAG=os.environ.get('SPACEQA_TEST_TAG','localhost/spaceqa-agent:0.6.1-final-r4-20261110')
def command(output,gpu=False,input_root=None):
 output.mkdir(parents=True,exist_ok=True)
 opts=['run','--shm-size=2g','--rm','--pull=never','--network=none','--read-only','--cgroups=disabled','--user=0:0','--security-opt=no-new-privileges','--cap-drop=all','--tmpfs','/tmp:rw,exec,nosuid,size=4g','-e','HOME=/tmp','-v',str(input_root or R/'inputs')+':/input:ro','-v',str(output)+':/output:rw']
 if gpu:
  for dev in ('nvidia1','nvidiactl','nvidia-uvm','nvidia-uvm-tools'):
   if Path('/dev/'+dev).exists():opts+=['--device','/dev/'+dev]
  opts+=['-e','CUDA_VISIBLE_DEVICES='+HOST_CONFIG.get('gpuUuid','GPU-af7438bd-6b58-2ddf-3706-087caafd0b5f'),'-e','NVIDIA_VISIBLE_DEVICES=void','-e','LD_LIBRARY_PATH=/host-driver:/opt/spaceqa/system-libs:/opt/spaceqa/driver-libs:/usr/local/cuda/lib64','-v',HOST_CONFIG.get('driverPath',str(R/'host-driver'))+':/host-driver:ro']
 return b.command()+opts+[TAG]
def execute(name,args,gpu=False,input_root=None):
 output=R/'acceptance/output';t=time.time();cmd=command(output,gpu,input_root)+args
 with (R/'acceptance'/ (name+'.log')).open('w') as f:r=subprocess.run(cmd,env=b.env(),stdout=f,stderr=subprocess.STDOUT)
 row=dict(name=name,returncode=r.returncode,seconds=time.time()-t,command=cmd)
 (R/'acceptance'/ (name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2));print(json.dumps(row),flush=True);return r.returncode
if __name__=='__main__':
 if sys.argv[1]=='help':raise SystemExit(execute('evaluate_help',['evaluate','--help']))
 family=sys.argv[1];package=next(p.name for p in (R/'inputs').iterdir() if p.name.startswith(family))
 raise SystemExit(execute('release4_smoke_'+family,['evaluate','--input','/input/'+package,'--family',family,'--output','/output/release4_'+family,'--release-profile','roadshow-b-v1'],True))
