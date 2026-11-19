"""After the control experiment finishes, reuse its GPU0 for 32B training."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/staff/tangjiarui/OPC');WORK=ROOT/'jwvl_optimization_20261104'
assert os.uname().nodename.split('.')[0]=='icarus1'
state=WORK/'qwen32_queue.json';start=time.time()
while time.time()-start<72*3600:
 parent=WORK/'qwen8b_control_v1/state.json'
 if parent.exists():
  status=json.loads(parent.read_text()).get('status')
  if status=='failed':
   state.write_text(json.dumps({'status':'blocked_by_control_failure','reason':'Review control before changing training stack'}));raise SystemExit(2)
  if status=='complete':break
 state.write_text(json.dumps({'status':'waiting_for_control_training_and_evaluation','gpu':'icarus1:0','updated_unix':time.time()}))
 time.sleep(30)
else:state.write_text(json.dumps({'status':'wait_expired'}));raise SystemExit(2)
receipt=json.loads((WORK/'qwen32_download.json').read_text());assert receipt['status']=='complete'
lock=(WORK/'icarus1_gpu0.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
for _ in range(3):
 r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 if int(r.stdout.strip())>=512:
  state.write_text(json.dumps({'status':'gpu_reclaimed_by_another_job','reason':'No sharing or termination'}));raise SystemExit(2)
 time.sleep(5)
out=WORK/'qwen32b_domain_v1';assert not out.exists()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
with (WORK/'qwen32b_domain_v1.log').open('ab') as f:
 proc=subprocess.Popen([str(ROOT/'.venv-qwen3vl/bin/python'),'-u',str(WORK/'code/train_qwen32.py'),'--variant','domain','--epochs','1','--output',str(out)],cwd=WORK/'code',env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock.fileno(),))
state.write_text(json.dumps({'status':'started','pid':proc.pid,'host':'icarus1','gpu':0,'output':str(out)}));print(state.read_text())
