"""Own two idle GPUs on icarus1 only; never terminates another process."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/staff/tangjiarui/OPC');WORK=ROOT/'jwvl_optimization_20261104'
assert os.uname().nodename.split('.')[0]=='icarus1'
held=[]
for index in [0,1]:
 lock=(WORK/f'icarus1_gpu{index}.lock').open('a+')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(lock)
for _ in range(3):
 r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
 used={int(row.split(',')[0]):int(row.split(',')[1]) for row in r.stdout.strip().splitlines()}
 assert used[0]<512 and used[1]<512,('GPU no longer idle',used)
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
 assert not p.stdout.strip(),'Compute process present; refuse to share GPU'
 time.sleep(5)
launched=[]
for index,variant in [(0,'control'),(1,'domain')]:
 out=WORK/('qwen8b_'+variant+'_v1');assert not out.exists(),'Already launched'
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(index),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
 py=ROOT/'.venv-qwen3vl/bin/python'
 with (WORK/f'qwen8b_{variant}_v1.log').open('ab') as f:
  proc=subprocess.Popen([str(py),'-u',str(WORK/'code/train_grounded.py'),'--variant',variant,'--output',str(out)],env=env,cwd=WORK/'code',stdout=f,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(held[index].fileno(),))
 launched.append(dict(host='icarus1',gpu=index,variant=variant,pid=proc.pid,output=str(out)))
(WORK/'gpu_launch.json').write_text(json.dumps(launched,indent=2));print(json.dumps(launched))
