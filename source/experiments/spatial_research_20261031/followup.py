"""Wait for this task's first runner, then run an isolated 448px ablation."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/spatial_research_20261031'
FIRST=ROOT/'runs/spatial-research-20261031'
OUT=ROOT/'runs/spatial-research-448-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state=dict(status='waiting_for_first_run',pid=os.getpid(),image_size=448,budget_minutes=60,stages=[],
        protocol_sha256=hashlib.sha256((SRC/'protocol_highres.md').read_bytes()).hexdigest())
    def save():(OUT/'state.json').write_text(json.dumps(state,indent=2))
    save();deadline=time.monotonic()+90*60
    while time.monotonic()<deadline:
        first=json.loads((FIRST/'state.json').read_text())
        if first['status']!='running':break
        time.sleep(10)
    else:state['status']='first_run_wait_timeout';save();return
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',
        OPC_SPATIAL_SIZE='448',OPC_SPATIAL_OUT=str(OUT),TOKENIZERS_PARALLELISM='false')
    state['status']='running';save();start=time.monotonic()
    for stage in ('prepare','unet','dino','evaluate'):
        remaining=60*60-(time.monotonic()-start)
        if remaining<=0:state['status']='budget_exhausted';break
        entry=dict(stage=stage,status='running');state['stages'].append(entry);save();ts=time.monotonic()
        try:
            with (OUT/(stage+'.log')).open('w') as log:
                p=subprocess.run([PY,'-u',str(SRC/'spatial.py'),stage],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,1800))
            entry.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-ts)
        except subprocess.TimeoutExpired:entry.update(status='timeout',seconds=time.monotonic()-ts)
        save()
        if stage=='prepare' and entry['status']!='complete':state['status']='failed';break
    else:state['status']='completed' if all(s['status']=='complete' for s in state['stages']) else 'completed_with_failures'
    state['seconds']=time.monotonic()-start;save();print(json.dumps(state),flush=True)


if __name__=='__main__':main()
