"""Bounded, restart-observable sequential GPU1 experiment runner."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/spatial-research-20261031'
SRC=ROOT/'experiments/spatial_research_20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'


def main():
    OUT.mkdir(parents=True,exist_ok=True);env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    state=dict(status='running',pid=os.getpid(),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       budget_minutes=90,gpu=1,preserve_service_port=8001,stages=[],formal_score_enabled=False,
       protocol_sha256=hashlib.sha256((SRC/'protocol.md').read_bytes()).hexdigest())
    def save():
        temp=OUT/'state.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'state.json')
    start=time.monotonic();save()
    for stage in ('prepare','unet','dino','evaluate'):
        remaining=90*60-(time.monotonic()-start)
        if remaining<=0:state['status']='budget_exhausted';break
        if stage=='dino' and not (ROOT.parent/'models/dinov2-small/download_manifest.json').exists():
            state['stages'].append(dict(stage=stage,status='skipped_model_download_not_ready'));save();continue
        entry=dict(stage=stage,status='running');state['stages'].append(entry);save();ts=time.monotonic()
        try:
            with (OUT/(stage+'.log')).open('w') as log:
                p=subprocess.run([PY,'-u',str(SRC/'spatial.py'),stage],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,2400))
            entry.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-ts)
        except subprocess.TimeoutExpired:entry.update(status='timeout',seconds=time.monotonic()-ts)
        save()
        if stage=='prepare' and entry['status']!='complete':state['status']='failed';break
    else:state['status']='completed' if all(s['status']=='complete' for s in state['stages']) else 'completed_with_failures'
    state['seconds']=time.monotonic()-start;save();print(json.dumps(state),flush=True)


if __name__=='__main__':
    if '--launch' in sys.argv:
        OUT.mkdir(parents=True,exist_ok=True)
        if (OUT/'state.json').exists():raise SystemExit('Existing run detected; inspect it before relaunch.')
        with (OUT/'runner.log').open('w') as log:
            p=subprocess.Popen([PY,'-u',str(Path(__file__))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        (OUT/'runner.pid').write_text(str(p.pid));print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:main()
