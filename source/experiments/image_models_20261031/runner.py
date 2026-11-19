"""Bounded single-GPU queue; each stage's outcome survives SSH disconnects."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/image-models-20261031'
SRC=ROOT/'experiments/image_models_20261031'
PY='/staff/tangjiarui/OPC/.venv-image-models/bin/python'
ROUTES=['zero_shot_yolo11seg','yolo11_detect','yolo11_segment','yolov8_segment',
        'attention_unet','unetpp','resnet18_unet','deeplab_mobilenet']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    state=dict(status='waiting_bootstrap',pid=os.getpid(),gpu=1,budget_minutes=150,stages=[],
        preserve_service_port=8001,formal_scoring_enabled=False,
        protocol_sha256=hashlib.sha256((SRC/'protocol.md').read_bytes()).hexdigest())
    def save():
        p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
    save();deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if (OUT/'bootstrap_complete.json').exists():break
        time.sleep(5)
    else:state['status']='bootstrap_timeout';save();return
    env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        YOLO_CONFIG_DIR=str(OUT/'ultralytics_config'),YOLO_AUTOINSTALL='false',MPLCONFIGDIR=str(OUT/'matplotlib_config'))
    state['status']='running';start=time.monotonic();save()
    jobs=[('prepare',['prepare.py'],300)]+[(r,['train.py',r],1200) for r in ROUTES]+[('evaluate',['evaluate.py'],1800),('validate',['validate.py'],300)]
    for name,args,limit in jobs:
        remaining=150*60-(time.monotonic()-start)
        if remaining<=0:state['status']='budget_exhausted';break
        entry=dict(stage=name,status='running');state['stages'].append(entry);save();ts=time.monotonic()
        try:
            with (OUT/(name+'.log')).open('w') as f:
                p=subprocess.run([PY,'-u',str(SRC/args[0]),*args[1:]],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(limit,remaining))
            entry.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-ts)
        except subprocess.TimeoutExpired:entry.update(status='timeout',seconds=time.monotonic()-ts)
        save()
        if name=='prepare' and entry['status']!='complete':state['status']='failed';break
    else:state['status']='completed' if all(s['status']=='complete' for s in state['stages']) else 'completed_with_failures'
    state['seconds']=time.monotonic()-start;save();print(json.dumps(state),flush=True)


if __name__=='__main__':
    if '--launch' in sys.argv:
        OUT.mkdir(parents=True,exist_ok=True)
        if (OUT/'state.json').exists():raise SystemExit('Existing run detected; inspect before relaunch')
        with (OUT/'runner.log').open('w') as f:
            p=subprocess.Popen([PY,'-u',str(Path(__file__))],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,output=str(OUT))))
    else:main()
