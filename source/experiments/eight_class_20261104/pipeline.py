"""Bounded automatic continuation: labels -> train -> all-30 test -> roadshow.

Uses separate outputs and never stops existing model/web services. Processes here
are experiment-owned children; no API or GPU0 inference service is managed here.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/eight_class_20261104')
PY='/staff/tangjiarui/OPC/.venv-image-models/bin/python'


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main(args):
    start=time.monotonic();deadline=start+23*3600
    env=os.environ.copy();env.update(OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',
        PYTHONPATH='/staff/tangjiarui/OPC/rescue_20261104/code_final/src')
    state_path=ROOT/(args.state_name+'.json')
    state=dict(start=dt.datetime.now().isoformat(),deadline=(dt.datetime.now().astimezone()+dt.timedelta(hours=23)).isoformat(),stage='waiting_targets')
    write(state_path,state)
    while not (ROOT/args.targets/'protocol.json').is_file():
        pid=json.loads((ROOT/(args.targets+'_launch.json')).read_text())['pid']
        if not Path('/proc/'+str(pid)).exists():raise RuntimeError('Target preparation exited without completion; inspect targets_v2.log')
        if time.monotonic()>deadline:raise TimeoutError('23 hour pipeline budget reached')
        time.sleep(15)
    commands=[(args.fusion,'fusion.py',['--targets',str(ROOT/args.targets),'--output',str(ROOT/args.fusion)]),
              (args.roadshow,'roadshow.py',['--fusion',str(ROOT/args.fusion),'--output',str(ROOT/args.roadshow)])]
    for name,script,args in commands:
        if time.monotonic()>deadline:raise TimeoutError('23 hour pipeline budget reached')
        with (ROOT/(name+'.log')).open('ab') as log:
            p=subprocess.Popen([PY,'-u',str(ROOT/'code'/script)]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage=name,pid=p.pid,elapsed_seconds=time.monotonic()-start)
            write(state_path,state)
            p.wait(timeout=max(1,deadline-time.monotonic()))
            if p.returncode:raise RuntimeError(name+' failed with exit '+str(p.returncode))
    state.update(stage='classification_and_roadshow_complete',elapsed_seconds=time.monotonic()-start,
        localization='independent GPU1 job; completion and localization quality must be checked separately')
    write(state_path,state)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--targets',default='targets_v2');ap.add_argument('--fusion',default='fusion_v1')
    ap.add_argument('--roadshow',default='roadshow_v1');ap.add_argument('--state-name',default='pipeline_state');args=ap.parse_args()
    try:main(args)
    except BaseException as exc:
        write(ROOT/(args.state_name+'_failure.json'),dict(type=type(exc).__name__,message=str(exc)));raise
