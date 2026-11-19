"""Start only task-owned bounded jobs; leave the existing API untouched."""
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/constraint-optimization-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'
SRC=ROOT/'experiments/constraint_optimization_20261031'
OUT.mkdir(exist_ok=True,parents=True)
jobs=[('followup','followup.py',[])] if 'followup' in sys.argv else [('rules','runtime_full.py',[]),('students_prepare','optimize_students.py',['prepare'])]
for tag,script,args in jobs:
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
    with (OUT/(tag+'.log')).open('w') as f:
        p=subprocess.Popen([PY,'-u',str(SRC/script)]+args,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    (OUT/(tag+'.pid')).write_text(str(p.pid))
    print(tag,p.pid)
