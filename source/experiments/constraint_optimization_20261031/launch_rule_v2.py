import subprocess
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/constraint_optimization_20261031'
OUT=ROOT/'runs/constraint-optimization-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'
for name,args in [('rules_release',[str(SRC/'runtime_full.py'),'--name','full-rule-release']),
                  ('rule_regression',[str(SRC/'rule_regression.py')])]:
    with (OUT/(name+'.log')).open('w') as f:
        p=subprocess.Popen([PY,'-u']+args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/(name+'.pid')).write_text(str(p.pid))
    print(name,p.pid)
