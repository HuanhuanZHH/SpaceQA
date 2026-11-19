import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

BASE=Path('/staff/tangjiarui/OPC/eight_class_20261104')
OUT=BASE/'refresh_20261105'
PY='/staff/tangjiarui/OPC/.venv-image-models/bin/python'
assert json.loads((OUT/'classification/complete.json').read_text())['records']==24782
assert not (OUT/'grounded').exists()
if (OUT/'failure.json').exists():(OUT/'failure.json').rename(OUT/'initial_launch_failure.json')
env=dict(os.environ)
for key in list(env):
    if 'API_KEY' in key or key.startswith('SPACEQA_CHAT_LLM_'):env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONPATH=str(BASE/'runtime/src'),PYTHONUNBUFFERED='1')
command=[PY,'-u',str(BASE/'refresh_code_20261105/ground_all_refresh.py'),'--roadshow',str(OUT/'classification'),'--localizer',str(BASE/'localizer_v2'),'--comparison',str(BASE/'candidate_comparison_v1'),'--output',str(OUT/'grounded'),'--batch','16']
log=BASE/'refresh_20261105.ground.log'
with log.open('xb') as stream:
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
state={'stage':'all_readable_image_forward','status':'running','pid':child.pid,'gpu':1,'log':str(log),'startedAt':dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()}
(OUT/'state.json').write_text(json.dumps(state,indent=2))
(OUT/'ground_launch.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state))
