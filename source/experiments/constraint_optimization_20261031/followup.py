"""Run expanded-student training, then isolated VLM runtime, then stop it."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/constraint_optimization_20261031'
OUT=ROOT/'runs/constraint-optimization-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
with (OUT/'students_train.log').open('w') as f:
    subprocess.run([PY,'-u',str(SRC/'optimize_students.py'),'train'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
env.update(QWEN_GPU='1',QWEN_PORT='8002',QWEN_GPU_MEMORY_UTILIZATION='0.35',QWEN_MAX_NUM_SEQS='32',QWEN_MIN_FREE_MIB='14000')
start=time.monotonic()
with (OUT/'batch_server.log').open('w') as f:
    p=subprocess.Popen(['bash',str(ROOT/'deploy/qwen3_vl/serve.sh')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(OUT/'batch_server.pid').write_text(str(p.pid))
try:
    ready=False
    for _ in range(360):
        if p.poll() is not None:raise RuntimeError('Task-owned VLM failed to start')
        try:
            with urlopen('http://127.0.0.1:8002/health',timeout=2) as f:ready=f.status==200
        except Exception:pass
        if ready:break
        time.sleep(1)
    assert ready,'VLM startup timed out'
    (OUT/'startup.json').write_text(json.dumps(dict(model_startup_seconds=time.monotonic()-start)))
    with (OUT/'all_family_runtime.log').open('w') as f:
        subprocess.run([PY,'-u',str(SRC/'all_family_runtime.py')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
finally:
    if p.poll() is None:
        os.killpg(os.getpgid(p.pid),signal.SIGTERM)
        try:p.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(os.getpgid(p.pid),signal.SIGKILL)
