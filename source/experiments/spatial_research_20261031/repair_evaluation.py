"""Resume evaluation after correcting repeated lazy archive decompression."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/spatial_research_20261031'
OUT=ROOT/'runs/spatial-research-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'

def main():
    state=json.loads((OUT/'state.json').read_text())
    if state['status']=='running' and state['stages'][-1]['stage']=='evaluate':
        # Only stop this runner's own evaluation child, validated via PPid and argv.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                status=(p/'status').read_text();argv=(p/'cmdline').read_bytes().split(b'\0')
                ppid=int(next(s.split()[1] for s in status.splitlines() if s.startswith('PPid:')))
                if ppid==state['pid'] and str(SRC/'spatial.py').encode() in argv and b'evaluate' in argv:
                    os.kill(int(p.name),signal.SIGTERM)
            except (OSError,StopIteration):pass
    for _ in range(30):
        state=json.loads((OUT/'state.json').read_text())
        if state['status']!='running':break
        time.sleep(1)
    start=time.monotonic();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4')
    with (OUT/'evaluate-repaired.log').open('w') as f:
        p=subprocess.run([PY,'-u',str(SRC/'spatial.py'),'evaluate'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
    state=json.loads((OUT/'state.json').read_text())
    state['stages'].append(dict(stage='evaluate_repaired',status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-start,
        reason='Cache archive arrays once; previous evaluation repeatedly decompressed full NPZ members. No metrics or protocol changed.'))
    state['status']='completed_after_evaluation_repair' if p.returncode==0 else 'failed'
    (OUT/'state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)

if __name__=='__main__':main()
