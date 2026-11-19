import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/spatial_research_20261031'
OUT=ROOT/'runs/spatial-research-20261031'
WAIT=ROOT/'runs/spatial-research-448-20261031'
PY='/staff/tangjiarui/OPC/.venv-qwen3vl/bin/python'

def main():
    state=dict(status='waiting_for_448',pid=os.getpid(),stages=[])
    def save():(OUT/'jepa_state.json').write_text(json.dumps(state,indent=2))
    save();deadline=time.monotonic()+75*60
    while time.monotonic()<deadline:
        if (WAIT/'state.json').exists() and json.loads((WAIT/'state.json').read_text())['status'] not in ('running','waiting_for_first_run'):break
        time.sleep(10)
    else:state['status']='wait_timeout';save();return
    state['status']='crossfit_running';save();ts=time.monotonic()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (OUT/'crossfit.log').open('w') as f:
        try:
            p=subprocess.run([PY,'-u',str(SRC/'crossfit.py')],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
            state['stages'].append(dict(stage='crossfit',status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-ts))
        except subprocess.TimeoutExpired:state['stages'].append(dict(stage='crossfit',status='timeout',seconds=time.monotonic()-ts))
    state['status']='waiting_for_download';save();deadline=time.monotonic()+30*60
    while time.monotonic()<deadline:
        if (ROOT.parent/'models/ijepa_vith14_1k/download_manifest.json').exists():break
        time.sleep(10)
    else:state['status']='download_timeout';save();return
    state['status']='running';save()
    if (OUT/'detector_results.json').exists():shutil.copyfile(OUT/'detector_results.json',OUT/'detector_results_before_jepa.json')
    for stage,timeout in [('ijepa',1800),('evaluate',900)]:
        entry=dict(stage=stage,status='running');state['stages'].append(entry);save();start=time.monotonic()
        try:
            with (OUT/(stage+'-jepa.log')).open('w') as f:
                p=subprocess.run([PY,'-u',str(SRC/'spatial.py'),stage],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
            entry.update(status='complete' if p.returncode==0 else 'failed',exit_code=p.returncode,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:entry.update(status='timeout',seconds=time.monotonic()-start)
        save()
        if entry['status']!='complete':state['status']='failed';save();return
    state['status']='completed';save()

if __name__=='__main__':main()
