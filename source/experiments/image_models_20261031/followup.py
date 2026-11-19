"""Wait for initial sweep, then run preregistered validation-selected refinements."""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/staff/tangjiarui/OPC/opc-agent')
OUT = ROOT/'runs/image-models-20261031'
SRC = ROOT/'experiments/image_models_20261031'
PY = '/staff/tangjiarui/OPC/.venv-image-models/bin/python'


def main():
    state = dict(status='waiting_initial_queue', pid=os.getpid(), stages=[], gpu=1,
                 protocol_sha256=hashlib.sha256((SRC/'protocol_tiers.md').read_bytes()).hexdigest())
    def save():
        tmp = OUT/'followup_state.tmp'; tmp.write_text(json.dumps(state, indent=2)); tmp.replace(OUT/'followup_state.json')
    save()
    deadline = time.monotonic()+160*60
    while time.monotonic() < deadline:
        initial = json.loads((OUT/'state.json').read_text())
        if initial['status'] not in ['waiting_bootstrap', 'running']:
            break
        time.sleep(10)
    else:
        state['status'] = 'initial_wait_timeout'; save(); return
    env = {k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               YOLO_CONFIG_DIR=str(OUT/'ultralytics_config'), YOLO_AUTOINSTALL='false', MPLCONFIGDIR=str(OUT/'matplotlib_config'))
    start = time.monotonic(); state['status'] = 'running'; save()
    def run(name, args, timeout):
        remaining = 150*60-(time.monotonic()-start)
        if remaining < 30:
            state['status'] = 'budget_exhausted'; save(); return False
        entry = dict(stage=name, status='running'); state['stages'].append(entry); save(); ts = time.monotonic()
        try:
            with (OUT/('followup_'+name+'.log')).open('w') as f:
                p = subprocess.run([PY, '-u', str(SRC/args[0]), *args[1:]], env=env, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=min(remaining, timeout), start_new_session=True)
            entry.update(status='complete' if p.returncode == 0 else 'failed', exit_code=p.returncode)
        except subprocess.TimeoutExpired:
            entry.update(status='timeout')
        entry['seconds'] = time.monotonic()-ts; save(); return entry['status'] == 'complete'
    if not (OUT/'selection.json').exists():
        if not run('initial_evaluate', ['evaluate.py'], 2400):
            state['status'] = 'evaluation_failed'; save(); return
    initial_selection = json.loads((OUT/'selection.json').read_text())
    (OUT/'initial_selection.json').write_text(json.dumps(initial_selection, indent=2))
    state['architectures'] = initial_selection['tuning_architectures']; save()
    for option in ['detail', 'long48', 'seed20261101', 'seed20261102']:
        for arch in state['architectures']:
            route = arch+'__'+option
            if not (OUT/'training'/(route+'.json')).exists():
                run(route, ['train.py', route], 1500)
    run('evaluate', ['evaluate.py'], 3000)
    run('validate', ['validate.py'], 600)
    state['status'] = 'complete' if all(s['status'] == 'complete' for s in state['stages']) else 'complete_with_failures'
    state['seconds'] = time.monotonic()-start; save(); print(json.dumps(state), flush=True)


if __name__ == '__main__':
    if '--launch' in sys.argv:
        if (OUT/'followup_state.json').exists():
            raise SystemExit('Existing followup queue; inspect before relaunch')
        with (OUT/'followup_runner.log').open('w') as f:
            p = subprocess.Popen([PY, '-u', str(Path(__file__))], stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        print(json.dumps(dict(pid=p.pid, output=str(OUT))))
    else:
        main()
