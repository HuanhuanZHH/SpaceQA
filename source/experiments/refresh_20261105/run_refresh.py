"""Re-evaluate real roadshow inputs and run a new RGB forward on every image."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/staff/tangjiarui/OPC/eight_class_20261104')
OUT = BASE / 'refresh_20261105'
PYTHON = '/staff/tangjiarui/OPC/.venv-image-models/bin/python'
CODE = BASE / 'refresh_code_20261105'


def write(name, value):
    target = OUT / name
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(target)


def main():
    OUT.mkdir(exist_ok=False)
    env = dict(os.environ)
    for key in list(env):
        if ('API_KEY' in key or key.startswith('SPACEQA_CHAT_LLM_')):
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6', PYTHONPATH=str(BASE / 'runtime/src'), PYTHONUNBUFFERED='1')
    started = time.monotonic()
    plan = dict(startedAt=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(), gpu='physical GPU1', batch=16,
                fresh_rule_revalidation=True, new_eight_class_fusion_forward=True,
                original_21_expert_predictions='verified frozen cache reused, explicitly disclosed',
                fresh_localizer_input='every readable AIA/EPB raw RGB image, no labels entering neural network',
                threshold_policy=str(BASE / 'candidate_comparison_v1/deployment_policy.json'),
                no_training=True, no_provider_api_calls=True,
                stages=['fresh_rules_and_eight_heads', 'all_readable_image_forward', 'localization_completion_and_summary'])
    write('plan.json', plan)
    commands = [
        ('fresh_rules_and_eight_heads', [PYTHON, '-u', str(BASE / 'code/roadshow.py'), '--fusion', str(BASE / 'fusion_v3_missing4'), '--output', str(OUT / 'classification')]),
        ('all_readable_image_forward', [PYTHON, '-u', str(CODE / 'ground_all_refresh.py'), '--roadshow', str(OUT / 'classification'), '--localizer', str(BASE / 'localizer_v2'), '--comparison', str(BASE / 'candidate_comparison_v1'), '--output', str(OUT / 'grounded'), '--batch', '16']),
    ]
    for stage, command in commands:
        child = subprocess.Popen(command, env=env, cwd=str(BASE / 'code'))
        while child.poll() is None:
            write('state.json', dict(stage=stage, childPid=child.pid, elapsedSeconds=round(time.monotonic()-started,2), status='running'))
            time.sleep(3)
        if child.returncode:
            raise RuntimeError(f'{stage} failed with code {child.returncode}')
    write('state.json', dict(stage='fresh_inference_complete_pending_postprocessing', status='inference_complete', elapsedSeconds=round(time.monotonic()-started,2)))
    print(json.dumps(dict(stage='fresh_inference_complete_pending_postprocessing', seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            write('failure.json', dict(type=type(error).__name__, message=str(error)))
        raise
