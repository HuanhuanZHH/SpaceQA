"""Resume only transport failures; freeze all completed model answers, including errors."""
from __future__ import annotations
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import random
import shutil
import threading
import time
import urllib.error
import urllib.request
import benchmark as b

ROOT = b.ROOT / 'planner_pilot_20261105'
INITIAL = ROOT / 'run_v1'
OUT = ROOT / 'run_v2'

class PacedAPI(b.API):
    def __init__(self, profile):
        super().__init__(profile)
        self.lock = threading.Lock()
        self.next_allowed = 0.

    def pace(self):
        while True:
            with self.lock:
                wait = self.next_allowed - time.monotonic()
                if wait <= 0:
                    self.next_allowed = time.monotonic() + 8.
                    return
            time.sleep(min(wait, 1.))

    def call(self, messages):
        payload = {'model': self.profile['model'], 'messages': messages, 'temperature': 0,
                   'max_tokens': 900, 'response_format': {'type': 'json_object'}}
        receipt = {'request_sha256': b.digest(payload), 'attempts': [], 'transport_policy': 'shared-8s-interval-60s-429-cooldown'}
        start = time.monotonic()
        for attempt in range(3):
            self.pace()
            try:
                request = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.profile['api_key']})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=60) as response:
                    result = json.loads(response.read(1024 * 1024))
                content = result['choices'][0]['message'].get('content')
                receipt.update(seconds=time.monotonic() - start, model=result.get('model'), usage=result.get('usage', {}),
                               content=content, finish_reason=result['choices'][0].get('finish_reason'), response_id=result.get('id'))
                receipt['attempts'].append({'attempt': attempt + 1, 'status': 'success'})
                if not isinstance(content, str):
                    raise ValueError('empty_content')
                text = content.strip()
                if text.startswith('```'):
                    text = b.re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError('not_object')
                return value, receipt
            except urllib.error.HTTPError as exc:
                receipt['attempts'].append({'attempt': attempt + 1, 'status': 'http_error', 'http_code': exc.code})
                if exc.code in [429, 500, 502, 503, 504] and attempt < 2:
                    with self.lock:
                        self.next_allowed = max(self.next_allowed, time.monotonic() + (60 if exc.code == 429 else 8))
                    continue
                receipt.update(seconds=time.monotonic() - start, error='http_error', http_code=exc.code)
                return None, receipt
            except (TimeoutError, OSError) as exc:
                receipt['attempts'].append({'attempt': attempt + 1, 'status': type(exc).__name__})
                if attempt < 2:
                    continue
                receipt.update(seconds=time.monotonic() - start, error=type(exc).__name__)
                return None, receipt
            except Exception as exc:
                receipt.update(seconds=time.monotonic() - start, error=type(exc).__name__)
                return None, receipt
        raise RuntimeError('unreachable')

def finish_metadata(task, result, started, previous_seconds):
    result['seconds'] = previous_seconds + time.monotonic() - started
    result['logical_model_calls'] = len(result['calls'])
    result['successful_model_responses'] = sum(isinstance(c.get('content'), str) and not c.get('error') for c in result['calls'])
    result['actual_http_attempts'] = sum(len(c.get('attempts', [])) for c in result['calls'])
    result['token_usage'] = {key: sum(c.get('usage', {}).get(key, 0) for c in result['calls']) for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']}
    result['messages_sha256'] = b.digest(result['messages'])
    b.score(task, result)
    return result

def resume_arm(task, arm, api):
    target = OUT / 'runs' / (task['task_id'] + '__' + arm + '.json')
    if target.exists():
        return b.read(target)
    old = INITIAL / 'runs' / target.name
    if not old.exists():
        return b.run_arm(task, arm, api, OUT)
    result = b.read(old)
    assert result['answer'] is None
    assert result['calls'] and result['calls'][-1].get('error') in ['http_error', 'TimeoutError', 'URLError', 'OSError']
    result['transport_resume_from'] = str(old)
    result['retained_successful_decisions'] = sum(isinstance(c.get('content'), str) and not c.get('error') for c in result['calls'])
    previous_seconds, started = result['seconds'], time.monotonic()
    messages = result['messages']
    # The last failed request had no model content. Re-send that exact context,
    # preserving every earlier tool and model decision. Do not retry a wrong verdict.
    successful = result['retained_successful_decisions']
    resolved = any(t['name'] == 'resolve_data' and 'error' not in t['result'] for t in result['tools'])
    remaining = 1 if arm == 'fixed' else max(0, 5 - successful)
    for step in range(remaining):
        answer, receipt = api.call(messages)
        result['calls'].append(receipt)
        if answer is None:
            break
        messages.append({'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)})
        if answer.get('action') == 'finish':
            result['answer'] = answer
            break
        if arm == 'fixed' or answer.get('action') != 'tool' or len(result['tools']) >= 4:
            messages.append({'role': 'user', 'content': 'Invalid action. Use one allowed tool action or finish; budget is unchanged.'})
            continue
        name, args = answer.get('name'), answer.get('arguments', {})
        value, resolved = b.tool(task, name, args, resolved)
        observation = {'evidence_id': 'E' + str(len(result['tools']) + 1), 'name': name, 'arguments': args, 'result': value}
        result['tools'].append(observation)
        messages.append({'role': 'user', 'content': 'Tool observation: ' + json.dumps(observation, ensure_ascii=False)})
        if len(result['tools']) >= 4:
            messages.append({'role': 'user', 'content': 'Budget reached. Return action=finish with checks now; do not request another tool.'})
    finish_metadata(task, result, started, previous_seconds)
    b.write(target, result)
    with b.LOCK:
        finished = len(list((OUT / 'runs').glob('*.json')))
        b.write(OUT / 'state.json', {'status': 'running', 'completed_arms': finished, 'total_arms': 72,
                                 'updated_at': dt.datetime.now().astimezone().isoformat()})
        print(json.dumps({'completed_arms': finished, 'arm': arm, 'task_id': task['task_id'],
                          'correct': result['exact_correct'], 'resumed_transport_only': True}), flush=True)
    return result

def main():
    OUT.mkdir(exist_ok=True)
    (OUT / 'runs').mkdir(exist_ok=True)
    protocol = b.read(INITIAL / 'protocol.json')
    assert b.sha(ROOT / 'code/benchmark.py') == protocol['script_sha256']
    assert b.sha(INITIAL / 'tasks_private.json') == protocol['tasks_sha256']
    for name in ['tasks_private.json', 'protocol.json', 'preparation.json']:
        if not (OUT / name).exists():
            shutil.copy2(INITIAL / name, OUT / name)
    tasks = b.read(OUT / 'tasks_private.json')
    retained, retry = [], []
    for p in (INITIAL / 'runs').glob('*.json'):
        item = b.read(p)
        transport_failure = item['answer'] is None and item['calls'] and item['calls'][-1].get('error') in ['http_error', 'TimeoutError', 'URLError', 'OSError']
        if transport_failure:
            retry.append(p.name)
        else:
            retained.append(p.name)
            if not (OUT / 'runs' / p.name).exists():
                shutil.copy2(p, OUT / 'runs' / p.name)
    b.write(OUT / 'execution_override.json', {'frozen_at': dt.datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
        'reason': 'Observed real HTTP429 during 4-concurrency initial run; preserve model decisions and resume transport failures only',
        'concurrency': 2, 'minimum_request_interval_seconds': 8, 'http429_cooldown_seconds': 60,
        'max_http_attempts_per_logical_request': 3, 'unchanged': ['task_set', 'reference', 'prompts', 'model', 'tool_budget', 'successful_model_decisions'],
        'retained_completed_arms': retained, 'transport_failure_arms_to_resume': retry,
        'source_script_sha256': b.sha(__file__),
        'inflight_limit': 'At termination of initial worker, up to four in-flight responses may lack receipts; their token consumption is unknown and not invented'})
    api = PacedAPI(b.provider())
    b.write(OUT / 'launch.json', {'pid': os.getpid(), 'model': api.profile['model'],
          'started_at': dt.datetime.now().astimezone().isoformat(), 'concurrency': 2, 'gpu_used': False})
    jobs = [(t, arm) for t in tasks for arm in ['fixed', 'agent'] if not (OUT/'runs'/(t['task_id']+'__'+arm+'.json')).exists()]
    random.Random(b.SEED + 2).shuffle(jobs)
    print(json.dumps({'retained_answers': len(retained), 'transport_failures_to_resume': len(retry), 'remaining_arms': len(jobs)}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(resume_arm, task, arm, api) for task, arm in jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    print(json.dumps(b.summarize(OUT), ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
