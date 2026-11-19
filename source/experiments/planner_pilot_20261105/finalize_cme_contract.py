"""Apply a source-schema correction to every affected CME task in both arms."""
import concurrent.futures
import datetime as dt
import json
from pathlib import Path
import shutil
import subprocess
import time
import benchmark as b
from resume_rate_limited import PacedAPI

ROOT = b.ROOT/'planner_pilot_20261105'
PRIOR = ROOT/'run_v2'
FINAL = ROOT/'run_v3'
NOTE = '''CME schema clarification: CME event records may have an implicit event type and no explicit category field. For CME only, when there are nonempty records with null categories, do NOT turn the missing explicit category into a type-set contradiction: skip the event_types set comparison. Still compare event_count against record length and has_events against record presence; these remain applicable even if the event list is empty. A zero count, false has_events and empty record list are consistent (pass), not not_applicable.'''

def main():
    started = time.monotonic()
    while True:
        if (PRIOR/'summary.json').exists() and b.read(PRIOR/'summary.json').get('status') == 'complete':
            break
        if time.monotonic()-started > 3600:
            raise RuntimeError('Previous paired phase did not complete within one hour; preserve evidence')
        time.sleep(5)
    FINAL.mkdir(exist_ok=False)
    (FINAL/'runs').mkdir()
    tasks = b.read(PRIOR/'tasks_private.json')
    affected = []
    changes = []
    for task in tasks:
        if task['source']['family'] != 'CME_C2_sample' or 'declaration_conflict' not in task['checks']:
            continue
        affected.append(task['task_id'])
        summary = task['header'].get('event_summary', {})
        records = task['records']['event_records']
        bad = ('event_count' in summary and summary['event_count'] != len(records)) or ('has_events' in summary and bool(summary['has_events']) != bool(records))
        if not records or all(r['category'] is not None for r in records):
            if isinstance(summary.get('event_types'), list):
                bad = bad or set(map(str, summary['event_types'])) != {str(r['category']) for r in records}
        old = task['expected']['declaration_conflict']
        task['expected']['declaration_conflict'] = 'fail' if bad else 'pass'
        changes.append({'task_id':task['task_id'],'old_reference':old,'corrected_reference':task['expected']['declaration_conflict']})
    assert len(affected) == 4
    for path in (PRIOR/'runs').glob('*.json'):
        item = b.read(path)
        if item['task_id'] not in affected:
            shutil.copy2(path,FINAL/'runs'/path.name)
    b.write(FINAL/'tasks_private.json',tasks)
    b.write(FINAL/'schema_amendment.json',{
        'frozen_at':dt.datetime.now().astimezone().isoformat(),'pid':__import__('os').getpid(),
        'reason':'Raw CME records contain no explicit class field. Absence of that optional field must not be treated as proof of a type error.',
        'affected_tasks':affected,'reference_changes':changes,
        'policy':'Re-run every affected task in both arms, regardless of earlier correctness. Preserve all original records. No model weights or thresholds changed.',
        'system_clarification':NOTE,'source_script_sha256':b.sha(__file__)})
    protocol=b.read(PRIOR/'protocol.json')
    protocol.update(version='planner-factual-pilot-1.1-CME-schema',tasks_sha256=b.sha(FINAL/'tasks_private.json'),
                    schema_amendment='schema_amendment.json',source_original='run_v1',
                    final_transport='2 concurrency; 8s shared request spacing; bounded HTTP retries')
    b.write(FINAL/'protocol.json',protocol)
    b.write(FINAL/'preparation.json',{'status':'pass','tasks':36,'groups':{'direct':12,'dependency':12,'multi_tool':12},
         'reference_failed_tasks':sum('fail' in t['expected'].values() for t in tasks),'schema_corrected_tasks':4})
    b.SYSTEM=b.SYSTEM+'\n'+NOTE
    api=PacedAPI(b.provider())
    b.write(FINAL/'launch.json',{'pid':__import__('os').getpid(),'model':api.profile['model'],
         'started_at':dt.datetime.now().astimezone().isoformat(),'concurrency':2,'rerun_tasks':4,'rerun_arms':8})
    jobs=[(t,arm) for t in tasks if t['task_id'] in affected for arm in ['fixed','agent']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(b.run_arm,t,arm,api,FINAL) for t,arm in jobs]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    value=b.summarize(FINAL)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    subprocess.run([str(b.ROOT/'.venv-image-models/bin/python'),str(ROOT/'code/audit_sources.py'),str(FINAL)],check=True)

if __name__=='__main__':
    main()
