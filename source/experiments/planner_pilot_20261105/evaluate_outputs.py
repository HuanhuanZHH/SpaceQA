"""Identical, syntax-only answer extraction for both arms; never repair verdicts."""
import json
from pathlib import Path
import re

def terminal_checks(result, names):
    # Pick the FIRST complete terminal verdict, never the best-scoring verdict.
    # Some OpenAI-compatible responses use {type:json_object,checks:...} rather
    # than the requested action wrapper. That is a format fault, not a changed
    # mechanical verdict. Keep protocol success separate from factual accuracy.
    for index, receipt in enumerate(result['calls']):
        content = receipt.get('content')
        if not isinstance(content, str):
            continue
        text = content.strip()
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or obj.get('action') == 'tool':
            continue
        checks = obj.get('checks')
        if checks is None and set(obj)-{'type','action'} == set(names):
            checks = {name:obj[name] for name in names}
        if not isinstance(checks, dict) or set(checks) != set(names):
            continue
        if not all(isinstance(checks[name], dict) and checks[name].get('status') in
                   ['pass','fail','not_applicable','undetermined'] for name in names):
            continue
        return checks, {'call_index':index,'syntax_normalized':obj.get('action') != 'finish' or 'checks' not in obj}
    return {}, {'call_index':None,'syntax_normalized':False}

def evaluate(root):
    root=Path(root)
    tasks={t['task_id']:t for t in json.loads((root/'tasks_private.json').read_text())}
    rows=[]
    for path in sorted((root/'runs').glob('*.json')):
        result=json.loads(path.read_text())
        task=tasks[result['task_id']]
        checks,meta=terminal_checks(result,task['checks'])
        exact=bool(checks) and all(checks[k]['status']==v for k,v in task['expected'].items())
        evidence={t['evidence_id'] for t in result['tools'] if 'error' not in t['result']}
        references_valid=bool(checks) and all(isinstance(checks[k].get('evidence_ids'),list) and checks[k]['evidence_ids']
                      and all(isinstance(e,str) and e in evidence for e in checks[k]['evidence_ids']) for k in checks)
        rows.append({'task_id':result['task_id'],'arm':result['arm'],'group':result['group'],
                     'checks':checks,'exact_correct':exact,'syntax':meta,
                     'raw_protocol_exact_correct':result['exact_correct'],
                     'raw_terminal_action_valid':result.get('answer') is not None,
                     'references_resolve':bool(references_valid)})
    return {'method':'Same first-complete-verdict extraction for both arms; action/type wrapper normalization only; no label/value/explanation repair',
            'rows':rows}

if __name__=='__main__':
    import sys
    target=Path(sys.argv[1])
    result=evaluate(target)
    (target/'normalized_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'arms':len(result['rows']),'normalized_terminal_answers':sum(bool(r['checks']) for r in result['rows']),
                      'syntax_normalized':sum(r['syntax']['syntax_normalized'] for r in result['rows'])}))
