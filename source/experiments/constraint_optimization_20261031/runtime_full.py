"""Measure the existing full evaluator, including evidence writes and audit."""
import collections
import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path('/staff/tangjiarui/OPC/opc-agent')
sys.path.insert(0, str(ROOT))
import opc_agent as agent
from tools.audit_run import audit

OUT = ROOT / 'runs/constraint-optimization-20261031'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='full-rule-run')
    args=parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stages = []
    for name in ('inventory_dataset', 'base_analysis', 'apply_reference_checks', 'write_outputs'):
        original = getattr(agent, name)
        def timed(*args, _name=name, _fn=original, **kwargs):
            start = time.monotonic()
            result = _fn(*args, **kwargs)
            record = dict(stage=_name, seconds=time.monotonic()-start)
            if _name in ('inventory_dataset', 'base_analysis'):
                record['dataset'] = str(getattr(args[0], 'dataset_id', args[0]))
            stages.append(record)
            print(json.dumps(record), flush=True)
            return result
        setattr(agent, name, timed)
    start = time.monotonic()
    summary = agent.run(ROOT.parent/'opc1018/newpublish', OUT/args.name,
                        20261021, set(agent.ALL_CHECKS))
    evaluation_seconds = time.monotonic()-start
    audit_start = time.monotonic()
    checked = audit(OUT/args.name)
    result = dict(summary=summary, stages=stages, evaluation_seconds=evaluation_seconds,
                  audit_seconds=time.monotonic()-audit_start, audit=checked,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='All 30 packages, existing 12-check evaluator; no VLM calls.',
                  limitations=['Engineering timing, not independent scientific accuracy.',
                               'Default reference geometry lacks full image transform compensation.',
                               'A100 server CPU/storage; final A10 host not tested.'])
    (OUT/(args.name+'_runtime.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert checked['status'] == 'pass', checked


if __name__ == '__main__':
    main()
