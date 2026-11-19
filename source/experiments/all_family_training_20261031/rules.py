"""Map independently executed blind rules to the shared three coarse groups."""
import collections
import csv
import json
from common import *


def predictions(rows):
    root=SOURCE/'blind-rules';label={r['label_path']:r['id'] for r in rows};data={r['data_path']:r['id'] for r in rows}
    with (root/'07_证据索引/evidence_index.csv').open(encoding='utf-8-sig') as f:
        evidence={r['evidenceId']:r['locator'] for r in csv.DictReader(f)}
    output={r['id']:[0,0,0] for r in rows};counts=collections.Counter();unmapped=[];other=collections.Counter()
    for r in lines(root/'06_问题清单/findings.jsonl'):
        assert r['checkId']!='C10','Reference-assisted rule must not enter blind baseline'
        if r['status']!='不通过':continue
        code=r['defectType'];locator=evidence[r['evidenceRef']].split('#')[0]
        id=label.get(locator) or data.get(locator)
        if code.startswith('L0-') or code=='SAMPLE_DUPLICATE':other[code]+=1;continue
        if code.startswith('GEOMETRY_') or code in ['L2-G','L3-MIS']:k=0
        elif code in ['L3-06','L3-07']:k=1
        elif code in ['L2-P','L2-F','L2-M','L2-T','ANNOTATION_STRUCTURE_CONFLICT']:k=2
        else:other[code]+=1;continue
        if id is None:unmapped.append(dict(findingId=r['findingId'],code=code,locator=locator));continue
        output[id][k]=1;counts[code]+=1
    write(OUT/'rule_mapping.json',dict(mapped=dict(counts),separate_nonsemantic=dict(other),unmapped=unmapped,
        reference_check_C10=False,structure_conflict_group='D: current annotation summary/record consistency'))
    assert not unmapped,'Some blind-rule findings could not be located'
    return output
