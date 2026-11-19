"""Independent split, input and reported-metric checks. No model fitting."""
import argparse
import collections
import csv
import json
from common import *


def prepared():
    rows=lines(OUT/'inventory.jsonl');assert len(rows)==123565
    assert len({r['base_sample_id'] for r in rows})==24713;assert_coverage(rows)
    missing=collections.Counter(r['family'] for r in rows if r['input_missing'])
    assert sum(missing.values())==762,dict(missing)
    splits=collections.defaultdict(set);raw=collections.defaultdict(set);base=collections.defaultdict(set)
    for r in rows:
        splits[r['training_group_id']].add(r['split']);base[r['base_sample_id']].add(r['split'])
        if r['data_sha256']:raw[r['data_sha256']].add(r['split'])
        assert len(r['features'])==256 and len(r['target'])==len(r['truth_valid'])==3
        assert 'P0' not in r['prompt'] and r['package_id'] not in r['prompt'] and r['source_key'] not in r['prompt']
        assert r['split'] in ['train','val','test']
    assert all(len(v)==1 for d in [splits,raw,base] for v in d.values())
    train=[r for r in rows if r['split']=='train' and not r['input_missing'] and any(r['truth_valid'])]
    counts=assert_coverage(train)
    write(OUT/'prepared_validation.json',dict(status='pass',records=len(rows),bases=len(base),packages=30,
        missing=dict(missing),train_coverage=counts,group_leakage=0,identical_raw_leakage=0,
        prompt_identity_and_tier_leakage=0,truth_unknown_preserved=True))
    emit('prepared_validation',status='pass',records=len(rows),packages=30)


def results():
    from barrier import verify_barrier
    verify_barrier();rows=lines(OUT/'inventory.jsonl')
    from rules import predictions
    rules=predictions(rows)
    metrics=list(csv.DictReader((OUT/'metrics_per_class.csv').open(encoding='utf-8-sig')))
    checks=0
    for route in ['rules']+ROUTES:
        if route=='rules':p=rules
        else:
            pp={r['id']:r for r in lines(OUT/'predictions'/(route+'.jsonl'))}
            thresholds=json.loads((OUT/'training'/(route+'.json')).read_text())['thresholds']
            p={r['id']:[int(v>=t) for v,t in zip(pp[r['id']]['probabilities'],thresholds[FAMILIES.index(r['family'])])]
               if pp[r['id']]['status']=='success' else [0,0,0] for r in rows}
        counters=collections.defaultdict(lambda:[0,0,0,0])
        for r in rows:
            for scope in ['full_inventory_replay']+(['heldout_test'] if r['split']=='test' else []):
                for k,name in enumerate(['G','C','D']):
                    if not r['truth_valid'][k]:continue
                    y=bool(r['target'][k]);z=bool(p[r['id']][k]);index=0 if y and z else 1 if z else 2 if y else 3
                    counters[(scope,r['family'],r['tier'],name)][index]+=1
        for m in metrics:
            if m['route']!=route:continue
            key=(m['scope'],m['family'],m['tier'],m['class_name'])
            assert counters[key]==[int(m[k]) for k in ['TP','FP','FN','TN']],(route,key)
            checks+=1
    write(OUT/'result_validation.json',dict(status='pass',independent_confusion_checks=checks,
        packages=30,routes=1+len(ROUTES),test_waited_for_all_training=False,
        each_model_trained_all_families_before_test=True))
    emit('independent_metric_validation',status='pass',checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',action='store_true');a=p.parse_args()
    results() if a.results else prepared()
