"""Freeze, evaluate and independently verify one fully trained six-family model."""
import argparse
import collections
import csv
import json
import os
import time
import numpy as np
from common import *
from barrier import build_route_barrier,verify_barrier
from scheduling import complete,lock,unlock


def csv_atomic(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def refresh():
    held=lock('evaluation_progress',blocking=True)
    try:
        ready=[r for r in ROUTES if complete(OUT/'evaluations'/r/'complete.json')]
        if not ready:return
        progress=OUT/'progress';progress.mkdir(exist_ok=True)
        for name in ['metrics_long.csv','metrics_per_class.csv','heldout_test_expanded.csv','heldout_test_F1.csv']:
            combined={}
            for route in ready:
                with (OUT/'evaluations'/route/name).open(encoding='utf-8-sig',newline='') as stream:
                    for row in csv.DictReader(stream):
                        key=tuple(row.get(k) for k in ['scope','route','family','tier','class_name'])
                        if key in combined:assert combined[key]==row,(name,key)
                        combined[key]=row
            csv_atomic(progress/name,list(combined.values()))
        write(progress/'status.json',dict(status='partial' if len(ready)<len(ROUTES) else 'all_models_evaluated',
            evaluated_routes=ready,evaluated_count=len(ready),planned_count=len(ROUTES),packages=30,
            primary='heldout_test',seed_rows='individual; not a three-seed mean',updated_unix=time.time()))
    finally:unlock(held)


def score_route(route):
    from evaluate import confusion
    from rules import predictions as rule_predictions
    barrier=verify_barrier(route);rows=lines(OUT/'inventory.jsonl');folder=OUT/'evaluations'/route
    check=json.loads((OUT/'predictions'/(route+'.complete.json')).read_text())
    assert check['prediction_hash']==sha(OUT/'predictions'/(route+'.jsonl'))
    assert check['barrier_hash']==sha(OUT/'route_barriers'/(route+'.json'))
    predicted={r['id']:r for r in lines(OUT/'predictions'/(route+'.jsonl'))}
    assert set(predicted)=={r['id'] for r in rows}
    thresholds=json.loads((OUT/'training'/(route+'.json')).read_text())['thresholds']
    rules=rule_predictions(rows);decisions={route:{},route+'_fixed05':{},'rules':rules}
    states={r['id']:predicted[r['id']]['status'] for r in rows}
    statuses={route:states,route+'_fixed05':states,'rules':{r['id']:'success' for r in rows}}
    for r in rows:
        p=predicted[r['id']]
        for name,ts in [(route,thresholds[FAMILIES.index(r['family'])]),(route+'_fixed05',[.5]*3)]:
            decisions[name][r['id']]=[int(a>=b) for a,b in zip(p['probabilities'],ts)] if p['status']=='success' else [0,0,0]
    metrics=[];classes=[]
    for scope in ['heldout_test','full_inventory_replay']:
        for name,pred in decisions.items():
            for family in FAMILIES:
                for tier in TIERS:
                    subset=[r for r in rows if r['family']==family and r['tier']==tier and (scope!='heldout_test' or r['split']=='test')]
                    summary,perclass=confusion(subset,pred,statuses[name]);common=dict(scope=scope,route=name,family=family,tier=tier)
                    metrics.append(dict(common,**summary));classes.extend(dict(common,**m) for m in perclass)
    csv_atomic(folder/'metrics_long.csv',metrics);csv_atomic(folder/'metrics_per_class.csv',classes)
    for scope in ['heldout_test','full_inventory_replay']:
        wide=[];overview=[]
        for name in decisions:
            expanded=dict(route=name);f1=dict(route=name)
            for r in metrics:
                if r['scope']!=scope or r['route']!=name:continue
                prefix=r['family']+'_'+r['tier']
                for m in ['F1','precision','recall','FPR','n']:expanded[prefix+'_'+m]=r[m]
                f1[prefix]=r['F1']
            assert len(expanded)==151 and len(f1)==31
            wide.append(expanded);overview.append(f1)
        csv_atomic(folder/(scope+'_expanded.csv'),wide);csv_atomic(folder/(scope+'_F1.csv'),overview)
    # Independent scalar confusion accumulation, then compare every written class row.
    counters=collections.defaultdict(lambda:[0,0,0,0])
    for r in rows:
        for scope in ['full_inventory_replay']+(['heldout_test'] if r['split']=='test' else []):
            for name,pred in decisions.items():
                for k,label in enumerate(['G','C','D']):
                    if not r['truth_valid'][k]:continue
                    y=bool(r['target'][k]);z=bool(pred[r['id']][k])
                    index=0 if y and z else 1 if z else 2 if y else 3
                    counters[(scope,name,r['family'],r['tier'],label)][index]+=1
    with (folder/'metrics_per_class.csv').open(encoding='utf-8-sig') as stream:
        written=list(csv.DictReader(stream))
    assert len(written)==540
    for m in written:
        key=tuple(m[k] for k in ['scope','route','family','tier','class_name'])
        assert counters[key]==[int(m[k]) for k in ['TP','FP','FN','TN']],key
    verification=dict(status='pass',independent_confusion_checks=len(written),packages=30,
        trained_all_six_families_before_evaluation=True,other_models_may_still_be_training=True)
    write(folder/'validation.json',verification)
    heldout=[m for m in metrics if m['scope']=='heldout_test' and m['route']==route]
    tier_summary={t:float(np.mean([m['F1'] for m in heldout if m['tier']==t and m['F1'] is not None]))
                  for t in TIERS if any(m['tier']==t and m['F1'] is not None for m in heldout)}
    return dict(status='complete',route=route,packages=30,records=len(rows),heldout_records=sum(r['split']=='test' for r in rows),
        train_records=barrier['train_records'],heldout_tier_macro_f1=tier_summary,
        barrier_sha256=sha(OUT/'route_barriers'/(route+'.json')),prediction_sha256=check['prediction_hash'],
        validation_sha256=sha(folder/'validation.json'),truth='public P0 technical proxy; unknown bits masked; not official hidden score',
        threshold_selection_split='val',test_feedback_into_training=False)


def evaluate_route(route):
    assert route in ROUTES
    held=lock('evaluate_route_'+route,blocking=True);start=time.monotonic()
    try:
        folder=OUT/'evaluations'/route;folder.mkdir(parents=True,exist_ok=True)
        if complete(folder/'complete.json'):
            verify_barrier(route);refresh();return
        state=dict(status='checking_training',route=route,pid=os.getpid(),started_unix=time.time())
        write(folder/'state.json',state)
        if (OUT/'route_barriers'/(route+'.json')).exists():verify_barrier(route)
        else:build_route_barrier(route)
        if not complete(OUT/'predictions'/(route+'.complete.json')):
            state['status']='evaluating_30_packages';write(folder/'state.json',state)
            from evaluate import inference
            inference(route)
        state['status']='scoring_and_independent_verification';write(folder/'state.json',state)
        result=score_route(route);result['seconds']=time.monotonic()-start
        write(folder/'complete.json',result);write(folder/'state.json',result);refresh()
        emit('model_evaluation_complete',**result)
    except BaseException as exc:
        write(OUT/'evaluations'/route/'state.json',dict(status='failed',route=route,error=repr(exc),pid=os.getpid()))
        raise
    finally:unlock(held)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('routes',nargs='+');args=p.parse_args()
    for route in args.routes:evaluate_route(route)
