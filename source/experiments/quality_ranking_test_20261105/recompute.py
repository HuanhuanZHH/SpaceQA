"""Recompute held-out quality-order diagnostics from frozen eight-class outputs.

No inference, training, threshold fitting, or roadshow pseudo-tier assignment.
Reference masks and P0-derived labels stay in evaluation only.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

from scipy.stats import spearmanr

ROOT = Path('/staff/tangjiarui/OPC')
FINE = ROOT/'eight_class_20261104'
BASE = ROOT/'opc-agent/runs/all-family-training-20261031'
FAMILIES = ['AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample']
SHORT = dict(zip(FAMILIES, ['AIA','HMI','CME','EPB','DMSP','ULF']))
CLASSES = ['L0-01','L2-G','L2-P','L2-T','L2-F','L3-06','L3-07','L3-MIS']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def csv_write(path, records):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)


def counts(target, mask, prediction):
    c = collections.Counter()
    for y, valid, p in zip(target, mask, prediction):
        if valid: c['TP' if y and p else 'FN' if y else 'FP' if p else 'TN'] += 1
    return c


def metric(c):
    tp, fp, fn, tn = (int(c[k]) for k in ['TP','FP','FN','TN'])
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,
        precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/(tp+fn) if tp+fn else None,
        F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        FPR=fp/(fp+tn) if fp+tn else None)


def rho(xs, ys):
    if len(set(ys)) < 2: return None
    value = float(spearmanr(xs,ys).statistic)
    return value if math.isfinite(value) else None


def order_summary(packages, value_key):
    within = []
    for family in sorted({p['family'] for p in packages}):
        group = sorted([p for p in packages if p['family']==family and p['tier']!='P0'],key=lambda p:p['tier'])
        assert len(group)==4
        good = wrong = tied = 0
        for i,a in enumerate(group):
            for b in group[i+1:]:
                difference = b[value_key]-a[value_key]
                if abs(difference)<1e-12: tied+=1
                elif difference>0: good+=1
                else: wrong+=1
        within.append(dict(family=family,name=SHORT[family],
            spearman_P1_P4=rho([1,2,3,4],[p[value_key] for p in group]),
            correct_pairs=good,reversed_pairs=wrong,tied_pairs=tied,total_pairs=6))
    valid = [r['spearman_P1_P4'] for r in within if r['spearman_P1_P4'] is not None]
    nonclean = [p for p in packages if p['tier']!='P0']
    return dict(metric=value_key,per_family=within,
        mean_within_family_spearman=sum(valid)/len(valid) if valid else None,
        defined_families=len(valid),
        global_spearman_P1_P4=rho([int(p['tier'][1]) for p in nonclean],[p[value_key] for p in nonclean]),
        correct_pairs=sum(r['correct_pairs'] for r in within),
        reversed_pairs=sum(r['reversed_pairs'] for r in within),
        tied_pairs=sum(r['tied_pairs'] for r in within),
        total_pairs=sum(r['total_pairs'] for r in within))


def main(out):
    started = time.monotonic()
    out.mkdir(parents=True,exist_ok=False)
    policy_path = FINE/'fusion_v2/frozen_policy.json'
    policy = read(policy_path)
    assert policy['classes']==CLASSES
    assert sha(FINE/'fusion_v2/models.pkl')==policy['models_sha256']
    sources = dict(test_source=BASE/'test.jsonl',prediction=FINE/'fusion_v2/test_predictions.jsonl',
                   target=FINE/'targets_v3/test.jsonl',policy=policy_path,
                   models=FINE/'fusion_v2/models.pkl',target_protocol=FINE/'targets_v3/protocol.json',
                   recorded_metrics=FINE/'fusion_v2/all30_strict_mask_test.csv')
    source_hashes = {name:dict(path=str(path),sha256=sha(path)) for name,path in sources.items()}
    assert source_hashes['test_source']['sha256']==policy['source_hashes']['test']
    predictions = {r['id']:r for r in rows(sources['prediction'])}
    labels = {r['id']:r for r in rows(sources['target'])}
    original = list(rows(sources['test_source']))
    assert len(predictions)==len(labels)==len(original)==18450
    assert set(predictions)==set(labels)=={r['id'] for r in original}
    test_groups = {r['training_group_id'] for r in original}
    for split in ['train','val']:
        path = BASE/(split+'.jsonl')
        digest = sha(path)
        assert digest==policy['source_hashes'][split]
        groups = {r['training_group_id'] for r in rows(path)}
        assert not test_groups.intersection(groups)
        source_hashes[split+'_source'] = dict(path=str(path),sha256=digest)
    counters = collections.defaultdict(collections.Counter)
    sums = collections.defaultdict(collections.Counter)
    class_counts = collections.defaultdict(collections.Counter)
    with gzip.open(out/'paired_test_samples.jsonl.gz','wt',encoding='utf-8') as fp:
        for record in original:
            sid = record['id'];pred = predictions[sid];reference = labels[sid]
            assert pred['family']==reference['family']==record['family']
            assert pred['tier']==reference['tier']==record['tier']
            assert pred['source_input_sha256']==record['input_sha256']
            assert pred['rule']==reference['rule'] and pred['known_target']==reference['valid']
            mask = reference.get('strict_valid',reference['valid'])
            family = record['family'];key = (family,record['tier'])
            fi = FAMILIES.index(family)
            reconstructed=[];weights=[];routes=[]
            for b,code in enumerate(CLASSES):
                route = policy['selected'][family][code];routes.append(route)
                probability = float(pred['rule'][b]) if route=='rules' else pred['probabilities_by_variant'][route][b]
                flag = pred['rule'][b] or (route!='rules' and probability>=policy['thresholds'][route][fi][b])
                if record['input_missing']: flag = pred['rule'][b]
                reconstructed.append(bool(flag))
                weights.append(1. if pred['rule'][b] else probability if flag else 0.)
            assert reconstructed==pred['prediction'], sid
            p = pred['prediction'];y=reference['target'];rule=pred['rule']
            counters[key].update(counts(y,mask,p))
            for b,code in enumerate(CLASSES):class_counts[(family,record['tier'],code)].update(counts([y[b]],[mask[b]],[p[b]]))
            s=sums[key];s['samples']+=1;s['detected_samples']+=int(any(p))
            s['detected_known_samples']+=int(any(a and valid for a,valid in zip(p,mask)))
            s['confirmed_rule_samples']+=int(any(rule))
            s['model_candidate_samples']+=int(any(a and not r for a,r in zip(p,rule)))
            s['reference_known_positive_samples']+=int(any(a and valid for a,valid in zip(y,mask)))
            s['known_class_slots']+=sum(mask);s['unknown_class_slots']+=8-sum(mask)
            s['weighted_burden_sum']+=sum(weights)
            fp.write(json.dumps(dict(id=sid,family=family,tier=record['tier'],
                training_group_id=record['training_group_id'],prediction=p,target=y,strict_valid=mask,
                rule=rule,routed_models=routes,weighted_burden=sum(weights),
                label_path=record['label_path'],data_path=record['data_path'],
                label_sha256=record['label_sha256'],data_sha256=record.get('data_sha256'),
                source_input_sha256=record['input_sha256']),ensure_ascii=False)+'\n')
    recorded = {}
    with sources['recorded_metrics'].open(encoding='utf-8-sig',newline='') as fp:
        for r in csv.DictReader(fp):
            if r['variant']=='selected_eight':recorded[(r['family'],r['tier'])]=r
    assert len(recorded)==30
    packages=[]
    for family in FAMILIES:
        for tier in ['P0','P1','P2','P3','P4']:
            key=(family,tier);s=sums[key];n=int(s['samples']);m=metric(counters[key])
            for k in ['TP','FP','FN','TN']:assert m[k]==int(recorded[key][k])
            packages.append(dict(family=family,name=SHORT[family],tier=tier,samples=n,
                detected_samples=int(s['detected_samples']),detected_sample_rate=s['detected_samples']/n,
                detected_known_samples=int(s['detected_known_samples']),
                detected_known_sample_rate=s['detected_known_samples']/n,
                confirmed_rule_samples=int(s['confirmed_rule_samples']),
                model_candidate_samples=int(s['model_candidate_samples']),
                reference_known_positive_samples=int(s['reference_known_positive_samples']),
                reference_known_positive_rate=s['reference_known_positive_samples']/n,
                weighted_pollution_index=s['weighted_burden_sum']/n,
                known_class_slots=int(s['known_class_slots']),unknown_class_slots=int(s['unknown_class_slots']),
                reference_slot_coverage=s['known_class_slots']/(8*n),**m))
    def scope(families):
        items=[p for p in packages if p['family'] in families]
        pos=[p for p in items if p['tier']!='P0'];clean=[p for p in items if p['tier']=='P0']
        total=collections.Counter()
        for p in pos:total.update({k:p[k] for k in ['TP','FP','FN','TN']})
        f1=[p['F1'] for p in pos if p['F1'] is not None]
        clean_n=sum(p['samples'] for p in clean)
        clean_flagged=sum(p['detected_known_samples'] for p in clean)
        assert not sum(p['reference_known_positive_samples'] for p in clean)
        return dict(families=families,total_samples=sum(p['samples'] for p in items),packages=len(items),
            P1_P4_micro=metric(total),P1_P4_macro_package_F1=sum(f1)/len(f1) if f1 else None,
            macro_defined_packages=len(f1),macro_total_packages=len(pos),
            P0=dict(known_false_positive_samples=clean_flagged,samples=clean_n,
                    known_sample_false_positive_rate=clean_flagged/clean_n,
                    all_flagged_samples=sum(p['detected_samples'] for p in clean)),
            detected_rate_ranking=order_summary(items,'detected_sample_rate'),
            weighted_index_ranking=order_summary(items,'weighted_pollution_index'))
    result=dict(status='complete',generated_at=dt.datetime.now().astimezone().isoformat(),
        source='Full original grouped test split; frozen fusion_v2 selected_eight predictions',
        records=18450,packages=30,classes=CLASSES,
        all_six=scope(FAMILIES),three_for_original_slide=scope(['AIA_AR_sample','ULF_sample','EPB_seg_sample']),
        sources=source_hashes,source_groups_disjoint_from_train_and_val=True,
        frozen_policy_and_model_hash_verified=True,all_predictions_reconstructed=True,
        all_package_counts_match_existing_strict_test_table=True,
        primary_ranking='Mean of six within-family Spearman correlations of detected sample rate vs P1-P4 ordinal tier',
        global_ranking='Across all 24 family/tier packages; report separately from within-family mean',
        weighted_index='Per sample sum of above-threshold selected class probabilities; current confirmed rule flags use unit weight; divide by all samples. Range 0..8, not a calibrated true pollution fraction.',
        limitations=['Frozen prediction re-evaluation, zero new model forward passes',
                     'Historical public test, not a newly sealed blind set',
                     'Strict P0-derived technical labels with unknowns masked; not official eight-class localization metrics',
                     'Detected samples include model candidates; not all confirmed scientific defects',
                     'Tier-order correlation is not actual defect-proportion estimation error or quality-grade classification accuracy',
                     'P0 ratio is within known proxy checks, not a measured before/after evidence-gate gain'],
        seconds=time.monotonic()-started)
    write(out/'results.json',result)
    csv_write(out/'all30_packages.csv',packages)
    csv_write(out/'ranking_by_family.csv',result['all_six']['detected_rate_ranking']['per_family'])
    csv_write(out/'all30_classes.csv',[dict(family=family,tier=tier,code=code,**metric(c))
        for (family,tier,code),c in class_counts.items()])
    write(out/'verification.json',dict(status='pass',records=18450,packages=30,
        input_hashes_verified=True,source_group_overlap_with_train_val=0,
        frozen_predictions_reconstructed=18450,package_count_vectors_independently_matched=30,
        paired_ledger_sha256=sha(out/'paired_test_samples.jsonl.gz'),
        source_files_unchanged=all(sha(v['path'])==v['sha256'] for v in source_hashes.values())))
    assert read(out/'verification.json')['source_files_unchanged']
    print(json.dumps(dict(status='complete',records=18450,seconds=time.monotonic()-started,
        all_six=result['all_six'],three_for_original_slide=result['three_for_original_slide']),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
