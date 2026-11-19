"""Independent experiment bookkeeping and numerical metric verification."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/constraint-optimization-20261031'
SRC=Path(__file__).resolve().parent
checks=[]


def check(name, condition):
    checks.append(dict(name=name,passed=bool(condition)))


def load(name):return json.loads((OUT/name).read_text())


def main():
    rows=[json.loads(l) for l in (OUT/'expanded_dataset.jsonl').open()]
    check('unique_sample_ids',len(rows)==len({r['id'] for r in rows}))
    sets={s:{r['split_group'] for r in rows if r['split']==s} for s in ('train','val','test')}
    for a,b in [('train','val'),('train','test'),('val','test')]:check('group_disjoint_'+a+'_'+b,not sets[a]&sets[b])
    check('train_records_3501',sum(r['split']=='train' for r in rows)==3501)
    check('no_synthetic_records',all('synthetic' not in r['source'] for r in rows))
    expected={r['id']:r['target'] for r in map(json.loads,(ROOT/'runs/model-comparison-20261031-v2/extended_test.jsonl').open())}
    actual={r['id']:r['target'] for r in rows if r['split']=='test' and r.get('tier')}
    check('published_test_cohort_identical',expected==actual)
    saved=np.load(OUT/'student_probabilities.npz');lookup={str(v):i for i,v in enumerate(saved['ids'])}
    summary=load('student_optimization.json')
    for route,r in summary.items():
        if 'selected_epoch' not in r:continue
        check(route+'_checkpoint_exists',(OUT/(route+'.pt')).is_file())
        check(route+'_35_updates_in_trace',len(r['trace'])==35)
        for level in ('0.0','0.01','0.05'):
            gates=r['variants'][level]['thresholds']
            for split in ('P1','P2_P4'):
                rr=[r for r in rows if r['split']=='test' and (not r.get('tier') if split=='P1' else r.get('tier'))]
                y=np.array([r['target'] for r in rr]);p=[]
                for x in rr:
                    prob=saved[route][lookup[x['id']]];k=int(prob.argmax())
                    p.append(x['rule_prediction'] or (k if k in (1,2) and prob[k]>=gates[k-1] else 0))
                p=np.array(p);tp=int(((y>0)&(y==p)).sum());fp=int(((p>0)&(y!=p)).sum());fn=int(((y>0)&(y!=p)).sum())
                f1=2*tp/max(2*tp+fp+fn,1)
                check(route+'_'+level+'_'+split+'_F1',abs(f1-r['variants'][level][split]['defect_f1'])<1e-12)
    runtime=load('all_family_vlm_runtime.json')['results']
    check('five_families_two_sizes',len(runtime)==10 and len({r['family'] for r in runtime})==5)
    check('all_640_json_requests_success',sum(r['success'] for r in runtime)==640==sum(r['requests'] for r in runtime))
    check('64_distinct_payloads_each_cell',all(r['unique_input_bytes']==64 for r in runtime))
    for r in runtime:
        check(r['family']+str(r['composite_size'])+'_throughput',abs(r['end_to_end_samples_per_second']-r['requests']/r['combined_seconds'])<1e-9)
    regression=load('fixed_field_regression.json')
    check('all_six_families_reference_regression',len([k for k in regression if not k.startswith('_')])==6)
    for family,cells in regression.items():
        if family.startswith('_'):continue
        for code,m in cells['after'].items():
            check(family+'_'+code+'_reference_P_R_1',m['fp']==0 and m['fn']==0)
    package=load('full-rule-release_runtime.json')
    check('final_full_scan_30_packages',package['summary']['datasetCount']==30)
    check('final_output_audit_pass',package['audit']['status']=='pass' and not package['audit']['errors'])
    check('final_all_12_checks_360_pairs',package['audit']['enabledCheckCount']==12 and package['audit']['completedLogPairs']==360)
    check('full_30_total_less_than_3h',package['evaluation_seconds']+package['audit_seconds']<10800)
    feature=load('feature_export_validation.json')
    check('portable_forest_probabilities_equivalent',feature['portable_probability_max_error']<1e-12)
    check('portable_forest_5923_cases',feature['portable_eval_records']==5923)
    config=load('selected_candidate/config.json')
    ablation=load('selection_and_ablation.json')
    for split in ('P1','P2_P4'):
        check(split+'_saved_CNN_reproduces',ablation[split+'_matched_image']['metrics']==config[split])
    queue=[json.loads(l) for l in (OUT/'localization_review_queue.jsonl').open()]
    check('review_labels_not_claimed_independent',len(queue)==127 and all(r['review_status']=='pending_expert' for r in queue))
    sys.path.insert(0,str(ROOT));import opc_agent
    spec=json.loads((ROOT/'spec/quality_spec.yaml').read_text())
    check('spec_rule_version_synced',spec['ruleVersion']==opc_agent.RULE_VERSION=='2026.10.31-7')
    result=dict(passed=sum(r['passed'] for r in checks),total=len(checks),checks=checks,
        scope='Numerical and artifact consistency only; no claim of expert annotation validity or official hidden score.',
        final_source_sha256=hashlib.sha256((ROOT/'opc_agent.py').read_bytes()).hexdigest())
    (OUT/'optimization_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    assert all(r['passed'] for r in checks),[r for r in checks if not r['passed']]


if __name__=='__main__':main()
