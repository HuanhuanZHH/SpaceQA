"""Independent artifact/coverage audit of completed whole-package benchmarks."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(s) for s in stream if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_candidate(candidate,record,routes):
    errors=[];expected=[0,0,0]
    if candidate['sample_id']!=record['id']:errors.append('sample_identifier')
    if candidate['label_path']!=record['label_path'] or candidate['data_path']!=record['data_path']:errors.append('sample_paths')
    if candidate['model_findings_score_eligible'] is not False:errors.append('candidate_promoted_to_score')
    if candidate['scientific_verdict']!='undetermined':errors.append('unsupported_scientific_verdict')
    models=candidate['models']
    if [m['route'] for m in models]!=[r['route'] for r in routes]:errors.append('route_set')
    for model,route in zip(models,routes):
        if model['thresholds']!=route['thresholds']:errors.append('frozen_threshold_changed')
        probability=model['probabilities']
        if record['input_missing']:
            if probability is not None:errors.append('missing_input_probability')
            bits=[0,0,0]
        else:
            if not isinstance(probability,list) or len(probability)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in probability):
                errors.append('invalid_probability');continue
            bits=[int(p>=t) for p,t in zip(probability,route['thresholds'])]
        if model['candidate_bits']!=bits:errors.append('candidate_threshold_recompute')
        expected=[int(a or b) for a,b in zip(expected,bits)]
    if candidate['model_candidate_bits']!=expected:errors.append('expert_union_recompute')
    if any(candidate.get(k) is not None for k in ('F1','recall','precision')):errors.append('unavailable_accuracy_claim')
    return errors


def audit(folder,plan):
    result=read(folder/'complete.json');request=read(folder/'request.json');fresh=folder/'fresh_experts'
    records=rows(fresh/'inventory.jsonl');candidates=rows(fresh/'selected_candidates.jsonl');completion=read(fresh/'complete.json')
    before=read(folder/'input_inventory_before.json');after=read(folder/'input_inventory_after.json');errors=[]
    package=result['package'];lock=read(plan['lock']);routes=lock['family_routes'][package['family']]['experts']
    expected_labels={r['path'] for r in before['files'] if Path(r['path']).suffix.lower()=='.json'}
    expected_data={r['path'] for r in before['files'] if Path(r['path']).suffix.lower() in ('.png','.csv')}
    if len({r['id'] for r in records})!=len(records):errors.append('duplicate_inventory_ids')
    if {r['label_path'] for r in records if r['label_path']}!=expected_labels:errors.append('incomplete_label_coverage')
    if {r['data_path'] for r in records if Path(r['data_path']).is_file()}!=expected_data:errors.append('incomplete_existing_data_coverage')
    if len(records)!=len(candidates):errors.append('candidate_record_count')
    if before!=after:errors.append('input_file_set_or_fingerprint_changed')
    present=sum(not r['input_missing'] for r in records)
    for route in routes:
        name=route['route']
        if completion['fresh_forward_samples'].get(name)!=present:errors.append('fresh_forward_sample_coverage:'+name)
        if completion['fresh_forward_batches'].get(name)!=math.ceil(present/4):errors.append('fresh_batch_count:'+name)
    if completion['cached_probability_reuse']!=0:errors.append('probability_cache_reused')
    if completion['preparation_failures']!=0:errors.append('preparation_failed')
    if completion['input_hashes_unchanged'] is not True:errors.append('source_sha_changed')
    for i,(candidate,record) in enumerate(zip(candidates,records)):
        errors.extend(str(i)+':'+e for e in check_candidate(candidate,record,routes))
    if result['rules_and_fresh_candidates_integrated'] is not False:errors.append('unsupported_integrated_delivery_claim')
    if result['official_deployment_acceptance'] is not False:errors.append('unsupported_hardware_delivery_claim')
    if result['rule_audit']['status']!='pass':errors.append('rules_audit_failed')
    if completion['seconds']>result['seconds']:errors.append('root_timer_shorter_than_fresh_pipeline')
    if not math.isfinite(result['seconds']) or result['seconds']<0:errors.append('invalid_root_time')
    if result['under_3h']!=(result['seconds']<=10800):errors.append('incorrect_per_package_limit')
    if completion['gpu_peak_allocated_bytes']>4*1024**3:errors.append('gpu_allocator_exceeds_budget')
    counts=collections.Counter(r['status'] for r in records)
    return dict(status='pass' if not errors else 'fail',package=package['package'],family=package['family'],
        seconds=result['seconds'],under_3h_on_measured_host=result['seconds']<=10800,records=len(records),
        current_labels=len(expected_labels),current_existing_data=len(expected_data),fresh_samples_per_expert=present,
        status_counts=dict(counts),model_route_names=[r['route'] for r in routes],errors=errors,
        fresh_candidates_separate_from_rule_scorecards=True,A10_acceptance_established=False,official_accuracy_metrics=None,
        source_sha256={str(p):sha(p) for p in [folder/'complete.json',folder/'request.json',fresh/'complete.json',
            fresh/'inventory.jsonl',fresh/'selected_candidates.jsonl',folder/'input_inventory_before.json',folder/'input_inventory_after.json']},
        auditor_sha256=sha(__file__))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=read(a.plan)
    results=[]
    for item in plan['packages']:
        folder=Path(plan['output_root'])/'runs'/item['package']
        if not (folder/'complete.json').exists():continue
        report=audit(folder,plan);target=folder/'independent_audit_v1.json'
        if target.exists():
            if read(target)!=report:raise ValueError('Existing independent audit differs')
        else:
            with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
        results.append({k:v for k,v in report.items() if k not in ('source_sha256',)})
    print(json.dumps(results,ensure_ascii=False))
    if any(r['status']!='pass' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
