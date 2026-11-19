"""Recompute measured pilot tables. Calibration uses validation only."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from learn_model_comparison import OUT,load,metrics,write


def main():
    train,val,test=load()
    teachers={r['id']:r for r in map(json.loads,(OUT/'teacher_predictions.jsonl').open()) if r['success']}
    result={}
    def add(name,preds,extra=None):
        result[name]=dict(metrics=metrics(test,preds),by_family={
            family:metrics([r for r in test if r['family']==family],preds)
            for family in sorted({r['family'] for r in test})},**(extra or {}))
    add('blind_structure_rules',{r['id']:r['rule_prediction'] for r in test})
    from prepare_model_comparison import ROOT,shapes
    references={}
    for directory in ['epb_p0_p1_audit_20261027','dmsp_p0_p1_audit_20261029']:
        for line in (ROOT/'phase2_visual_audit_xzj'/directory/'sample_comparisons.jsonl').open():
            rr=json.loads(line);references[rr['base_sample_id']]=rr['paths']
    reference_predictions={}
    for r in test:
        paths=references[r['base_sample_id']]
        a=shapes(json.load(open(paths['P0']['label'])))
        b=shapes(json.load(open(paths['P1']['label'])))
        def geometry(s):return tuple(sorted(tuple(p) for p in s.get('points',[])))
        ga=sorted(geometry(s) for s in a);gb=sorted(geometry(s) for s in b)
        pred=r['rule_prediction']
        if not pred:
            if len(a)!=len(b):pred=2
            elif ga!=gb:pred=1
            elif sorted((geometry(s),s.get('label')) for s in a)!=sorted((geometry(s),s.get('label')) for s in b):pred=3
        reference_predictions[r['id']]=pred
    add('P0_reference_difference_rules',reference_predictions,dict(
        limitation='Uses clean reference at inference; labels are P0-assisted too. Consistency check, not independent accuracy.'))
    add('always_none',{r['id']:0 for r in test})
    add('qwen8b_original_api',{k:r['prediction'] for k,r in teachers.items()})
    # A model confidence threshold is selected on validation, never test.
    candidates=[]
    for threshold in [0,.25,.5,.6,.7,.8,.85,.9,.95,.98,.99,.995,.999,1.01]:
        def pred(row):
            if row['rule_prediction']!=0:return row['rule_prediction']
            value=teachers.get(row['id'])
            if value is None:return 0
            probs=value['probabilities'];index=max(range(1,4),key=lambda j:probs[j])
            return index if probs[index]>=threshold else 0
        vm=metrics(val,{r['id']:pred(r) for r in val})
        if vm['clean_false_positive_rate']<=.05:
            candidates.append((vm['defect_f1'],threshold,vm))
    _,threshold,vm=max(candidates,key=lambda v:(v[0],v[1]))
    calibrated={}
    for r in test:
        probs=teachers[r['id']]['probabilities'];j=max(range(1,4),key=lambda j:probs[j])
        calibrated[r['id']]=r['rule_prediction'] or (j if probs[j]>=threshold else 0)
    add('rules_plus_calibrated_qwen8b',calibrated,dict(validation_threshold=threshold,validation_metrics=vm))
    for path in OUT.glob('qlora_*.json'):
        if path.stem not in ('qlora_before','qlora_after'):continue
        data=json.loads(path.read_text());add(path.stem,data['predictions'])
    for path in sorted(OUT.glob('student_*seed*.json')):
        data=json.loads(path.read_text())
        for name,value in data.items():
            if isinstance(value,dict) and 'predictions' in value:
                add(path.stem+'_'+name,value['predictions'],{k:value[k] for k in ('parameters','artifact_bytes')})
    proof=OUT/'evidence_review_predictions.jsonl'
    if proof.exists():
        add('qwen8b_evidence_first',{r['id']:r['prediction'] for r in map(json.loads,proof.open()) if r['success']})
    qpath=OUT/'qlora_after.json'
    if qpath.exists():
        qdata=json.loads(qpath.read_text());probs=qdata['probabilities'];best=None
        def gated(rows,tb,tc):
            result={}
            for row in rows:
                p=probs[row['id']];index=max(range(4),key=lambda j:p[j]);threshold=tb if index==1 else tc
                result[row['id']]=row['rule_prediction'] or (index if index in (1,2) and p[index]>=threshold else 0)
            return result
        for tb in [.4,.5,.6,.7,.8,.9,.95,.98,.99,.999,1.01]:
            for tc in [.4,.5,.6,.7,.8,.9,.95,.98,.99,.999,1.01]:
                vm=metrics(val,gated(val,tb,tc))
                if vm['clean_false_positive_rate']>.05:continue
                if best is None or (vm['defect_f1'],tb+tc)>best[0]:best=((vm['defect_f1'],tb+tc),tb,tc,vm)
        _,tb,tc,vm=best
        extra=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
        add('rules_plus_calibrated_qlora',gated(test,tb,tc),dict(thresholds=[tb,tc],validation_metrics=vm,
            extended_metrics=metrics(extra,gated(extra,tb,tc))))
    batches={}
    runtime_root=OUT.parent/'model-comparison-20261031'
    for path in sorted(runtime_root.glob('batch_*.jsonl')):
        rows=[json.loads(s) for s in path.open()]
        levels={}
        for c in [1,2,4,8,16,32]:
            rr=[r for r in rows if r['concurrency']==c]
            if not rr:continue
            n=sum(r['requests'] for r in rr);wall=sum(r['wall_seconds'] for r in rr)
            levels[c]=dict(repeats=len(rr),requests=n,success=sum(r['success'] for r in rr),
                          requests_per_second=n/wall,amortized_seconds=wall/n,
                          mean_p95_seconds=statistics.mean(r['latency_p95'] for r in rr),
                          peak_memory_mib=max((r['peak_memory_mib'] for r in rr if r['peak_memory_mib']),default=None),
                          tokens_per_request=statistics.mean(x['usage']['completion_tokens'] for r in rr for x in r['results'] if x.get('usage')))
        batches[path.stem]=levels
    write(OUT/'comparison_summary.json',dict(models=result,batches=batches,
          runtime_source=str(runtime_root),
          dataset_sha256=hashlib.sha256((OUT/'dataset.jsonl').read_bytes()).hexdigest(),
          test_support={c:sum(r['target']==c for r in test) for c in range(4)},
          limitation='Public P0-assisted sample-class pilot. No independent scientific truth, localization or official score.'))
    with (OUT/'accuracy_comparison.csv').open('w',newline='') as f:
        fields=['model','n','accuracy','macro_f1','defect_precision','defect_recall','defect_f1','clean_false_positive_rate','parameters','artifact_bytes']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for k,v in result.items():writer.writerow(dict(model=k,**{x:v['metrics'][x] for x in fields[1:8]},parameters=v.get('parameters'),artifact_bytes=v.get('artifact_bytes')))
    print(json.dumps({k:v['metrics'] for k,v in result.items()},indent=2))

if __name__=='__main__':main()
