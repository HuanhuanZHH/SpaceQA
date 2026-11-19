"""Independent artifact integrity checks, grouped splits and metric arithmetic."""
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/spatial-research-20261031'


def lines(path):return [json.loads(s) for s in path.open() if s.strip()]


def main():
    checks=[];notes=[]
    def check(name,condition,**details):checks.append(dict(name=name,passed=bool(condition),**details))
    for dirname in ('spatial-research-20261031','spatial-research-448-20261031'):
        out=ROOT/'runs'/dirname
        if not (out/'detector_results.json').exists():notes.append(dirname+': still pending');continue
        bases=lines(out/'bases.jsonl');rows=lines(out/'dataset.jsonl');summary=json.loads((out/'dataset_summary.json').read_text())
        check(dirname+':manifest_hash',hashlib.sha256((out/'dataset.jsonl').read_bytes()).hexdigest()==summary['data_sha256'])
        check(dirname+':unique_rows',len({r['id'] for r in rows})==len(rows)==summary['records'])
        check(dirname+':counts',len(bases)==1623 and summary['split_counts']=={'train':958,'val':495,'test':170})
        for field in ('base_sample_id','split_group','raw_sha256'):
            sets={s:{b[field] for b in bases if b['split']==s} for s in ('train','val','test')}
            check(dirname+':disjoint_'+field,not sets['train']&(sets['val']|sets['test']) and not sets['val']&sets['test'])
        check(dirname+':p0_negative',all(r['target']==[0,0,0] for r in rows if r['tier']=='P0'))
        check(dirname+':missing_retained_in_summary',summary['skipped']=={'missing_current_data':316})
        d=json.loads((out/'detector_results.json').read_text())
        def verify(m,name):
            if not m.get('samples'):return
            tp=sum(v['tp'] for v in m['per_class'].values());fp=sum(v['fp'] for v in m['per_class'].values());fn=sum(v['fn'] for v in m['per_class'].values())
            check(name+':confusion_sum',(tp,fp,fn)==(m['tp'],m['fp'],m['fn']))
            check(name+':f1_arithmetic',abs(m['micro_f1']-2*tp/max(2*tp+fp+fn,1))<1e-12)
        for route,r in d.items():
            verify(r['validation'],dirname+':'+route+':validation')
            check(dirname+':'+route+':val_fpr_budget',r['validation']['clean_fpr']<=.01)
            for cohort,v in r['cohorts'].items():verify(v['gated'],dirname+':'+route+':'+cohort)
            with np.load(out/(route+'_detector_probabilities.npz')) as z:
                probs=z['probabilities'];targets=z['targets'];ids=z['ids']
            check(dirname+':'+route+':prediction_row_identity',list(ids)==[r['id'] for r in rows] and
                  np.array_equal(targets,np.array([r['target'] for r in rows])))
            for protocol,thresholds in [('primary',r['thresholds']),('multi',r['multi_tier_calibration']['thresholds'])]:
                prediction=probs>=thresholds
                for tier in ('P1','P2_P4'):
                    ii=np.array([i for i,row in enumerate(rows) if row['split']=='test' and
                       (row['tier']=='P1' if tier=='P1' else row['tier'] in ('P2','P3','P4'))])
                    yy=targets[ii].astype(bool);pp=prediction[ii]
                    actual=((yy&pp).sum(),((~yy)&pp).sum(),(yy&~pp).sum())
                    expected=r['cohorts']['ALL:'+tier]['gated'] if protocol=='primary' else r['multi_tier_calibration']['cohorts']['ALL:'+tier]
                    check(dirname+':'+route+':reconstruct_'+protocol+'_'+tier,
                          actual==(expected['tp'],expected['fp'],expected['fn']))
            if route!='annotation_only':
                with np.load(out/(route+'_predictions.npz')) as z:
                    a=z['matched'];b=z['permuted']
                check(dirname+':'+route+':dense_shape',a.shape==(len(bases),summary['image_size'],summary['image_size']))
                check(dirname+':'+route+':finite_probs',np.isfinite(a).all() and a.min()>=0 and a.max()<=1)
                # Within-family/split permutation preserves the multiset of
                # pixel predictions exactly, including all per-base values.
                check(dirname+':'+route+':permutation_conservation',np.isclose(a.astype(float).sum(),b.astype(float).sum(),rtol=0,atol=1e-6))
        original={r['base_sample_id']:r for r in lines(ROOT/'runs/model-comparison-20261031-v2/dataset.jsonl') if r['source']=='public_P0_assisted_real_P1'}
        changes=[]
        for r in rows:
            if r['tier']!='P1':continue
            c=set(original[r['base_sample_id']]['codes']);expected=[int('L2-G' in c),int('L3-06'in c),int('L3-07'in c)]
            if expected!=r['target']:changes.append(dict(id=r['id'],previous=expected,current=r['target']))
        notes.append(dict(run=dirname,P1_truth_differences_from_previous_audit=changes,
            explanation='New proxy uses 0.5 native-pixel coordinate tolerance; prior audit can retain subpixel technical differences.'))
    result=dict(passed=all(c['passed'] for c in checks),checks=len(checks),failed=[c for c in checks if not c['passed']],
        notes=notes,check_details=checks,scientific_accuracy_certified=False)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='check_details'}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
