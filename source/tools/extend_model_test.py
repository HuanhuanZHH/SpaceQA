"""Additional P2-P4 weak-label stress set, frozen held-out base groups only."""
import json
from collections import Counter
from pathlib import Path
from prepare_model_comparison import ROOT,OUT,shapes,features,annotation_text,make_view,PROMPT


def geometry(s):
    return tuple(sorted(tuple(p) for p in s.get('points',[])))


def main():
    test={r['base_sample_id']:r for r in map(json.loads,(OUT/'dataset.jsonl').open()) if r['split']=='test'}
    manifest=ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    groups={}
    for r in map(json.loads,manifest.open()):
        if r['base_sample_id'] in test:groups.setdefault(r['base_sample_id'],{})[r['tier']]=r
    rows=[];skipped=Counter()
    for sid,tiers in groups.items():
        p0=tiers['P0'];a=json.load(open(p0['label_path']));sa=shapes(a)
        for tier in ['P2','P3','P4']:
            current=tiers[tier]
            if not current['data_exists']:
                skipped['missing_raw']+=1;continue
            if current['data_sha256']!=p0['data_sha256']:
                skipped['raw_not_identical']+=1;continue
            b=json.load(open(current['label_path']));sb=shapes(b);feat=features(b,current['dataset_family'])
            causes=set()
            if len(sa)!=len(sb):causes.add(2)
            elif sorted(geometry(s) for s in sa)!=sorted(geometry(s) for s in sb):causes.add(1)
            elif sorted((geometry(s),s.get('label')) for s in sa)!=sorted((geometry(s),s.get('label')) for s in sb):causes.add(3)
            if feat[2] or feat[8]:causes.add(3)
            if len(causes)>1:
                skipped['mixed_class']+=1;continue
            row=dict(test[sid]);row['id']=row['id'].replace('_real','_'+tier)
            row.update(target=next(iter(causes),0),source='P0_rule_derived_unreviewed_P2_P4',tier=tier,
                       features=feat,prompt=PROMPT+annotation_text(b,current['dataset_family']),
                       rule_prediction=3 if feat[2] or feat[8] else 1 if feat[3] else 0)
            image=OUT/'images'/(row['id']+'.jpg')
            if not image.exists():make_view(current['data_path'],b,image)
            row['image']=str(image)
            rows.append(row)
    with (OUT/'extended_test.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    summary=dict(records=len(rows),cells=dict(Counter(f"{r['family']}:{r['tier']}:{r['target']}" for r in rows)),
                 skipped=dict(skipped),limitation='Rule-derived labels; held-out groups but no manual review. P0-reference score is circular on this set.')
    (OUT/'extended_test_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
