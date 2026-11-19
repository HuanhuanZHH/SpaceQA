"""Eight binary heads: all-family current features and frozen-expert fusion.

Targets are conservative technical proxies, not official localized defect truth.
All choices are frozen on grouped validation before held-out test evaluation.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
from fine_labels import CLASSES,NAMES

ROOT=Path('/staff/tangjiarui/OPC')
MISSING_POSITIVE_WEIGHT=1.
sys.path.insert(0,str(ROOT/'jwvl_optimization_20261104/code'))
import ensemble as e


def csv_write(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def probabilities(models,x):
    result=np.zeros((len(x),8),np.float32)
    for b,model in enumerate(models):
        if model is None:continue
        pr=model.predict_proba(x)
        if 1 in model.classes_:result[:,b]=pr[:,list(model.classes_).index(1)]
    return result


def train_heads(x,y,v,fit,family,config,out,name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    models=[];coverage={}
    for b,code in enumerate(CLASSES):
        m=fit&v[:,b];n=int(m.sum());positive=int(y[m,b].sum())
        if n and positive and positive<n:
            model=HistGradientBoostingClassifier(**config,early_stopping=False,random_state=20261104)
            weights=np.ones(int(m.sum()),np.float32)
            if b==3:weights[(family[m]==2)&y[m,b]]=.5 # supplemental CME identity is weak
            if b==5:weights[y[m,b]]=MISSING_POSITIVE_WEIGHT
            model.fit(x[m],y[m,b],sample_weight=weights);models.append(model)
        else:models.append(None)
        coverage[code]=dict(records=n,positive=positive,negative=n-positive,learned=models[-1] is not None,
            family_records=dict(collections.Counter(e.FAMILIES[i] for i in family[m])),
            family_positive=dict(collections.Counter(e.FAMILIES[i] for i in family[m&y[:,b]])))
        e.log('fine_head_trained',variant=name,code=code,**coverage[code])
    e.write(out/(name+'_training_coverage.json'),coverage)
    return models


def threshold_key(y,v,p,tier):
    scores=[]
    for t in [1,2,3,4]:
        m=tier==t;score=e.metric(y[m],v[m],p[m])['F1']
        if score is not None:scores.append(score)
    score=e.metric(y,v,p)['F1']
    return (float(np.mean(scores)) if scores else 0.,score or 0.)


def calibrate(q,rule,y,v,fi,tier):
    th=np.full((6,8),1.01);grid=np.r_[np.arange(.02,1,.02),.99,1.01]
    support={}
    for b,code in enumerate(CLASSES):
        for family in [-1,0,1,2,3,4,5]:
            m=np.ones(len(y),bool) if family==-1 else fi==family
            yy=y[m,b];vv=v[m,b];tt=tier[m];rr=rule[m,b];qq=q[m,b]
            pos=int((yy&vv).sum());neg=int((~yy&vv).sum())
            if family!=-1 and (pos<8 or neg<16):
                support[e.FAMILIES[family]+'|'+code]='global_fallback: fewer than 8 positives or 16 negatives';continue
            best=(threshold_key(yy,vv,rr,tt)+(1.01,),1.01)
            clean=(tt==0)&vv&~yy
            baseline=float(rr[clean].mean()) if clean.any() else 0.
            for t in grid:
                p=rr|(qq>=t)
                if clean.any() and p[clean].mean()>max(.02,baseline)+1e-12:continue
                key=threshold_key(yy,vv,p,tt)+(float(t),)
                if key>best[0]:best=(key,float(t))
            if family==-1:th[:,b]=best[1]
            else:th[family,b]=best[1];support[e.FAMILIES[family]+'|'+code]=dict(positive=pos,negative=neg)
    return th,support


def feature_current(records,rule):
    # No paired reference, tier, ID, filename or target is introduced here.
    structural=np.asarray([r['features'] for r in records],np.float32)
    fi=np.eye(6,dtype=np.float32)[[e.FAMILIES.index(r['family']) for r in records]]
    return np.concatenate([structural,fi,rule.astype(np.float32)],axis=1)


def load_split(split,targets,policy,with_experts=True):
    base=ROOT/'opc-agent/runs/all-family-training-20261031'
    hashes=e.read(base/'training_barrier.json')['split_hashes'];target_protocol=e.read(targets/'protocol.json')
    assert target_protocol['source_split_hashes'][split]==hashes[split]==e.sha(base/(split+'.jsonl'))
    records=list(e.rows(base/(split+'.jsonl')));labels=list(e.rows(targets/(split+'.jsonl')))
    assert [r['id'] for r in records]==[r['id'] for r in labels]
    rule=np.array([r['rule'] for r in labels],bool)
    y=np.array([r['target'] for r in labels],bool);v=np.array([r['valid'] for r in labels],bool)
    family=np.array([e.FAMILIES.index(r['family']) for r in records]);tier=np.array([int(r['tier'][1]) for r in records])
    if with_experts:
        sys.path.insert(0,str(ROOT/'spaceqa-agent/experiments/prior_optimization_20261102'))
        import compare_versions as cv
        from compare_gbdt_extension import prior_from_file
        # These are current-input deterministic bits used by the original 391 features.
        loc,_=cv.load_old_rule_locators(ROOT/'opc-agent/runs/full-matrix-20261031/blind-rules')
        broad_rule=np.array(cv.retained_prior(prior_from_file(records,ROOT/'inference_prior_20261102/comparison_v1/prior_outputs'/(split+'.jsonl')),cv.old_bits(records,loc)),bool)
        p={}
        for route in e.ROUTES:
            p[route]=cv.load_probabilities(ROOT/'inference_prior_20261102/gbdt' if route=='gbdt_prior_v1' else base,route,records,policy['provenance'][route])
        prob=np.array([[p[n][i] if p[n][i] is not None else [np.nan]*3 for n in e.ROUTES] for i in range(len(records))],np.float32)
        x=e.features(records,prob,broad_rule)
        x=np.concatenate([x,rule.astype(np.float32)],axis=1) # 391 + 8 fine rule flags = 399
        assert x.shape[1]==399
    else:x=None
    return records,rule,y,v,family,tier,x


def summary(records,y,v,p,variant):
    rows=[];details=[]
    for f in e.FAMILIES:
        for tier in ['P0','P1','P2','P3','P4']:
            m=np.array([r['family']==f and r['tier']==tier for r in records])
            rows.append(dict(variant=variant,family=f,tier=tier,records=int(m.sum()),known_class_slots=int(v[m].sum()),
                unknown_class_slots=int((~v[m]).sum()),**e.metric(y[m],v[m],p[m])))
            for b,code in enumerate(CLASSES):
                details.append(dict(variant=variant,family=f,tier=tier,code=code,name=NAMES[b],records=int(m.sum()),
                    known=int(v[m,b].sum()),unknown=int((~v[m,b]).sum()),positives=int((y[m,b]&v[m,b]).sum()),
                    **e.metric(y[m,b],v[m,b],p[m,b])))
    nonclean=np.array([r['tier']!='P0' for r in records]);scientific=[1,3,5,6,7]
    return dict(variant=variant,scope='masked P0-derived eight-class technical proxy, NOT official localized F1',
        all_P1_P4=e.metric(y[nonclean],v[nonclean],p[nonclean]),
        scientific_proxy_P1_P4=e.metric(y[nonclean][:,scientific],v[nonclean][:,scientific],p[nonclean][:,scientific]),
        macro_package_F1=float(np.mean([r['F1'] for r in rows if r['F1'] is not None and r['tier']!='P0'])),
        known_class_slots=int(v.sum()),unknown_class_slots=int((~v).sum())),rows,details


def main(args):
    out=args.output.resolve();assert not out.exists();out.mkdir(parents=True);started=time.monotonic()
    policy=e.read(ROOT/'jwvl_optimization_20261104/ensemble_v1/frozen_policy.json')
    config7=dict(max_iter=150,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=5.,learning_rate=.05)
    config15=dict(max_iter=200,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=5.,learning_rate=.05)
    # A current-feature baseline learns from ALL original train groups first.
    tr,rr,yy,vv,ff,tt,_=load_split('train',args.targets,policy,False)
    train_groups={r['training_group_id'] for r in tr}
    current=train_heads(feature_current(tr,rr),yy,vv,np.ones(len(tr),bool),ff,config15,out,'current_HGB15_eight')
    del tr,rr,yy,vv,ff,tt
    records,rule,y,v,fi,tier,x=load_split('val',args.targets,policy)
    assert not train_groups.intersection(r['training_group_id'] for r in records)
    part=e.partition(records);fit=part=='fit';cal=part=='cal';select=part=='select'
    bundle={'current_HGB15_eight':current}
    for name,config in [('fusion_HGB7_eight',config7),('fusion_HGB15_eight',config15)]:
        bundle[name]=train_heads(x,y,v,fit,fi,config,out,name)
    with (out/'models.pkl').open('wb') as fp:pickle.dump(bundle,fp)
    e.log('all_training_complete',models=len(bundle),heads=8,records_all_train=86505,meta_fit=int(fit.sum()))
    q={name:probabilities(models,feature_current(records,rule) if name.startswith('current') else x) for name,models in bundle.items()}
    thresholds={};preds={'rules':rule.copy()};support={}
    missing=np.array([r['input_missing'] for r in records])
    for name,prob in q.items():
        th,su=calibrate(prob[cal],rule[cal],y[cal],v[cal],fi[cal],tier[cal]);thresholds[name]=th;support[name]=su
        pred=rule|(prob>=th[fi]);pred[missing]=rule[missing];preds[name]=pred
    selected={};selection_rows=[]
    for f,family in enumerate(e.FAMILIES):
        m=select&(fi==f);selected[family]={}
        for b,code in enumerate(CLASSES):
            candidates=[]
            for name,p in preds.items():
                key=threshold_key(y[m,b],v[m,b],p[m,b],tier[m]);candidates.append((key,name))
                selection_rows.append(dict(family=family,code=code,variant=name,macro_tier_F1=key[0],F1=key[1],
                    **{k:val for k,val in e.metric(y[m,b],v[m,b],p[m,b]).items() if k!='F1'}))
            selected[family][code]=max(candidates,key=lambda a:(a[0],a[1]=='rules',a[1]=='fusion_HGB7_eight',a[1]))[1]
    frozen=dict(status='frozen',test_read=False,classes=CLASSES,selected=selected,thresholds={k:v.tolist() for k,v in thresholds.items()},
        calibration_support=support,models_sha256=e.sha(out/'models.pkl'),targets_protocol_sha256=e.sha(args.targets/'protocol.json'),
        meta_split='whole training_group_id, original validation 60/20/20 fit/cal/select',feature_dimensions={'current':270,'fusion':399},
        time_supervision=e.read(args.targets/'protocol.json').get('time_supervision'),
        missing_positive_weight=MISSING_POSITIVE_WEIGHT,
        fine_rule_classes=['L0-01','L2-P vocabulary subset','L2-F'],source_policy_sha256=e.sha(ROOT/'jwvl_optimization_20261104/ensemble_v1/frozen_policy.json'),
        head_semantics=dict(zip(CLASSES,NAMES)),source_hashes=e.read(args.targets/'protocol.json')['source_split_hashes'],
        inference_inputs='frozen current-only expert probabilities + structural hash features + family schema + actual current rules',
        limitations=['Independent scientific fine labels unavailable','Historical test was previously reported for other models; not new blind data',
            'Original experts used the full original validation for checkpoint selection; meta holdout is not nested CV',
            'Unknown labels masked and coverage reported; fine classifiers alone do not localize defects'])
    e.write(out/'frozen_policy.json',frozen);freeze_sha=e.sha(out/'frozen_policy.json');csv_write(out/'selection.csv',selection_rows)
    # ALL train/fusion heads and selection are complete before parsing test.
    validation_groups={r['training_group_id'] for r in records}
    def choose(predictions,recs):
        result=np.zeros((len(recs),8),bool)
        for i,r in enumerate(recs):
            for b,code in enumerate(CLASSES):result[i,b]=predictions[selected[r['family']][code]][i,b]
        return result
    preds['selected_eight']=choose(preds,records)
    val_summary=[];vp=[];vd=[]
    held=[records[i] for i in np.where(select)[0]]
    for name,p in preds.items():
        s,a,b=summary(held,y[select],v[select],p[select],name);val_summary.append(s);vp.extend(a);vd.extend(b)
    e.write(out/'validation_summary.json',val_summary);csv_write(out/'all30_validation.csv',vp);csv_write(out/'all30_classes_validation.csv',vd)
    records,rule,y,v,fi,tier,x=load_split('test',args.targets,policy)
    assert not validation_groups.intersection(r['training_group_id'] for r in records)
    assert not train_groups.intersection(r['training_group_id'] for r in records)
    preds={'rules':rule.copy()};q={name:probabilities(models,feature_current(records,rule) if name.startswith('current') else x) for name,models in bundle.items()}
    missing=np.array([r['input_missing'] for r in records])
    for name,prob in q.items():
        pred=rule|(prob>=thresholds[name][fi]);pred[missing]=rule[missing];preds[name]=pred
    preds['selected_eight']=choose(preds,records)
    ts=[];tp=[];td=[];strict_summaries=[];strict_packages=[];strict_details=[]
    strict=np.array([r.get('strict_valid',r['valid']) for r in e.rows(args.targets/'test.jsonl')],bool)
    for name,p in preds.items():
        s,a,b=summary(records,y,v,p,name);ts.append(s);tp.extend(a);td.extend(b)
        s,a,b=summary(records,y,strict,p,name);strict_summaries.append(s);strict_packages.extend(a);strict_details.extend(b)
    csv_write(out/'all30_test.csv',tp);csv_write(out/'all30_classes_test.csv',td)
    csv_write(out/'all30_strict_mask_test.csv',strict_packages);csv_write(out/'all30_classes_strict_mask_test.csv',strict_details)
    with (out/'test_predictions.jsonl').open('w') as fp:
        for i,r in enumerate(records):fp.write(json.dumps(dict(id=r['id'],family=r['family'],tier=r['tier'],classes=CLASSES,
            prediction=preds['selected_eight'][i].tolist(),known_target=v[i].tolist(),rule=rule[i].tolist(),
            probabilities_by_variant={k:p[i].tolist() for k,p in q.items()},source_input_sha256=r['input_sha256']))+'\n')
    assert freeze_sha==e.sha(out/'frozen_policy.json')
    e.write(out/'comparison.json',dict(status='complete',seconds=time.monotonic()-started,validation=val_summary,test=ts,strict_mask_test=strict_summaries,
        test_records=len(records),packages=30,classes=8,selected=selected,frozen_policy_sha256=freeze_sha,
        independent_localized_metrics=None,official_critical_recall=None,scope='Eight-class masked P0-derived technical proxy'))
    e.log('complete',seconds=time.monotonic()-started,output=str(out),test=ts)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--targets',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--missing-positive-weight',type=float,default=1.)
    args=ap.parse_args();MISSING_POSITIVE_WEIGHT=args.missing_positive_weight
    try:main(args)
    except BaseException as exc:
        if args.output.exists():e.write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
