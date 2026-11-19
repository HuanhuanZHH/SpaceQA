"""All-family accuracy-first late fusion; never mutates frozen experiments."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, os, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/staff/tangjiarui/OPC')
FAMILIES = ['AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample']
ARCHS = ['unet','attention_unet','unetpp','resnet18_unet','deeplab_mobilenet','yolo11_detect_backbone','yolo11_segment_backbone','yolov8_segment_backbone','dino_finetune','ijepa_finetune']
ROUTES = ['qlora']+[f'{x}_seed{s}' for x in ['supervised','distilled','pruned'] for s in [20261031,20261101,20261102]]+ARCHS+['gbdt_prior_v1']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:
        for l in f:
            if l.strip(): yield json.loads(l)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def log(s,**kw):print(json.dumps(dict(status=s,**kw)),flush=True)

def partition(records):
    """Whole duplicate/source groups, including any cross-family groups."""
    membership=collections.defaultdict(set)
    for r in records:membership[r['training_group_id']].add(r['family'])
    by_family=collections.defaultdict(list)
    for g,fs in membership.items():by_family[min(fs)].append(g)
    assign={}
    for fs,gs in by_family.items():
        ordered=sorted(gs,key=lambda g:hashlib.sha256(('20261104:'+g).encode()).hexdigest())
        a,b=int(.6*len(gs)),int(.8*len(gs))
        for i,g in enumerate(ordered):assign[g]='fit' if i<a else 'cal' if i<b else 'select'
    return np.array([assign[r['training_group_id']] for r in records])

def features(records,p,rule):
    """Explicit whitelist, unavailable experts masked, no target/tier/IDs."""
    z=np.clip(np.nan_to_num(p,nan=.5),1e-5,1-1e-5)
    logits=np.log(z/(1-z))
    structural=np.asarray([r['features'] for r in records],np.float32)
    assert structural.shape==(len(records),256)
    family=np.eye(6,dtype=np.float32)[[FAMILIES.index(r['family']) for r in records]]
    return np.concatenate([z.reshape(len(p),-1),logits.reshape(len(p),-1),structural,family,rule],axis=1).astype(np.float32)

def metric(y,v,p):
    tp=int((y&v&p).sum());fp=int((~y&v&p).sum());fn=int((y&v&~p).sum());tn=int((~y&v&~p).sum())
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,FPR=fp/(fp+tn) if fp+tn else None)

def objective(y,v,p,tier):
    f=[metric(y[tier==t],v[tier==t],p[tier==t])['F1'] for t in range(1,5)]
    clean=tier==0
    return (float(np.mean([x for x in f if x is not None])) if any(x is not None for x in f) else 0.,f[0] or 0.,-float(p[clean].any(1).mean()) if clean.any() else 0.)

def calibrate(prob,rule,y,v,family,tier):
    th=np.full((6,3),1.01)
    grid=np.r_[np.arange(.02,1,.02),.99,1.01]
    for fi in range(6):
        m=family==fi;q=prob[m];r=rule[m];yy=y[m];vv=v[m];tt=tier[m]
        pred=r.copy();clean=tt==0
        cap=max(.02,float(r[clean].any(1).mean()) if clean.any() else 0.)
        for _ in range(2):
            for bit in range(3):
                key=objective(yy,vv,pred,tt)+(th[fi,bit],);best=th[fi,bit]
                for threshold in grid:
                    trial=pred.copy();trial[:,bit]=r[:,bit]|(q[:,bit]>=threshold)
                    if clean.any() and trial[clean].any(1).mean()>cap+1e-12:continue
                    k=objective(yy,vv,trial,tt)+(float(threshold),)
                    if k>key:key,best=k,float(threshold)
                th[fi,bit]=best;pred[:,bit]=r[:,bit]|(q[:,bit]>=best)
    return th

def decide(prob,rule,th,family): return rule|(prob>=th[family])

def top3_indices(p,y,v,family,fit):
    result=np.zeros((6,3,3),int)
    q=np.clip(np.nan_to_num(p,nan=.5),1e-5,1-1e-5)
    for fi in range(6):
        for b in range(3):
            m=fit&(family==fi)&v[:,b]
            yy=y[m,b,None];loss=-(yy*np.log(q[m,:,b])+(~yy)*np.log(1-q[m,:,b])).mean(0)
            result[fi,b]=np.argsort(loss,kind='stable')[:3]
    return result

def aggregate(p,family,top):
    q=np.clip(np.nan_to_num(p,nan=.5),1e-5,1-1e-5)
    mean=np.mean(q,axis=1);z=np.mean(np.log(q/(1-q)),axis=1)
    chosen=np.zeros((len(p),3))
    for fi in range(6):
        for b in range(3):chosen[family==fi,b]=q[family==fi,:,b][:,top[fi,b]].mean(1)
    return {'mean21':mean,'logit_mean21':1/(1+np.exp(-z)),'fit_top3':chosen}

def train_stack(x,y,v,fit,family,out):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    configs={'logistic_C01':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=20261104)),
             'logistic_C1':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=500,random_state=20261104)),
             'histgb_leaf7':lambda:HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=5.,learning_rate=.05,early_stopping=False,random_state=20261104),
             'histgb_leaf15':lambda:HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=5.,learning_rate=.05,early_stopping=False,random_state=20261104)}
    bundle={}
    coverage={}
    for name,ctor in configs.items():
        models=[]
        for b in range(3):
            m=fit&v[:,b];counts=collections.Counter(family[m].tolist());assert len(counts)==6
            model=ctor();model.fit(x[m],y[m,b]);models.append(model)
            coverage[f'{name}:{b}']={'n':int(m.sum()),'positives':int(y[m,b].sum()),'family_counts':dict(counts)}
        bundle[name]=models
        log('stack_trained',name=name)
    with (out/'stack_models.pkl').open('wb') as f:pickle.dump(bundle,f)
    return bundle,coverage

def stack_predict(bundle,x):return {name:np.stack([m.predict_proba(x)[:,1] for m in models],axis=1) for name,models in bundle.items()}

def bootstrap(records,y,v,a,b,replicates=1000):
    """Paired resample at group level, conditional on fixed trained policy."""
    groups=sorted({r['training_group_id'] for r in records});idx={g:i for i,g in enumerate(groups)}
    ca=np.zeros((len(groups),3),np.int32);cb=ca.copy()
    for j,r in enumerate(records):
        if r['tier']=='P0':continue
        k=idx[r['training_group_id']]
        for c,p in [(ca,a),(cb,b)]:c[k]+=[int((y[j]&v[j]&p[j]).sum()),int((~y[j]&v[j]&p[j]).sum()),int((y[j]&v[j]&~p[j]).sum())]
    rng=np.random.default_rng(20261104);delta=[]
    for _ in range(replicates):
        sample=rng.integers(0,len(groups),len(groups));aa=ca[sample].sum(0);bb=cb[sample].sum(0)
        f=lambda c:2*c[0]/max(1,2*c[0]+c[1]+c[2])
        delta.append(f(bb)-f(aa))
    return {'replicates':replicates,'unit':'training_group_id; P1-P4 micro F1','delta_95_percentile':np.quantile(delta,[.025,.975]).tolist(),'limitation':'Conditional on fixed learned policy; does not include training or selection uncertainty.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();assert out.is_relative_to(ROOT/'jwvl_optimization_20261104') and not out.exists();out.mkdir(parents=True)
    start=time.time()
    priorroot=ROOT/'spaceqa-agent/experiments/prior_optimization_20261102';sys.path.insert(0,str(priorroot))
    import compare_versions as cv
    from compare_gbdt_extension import prior_from_file,predictions_for
    from evaluate_roadshow_prior import selected_choices
    base=ROOT/'opc-agent/runs/all-family-training-20261031';gbdt=ROOT/'inference_prior_20261102/gbdt'
    comp=ROOT/'inference_prior_20261102/comparison_v1';ext=ROOT/'inference_prior_20261102/gbdt_extension_v1'
    hashes=read(base/'training_barrier.json')['split_hashes'];old=read(comp/'frozen_validation_policy.json');extension=read(ext/'frozen_validation_extension.json')
    locators,_=cv.load_old_rule_locators(ROOT/'opc-agent/runs/full-matrix-20261031/blind-rules')
    provenance={r:cv.verify_route(base,r,hashes) for r in ROUTES if r!='gbdt_prior_v1'}
    marker=read(gbdt/'predictions/gbdt_prior_v1.complete.json')
    provenance['gbdt_prior_v1']={'records':marker['records'],'prediction_sha256':marker['prediction_hash']}
    assert sha(gbdt/'training.json')==read(gbdt/'training_barrier.json')['training_sha256']
    train_groups={r['training_group_id'] for r in rows(base/'train.jsonl')}
    assert sha(base/'train.jsonl')==hashes['train']
    choices,gc,force=selected_choices(old)
    def load(split):
        records=cv.load_split(base/(split+'.jsonl'),split,hashes[split])
        # Feature whitelist loaded from source only after split integrity check.
        feature_map={r['id']:r['features'] for r in rows(base/(split+'.jsonl'))}
        for r in records:r['features']=feature_map[r['id']]
        assert not train_groups.intersection(r['training_group_id'] for r in records)
        pure=prior_from_file(records,comp/'prior_outputs'/(split+'.jsonl'))
        rule=np.array(cv.retained_prior(pure,cv.old_bits(records,locators)),bool)
        probs={}
        for name in ROUTES:
            probs[name]=cv.load_probabilities(gbdt if name=='gbdt_prior_v1' else base,name,records,provenance[name])
            log('cache_verified',split=split,expert=name)
        preds=predictions_for(records,rule.tolist(),probs,probs['gbdt_prior_v1'],choices,gc,force,extension['thresholds'])
        baseline=np.array([preds[extension['selected'][r['family']]['variant']][i] for i,r in enumerate(records)],bool)
        p=np.array([[probs[n][i] if probs[n][i] is not None else [np.nan]*3 for n in ROUTES] for i in range(len(records))],np.float32)
        y=np.array([r['target'] for r in records],bool);v=np.array([r['truth_valid'] for r in records],bool)
        family=np.array([FAMILIES.index(r['family']) for r in records]);tier=np.array([int(r['tier'][1]) for r in records])
        present=np.array([not r['input_missing'] for r in records])
        return records,rule,baseline,p,y,v,family,tier,present
    records,rule,baseline,p,y,v,family,tier,present=load('val')
    part=partition(records);fit=(part=='fit')&present;cal=part=='cal';select=part=='select'
    covers={}
    for s in ['fit','cal','select']:
        covers[s]=dict(collections.Counter(r['family']+'|'+r['tier'] for i,r in enumerate(records) if part[i]==s));assert len(covers[s])==30
    top=top3_indices(p,y,v,family,fit)
    bundle,coverage=train_stack(features(records,p,rule),y,v,fit,family,out)
    candidates=aggregate(p,family,top);candidates.update(stack_predict(bundle,features(records,p,rule)))
    thresholds={};val_preds={'release052':baseline};diagnostics=[]
    for name,prob in candidates.items():
        th=calibrate(prob[cal],rule[cal],y[cal],v[cal]&present[cal,None],family[cal],tier[cal]);thresholds[name]=th
        pred=decide(prob,rule,th,family);pred[~present]=rule[~present];val_preds[name]=pred
    selected={}
    for fi,f in enumerate(FAMILIES):
        m=select&(family==fi)
        options=[]
        for name,pred in val_preds.items():
            key=objective(y[m],v[m],pred[m],tier[m]);options.append((key,name));diagnostics.append(dict(family=f,variant=name,macro_F1=key[0],P1_F1=key[1],P0_FPR=-key[2]))
        selected[f]=max(options,key=lambda z:(z[0],z[1]=='release052',z[1]))[1]
    policy=dict(status='frozen',test_parsed=False,selection_source='group-held validation meta_select',selected=selected,thresholds={k:t.tolist() for k,t in thresholds.items()},top3_indices=top.tolist(),experts=ROUTES,split_hashes=hashes,meta_coverage=covers,head_training_coverage=coverage,provenance=provenance,script_sha256=sha(__file__),stack_sha256=sha(out/'stack_models.pkl'),feature_allowlist='expert probabilities/logits + current 256 features + family schema + rule bits',size_constraint=None,known_limitations=['Historical test reused for reporting; not fresh blind test','Original expert checkpoints selected with original validation; not nested CV','G/C/D P0-derived proxy; not official nine-class localization F1'])
    write(out/'frozen_policy.json',policy);freeze_hash=sha(out/'frozen_policy.json');cv.write_csv(out/'meta_selection.csv',diagnostics)
    log('frozen_before_test',selected=selected,sha256=freeze_hash)
    val_groups={r['training_group_id'] for r in records}
    records,rule,baseline,p,y,v,family,tier,present=load('test');assert not val_groups.intersection(r['training_group_id'] for r in records)
    candidates=aggregate(p,family,top);candidates.update(stack_predict(bundle,features(records,p,rule)))
    test_preds={'release052':baseline}
    for name,prob in candidates.items():
        pred=decide(prob,rule,thresholds[name],family);pred[~present]=rule[~present];test_preds[name]=pred
    test_preds['validation_selected_fusion']=np.array([test_preds[selected[r['family']]][i] for i,r in enumerate(records)])
    summaries=[];packages=[];groups=[]
    for name,pred in test_preds.items():
        summary,pack,gr=cv.summarize(records,pred.tolist(),name);summaries.append(summary);packages.extend(pack);groups.extend(gr)
        # Independently vectorized micro counts must equal legacy scalar evaluation.
        assert metric(y[tier>0],v[tier>0],pred[tier>0])==summary['overall']
    cv.write_csv(out/'all30_test.csv',packages);cv.write_csv(out/'groups_test.csv',groups)
    cv.write_csv(out/'summary_test.csv',[dict(variant=s['variant'],macro_F1_24=s['macro_F1_24'],P1_macro_F1_6=s['P1_macro_F1_6'],P0_sample_FPR=s['P0_sample_FPR'],**s['overall']) for s in summaries])
    with (out/'selected_test_predictions.jsonl').open('w') as f:
        for i,r in enumerate(records):f.write(json.dumps(dict(id=r['id'],input_sha256=r['input_sha256'],family=r['family'],tier=r['tier'],selected=selected[r['family']],prediction=test_preds['validation_selected_fusion'][i].tolist(),baseline=baseline[i].tolist(),truth_valid=r['truth_valid']))+'\n')
    assert sha(out/'frozen_policy.json')==freeze_hash
    result=dict(status='complete',seconds=time.time()-start,packages=30,experts=21,validation_records=len(part),test_records=len(records),test_used_for_selection=False,selected=selected,summaries=summaries,bootstrap=bootstrap(records,y,v,baseline,test_preds['validation_selected_fusion']),frozen_policy_sha256=freeze_hash,scope='P0-derived G/C/D grouped held-out technical proxy; cached expert outputs; no new forward-pass timing')
    write(out/'comparison.json',result);log('complete',output=str(out),summaries=[{k:s[k] for k in ['variant','overall','macro_F1_24','P1_macro_F1_6']} for s in summaries])

if __name__=='__main__':main()
