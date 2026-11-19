"""Real-only multi-tier training and validation-only precision calibration.

All tiers inherit the frozen event-day/duplicate split. Existing public tests are
reused benchmark sets, not a fresh confirmation set. No output enters scoring.
"""
import collections
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path('/staff/tangjiarui/OPC/opc-agent')
sys.path.insert(0, str(ROOT/'tools'))
from learn_model_comparison import Student, metrics
from prepare_model_comparison import shapes, features, make_view

OLD = ROOT/'runs/model-comparison-20261031-v2'
OUT = ROOT/'runs/constraint-optimization-20261031'
SEEDS = [20261031, 20261101, 20261102]


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def lines(path):
    return [json.loads(s) for s in path.open() if s.strip()]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    old = lines(OLD/'dataset.jsonl')
    original = [r for r in old if r['source']=='public_P0_assisted_real_P1']
    lookup = {r['base_sample_id']:r for r in original}
    groups = collections.defaultdict(dict)
    manifest = ROOT.parent/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    for r in lines(manifest):
        if r['base_sample_id'] in lookup:
            groups[r['base_sample_id']][r['tier']] = r
    rows, skipped = original[:], collections.Counter()
    def geometry(s):
        return tuple(sorted(tuple(p) for p in s.get('points',[])))
    for sid, tiers in sorted(groups.items()):
        a = json.loads(Path(tiers['P0']['label_path']).read_text())
        sa = shapes(a)
        for tier in ('P2', 'P3', 'P4'):
            current = tiers[tier]
            if not current['data_exists']:
                skipped['missing_data']+=1; continue
            if current['data_sha256'] != tiers['P0']['data_sha256']:
                skipped['changed_raw']+=1; continue
            b = json.loads(Path(current['label_path']).read_text())
            sb = shapes(b); feat = features(b,current['dataset_family'])
            causes=set()
            if len(sa)!=len(sb): causes.add(2)
            elif sorted(geometry(s) for s in sa)!=sorted(geometry(s) for s in sb): causes.add(1)
            elif sorted((geometry(s),s.get('label')) for s in sa)!=sorted((geometry(s),s.get('label')) for s in sb): causes.add(3)
            if feat[2] or feat[8]: causes.add(3)
            if len(causes)>1:
                skipped['mixed_class']+=1; continue
            row=dict(lookup[sid]); row['id']=row['id'].replace('_real','_'+tier)
            image=OUT/'images'/(row['id']+'.jpg')
            # Exactly preserve the already published test cohort and its views.
            prior=OLD/'images'/image.name
            if prior.exists(): image=prior
            elif not image.exists(): make_view(current['data_path'],b,image)
            row.update(target=next(iter(causes),0), source='P0_rule_derived_unreviewed_P2_P4',
                       tier=tier, features=feat, image=str(image),
                       rule_prediction=3 if feat[2] or feat[8] else 1 if feat[3] else 0)
            rows.append(row)
    split_groups = {s:{r['split_group'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not split_groups['train'] & (split_groups['val']|split_groups['test'])
    assert not split_groups['val'] & split_groups['test']
    oldextra = {r['id']:r['target'] for r in lines(OLD/'extended_test.jsonl')}
    newextra = {r['id']:r['target'] for r in rows if r['split']=='test' and r.get('tier')}
    assert oldextra == newextra
    with (OUT/'expanded_dataset.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    summary=dict(records=len(rows), split_counts=dict(collections.Counter(r['split'] for r in rows)),
                 cells=dict(collections.Counter(f"{r['split']}:{r.get('tier','P1')}:{r['target']}" for r in rows)),
                 skipped=dict(skipped), test_cohort_matches=True,
                 supervision='P1 technical audit + P0-derived weak labels for P2-P4; no expert scientific truth',
                 dataset_sha256=hashlib.sha256((OUT/'expanded_dataset.jsonl').read_bytes()).hexdigest())
    write('expanded_dataset_summary.json',summary)
    print(json.dumps(summary),flush=True)


def gates(rows, probabilities, fpr_limit):
    """Fit class-wise gates on real P1 validation only; rules cannot be overridden."""
    y=np.array([r['target'] for r in rows]); rule=np.array([r['rule_prediction'] for r in rows])
    cls=probabilities.argmax(1)
    choices=[.5,.6,.7,.8,.9,.95,.98,.99,.995,.999,1.01]
    best=None
    for t1 in choices:
        for t2 in choices:
            pred=rule.copy()
            for k,t in ((1,t1),(2,t2)):
                take=(rule==0)&(cls==k)&(probabilities[:,k]>=t)
                pred[take]=k
            fp=int(((y==0)&(pred>0)).sum())
            if fp>int(fpr_limit*int((y==0).sum())): continue
            tp=int(((y>0)&(pred==y)).sum()); wrong=int(((pred>0)&(pred!=y)).sum()); fn=int(((y>0)&(pred!=y)).sum())
            f1=2*tp/max(2*tp+wrong+fn,1)
            score=(f1,-fp,t1+t2)
            if best is None or score>best[0]: best=(score,[t1,t2],pred)
    return best[1]


def gate_predict(rows, pp, thresholds):
    result={}
    for r,p in zip(rows,pp):
        k=int(p.argmax())
        result[r['id']]=r['rule_prediction'] or (k if k in (1,2) and p[k]>=thresholds[k-1] else 0)
    return result


def train():
    torch.set_num_threads(4)
    rows=lines(OUT/'expanded_dataset.jsonl')
    device='cuda:0'
    start=time.monotonic()
    # Retain the original 256x128 student input for a controlled data ablation.
    ims=[]
    for r in rows:
        with Image.open(r['image']) as im:
            ims.append(np.array(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
    x=torch.tensor(np.array(ims),device=device); del ims
    feats=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device=device)
    targets=torch.tensor([r['target'] for r in rows],device=device)
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    vi=np.array([i for i,r in enumerate(rows) if r['split']=='val' and not r.get('tier')])
    test_indices={name:np.array([i for i,r in enumerate(rows) if r['split']=='test' and
                   (not r.get('tier') if name=='P1' else bool(r.get('tier')))]) for name in ('P1','P2_P4')}
    mean=feats[tr].mean(0); std=feats[tr].std(0).clamp_min(.01); ff=(feats-mean)/std
    ytr=targets[tr].cpu().numpy()
    summary={}; probabilities={}
    def infer(model, idx=None):
        model.eval()
        idx=np.arange(len(rows)) if idx is None else idx
        with torch.inference_mode():
            return torch.cat([model(x[ii],ff[ii]).softmax(1) for ii in
                              (idx[i:i+32] for i in range(0,len(idx),32))]).cpu().numpy()
    # 2 fixed training objectives x 3 seeds; model size stays unchanged.
    for objective in ('natural','balanced'):
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model=Student().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            weights=None
            if objective=='balanced':
                counts=np.bincount(ytr,minlength=4)
                weights=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32,device=device)
                weights/=weights.mean()
            best=None; trace=[]; ts=time.monotonic()
            for epoch in range(35):
                model.train(); perm=np.random.permutation(tr); losses=[]
                for offset in range(0,len(perm),32):
                    ii=perm[offset:offset+32];opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(x[ii],ff[ii]),targets[ii],weight=weights)
                    loss.backward(); opt.step(); losses.append(float(loss.detach()))
                pp=infer(model,vi)
                valrows=[rows[i] for i in vi]
                thresholds=gates(valrows,pp,.01)
                vm=metrics(valrows,gate_predict(valrows,pp,thresholds))
                raw=metrics(valrows,{r['id']:int(p.argmax()) for r,p in zip(valrows,pp)})
                key=(vm['defect_f1'],-vm['clean_false_positive_rate'],raw['macro_f1'])
                trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_f1=vm['defect_f1'],validation_raw_macro_f1=raw['macro_f1']))
                if best is None or key>best[0]:
                    best=(key,epoch+1,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            model.load_state_dict(best[2]); pp=infer(model)
            name=f'{objective}_seed{seed}';probabilities[name]=pp
            ckpt=OUT/(name+'.pt')
            torch.save(dict(state=best[2],widths=model.widths,hidden=model.hidden,feature_mean=mean.cpu(),feature_std=std.cpu()),ckpt)
            entry=dict(selected_epoch=best[1],training_seconds=time.monotonic()-ts,weights_bytes=ckpt.stat().st_size,
                       parameters=sum(p.numel() for p in model.parameters()),trace=trace,variants={})
            for limit in (.0,.01,.05):
                th=gates([rows[i] for i in vi],pp[vi],limit)
                sub=dict(thresholds=th,validation=metrics([rows[i] for i in vi],gate_predict([rows[i] for i in vi],pp[vi],th)))
                for label,idx in test_indices.items():
                    rr=[rows[i] for i in idx]
                    sub[label]=metrics(rr,gate_predict(rr,pp[idx],th))
                entry['variants'][str(limit)]=sub
            for label,idx in test_indices.items():
                rr=[rows[i] for i in idx]
                entry['raw_'+label]=metrics(rr,{r['id']:int(p.argmax()) for r,p in zip(rr,pp[idx])})
            summary[name]=entry;write('student_optimization.json',summary)
            print(json.dumps(dict(run=name,seconds=entry['training_seconds'],P1=entry['variants']['0.01']['P1'],P2_P4=entry['variants']['0.01']['P2_P4'])),flush=True)
    for objective in ('natural','balanced'):
        pp=np.mean([probabilities[f'{objective}_seed{s}'] for s in SEEDS],axis=0)
        entry={'variants':{}}
        for limit in (.0,.01,.05):
            th=gates([rows[i] for i in vi],pp[vi],limit)
            sub=dict(thresholds=th,validation=metrics([rows[i] for i in vi],gate_predict([rows[i] for i in vi],pp[vi],th)))
            for label,idx in test_indices.items():
                rr=[rows[i] for i in idx];sub[label]=metrics(rr,gate_predict(rr,pp[idx],th))
            entry['variants'][str(limit)]=sub
        summary[objective+'_ensemble3']=entry
    summary['_protocol']=dict(train_records=len(tr),P1_validation=len(vi),test_counts={k:len(v) for k,v in test_indices.items()},
           total_seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
           selection='P1 validation only; 1% validation FPR gate; raw validation macroF1 tie-break',
           status='Experimental sample classification candidates; no formal evidence localization',
           caveats=['Previously exposed test cohorts reused for regression, not fresh confirmation.',
                    'P2-P4 train and test labels are P0-derived and not expert reviewed.'])
    write('student_optimization.json',summary)
    np.savez_compressed(OUT/'student_probabilities.npz',ids=np.array([r['id'] for r in rows]),**probabilities)
    print(json.dumps(summary['_protocol']),flush=True)


if __name__=='__main__':
    prepare() if sys.argv[1]=='prepare' else train()
