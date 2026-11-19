"""Dense, image-only predictors; multi-label technical proxy evaluation.

No generated finding enters the competition evaluator. Test clean masks are
only evaluation truth. The saved per-base dense predictions use raw pixels.
"""
import collections
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesClassifier
from torch import nn
from torch.nn import functional as F

ROOT=Path('/staff/tangjiarui/OPC')
OUT=Path(os.environ.get('OPC_SPATIAL_OUT',str(ROOT/'opc-agent/runs/spatial-research-20261031')))
OLD=ROOT/'opc-agent/runs/model-comparison-20261031-v2'
SIZE=int(os.environ.get('OPC_SPATIAL_SIZE','224'))
SEED=20261031
CODES=['L2-G','L3-06','L3-07']
sys.path.insert(0,str(ROOT/'opc-agent/tools'))
from prepare_model_comparison import shapes, features


def lines(path):return [json.loads(s) for s in path.open() if s.strip()]
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));temp.replace(p)
def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw)),flush=True)
def label(r):return json.loads(Path(r['label_path']).read_text())


def mask(annotation,w,h):
    image=Image.new('L',(SIZE,SIZE),0);d=ImageDraw.Draw(image)
    for s in shapes(annotation):
        points=[(float(p[0])*SIZE/w,float(p[1])*SIZE/h) for p in s.get('points',[])]
        if len(points)<2:continue
        if s.get('shape_type')=='rectangle' and len(points)==2:
            xs=sorted(p[0] for p in points);ys=sorted(p[1] for p in points)
            d.rectangle((xs[0],ys[0],xs[1],ys[1]),fill=1)
        elif len(points)>=3:d.polygon(points,fill=1)
    return np.array(image,dtype=np.uint8)


def normalized_points(s):
    # Vertex order/winding is irrelevant for exact point-set comparison.
    return np.array(sorted(tuple(float(v) for v in p) for p in s.get('points',[])),dtype=float)


def technical_truth(clean,current,w,h):
    a,b=shapes(clean),shapes(current)
    ma=[mask({'science_annotation':{'specialShapes':[s]}},w,h) for s in a]
    mb=[mask({'science_annotation':{'specialShapes':[s]}},w,h) for s in b]
    pairs=[]
    if a and b:
        cost=np.zeros((len(a),len(b)))
        for i,x in enumerate(ma):
            pa=np.array(a[i]['points'],float).mean(0)
            for j,y in enumerate(mb):
                iou=(x&y).sum()/max(1,(x|y).sum())
                pb=np.array(b[j]['points'],float).mean(0)
                cost[i,j]=1-iou+.1*np.linalg.norm((pa-pb)/[w,h])
        aa,bb=linear_sum_assignment(cost);pairs=list(zip(aa.tolist(),bb.tolist()))
    geom=False
    for i,j in pairs:
        pa,pb=normalized_points(a[i]),normalized_points(b[j])
        if pa.shape!=pb.shape or not np.allclose(pa,pb,atol=.5,rtol=0):geom=True
    return [int(geom),int(len(a)>len(b)),int(len(a)<len(b))]


def annotation_features(ann,family,w,h):
    result=features(ann,family)
    boxes=[]
    for s in shapes(ann):
        pp=np.array(s.get('points',[]),float)
        if len(pp)>1:
            lo=pp.min(0)/[w,h];hi=pp.max(0)/[w,h]
            boxes.append([*lo,*hi,*(hi-lo)])
    a=np.array(boxes) if boxes else np.zeros((1,6))
    return result+a.mean(0).tolist()+a.std(0).tolist()


def prepare():
    old=[r for r in lines(OLD/'dataset.jsonl') if r['source']=='public_P0_assisted_real_P1']
    lookup={r['base_sample_id']:r for r in old};groups=collections.defaultdict(dict)
    for r in lines(ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'):
        if r['base_sample_id'] in lookup:groups[r['base_sample_id']][r['tier']]=r
    xs=[];ys=[];bases=[];rows=[];masks=[];truthmaps=[];skipped=collections.Counter()
    for sid,tiers in sorted(groups.items()):
        ref=tiers['P0'];prior=lookup[sid]
        if not ref['data_exists']:skipped['missing_clean_data']+=1;continue
        ann=label(ref);w,h=ref['image_width'],ref['image_height'];cleanmask=mask(ann,w,h)
        with Image.open(ref['data_path']) as im:
            xs.append(np.asarray(im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR)))
        ys.append(cleanmask)
        idx=len(bases)
        bases.append(dict(base_sample_id=sid,split=prior['split'],split_group=prior['split_group'],family=prior['family'],
                          raw_path=ref['data_path'],reference_label_path=ref['label_path'],raw_sha256=ref['data_sha256']))
        for tier in ('P0','P1','P2','P3','P4'):
            r=tiers[tier]
            if not r['data_exists']:skipped['missing_current_data']+=1;continue
            if r['data_sha256']!=ref['data_sha256']:skipped['changed_raw']+=1;continue
            current=label(r);m=mask(current,w,h)
            y=technical_truth(ann,current,w,h)
            rows.append(dict(id=sid+':'+tier,base_index=idx,base_sample_id=sid,tier=tier,split=prior['split'],
                split_group=prior['split_group'],family=prior['family'],label_path=r['label_path'],width=w,height=h,
                target=y,annotation_features=annotation_features(current,prior['family'],w,h)))
            masks.append(m);truthmaps.append(cleanmask^m)
    assert len(bases)==len(xs)==len(ys)
    sg={s:{b['split_group'] for b in bases if b['split']==s} for s in ('train','val','test')}
    assert not sg['train']&(sg['val']|sg['test']) and not sg['val']&sg['test']
    sha={s:{b['raw_sha256'] for b in bases if b['split']==s} for s in ('train','val','test')}
    assert not sha['train']&(sha['val']|sha['test']) and not sha['val']&sha['test']
    for name,rr in [('bases.jsonl',bases),('dataset.jsonl',rows)]:
        (OUT/name).write_text(''.join(json.dumps(r)+'\n' for r in rr))
    np.savez_compressed(OUT/'spatial_data.npz',raw=np.array(xs),clean=np.array(ys),current=np.array(masks),truth=np.array(truthmaps))
    summary=dict(bases=len(bases),records=len(rows),skipped=dict(skipped),image_size=SIZE,
        original_eligible_bases=len(old),split_counts=dict(collections.Counter(b['split'] for b in bases)),
        cells={f'{s}:{t}':dict(n=sum(r['split']==s and r['tier']==t for r in rows),
          positive_support=np.sum([r['target'] for r in rows if r['split']==s and r['tier']==t],axis=0).tolist())
          for s in ('train','val','test') for t in ('P0','P1','P2','P3','P4')},
        codes=CODES,group_and_hash_disjoint=True,test_status='exposed_regression',
        truth_status='P0_derived_technical_proxy; no independent scientific adjudication',
        multi_label=True,data_sha256=hashlib.sha256((OUT/'dataset.jsonl').read_bytes()).hexdigest())
    dump('dataset_summary.json',summary);emit('prepare',**summary)


class Conv(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())


class UNet(nn.Module):
    def __init__(self):
        super().__init__();self.e1=Conv(3,16);self.e2=Conv(16,32);self.e3=Conv(32,64);self.e4=Conv(64,128)
        self.d3=Conv(192,64);self.d2=Conv(96,32);self.d1=Conv(48,16);self.out=nn.Conv2d(16,1,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(F.max_pool2d(a,2));c=self.e3(F.max_pool2d(b,2));d=self.e4(F.max_pool2d(c,2))
        d=self.d3(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        d=self.d2(torch.cat([F.interpolate(d,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.d1(torch.cat([F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))


class DinoHead(nn.Sequential):
    def __init__(self,channels=384):
        super().__init__(nn.Conv2d(channels,64,1),nn.SiLU(),nn.Conv2d(64,64,3,padding=1),nn.SiLU(),nn.Conv2d(64,1,1))
    def forward(self,x):return F.interpolate(super().forward(x),size=(SIZE,SIZE),mode='bilinear',align_corners=False)


def seed():
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True


def dino_features(raw,route='dino'):
    from transformers import Dinov2Model,IJepaModel
    model_class=Dinov2Model if route=='dino' else IJepaModel
    model_dir='dinov2-small' if route=='dino' else 'ijepa_vith14_1k'
    start=time.monotonic();backbone=model_class.from_pretrained(ROOT/'models'/model_dir,local_files_only=True).cuda().eval()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    result=[]
    with torch.inference_mode():
        batch=32 if route=='dino' else 8
        for i in range(0,len(raw),batch):
            x=torch.tensor(raw[i:i+batch].transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255
            f=backbone(pixel_values=(x-mean)/std).last_hidden_state
            if route=='dino':f=f[:,1:]
            f=f.transpose(1,2).reshape(len(x),backbone.config.hidden_size,SIZE//14,SIZE//14)
            result.append(f.cpu().numpy().astype(np.float16))
    params=sum(p.numel() for p in backbone.parameters());del backbone;torch.cuda.empty_cache()
    result=np.concatenate(result);np.save(OUT/(route+'_features.npy'),result)
    dump(route+'_feature_runtime.json',dict(seconds=time.monotonic()-start,samples=len(raw),parameters=params,
        feature_shape=list(result.shape),batch=batch,raw_only=True))
    return result


def dice(p,y):
    return ((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()


def train(route):
    seed();start=time.monotonic();data=np.load(OUT/'spatial_data.npz');raw=data['raw'];clean=data['clean']
    bases=lines(OUT/'bases.jsonl');tr=np.array([i for i,r in enumerate(bases) if r['split']=='train'])
    va=np.array([i for i,r in enumerate(bases) if r['split']=='val'])
    yy=torch.tensor(clean[:,None],dtype=torch.float32,device='cuda')
    if route=='unet':
        model=UNet().cuda();xx=torch.tensor(raw.transpose(0,3,1,2),dtype=torch.float32,device='cuda')/255;epochs=24
    else:
        f=dino_features(raw,route);channels=f.shape[1];xx=torch.tensor(f,dtype=torch.float32,device='cuda');del f
        model=DinoHead(channels).cuda();epochs=60
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    trace=[];best=(-1,None);posweight=((1-yy[tr].mean())/yy[tr].mean().clamp_min(.01)).clamp(1,10)
    def infer(x=xx):
        model.eval()
        with torch.inference_mode():return torch.cat([model(x[i:i+24]).sigmoid() for i in range(0,len(x),24)])
    for epoch in range(epochs):
        model.train();perm=np.random.permutation(tr);losses=[]
        for i in range(0,len(perm),16):
            ii=perm[i:i+16];opt.zero_grad(set_to_none=True);p=model(xx[ii])
            loss=F.binary_cross_entropy_with_logits(p,yy[ii],pos_weight=posweight)+1-dice(p.sigmoid(),yy[ii])
            loss.backward();opt.step();losses.append(float(loss.detach()))
        with torch.inference_mode():
            model.eval();pp=torch.cat([model(xx[ii]).sigmoid() for ii in (va[i:i+24] for i in range(0,len(va),24))])
            vd=float(dice((pp>.5).float(),yy[va]))
        trace.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_clean_dice=vd))
        if vd>best[0]:best=(vd,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        if epoch%4==0:emit(route,**trace[-1])
    model.load_state_dict(best[1]);torch.cuda.synchronize();t=time.monotonic();pp=infer();torch.cuda.synchronize()
    seconds=time.monotonic()-t;pred=pp[:,0].cpu().numpy()
    # Counterfactual: shuffle raw predictions within family/split, never test labels.
    rng=np.random.RandomState(SEED);perm=np.arange(len(bases))
    for family in sorted({b['family'] for b in bases}):
        for split in ('train','val','test'):
            ii=np.array([i for i,b in enumerate(bases) if b['family']==family and b['split']==split]);perm[ii]=rng.permutation(ii)
    np.savez_compressed(OUT/(route+'_predictions.npz'),matched=pred.astype(np.float16),permuted=pred[perm].astype(np.float16))
    ckpt=OUT/(route+'.pt');torch.save(dict(state=best[1],image_size=SIZE,seed=SEED),ckpt)
    test=np.array([i for i,b in enumerate(bases) if b['split']=='test'])
    result=dict(route=route,parameters=sum(p.numel() for p in model.parameters()),checkpoint_bytes=ckpt.stat().st_size,
        total_seconds=time.monotonic()-start,inference_seconds=seconds,trace=trace,best_val_dice=best[0],
        test_clean_dice=float(dice((pp[test]>.5).float(),yy[test])),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),inference_samples=len(raw),
        note='Frozen backbone inference time here is head-only; backbone time separately recorded' if route!='unet' else 'cached resized raw; excludes decode',
        formal_scoring_enabled=False)
    dump(route+'_training.json',result);emit('trained',**{k:v for k,v in result.items() if k!='trace'})


def residual_features(p,m,ann,w,h):
    p=np.asarray(p,dtype=np.float32);binary=p>.5;inside=m>0;outside=~inside
    def avg(a):return float(np.mean(a)) if a.size else 0.
    area=max(int(inside.sum()),1);soft=float(p.sum())
    miss=binary&outside;extra=inside&~binary
    def components(a):
        n,_,stats,_=cv2.connectedComponentsWithStats(a.astype(np.uint8),8)
        sizes=stats[1:,cv2.CC_STAT_AREA] if n>1 else np.array([0])
        return [float(sizes.max())/(SIZE*SIZE),float(np.sum(sizes>8))]
    boundary=cv2.morphologyEx(inside.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    support=[]
    for s in shapes(ann):
        mm=mask({'science_annotation':{'specialShapes':[s]}},w,h)>0
        support.append(avg(p[mm]))
    return [avg(p[inside]),avg(p[outside]),soft/(SIZE*SIZE),float(binary.mean()),
            float((p*inside).sum())/max(soft+area-float((p*inside).sum()),1),
            float(miss.mean()),float(extra.mean()),avg(p[boundary]),min(support,default=0),max(support,default=0),
            *components(miss),*components(extra)]


def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,bool)
    if len(y)==0:return dict(samples=0)
    tp=(y&p).sum(0);fp=((~y)&p).sum(0);fn=(y&~p).sum(0)
    def f(a,b,c):return float(2*a/max(2*a+b+c,1))
    clean=~y.any(1)
    return dict(samples=len(y),micro_f1=f(tp.sum(),fp.sum(),fn.sum()),precision=float(tp.sum()/max((tp+fp).sum(),1)),
       recall=float(tp.sum()/max((tp+fn).sum(),1)),clean_samples=int(clean.sum()),
       clean_fpr=float(p[clean].any(1).mean()) if clean.any() else None,
       tp=int(tp.sum()),fp=int(fp.sum()),fn=int(fn.sum()),
       per_class={c:dict(support=int(tp[i]+fn[i]),tp=int(tp[i]),fp=int(fp[i]),fn=int(fn[i]),f1=f(tp[i],fp[i],fn[i])) for i,c in enumerate(CODES)})


def fit_gates(y,pp,guard_y=None,guard_pp=None):
    # Joint clean-sample budget, fixed threshold grid, validation only.
    choices=[.5,.7,.85,.95,.99,1.01];best=None
    for a in choices:
        for b in choices:
            for c in choices:
                pred=pp>=np.array([a,b,c]);mm=metrics(y,pred)
                gm=metrics(guard_y,guard_pp>=np.array([a,b,c])) if guard_y is not None else mm
                if gm['clean_fpr'] is not None and gm['clean_fpr']>.01:continue
                key=(mm['micro_f1'],-(gm['clean_fpr'] or 0),a+b+c)
                if best is None or key>best[0]:best=(key,[a,b,c])
    return best[1]


def evaluate():
    seed()
    with np.load(OUT/'spatial_data.npz') as z:
        data={k:z[k] for k in ('current','truth')}
    rows=lines(OUT/'dataset.jsonl');bases=lines(OUT/'bases.jsonl')
    y=np.array([r['target'] for r in rows]);ann=np.array([r['annotation_features'] for r in rows])
    tr=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    va=np.array([i for i,r in enumerate(rows) if r['split']=='val' and r['tier']=='P1'])
    results={};predictions={};probabilities={}
    configurations=[('annotation_only',None,'matched')]
    for route in ('unet','dino','unet_oof','ijepa'):
        if (OUT/(route+'_predictions.npz')).exists():configurations.append((route,route,'matched'))
    dense_cache={}
    for name,route,mode in configurations:
        if route:
            with np.load(OUT/(route+'_predictions.npz')) as z:
                pred={k:z[k] for k in ('matched','permuted')}
            dense_cache[route]=pred
            fs=[]
            for i,r in enumerate(rows):
                fs.append(residual_features(pred['matched'][r['base_index']],data['current'][i],label(r),r['width'],r['height']))
            xx=np.concatenate([ann,np.array(fs)],1)
        else:xx=ann
        classifiers=[];pp=np.zeros((len(rows),3));start=time.monotonic()
        for k in range(3):
            model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,class_weight='balanced',random_state=SEED,n_jobs=4)
            model.fit(xx[tr],y[tr,k]);cl=list(model.classes_)
            if 1 in cl:pp[:,k]=model.predict_proba(xx)[:,cl.index(1)]
            classifiers.append(model)
        # Parallel tree sums can vary at machine precision. Quantize far below
        # meaningful confidence precision before comparisons to fixed gates.
        pp=np.round(pp,12)
        thresholds=fit_gates(y[va],pp[va]);pg=pp>=thresholds;predictions[name]=pg;probabilities[name]=pp
        record=dict(thresholds=thresholds,validation=metrics(y[va],pg[va]),seconds=time.monotonic()-start,
            feature_count=xx.shape[1],cohorts={},probability_decimal_places=12)
        allval=np.array([i for i,r in enumerate(rows) if r['split']=='val'])
        multi_thresholds=fit_gates(y[allval],pp[allval],y[va],pp[va])
        multi_pred=pp>=multi_thresholds
        with (OUT/(name+'_detectors.pkl')).open('wb') as f:
            pickle.dump(dict(classifiers=classifiers,thresholds=thresholds,multi_tier_thresholds=multi_thresholds,
                feature_count=xx.shape[1],codes=CODES,image_size=SIZE,route=name),f,protocol=4)
        np.savez_compressed(OUT/(name+'_detector_probabilities.npz'),probabilities=pp,targets=y,
                            ids=np.array([r['id'] for r in rows]))
        record['multi_tier_calibration']=dict(thresholds=multi_thresholds,
            validation=metrics(y[allval],multi_pred[allval]),P1_validation_guard=metrics(y[va],multi_pred[va]),cohorts={})
        for family in ['ALL','EPB_seg_sample','DMSP_SSUSI']:
            for tier in ['P0','P1','P2','P3','P4','P2_P4']:
                ii=np.array([i for i,r in enumerate(rows) if r['split']=='test' and
                    (family=='ALL' or r['family']==family) and (r['tier'] in ['P2','P3','P4'] if tier=='P2_P4' else r['tier']==tier)])
                if not len(ii):continue
                record['cohorts'][family+':'+tier]=dict(gated=metrics(y[ii],pg[ii]),ungated=metrics(y[ii],pp[ii]>=.5))
                record['multi_tier_calibration']['cohorts'][family+':'+tier]=metrics(y[ii],multi_pred[ii])
        if route:
            fs=[residual_features(pred['permuted'][r['base_index']],data['current'][i],label(r),r['width'],r['height']) for i,r in enumerate(rows)]
            xxperm=np.concatenate([ann,np.array(fs)],1);pperm=np.zeros_like(pp)
            for k,model in enumerate(classifiers):
                cl=list(model.classes_)
                if 1 in cl:pperm[:,k]=model.predict_proba(xxperm)[:,cl.index(1)]
            pperm=np.round(pperm,12)
            test=np.array([i for i,r in enumerate(rows) if r['split']=='test' and r['tier']!='P0'])
            record['image_permutation']=dict(matched=metrics(y[test],pg[test]),permuted=metrics(y[test],pperm[test]>=thresholds),
                 changed_samples=int(np.any(pg[test]!=(pperm[test]>=thresholds),axis=1).sum()),test_samples=len(test))
            record['multi_tier_calibration']['image_permutation']=dict(matched=metrics(y[test],multi_pred[test]),
                permuted=metrics(y[test],pperm[test]>=multi_thresholds),
                changed_samples=int(np.any(multi_pred[test]!=(pperm[test]>=multi_thresholds),axis=1).sum()),test_samples=len(test))
            spatial_by_class={}
            for k,code in enumerate(CODES):
                ious=[]
                for i in test:
                    if not (multi_pred[i,k] and y[i,k]):continue
                    r=rows[i];res=(pred['matched'][r['base_index']]>.5)^(data['current'][i]>0);truth=data['truth'][i]>0
                    ious.append(float((res&truth).sum()/max(1,(res|truth).sum())))
                spatial_by_class[code]=dict(correct_class_samples=len(ious),mean_residual_iou=float(np.mean(ious)) if ious else None,
                    samples_iou_at_least_05=int(np.sum(np.array(ious)>=.5)))
            record['multi_tier_calibration']['localization_proxy_by_class']=spatial_by_class
            # Localized proposal quality measured only where the detector emits
            # at least one correct class. It is not official location credit.
            ious=[];proposals=[]
            for i in test:
                if not np.any(pg[i]&y[i].astype(bool)):continue
                r=rows[i];res=(pred['matched'][r['base_index']]>.5)^(data['current'][i]>0);truth=data['truth'][i]>0
                iou=float((res&truth).sum()/max(1,(res|truth).sum()));ious.append(iou)
                if len(proposals)<30:
                    # Small artifact: no copied raw data, only normalized region.
                    n,_,stats,_=cv2.connectedComponentsWithStats(res.astype(np.uint8),8)
                    regions=sorted(stats[1:].tolist(),key=lambda s:s[4],reverse=True)[:6]
                    proposals.append(dict(id=r['id'],codes=[CODES[k] for k in range(3) if pg[i,k]],
                        regions_xywh_model_pixels=[s[:4] for s in regions],model_image_size=SIZE,technical_residual_iou=iou,
                        status='review_only',raw_path=bases[r['base_index']]['raw_path'],label_path=r['label_path']))
            record['localization_proxy']=dict(correct_class_samples=len(ious),mean_residual_iou=float(np.mean(ious)) if ious else None,
                samples_iou_at_least_05=int(np.sum(np.array(ious)>=.5)),official_location_credit=False)
            (OUT/(route+'_region_proposals.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in proposals))
        results[name]=record;dump('detector_results.json',results);emit('detector',route=name,validation=record['validation'],
              P1=record['cohorts']['ALL:P1']['gated'],heavy=record['cohorts']['ALL:P2_P4']['gated'])
    # Choose solely on validation. Keep rules unchanged even if this wins.
    winner=max(results,key=lambda n:(results[n]['validation']['micro_f1'],-(results[n]['validation']['clean_fpr'] or 0),-results[n]['feature_count']))
    dump('selection.json',dict(selected=winner,selection='P1 validation F1 then FPR then fewer features',
        may_enter_formal_score=False,limitations=['Spatial classes only, not prior 4-way coarse F1.',
        'Technical clean-derived labels, not expert scientific truth.','Exposed test: exploratory regression.',
        '224px global mask may discard subtle native-resolution boundaries.']))


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='evaluate':evaluate()
    else:train(sys.argv[1])
