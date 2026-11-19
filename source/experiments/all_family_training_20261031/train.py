"""Train/validate only. This module never opens the held-out test file."""
import argparse
import collections
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

from common import *
from models import build,prune_student


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def read_fit_rows():
    train=[r for r in lines(OUT/'train.jsonl') if not r['input_missing'] and any(r['truth_valid'])]
    val=[r for r in lines(OUT/'val.jsonl') if not r['input_missing'] and any(r['truth_valid'])]
    assert_coverage(train);assert_coverage(val)
    assert not ({r['training_group_id'] for r in train}&{r['training_group_id'] for r in val})
    return train,val


class Records(Dataset):
    def __init__(self,rows,teacher=None,size=SIZE):self.rows=rows;self.teacher=teacher;self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        source=r['image'] if self.size==SPATIAL_SIZE else r['training_image']
        with Image.open(source) as im:x=np.asarray(im.resize((self.size[1],self.size[0]),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
        y=np.zeros(self.size,np.float32)
        if r['aux_mask_valid']:
            with Image.open(r['aux_mask']) as im:y=np.asarray(im.resize((self.size[1],self.size[0]),Image.Resampling.NEAREST),dtype=np.float32)
        return dict(x=x,f=np.array(r['features'],np.float32),target=np.array(r['target'],np.float32),
            valid=np.array(r['truth_valid'],np.float32),mask=y[None],mask_valid=np.float32(r['aux_mask_valid']),
            family=FAMILIES.index(r['family']),index=index,
            soft=self.teacher[r['id']] if self.teacher is not None else np.zeros(3,np.float32))


def f1_by_family(rows,probs,thresholds):
    y=np.asarray([r['target'] for r in rows],bool);v=np.asarray([r['truth_valid'] for r in rows],bool)
    p=np.zeros_like(y);scores=[]
    for fi,family in enumerate(FAMILIES):
        ii=np.array([r['family']==family for r in rows]);p[ii]=probs[ii]>=np.asarray(thresholds[fi])
        tp=int((p[ii]&y[ii]&v[ii]).sum());fp=int((p[ii]&~y[ii]&v[ii]).sum());fn=int((~p[ii]&y[ii]&v[ii]).sum())
        scores.append(2*tp/max(2*tp+fp+fn,1))
    return float(np.mean(scores))


def calibrate(rows,probs):
    thresholds=np.full((6,3),.5)
    y=np.array([r['target'] for r in rows],bool);valid=np.array([r['truth_valid'] for r in rows],bool)
    for fi,family in enumerate(FAMILIES):
        ii=np.array([r['family']==family for r in rows])
        for k in range(3):
            chosen=ii&valid[:,k]
            if y[chosen,k].sum()<3:continue
            best=(-1,0,.5)
            for t in np.arange(.1,.951,.05):
                p=probs[chosen,k]>=t;yy=y[chosen,k]
                tp=(p&yy).sum();fp=(p&~yy).sum();fn=(~p&yy).sum()
                key=(2*tp/max(2*tp+fp+fn,1),-abs(t-.5),float(t))
                if key>best:best=key
            thresholds[fi,k]=best[2]
    return thresholds.tolist()


def infer_validation(model,loader,mean,std):
    model.eval();results=[]
    with torch.inference_mode():
        for batch in loader:
            x=batch['x'].cuda(non_blocking=True);f=(batch['f'].cuda(non_blocking=True)-mean)/std
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(x,f)
            results.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(results)


def fit(route,epochs_override=None,smoke=False):
    torch.set_num_threads(4);runseed=int(route.split('_seed')[-1]) if '_seed' in route else SEEDS[0];seed(runseed)
    tr,va=read_fit_rows()
    if smoke:
        tr=[next(r for r in tr if r['family']==f and r['tier']==t) for f in FAMILIES for t in TIERS]
        va=[next(r for r in va if r['family']==f and r['tier']==t) for f in FAMILIES for t in TIERS]
    info=json.loads((OUT/'preparation.json').read_text());mean=torch.tensor(info['mean'],device='cuda',dtype=torch.float32)
    std=torch.tensor(info['std'],device='cuda',dtype=torch.float32)
    teacher=None
    if route.startswith('distilled'):
        meta=json.loads((OUT/'teacher_train.json').read_text())
        assert meta['split']=='train' and meta['split_hash']==sha(OUT/'train.jsonl')
        teacher={r['id']:np.array(r['probabilities'],np.float32) for r in lines(OUT/'teacher_train.jsonl')}
        assert {r['id'] for r in tr}<=set(teacher)
    model=build(route).cuda()
    encoder_init=None
    if not smoke and route.startswith(('supervised','distilled')):
        pretraining=json.loads((OUT/'pretraining/complete.json').read_text())
        checkpoint=OUT/'pretraining/raw_encoder.pt'
        assert pretraining['status']=='complete' and sha(checkpoint)==pretraining['checkpoint_sha256']
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.convs.load_state_dict(data['encoder']);encoder_init=pretraining['checkpoint_sha256']
    if route.startswith('pruned'):
        supervised='supervised_seed'+str(runseed)
        parent=build(supervised).cuda()
        if not smoke:
            data=torch.load(OUT/'checkpoints'/(supervised+'.pt'),map_location='cpu',weights_only=False)
            parent.load_state_dict(data['state'])
        model=prune_student(parent);del parent
    epochs=epochs_override or (6 if route.startswith('pruned') else 12)
    batchsize=8 if route=='ijepa_finetune' else 16 if route in ARCHS else 64
    if smoke:batchsize=4;epochs=1
    gen=torch.Generator().manual_seed(runseed)
    image_size=SPATIAL_SIZE if route in ARCHS else SIZE
    loader=DataLoader(Records(tr,teacher,image_size),batch_size=batchsize,shuffle=True,generator=gen,num_workers=4,
        pin_memory=True,persistent_workers=True,drop_last=False)
    val_loader=DataLoader(Records(va,size=image_size),batch_size=batchsize,shuffle=False,num_workers=2,pin_memory=True,persistent_workers=True)
    counts=collections.Counter(r['family'] for r in tr)
    fw=torch.tensor([len(tr)/(6*counts[f]) for f in FAMILIES],device='cuda')
    y=np.array([r['target'] for r in tr]);v=np.array([r['truth_valid'] for r in tr])
    positive=(y*v).sum(0);negative=((1-y)*v).sum(0)
    posweight=torch.tensor(np.clip(negative/np.maximum(positive,1),1,12),device='cuda',dtype=torch.float32)
    lr=1e-5 if route in ['dino_finetune','ijepa_finetune'] else 1e-4 if route in ['resnet18_unet','deeplab_mobilenet'] else 6e-4
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=lr/10)
    start=time.monotonic();best=-1.;best_state=None;best_epoch=0;trace=[];seen=set();total_updates=0;actual=collections.Counter()
    resume=OUT/'training'/(route+'.resume.pt')
    first_epoch=0
    if resume.exists() and not smoke:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['train_hash']==info['split_hashes']['train']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        first_epoch=saved['epoch'];best=saved['best'];best_state=saved['best_state'];best_epoch=saved['best_epoch']
        trace=saved['trace'];seen=set(saved['seen']);actual.update(saved['coverage']);total_updates=saved['updates']
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);gen.set_state(saved['loader_rng'])
    for epoch in range(first_epoch,epochs):
        model.train();losses=[];epoch_coverage=collections.Counter();epoch_seen=set()
        for batch in loader:
            x=batch['x'].cuda(non_blocking=True);f=(batch['f'].cuda(non_blocking=True)-mean)/std
            if len(x)==1:
                for layer in model.modules():
                    if isinstance(layer,nn.BatchNorm2d):layer.eval()
            target=batch['target'].cuda();valid=batch['valid'].cuda();weights=fw[batch['family'].cuda()]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):logits,dense=model(x,f)
            logits=logits.float()
            terms=F.binary_cross_entropy_with_logits(logits,target,pos_weight=posweight,reduction='none')*valid
            loss=(terms.sum(1)/valid.sum(1).clamp_min(1)*weights).mean()
            if teacher is not None:
                q=batch['soft'].cuda().clamp(1e-5,1-1e-5)
                softened=torch.sigmoid(torch.logit(q)/2)
                distill=F.binary_cross_entropy_with_logits(logits/2,softened,reduction='none').mean(1)
                loss=.5*loss+.5*4*(distill*weights).mean()
            if dense is not None:
                mask=batch['mask'].cuda();mv=batch['mask_valid'].cuda()*weights
                dense=dense.float();p=dense.sigmoid()
                bce=F.binary_cross_entropy_with_logits(dense,mask,reduction='none').mean((1,2,3))
                dice=1-(2*(p*mask).sum((1,2,3))+1)/(p.sum((1,2,3))+mask.sum((1,2,3))+1)
                loss=loss+.2*((bce+dice)*mv).sum()/mv.sum().clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss '+route)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total_updates+=1
            losses.append(float(loss.detach()))
            for i in batch['index'].tolist():
                r=tr[i];key=r['family']+'|'+r['tier'];epoch_coverage[key]+=1;actual[key]+=1;epoch_seen.add(r['id']);seen.add(r['id'])
        assert epoch_seen=={r['id'] for r in tr},'An epoch did not visit every train record'
        assert len(epoch_coverage)==30
        pp=infer_validation(model,val_loader,mean,std)
        score=f1_by_family(va,pp,np.full((6,3),.5))
        if score>best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
        scheduler.step()
        entry=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation_family_macro_f1_at_05=score,
            actual_train_coverage=dict(epoch_coverage),unique_train_records=len(epoch_seen),seconds=time.monotonic()-start)
        trace.append(entry);emit('train',route=route,**{k:v for k,v in entry.items() if k!='actual_train_coverage'})
        state=dict(status='training',route=route,epoch=epoch+1,epochs=epochs,best_epoch=best_epoch,
            unique_records_updated=len(seen),actual_train_coverage=dict(actual),updates=total_updates,trace=trace,pid=os.getpid())
        write(OUT/'training'/(route+'.state.json'),state)
        if not smoke:
            temp=resume.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),epoch=epoch+1,best=best,best_state=best_state,best_epoch=best_epoch,
                trace=trace,seen=sorted(seen),coverage=dict(actual),updates=total_updates,train_hash=info['split_hashes']['train'],
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=gen.get_state()),temp);temp.replace(resume)
    model.load_state_dict(best_state);model.eval();pp=infer_validation(model,val_loader,mean,std);thresholds=calibrate(va,pp)
    report=dict(status='smoke_passed' if smoke else 'complete',route=route,epochs=epochs,selected_epoch=best_epoch,
        selection='family-macro validation F1 at .5',validation_macro_f1_at_05=best,
        validation_macro_f1_calibrated=f1_by_family(va,pp,thresholds),thresholds=thresholds,
        training_records=len(tr),training_ids_sha256=digest(sorted(seen)),actual_train_coverage=dict(actual),
        unique_records_updated=len(seen),updates=total_updates,training_families=FAMILIES,
        train_split_sha256=info['split_hashes']['train'],val_split_sha256=info['split_hashes']['val'],
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),trace=trace,seed=runseed,
        image_size=list(image_size),
        encoder_initialization_sha256=encoder_init,
        representation='current waveform and event-window view for ULF; current-only metadata; reference masks training only',
        test_read=False)
    if not smoke:
        checkpoint=OUT/'checkpoints'/(route+'.pt');temp=checkpoint.with_suffix('.tmp')
        torch.save(dict(state=best_state,route=route,mean=mean.cpu(),std=std.cpu(),thresholds=thresholds,
            widths=getattr(model,'widths',None),hidden=getattr(model,'hidden',None)),temp);temp.replace(checkpoint)
        report['checkpoint_hashes']={str(checkpoint.relative_to(OUT)):sha(checkpoint)}
        save_lines(OUT/'training'/(route+'.updated_ids.jsonl'),[{'id':i} for i in sorted(seen)])
        np.savez_compressed(OUT/'training'/(route+'.validation.npz'),ids=np.array([r['id'] for r in va]),probabilities=pp)
        write(OUT/'training'/(route+'.json'),report)
    else:write(OUT/'smoke'/(route+'.json'),report)
    emit('train_complete',route=route,smoke=smoke,records=len(tr),families=6,epochs=epochs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('route');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    fit(a.route,smoke=a.smoke)
