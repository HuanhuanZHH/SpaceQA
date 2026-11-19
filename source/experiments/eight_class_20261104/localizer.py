"""Joint four-family RGB localization, presence gating, and frozen evaluation.

P0 masks are weak annotation supervision, not independently audited science truth.
No current annotation or paired clean image is supplied to the neural network.
"""
from __future__ import annotations
import argparse, collections, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/staff/tangjiarui/OPC')
SEED = 20261104


def helper():
    path=Path(__file__).with_name('pilot_helpers.py')
    spec=importlib.util.spec_from_file_location('previous_localizer',path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def network(width=32):
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Block(nn.Sequential):
        def __init__(self,a,b):
            super().__init__(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU(),
                             nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU())
    class Net(nn.Module):
        def __init__(self):
            super().__init__();w=width
            self.a=Block(3,w);self.b=Block(w,2*w);self.c=Block(2*w,4*w);self.d=Block(4*w,8*w)
            self.u2=Block(12*w,4*w);self.u1=Block(6*w,2*w);self.u0=Block(3*w,w)
            self.mask=nn.Conv2d(w,4,1);self.presence=nn.Linear(8*w,4)
        def forward(self,x):
            a=self.a(x);b=self.b(F.max_pool2d(a,2));c=self.c(F.max_pool2d(b,2));d=self.d(F.max_pool2d(c,2))
            def up(t,s):return F.interpolate(t,size=s.shape[-2:],mode='bilinear',align_corners=False)
            z=self.u2(torch.cat([up(d,c),c],1));z=self.u1(torch.cat([up(z,b),b],1))
            return self.mask(self.u0(torch.cat([up(z,a),a],1))),self.presence(d.mean((2,3)))
    return Net()


def run(args):
    import torch
    from PIL import Image
    from torch.nn import functional as F
    h=helper();out=args.output.resolve()
    h.validate_output_path(out,ROOT);out.mkdir(parents=True)
    started=time.monotonic();deadline=started+args.budget_seconds
    def state(stage,**kw):
        v=dict(stage=stage,elapsed_seconds=round(time.monotonic()-started,2),**kw)
        h.write(out/'state.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
    h.write(out/'plan.json',dict(seed=SEED,epochs=args.epochs,batch=args.batch,size=args.size,width=args.width,
        families=h.FAMILIES,not_pixel_tasks=h.NA_FAMILIES,device='physical GPU1 via CUDA_VISIBLE_DEVICES=1',
        training_input='RGB/255 only; jointly all four pixel families; no overlay or annotation features',
        weak_target='Original P0 regions, not independent scientific truth',
        split='Existing train/val/test groups; all family training finishes before validation/test',
        budget_seconds=args.budget_seconds,source_sha256=h.sha(__file__),
        routes=['presence_gated','no_presence_loss'],test_for_selection=False))
    torch.set_num_threads(6);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    torch.backends.cudnn.benchmark=False
    state('preparing_train')
    old=ROOT/'research_optimization_20261103/localizer/pilot_v1/train_inventory.json'
    inventory=json.loads(old.read_text());source=ROOT/'opc-agent/runs/all-family-training-20261031/train.jsonl'
    oldaudit=json.loads((old.parent/'train_audit.json').read_text())
    assert h.sha(source)==oldaudit['source_sha256']
    selected=h.capped(inventory,args.train_cap,'extended_20261104')
    assert set(r['family'] for r in selected)==set(h.FAMILIES)
    h.write(out/'train_inventory.json',selected)
    state('verifying_train_sources',records=len(selected),families=h.summarize(inventory))
    snapshot=h.verify_sources(selected,out/'sources_train_before.json')
    samples=h.prepare_train_cache(selected,out,args.size)
    state('train_cache_ready',records=len(selected),views=len(samples))

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):return len(samples)
        def __getitem__(self,i):
            r=samples[i]
            with Image.open(r['image']) as im:x=np.array(im.convert('RGB'),dtype=np.float32)/255
            with Image.open(r['mask']) as im:y=np.array(im,dtype=np.float32)/255
            # Geometric augmentation is applied to image AND target.
            if random.random()<.5:x=x[:,::-1].copy();y=y[:,::-1].copy()
            if random.random()<.5:x=x[::-1].copy();y=y[::-1].copy()
            return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y[None].copy()),h.FAMILIES.index(r['family'])
    class Sampler(torch.utils.data.Sampler):
        epoch=0
        def __iter__(self):
            ix=h.balanced_epoch_indices(samples,self.epoch,SEED);self.epoch+=1;return iter(ix)
        def __len__(self):return len(h.balanced_epoch_indices(samples,0,SEED))
    fractions=[np.mean([r['foreground_fraction'] for r in samples if r['family']==f]) for f in h.FAMILIES]
    pw=torch.tensor([min(10.,max(1.,(1-v)/max(.001,v))) for v in fractions],device='cuda')
    records=[]
    for route in ['presence_gated','no_presence_loss']:
        torch.manual_seed(SEED);model=network(args.width).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs,eta_min=3e-5)
        loader=torch.utils.data.DataLoader(Dataset(),batch_size=args.batch,sampler=Sampler(),num_workers=6,
            pin_memory=True,persistent_workers=True)
        history=[];tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train();losses=[];seen=collections.Counter()
            for step,(x,y,fi) in enumerate(loader):
                if time.monotonic()>deadline-600:raise TimeoutError('Training budget exhausted; no partial model promoted')
                x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);fi=fi.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    z,pr=model(x);idx=torch.arange(len(x),device='cuda');z=z[idx,fi][:,None];pr=pr[idx,fi]
                    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none')
                    bce=(bce*torch.where(y>.5,pw[fi,None,None,None],1.)).mean()
                    p=z.sigmoid();dice=1-((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).mean()
                    presence=F.binary_cross_entropy_with_logits(pr,y.flatten(1).any(1).float())
                    loss=bce+dice+(.5*presence if route=='presence_gated' else 0.)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                losses.append(float(loss));seen.update(h.FAMILIES[i] for i in fi.cpu().tolist())
                if step%100==0:state('training',route=route,epoch=epoch+1,epochs=args.epochs,step=step,steps=len(loader),
                    loss=float(np.mean(losses[-100:])),training_seconds=time.monotonic()-tick,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
            assert set(seen)==set(h.FAMILIES)
            scheduler.step();history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),family_seen=dict(seen),seconds=time.monotonic()-tick))
            ckpt=dict(state_dict=model.state_dict(),width=args.width,size=args.size,families=h.FAMILIES,seed=SEED,
                epoch=epoch+1,route=route,train_group_ids=sorted({r['training_group_id'] for r in selected}))
            torch.save(ckpt,out/(route+'.pt.tmp'));(out/(route+'.pt.tmp')).replace(out/(route+'.pt'))
            h.write(out/(route+'_training.json'),history)
            state('epoch_finished',route=route,**history[-1])
        records.append(dict(route=route,sha256=h.sha(out/(route+'.pt')),parameters=sum(p.numel() for p in model.parameters()),
            bytes=(out/(route+'.pt')).stat().st_size,history=history,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        del model,opt,loader;torch.cuda.empty_cache()
    h.check_sources_after(snapshot,out/'sources_train_after.json')
    h.write(out/'training_complete.json',dict(all_supported_families_trained=True,models=records,
        source_train_sha256=h.sha(source),weak_supervision=True,training_groups=sorted({r['training_group_id'] for r in selected})))
    state('all_training_finished',models=[r['route'] for r in records])
    evaluate(args,h,selected,out,state)


def choose_thresholds(items):
    """Only validation P0; limit false alarms on empty masks when supported."""
    best=None
    for pixel in [.2,.35,.5,.65,.8,.9]:
        for gate in [0.,.1,.3,.5,.7,.9,.95,.99,1.01]:
            met=[mask_stats((r['prob']>=pixel)&(r['presence']>=gate),r['truth']) for r in items]
            pos=[r['dice'] for r in met if r['truth_positive']];empty=[r['prediction_positive'] for r in met if not r['truth_positive']]
            fpr=float(np.mean(empty)) if empty else None
            allowed=max(.05,1/len(empty)) if empty else 1.
            if fpr is not None and fpr>allowed:continue
            key=(float(np.mean(pos)) if pos else 0.,-(fpr or 0.),pixel,gate)
            if best is None or key>best[0]:best=(key,pixel,gate)
    return dict(pixel=best[1],presence=best[2],validation_objective='max positive mean Dice subject to empty FPR <= max(5%,1/n_empty)')


def mask_stats(pred,truth):
    truth=truth.astype(bool);pred=pred.astype(bool)
    tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
    return dict(tp=tp,fp=fp,fn=fn,dice=2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.,
        iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,truth_positive=bool(truth.any()),prediction_positive=bool(pred.any()))


def summarize(items):
    pos=[r for r in items if r['truth_positive']];neg=[r for r in items if not r['truth_positive']]
    return dict(records=len(items),positive_records=len(pos),empty_records=len(neg),
        positive_mean_iou=float(np.mean([r['iou'] for r in pos])) if pos else None,
        positive_mean_dice=float(np.mean([r['dice'] for r in pos])) if pos else None,
        empty_prediction_rate=float(np.mean([r['prediction_positive'] for r in neg])) if neg else None,
        tp=sum(r['tp'] for r in items),fp=sum(r['fp'] for r in items),fn=sum(r['fn'] for r in items))


def evaluate(args,h,train,out,state):
    import torch
    from PIL import Image
    ann,_=h.annotation_module(ROOT);evaluations={};thresholds={}
    # Evaluation consumes ALL usable P0 held-out images, not a single-family pilot.
    for split in ['val','test']:
        raw,source=h.load_rows(ROOT,split);inspected=[h.inspect_row(r,ann) for r in raw]
        usable=[r for r in inspected if r['usable']];h.assert_disjoint(train,usable)
        if split=='test':h.assert_disjoint(val_rows,usable)
        else:val_rows=usable
        h.write(out/(split+'_inventory.json'),inspected)
        state('evaluation_inputs',split=split,families=h.summarize(inspected))
        for route in ['presence_gated','no_presence_loss']:
            saved=torch.load(out/(route+'.pt'),map_location='cpu',weights_only=False)
            model=network(args.width).cuda();model.load_state_dict(saved['state_dict']);model.eval()
            per=collections.defaultdict(list);predictions=[]
            for j,r in enumerate(usable):
                with Image.open(r['data_path']) as im:x=np.array(im.convert('RGB').resize((args.size,args.size),Image.Resampling.BILINEAR),dtype=np.float32)/255
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    z,pr=model(torch.from_numpy(x.transpose(2,0,1).copy())[None].cuda())
                fi=h.FAMILIES.index(r['family']);prob=z[0,fi].sigmoid().float().cpu().numpy()
                presence=float(pr[0,fi].sigmoid()) if route=='presence_gated' else 1.
                truth=h.region_mask(r['geometry'],r['width'],r['height'],args.size).astype(bool)
                per[r['family']].append(dict(prob=prob,presence=presence,truth=truth,id=r['id']))
                if j%100==0:state('evaluating',split=split,route=route,completed=j,total=len(usable))
            if split=='val':
                thresholds[route]={f:choose_thresholds(per[f]) for f in h.FAMILIES}
                # Checkpoint and thresholds frozen BEFORE test is read.
                h.write(out/(route+'_frozen_policy.json'),dict(thresholds=thresholds[route],checkpoint_sha256=h.sha(out/(route+'.pt')),
                    source='all usable grouped P0 validation; independent scientific truth unavailable',test_read=False))
            metrics={}
            for f,items in per.items():
                th=thresholds[route][f]
                stats=[mask_stats((r['prob']>=th['pixel'])&(r['presence']>=th['presence']),r['truth']) for r in items]
                metrics[f]=summarize(stats)
                for r,s in zip(items,stats):predictions.append(dict(id=r['id'],family=f,presence=r['presence'],**s))
            h.write(out/(route+'_'+split+'_metrics.json'),metrics)
            with (out/(route+'_'+split+'_predictions.jsonl')).open('w') as fp:
                for r in predictions:fp.write(json.dumps(r)+'\n')
            evaluations[route+'|'+split]=metrics
            del model;torch.cuda.empty_cache()
    # Choose route on VALIDATION only. Report test of both and selected.
    selected={}
    for f in h.FAMILIES:
        selected[f]=max(['presence_gated','no_presence_loss'],key=lambda r:(evaluations[r+'|val'][f]['positive_mean_dice'] or 0.,-(evaluations[r+'|val'][f]['empty_prediction_rate'] or 0.)))
    h.write(out/'comparison.json',dict(status='complete',selected_on_validation=selected,metrics=evaluations,
        independent_fine_defect_F1=None,scope='P0 weak-mask localization; raw RGB-only global inference; no official defect truth',
        excluded={'CME_C2_sample':'angular-sector/time branch, not pixel masks','ULF_sample':'time-series branch, not pixel masks'}))
    state('complete',selected=selected)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch',type=int,default=16)
    ap.add_argument('--size',type=int,default=384);ap.add_argument('--width',type=int,default=32)
    ap.add_argument('--train-cap',type=int,default=2048);ap.add_argument('--budget-seconds',type=int,default=36000)
    a=ap.parse_args()
    try:run(a)
    except BaseException as exc:
        if a.output.exists():helper().write(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
