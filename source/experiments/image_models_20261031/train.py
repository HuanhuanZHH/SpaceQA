"""Train raw-image spatial predictors without test-label feedback."""
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from common import OUT,ROOT,SIZE,SEED,lines,dump,emit,dice,seed,dense_instances
from models import build,WEIGHTS


def summarize_dense(route,path,metadata,instances=None):
    bases=lines(OUT/'bases.jsonl');pred=np.load(path,mmap_mode='r');truth=np.load(OUT/'clean.npy',mmap_mode='r')
    records=[]
    for split in ['val','test']:
        for family in ['ALL','EPB_seg_sample','DMSP_SSUSI']:
            indices=[i for i,b in enumerate(bases) if b['split']==split and (family=='ALL' or b['family']==family)]
            ds=[];js=[]
            for i in indices:
                p=pred[i]>.5;y=truth[i]>0;inter=int((p&y).sum());a=int(p.sum());b=int(y.sum())
                ds.append((2*inter+1)/(a+b+1));js.append((inter+1)/(int((p|y).sum())+1))
            records.append(dict(split=split,family=family,samples=len(indices),dice=float(np.mean(ds)),iou=float(np.mean(js))))
    if instances is None:
        instances=[dict(base_index=i,instances=dense_instances(p)) for i,p in enumerate(pred)]
    (OUT/'instances').mkdir(exist_ok=True)
    (OUT/'instances'/(route+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in instances))
    metadata.update(route=route,dense_metrics=records,prediction_path=str(path),
        test_truth_used_only_for_reporting=True,formal_scoring_enabled=False)
    dump('training/'+route+'.json',metadata);emit('trained',**{k:v for k,v in metadata.items() if k!='trace'})


def configure(route):
    arch,*suffix=route.split('__')
    option=suffix[0] if suffix else 'default'
    return arch,48 if option=='long48' else 24,'detail' if option=='detail' else 'default',int(option[4:]) if option.startswith('seed') else SEED


def set_seed(value):
    seed();random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def segmentation(route):
    arch,epochs,objective,runseed=configure(route)
    set_seed(runseed);start=time.monotonic();bases=lines(OUT/'bases.jsonl')
    tr=np.array([i for i,b in enumerate(bases) if b['split']=='train']);va=np.array([i for i,b in enumerate(bases) if b['split']=='val'])
    raw=np.load(OUT/'raw.npy',mmap_mode='r');clean=np.load(OUT/'clean.npy',mmap_mode='r')
    x=torch.tensor(np.array(raw).transpose(0,3,1,2),dtype=torch.float32,device='cuda')/255
    y=torch.tensor(np.array(clean)[:,None],dtype=torch.float32,device='cuda')
    model=build(arch).cuda();lr=1e-4 if arch in ('resnet18_unet','deeplab_mobilenet') else 1e-3
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    posweight=((1-y[tr].mean())/y[tr].mean().clamp_min(.01)).clamp(1,10)
    trace=[];best=(-1,None,0);batch=16
    for epoch in range(epochs):
        model.train();losses=[];perm=np.random.permutation(tr)
        for off in range(0,len(perm),batch):
            ii=perm[off:off+batch];opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x[ii])
            p=p.float();loss=F.binary_cross_entropy_with_logits(p,y[ii],pos_weight=posweight)+1-dice(p.sigmoid(),y[ii])
            if objective=='detail':
                probability=p.sigmoid()
                boundary=(F.max_pool2d(probability,3,1,1)+F.max_pool2d(-probability,3,1,1)).clamp(1e-4,1-1e-4)
                target_boundary=(F.max_pool2d(y[ii],3,1,1)+F.max_pool2d(-y[ii],3,1,1)).clamp(0,1)
                loss=loss+.25*F.binary_cross_entropy(boundary,target_boundary)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval();value=0.
        with torch.inference_mode():
            for off in range(0,len(va),batch):
                ii=va[off:off+batch]
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x[ii])
                value+=float(dice((p.sigmoid()>.5).float(),y[ii]))*len(ii)
        value/=len(va);trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),val_clean_dice=value))
        if value>best[0]:best=(value,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch+1)
        if epoch%4==0:emit(route,**trace[-1])
    model.load_state_dict(best[1]);model.eval();peak=torch.cuda.max_memory_reserved()
    checkpoint=OUT/'checkpoints'/(route+'.pt');checkpoint.parent.mkdir(exist_ok=True)
    torch.save(dict(state=best[1],route=route,architecture=arch,image_size=SIZE,seed=runseed,epoch=best[2]),checkpoint)
    (OUT/'predictions').mkdir(exist_ok=True);path=OUT/'predictions'/(route+'.npy')
    pred=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(bases),SIZE,SIZE))
    ts=time.monotonic()
    with torch.inference_mode():
        for off in range(0,len(bases),batch):
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(x[off:off+batch])
            pred[off:off+batch]=p.sigmoid()[:,0].float().cpu().numpy()
    torch.cuda.synchronize();inference=time.monotonic()-ts;pred.flush();del pred
    summarize_dense(route,path,dict(parameters=sum(p.numel() for p in model.parameters()),weights_bytes=checkpoint.stat().st_size,
        training_seconds=time.monotonic()-start,cached_raw_inference_seconds=inference,peak_reserved_bytes=peak,
        selected_epoch=best[2],selection='P0 validation Dice only',precision='BF16 autocast; FP32 BCE+Dice',
        trace=trace,learning_rate=lr,batch=batch,epochs=epochs,objective=objective,seed=runseed,
        architecture=arch,pretrained=arch in ('resnet18_unet','deeplab_mobilenet')))


def yolo(route):
    (OUT/'ultralytics_config').mkdir(exist_ok=True)
    from ultralytics import YOLO
    arch,epochs,objective,runseed=configure(route)
    set_seed(runseed);start=time.monotonic();zero=arch=='zero_shot_yolo11seg'
    weight={'yolo11_detect':'yolo11n.pt','yolo11_segment':'yolo11n-seg.pt','yolov8_segment':'yolov8n-seg.pt',
            'zero_shot_yolo11seg':'yolo11n-seg.pt'}[arch]
    net=YOLO(str(WEIGHTS/weight));task='detect' if arch=='yolo11_detect' else 'segment'
    training=0.
    if not zero:
        ts=time.monotonic()
        detail=dict(box=10.,dfl=2.,**(dict(mask_ratio=2) if task=='segment' else {})) if objective=='detail' else {}
        net.train(data=str(OUT/f'yolo_{task}.yaml'),epochs=epochs,imgsz=SIZE,batch=16,device=1,workers=4,
            seed=runseed,deterministic=True,optimizer='AdamW',lr0=.001,weight_decay=.01,amp=False,
            project=str(OUT/'ultralytics_runs'),name=route,exist_ok=False,plots=False,save=True,
            patience=100,cache=False,verbose=False,close_mosaic=0,mosaic=0.,mixup=0.,copy_paste=0.,
            degrees=0.,translate=0.,scale=0.,shear=0.,perspective=0.,flipud=0.,fliplr=0.,
            hsv_h=0.,hsv_s=0.,hsv_v=0.,**detail)
        training=time.monotonic()-ts
        checkpoint=OUT/'checkpoints'/(route+'.pt');checkpoint.parent.mkdir(exist_ok=True)
        shutil.copyfile(net.trainer.best,checkpoint);net=YOLO(str(checkpoint))
    else:checkpoint=WEIGHTS/weight
    bases=lines(OUT/'bases.jsonl');(OUT/'predictions').mkdir(exist_ok=True)
    path=OUT/'predictions'/(route+'.npy');pred=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(bases),SIZE,SIZE))
    instances=[];ts=time.monotonic()
    for off in range(0,len(bases),16):
        paths=[str(OUT/'yolo_images_all'/f'{i:05d}.png') for i in range(off,min(off+16,len(bases)))]
        results=net.predict(source=paths,imgsz=SIZE,device=1,batch=16,conf=.05,iou=.7,max_det=100,
            retina_masks=True,verbose=False,save=False,stream=False)
        for j,r in enumerate(results):
            p=np.zeros((SIZE,SIZE),dtype=np.float32);objects=[]
            xy=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0,4))
            scores=r.boxes.conf.cpu().numpy() if r.boxes is not None else []
            masks=r.masks.data.cpu().numpy() if r.masks is not None else None
            for k,(b,score) in enumerate(zip(xy,scores)):
                objects.append(dict(xyxy=(b/SIZE).tolist(),confidence=float(score)))
                if masks is not None:
                    m=cv2.resize(masks[k].astype(np.float32),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
                else:
                    m=np.zeros_like(p);x0,y0,x1,y1=np.clip(np.rint(b).astype(int),0,SIZE)
                    m[y0:y1,x0:x1]=1
                p=np.maximum(p,m*float(score))
            pred[off+j]=p;instances.append(dict(base_index=off+j,instances=objects))
    pred.flush();del pred
    summarize_dense(route,path,dict(parameters=sum(p.numel() for p in net.model.parameters()),
        weights_bytes=checkpoint.stat().st_size,training_seconds=training,total_seconds=time.monotonic()-start,
        resized_png_inference_seconds=time.monotonic()-ts,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        selection='framework best P0 validation mAP' if not zero else 'unadapted COCO model; foreground proposals only',
        task=task,epochs=0 if zero else epochs,seed=runseed,objective=objective,architecture=arch,
        mask_representation='rectangles only' if task=='detect' else 'confidence weighted instance masks'),instances)


if __name__=='__main__':
    route=sys.argv[1]
    if route.startswith('yolo') or route.startswith('zero_'):yolo(route)
    else:segmentation(route)
