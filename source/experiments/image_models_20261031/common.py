import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/image-models-20261031'
PREVIOUS=ROOT/'runs/spatial-research-448-20261031'
SIZE=448
SEED=20261031
os.environ['OPC_SPATIAL_SIZE']='448'
sys.path.insert(0,str(ROOT/'experiments/spatial_research_20261031'))
from spatial import residual_features,metrics,fit_gates,shapes,dice,seed


def lines(p):return [json.loads(s) for s in p.open() if s.strip()]
def dump(name,obj):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(obj,indent=2,ensure_ascii=False));q.replace(p)
def emit(stage,**kwargs):print(json.dumps(dict(stage=stage,**kwargs)),flush=True)


def annotation_boxes(ann,w,h):
    result=[]
    for s in shapes(ann):
        p=np.array(s.get('points',[]),float)
        if len(p)<2:continue
        lo=np.clip(p.min(0)/[w,h],0,1);hi=np.clip(p.max(0)/[w,h],0,1)
        if np.all(hi>lo):result.append([*lo,*hi])
    return np.array(result,dtype=float).reshape(-1,4)


def ious(a,b):
    if len(a)==0 or len(b)==0:return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    overlap=np.maximum(hi-lo,0).prod(2)
    aa=np.maximum(a[:,2:]-a[:,:2],0).prod(1);bb=np.maximum(b[:,2:]-b[:,:2],0).prod(1)
    return overlap/np.maximum(aa[:,None]+bb[None,:]-overlap,1e-10)


def dense_instances(p):
    n,regions,stats,_=cv2.connectedComponentsWithStats((p>.5).astype(np.uint8),8)
    result=[]
    for j,s in enumerate(stats[1:],1):
        x,y,w,h,area=map(int,s)
        if area<4:continue
        conf=float(np.asarray(p[y:y+h,x:x+w],float)[regions[y:y+h,x:x+w]==j].mean())
        result.append(dict(xyxy=[x/SIZE,y/SIZE,(x+w)/SIZE,(y+h)/SIZE],confidence=conf))
    return sorted(result,key=lambda r:r['confidence'],reverse=True)[:100]


def permutation(bases):
    result=np.arange(len(bases));rng=np.random.RandomState(SEED)
    for family in sorted({r['family'] for r in bases}):
        for split in ('train','val','test'):
            ii=np.array([i for i,b in enumerate(bases) if b['family']==family and b['split']==split]);result[ii]=rng.permutation(ii)
    return result
