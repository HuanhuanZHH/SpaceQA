import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC')
REPO=ROOT/'opc-agent'
SOURCE=REPO/'runs/full-matrix-20261031'
OUT=Path(os.environ.get('OPC_ALL_FAMILY_OUT',str(REPO/'runs/all-family-training-20261031')))
CODE=Path(__file__).resolve().parent
FAMILIES=['AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample']
TIERS=['P0','P1','P2','P3','P4']
SEEDS=[20261031,20261101,20261102]
ARCHS=['unet','attention_unet','unetpp','resnet18_unet','deeplab_mobilenet',
       'yolo11_detect_backbone','yolo11_segment_backbone','yolov8_segment_backbone',
       'dino_finetune','ijepa_finetune']
ROUTES=['qlora']+[f'{s}_seed{seed}' for s in ['supervised','distilled','pruned'] for seed in SEEDS]+ARCHS
SIZE=(128,256)
SPATIAL_SIZE=(384,768)
FEATURES=256
MANIFEST=ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def lines(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)
def save_lines(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    tmp.replace(path)
def emit(stage,**kw):print(json.dumps(dict(stage=stage,**kw)),flush=True)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def checkpoint_files(route):
    p=OUT/'checkpoints'/route
    return sorted([x for x in p.rglob('*') if x.is_file()]) if p.is_dir() else [p.with_suffix('.pt')]
def assert_coverage(rows,keys=('family','tier')):
    import collections
    c=collections.Counter(tuple(r[k] for k in keys) for r in rows)
    assert len(c)==30 and all(c[(f,t)]>0 for f in FAMILIES for t in TIERS),dict(c)
    return {'|'.join(k):n for k,n in sorted(c.items())}
