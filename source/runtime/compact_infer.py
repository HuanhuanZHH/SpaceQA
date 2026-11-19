"""Offline eager inference with all frozen experts; no dataset-specific caches."""
from pathlib import Path
import os,sys,json,time,gc,pickle,hashlib,contextlib,argparse
import numpy as np
R=Path('/staff/tangjiarui/OPC')
ROUTES=['qlora']+[f'{s}_seed{seed}' for s in ['supervised','distilled','pruned'] for seed in [20261031,20261101,20261102]]+['unet','attention_unet','unetpp','resnet18_unet','deeplab_mobilenet','yolo11_detect_backbone','yolo11_segment_backbone','yolov8_segment_backbone','dino_finetune','ijepa_finetune','gbdt_prior_v1']
FAMILIES=['AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample']
CLASSES=['L0-01','L2-G','L2-P','L2-T','L2-F','L3-06','L3-07','L3-MIS']
CHECKPOINTS=R/'opc-agent/runs/all-family-training-20261031/checkpoints'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def setup_models():
 code=R/'opc-agent/experiments/all_family_training_20261031'
 sys.path.insert(0,str(code))
 import common,models
 return common,models
def infer(records,out,batch=16,qbatch=32,device='cuda:0',routes=None):
 import torch
 from PIL import Image
 from torch.utils.data import Dataset,DataLoader
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4)
 common,models=setup_models();routes=routes or ROUTES
 present=[i for i,r in enumerate(records) if not r['input_missing']]
 timings={};p=np.full((len(records),len(ROUTES),3),np.nan,np.float32)
 class Inputs(Dataset):
  def __init__(self,spatial):self.spatial=spatial
  def __len__(self):return len(present)
  def __getitem__(self,k):
   r=records[present[k]];size=(384,768) if self.spatial else (128,256)
   with Image.open(r['image'] if self.spatial else r['training_image']) as im:
    x=np.array(im.resize((size[1],size[0]),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
   return x,np.array(r['features'],np.float32),present[k]
 for route in routes:
  j=ROUTES.index(route);dest=out/(route+'.npz');meta=out/(route+'.json')
  if dest.exists() and meta.exists():
   data=np.load(dest);assert np.array_equal(data['ids'],[r['id'] for r in records]);p[:,j]=data['prob'];timings[route]=read(meta);continue
  start=time.monotonic();batches=0;items=0
  if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
  if route=='gbdt_prior_v1':
   sys.path.insert(0,str(R/'spaceqa-agent/experiments/prior_optimization_20261102/gbdt'))
   from train_gbdt import features
   g=R/'inference_prior_20261102/gbdt';vectorizer=pickle.load(open(g/'vectorizer.pkl','rb'))
   use_anchor=read(g/'training.json').get('additional_CME_input')!='none'
   def ff(r):
    anchor=(read(r['label_path']).get('hek_event_annotation') or {}).get('image_time') if use_anchor and r['family']=='CME_C2_sample' else None
    return features(r,cme_image_time=anchor)
   x=vectorizer.transform([ff(records[i]) for i in present])
   p[present,j]=np.stack([pickle.load(open(g/('model_'+b+'.pkl'),'rb')).predict_proba(x)[:,1] for b in ['G','C','D']],1)
   batches=3;items=len(present)
  elif route=='qlora':
   if not device.startswith('cuda'):raise ValueError('NF4 vision expert requires CUDA')
   from qwen import load,predict
   bundle=load(CHECKPOINTS/'qlora',False)
   for off in range(0,len(present),qbatch):
    ix=present[off:off+qbatch];pp=predict(bundle,[records[i] for i in ix],qbatch)
    p[ix,j]=pp;batches+=1;items+=len(ix)
    if batches%10==0:print(json.dumps(dict(stage='fresh_forward',route=route,completed=items,total=len(present),seconds=time.monotonic()-start)),flush=True)
   del bundle
  else:
   saved=torch.load(CHECKPOINTS/(route+'.pt'),map_location='cpu',weights_only=False)
   model=models.build(route,pretrained=False,widths=saved.get('widths'),hidden=saved.get('hidden')).to(device).eval()
   model.load_state_dict(saved['state'],strict=True);mean=saved['mean'].to(device);std=saved['std'].to(device)
   spatial=route in common.ARCHS
   loader=DataLoader(Inputs(spatial),batch_size=batch if spatial else max(batch,64),num_workers=4,pin_memory=device.startswith('cuda'),shuffle=False)
   with torch.inference_mode():
    for x,f,ix in loader:
     with torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') else contextlib.nullcontext():
      logits,_=model(x.to(device,non_blocking=True),(f.to(device,non_blocking=True)-mean)/std)
     pp=logits.float().sigmoid().cpu().numpy();assert np.isfinite(pp).all()
     p[ix.numpy(),j]=pp;batches+=1;items+=len(ix)
     if batches%100==0:print(json.dumps(dict(stage='fresh_forward',route=route,completed=items,total=len(present),seconds=time.monotonic()-start)),flush=True)
   del model,saved,loader
  if device.startswith('cuda'):torch.cuda.synchronize()
  receipt=dict(route=route,status='complete',seconds=time.monotonic()-start,fresh_forward_samples=items,batches=batches,
   peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
   missing_records=len(records)-len(present),cache_reuse_from_training_or_demo=False)
  np.savez_compressed(dest,ids=np.array([r['id'] for r in records]),prob=p[:,j]);write(meta,receipt);timings[route]=receipt
  print(json.dumps(dict(stage='expert_complete',**receipt)),flush=True)
  gc.collect()
  if device.startswith('cuda'):torch.cuda.empty_cache()
 return p,timings
def classify(records,p,broad_rule,fine_rule):
 policy=read(R/'eight_class_20261104/fusion_v3_missing4/frozen_policy.json')
 bundle=pickle.load(open(R/'eight_class_20261104/fusion_v3_missing4/models.pkl','rb'))
 z=np.clip(np.nan_to_num(p,nan=.5),1e-5,1-1e-5);fi=np.array([FAMILIES.index(r['family']) for r in records])
 structural=np.asarray([r['features'] for r in records],np.float32);family=np.eye(6,dtype=np.float32)[fi]
 current=np.concatenate([structural,family,fine_rule],1).astype(np.float32)
 x=np.concatenate([z.reshape(len(p),-1),np.log(z/(1-z)).reshape(len(p),-1),structural,family,broad_rule,fine_rule],1).astype(np.float32)
 assert x.shape[1]==399 and current.shape[1]==270
 probs={}
 for name,heads in bundle.items():
  qq=np.zeros((len(records),8),np.float32)
  for b,model in enumerate(heads):
   if model is not None and 1 in model.classes_:qq[:,b]=model.predict_proba(current if name.startswith('current') else x)[:,list(model.classes_).index(1)]
  probs[name]=qq
 prediction=np.zeros_like(fine_rule,dtype=bool);score=np.zeros_like(fine_rule,dtype=np.float32)
 for f,family in enumerate(FAMILIES):
  m=fi==f
  for b,code in enumerate(CLASSES):
   name=policy['selected'][family][code]
   if name=='rules':prediction[m,b]=fine_rule[m,b];score[m,b]=fine_rule[m,b]
   else:score[m,b]=probs[name][m,b];prediction[m,b]=fine_rule[m,b]|(score[m,b]>=policy['thresholds'][name][f][b])
 missing=np.array([r['input_missing'] for r in records]);prediction[missing]=fine_rule[missing]
 return prediction,score,x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--records',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--batch',type=int,default=16);ap.add_argument('--qbatch',type=int,default=32);ap.add_argument('--routes',nargs='*');args=ap.parse_args()
 records=[json.loads(s) for s in args.records.read_text().splitlines() if s.strip()]
 infer(records,args.output,args.batch,args.qbatch,routes=args.routes)
if __name__=='__main__':main()
