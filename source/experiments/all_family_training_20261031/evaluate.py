"""Evaluate each model only after its complete all-family training is frozen."""
import argparse
import collections
import csv
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader

from common import *
from barrier import verify_barrier
from train import f1_by_family


class CurrentInputs(Dataset):
    def __init__(self,rows,size=SIZE):self.rows=rows;self.size=size
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        source=r['image'] if self.size==SPATIAL_SIZE else r['training_image']
        with Image.open(source) as im:x=np.array(im.resize((self.size[1],self.size[0]),Image.Resampling.BILINEAR),dtype=np.float32).transpose(2,0,1)/255
        return x,np.array(r['features'],np.float32),i


def inference(route):
    barrier=verify_barrier(route)  # Must precede reading any held-out input.
    rows=lines(OUT/'inventory.jsonl');path=OUT/'predictions'/(route+'.jsonl');path.parent.mkdir(exist_ok=True)
    old={r['id']:r for r in lines(path)} if path.exists() else {}
    rr=[r for r in rows if r['id'] not in old];present=[r for r in rr if not r['input_missing']]
    start=time.monotonic();cache={r['input_sha256']:r for r in old.values() if r['status']=='success'}
    stream=path.open('a')
    def record(r,prob,status,elapsed):
        item=dict(id=r['id'],input_sha256=r['input_sha256'],probabilities=prob,status=status,seconds=elapsed)
        stream.write(json.dumps(item)+'\n');old[r['id']]=item
        if status=='success':cache[r['input_sha256']]=item
    for r in rr:
        if r['input_missing']:record(r,None,'input_missing',0.)
    if route=='qlora':
        from qwen import load,predict
        bundle=load(OUT/'checkpoints/qlora',False)
        for off in range(0,len(present),32):
            batch=present[off:off+32];todo={r['input_sha256']:r for r in batch if r['input_sha256'] not in cache}
            if todo:
                unique=list(todo.values());ts=time.monotonic();pp=predict(bundle,unique,32);elapsed=time.monotonic()-ts
                for r,p in zip(unique,pp):cache[r['input_sha256']]=dict(probabilities=p.tolist(),seconds=elapsed/len(unique))
            for r in batch:
                p=cache[r['input_sha256']];record(r,p['probabilities'],'success',p['seconds'])
            stream.flush()
            if off%1024==0:
                emit('model_evaluation',route=route,records=len(old),total=len(rows))
                write(OUT/'predictions'/(route+'.state.json'),dict(status='evaluating',route=route,records=len(old),total=len(rows)))
    else:
        from models import build
        saved=torch.load(OUT/'checkpoints'/(route+'.pt'),map_location='cpu',weights_only=False)
        model=build(route,pretrained=False,widths=saved.get('widths'),hidden=saved.get('hidden')).cuda().eval()
        model.load_state_dict(saved['state']);mean=saved['mean'].cuda();std=saved['std'].cuda()
        image_size=SPATIAL_SIZE if route in ARCHS else SIZE
        loader=DataLoader(CurrentInputs(present,image_size),batch_size=16 if route in ARCHS else 64,num_workers=4,pin_memory=True,shuffle=False)
        with torch.inference_mode():
            for batch_index,(x,f,indices) in enumerate(loader):
                ts=time.monotonic()
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,_=model(x.cuda(),(f.cuda()-mean)/std)
                pp=logits.float().sigmoid().cpu().numpy();elapsed=time.monotonic()-ts
                assert np.isfinite(pp).all()
                for i,p in zip(indices.tolist(),pp):record(present[i],p.tolist(),'success',elapsed/len(indices))
                stream.flush()
                if batch_index%64==0:
                    write(OUT/'predictions'/(route+'.state.json'),dict(status='evaluating',route=route,records=len(old),total=len(rows)))
    stream.close();assert set(old)=={r['id'] for r in rows}
    write(OUT/'predictions'/(route+'.complete.json'),dict(status='complete',route=route,records=len(old),
        prediction_hash=sha(path),barrier_hash=sha(OUT/'route_barriers'/(route+'.json')),seconds=time.monotonic()-start,
        missing_inputs=sum(r['input_missing'] for r in rows),hardware=__import__('socket').gethostname()+' A100 40GB; not A10 acceptance'))


def confusion(rows,decisions,statuses):
    y=np.array([r['target'] for r in rows],bool);v=np.array([r['truth_valid'] for r in rows],bool)
    p=np.array([decisions[r['id']] for r in rows],bool)
    tp=int((p&y&v).sum());fp=int((p&~y&v).sum());fn=int((~p&y&v).sum());tn=int((~p&~y&v).sum())
    full=v.all(1);negative=full&~y.any(1);pos=int((y&v).sum())
    out=dict(n=len(rows),known_positive_bits=pos,fully_known_samples=int(full.sum()),partial_or_unknown_samples=int((~full).sum()),
        TP=tp,FP=fp,FN=fn,TN=tn,F1=2*tp/(2*tp+fp+fn) if pos else None,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if pos else None,
        FPR=float(p[negative].any(1).mean()) if negative.any() else None,fully_known_negative_samples=int(negative.sum()),
        missing_inputs=sum(r['input_missing'] for r in rows),abstentions=sum(statuses[r['id']]!='success' for r in rows))
    perclass=[]
    for k,name in enumerate(['G','C','D']):
        a=int((p[:,k]&y[:,k]&v[:,k]).sum());b=int((p[:,k]&~y[:,k]&v[:,k]).sum());c=int((~p[:,k]&y[:,k]&v[:,k]).sum());d=int((~p[:,k]&~y[:,k]&v[:,k]).sum())
        perclass.append(dict(class_name=name,TP=a,FP=b,FN=c,TN=d,known_positive_bits=a+c,unknown=int((~v[:,k]).sum()),
            F1=2*a/(2*a+b+c) if a+c else None,precision=a/(a+b) if a+b else None,recall=a/(a+c) if a+c else None,FPR=b/(b+d) if b+d else None))
    return out,perclass


def csvwrite(name,rows,columns=None):
    path=OUT/name;path.parent.mkdir(exist_ok=True);columns=columns or list(rows[0])
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)


def score():
    verify_barrier();rows=lines(OUT/'inventory.jsonl');assert len(rows)==123565
    from rules import predictions as rule_predict
    rules=rule_predict(rows);decisions={'rules':rules};statuses={'rules':{r['id']:'success' for r in rows}}
    for route in ROUTES:
        check=json.loads((OUT/'predictions'/(route+'.complete.json')).read_text())
        assert check['prediction_hash']==sha(OUT/'predictions'/(route+'.jsonl'))
        pred={r['id']:r for r in lines(OUT/'predictions'/(route+'.jsonl'))};assert set(pred)=={r['id'] for r in rows}
        thresholds=np.array(json.loads((OUT/'training'/(route+'.json')).read_text())['thresholds'])
        decisions[route]={r['id']:(np.array(pred[r['id']]['probabilities'])>=thresholds[FAMILIES.index(r['family'])]).astype(int).tolist()
            if pred[r['id']]['status']=='success' else [0,0,0] for r in rows}
        statuses[route]={r['id']:pred[r['id']]['status'] for r in rows}
    # Confidence gate is calibrated using validation rows only, then frozen for test scoring.
    qwen={r['id']:r for r in lines(OUT/'predictions/qlora.jsonl')}
    va=[r for r in rows if r['split']=='val'];gate=np.full((6,3),1.01)
    for fi,family in enumerate(FAMILIES):
        for k in range(3):
            choices=[]
            for t in [1.01,.95,.9,.8,.7,.6,.5]:
                trial=gate.copy();trial[fi,k]=t
                p=[]
                for r in va:
                    q=qwen[r['id']]['probabilities'];v=np.array(rules[r['id']],bool)
                    if q is not None:v|=np.array(q)>=trial[FAMILIES.index(r['family'])]
                    p.append(v.astype(float))
                choices.append((f1_by_family(va,np.array(p),np.full((6,3),.5)),t))
            gate[fi,k]=max(choices)[1]
    decisions['rules_plus_qlora']={}
    for r in rows:
        q=qwen[r['id']]['probabilities'];p=np.array(rules[r['id']],bool)
        if q is not None:p|=np.array(q)>=gate[FAMILIES.index(r['family'])]
        decisions['rules_plus_qlora'][r['id']]=p.astype(int).tolist()
    statuses['rules_plus_qlora']=statuses['qlora'];write(OUT/'gate_validation.json',dict(source_split='val',thresholds=gate.tolist()))
    packages=[(f,t) for f in FAMILIES for t in TIERS];long=[];classes=[]
    for scope in ['heldout_test','full_inventory_replay']:
        for route,pred in decisions.items():
            for family,tier in packages:
                subset=[r for r in rows if r['family']==family and r['tier']==tier and (scope!='heldout_test' or r['split']=='test')]
                metrics,perclass=confusion(subset,pred,statuses[route]);common=dict(scope=scope,route=route,family=family,tier=tier)
                long.append(dict(common,**metrics));classes.extend(dict(common,**c) for c in perclass)
    csvwrite('metrics_long.csv',long);csvwrite('metrics_per_class.csv',classes)
    for scope in ['heldout_test','full_inventory_replay']:
        wide=[];overview=[]
        for route in ['rules','qlora','supervised','distilled','pruned','rules_plus_qlora']+ARCHS:
            members=[route+'_seed'+str(s) for s in SEEDS] if route in ['supervised','distilled','pruned'] else [route]
            expanded={'route':route};f1={'route':route}
            for family,tier in packages:
                rr=[r for r in long if r['scope']==scope and r['route'] in members and r['family']==family and r['tier']==tier]
                assert len(rr)==len(members)
                for metric in ['F1','precision','recall','FPR','n']:
                    vals=[r[metric] for r in rr if r[metric] is not None]
                    expanded[family+'_'+tier+'_'+metric]=float(np.mean(vals)) if vals else None
                f1[family+'_'+tier]=expanded[family+'_'+tier+'_F1']
            wide.append(expanded);overview.append(f1)
        csvwrite(scope+'_expanded.csv',wide);csvwrite(scope+'_F1.csv',overview)
    write(OUT/'evaluation_complete.json',dict(status='complete',records=123565,packages=30,routes=len(decisions),
        primary='heldout_test',secondary='full_inventory_replay includes training and validation observations',
        truth='P0-derived technical proxy with unknown masks; not official hidden score',
        coarse_classes=['G','C','D'],student_summary='mean of three independent seed metrics, not ensemble',
        barrier_sha256=sha(OUT/'training_barrier.json')))
    report=['# 全六类训练与逐模型评价汇总','','每个模型完成六类训练并通过独立屏障后立即评价；本表为全部路线完成后的汇总。',
        '主成绩是分组留出的 test；全清单回放包括训练数据，不能称泛化成绩。',
        '每包 F1 是已知真值位上的三类技术参考指标；未知位、缺文件及覆盖率见明细，不能称官方隐藏测试分数。','',
        '- `heldout_test_F1.csv`：30包F1概览；', '- `heldout_test_expanded.csv`：30包逐指标展开；',
        '- `metrics_long.csv`、`metrics_per_class.csv`：覆盖率、逐种子与混淆矩阵；',
        '- `training_barrier.json`：全部训练覆盖与冻结权重核验。','',
        'ULF 是波形视图适配；YOLO 是骨干多任务适配，未混报原生检测/实例分割 mAP。',
        'A100 时间只代表本机；每包3小时和最终镜像20GB仍需独立验收。']
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('route');a=p.parse_args()
    score() if a.route=='score' else inference(a.route)
