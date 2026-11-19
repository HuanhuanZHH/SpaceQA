"""Small visual student, genuine structured pruning, and 8B QLoRA pilot."""
import argparse
import copy
import json
import math
import os
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F

ROOT=Path('/staff/tangjiarui/OPC')
OUT=ROOT/'opc-agent/runs/model-comparison-20261031-v2'


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def metrics(rows,predictions):
    valid=[(r,predictions.get(r['id'],-1)) for r in rows]
    y=np.array([r['target'] for r,p in valid]); p=np.array([p for r,p in valid])
    precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=list(range(4)),zero_division=0)
    tp=int(((y>0)&(p==y)).sum())
    fp=int(((p>0)&(p!=y)).sum())
    fn=int(((y>0)&(p!=y)).sum())
    return dict(n=len(valid),missing=sum(r['id'] not in predictions for r in rows),abstention_rate=float((p<0).mean()),accuracy=float(accuracy_score(y,p)),
                macro_f1=float(f1.mean()),balanced_accuracy=float(recall[support>0].mean()),
                defect_precision=tp/max(tp+fp,1),defect_recall=tp/max(tp+fn,1),
                defect_f1=2*tp/max(2*tp+fp+fn,1),clean_false_positive_rate=float((p[y==0]>0).mean()) if (y==0).any() else None,
                per_class=[dict(class_id=i,precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i in range(4)],
                confusion_matrix=confusion_matrix(y,p,labels=list(range(4))).tolist())


def load():
    rows=[json.loads(s) for s in (OUT/'dataset.jsonl').open()]
    selected=set(json.loads((OUT/'teacher_selection.json').read_text()))
    train=[r for r in rows if r['split']=='train' and r['id'] in selected]
    val=[r for r in rows if r['split']=='val']
    test=[r for r in rows if r['split']=='test']
    return train,val,test


def optimize_patch_embed(model):
    """Conv3d over one complete patch is an exactly equivalent linear projection."""
    modules=[m for m in model.modules() if m.__class__.__name__=='Qwen3VLVisionPatchEmbed']
    assert len(modules)==1
    patch=modules[0]
    weight=patch.proj.weight.detach().float().cpu();bias=patch.proj.bias.detach().float().cpu()
    probe=torch.randn(2,*weight.shape[1:])
    expected=F.conv3d(probe,weight,bias,stride=tuple(weight.shape[2:])).flatten(1)
    actual=F.linear(probe.flatten(1),weight.flatten(1),bias)
    torch.testing.assert_close(expected,actual,atol=1e-5,rtol=1e-4)
    write(OUT/'patch_projection_equivalence.json',dict(max_abs_error=float((expected-actual).abs().max()),
          parameters_removed=0,note='Equivalent patch projection; original parameter names and checkpoint unchanged'))
    def forward(self,hidden_states):
        return F.linear(hidden_states.to(self.proj.weight.dtype),self.proj.weight.flatten(1),self.proj.bias)
    patch.forward=types.MethodType(forward,patch)


class Student(nn.Module):
    def __init__(self,widths=(32,64,128),hidden=128):
        super().__init__()
        self.widths=list(widths); self.hidden=hidden
        self.convs=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip([3]+list(widths[:-1]),widths)])
        self.fc1=nn.Linear(widths[-1]*32+9,hidden)
        self.fc2=nn.Linear(hidden,4)
    def forward(self,x,f):
        for conv in self.convs:
            x=F.max_pool2d(F.gelu(conv(x)),2)
        x=F.adaptive_avg_pool2d(x,(4,8)).flatten(1)
        x=torch.cat([x,f],dim=1)
        return self.fc2(F.dropout(F.gelu(self.fc1(x)),.1,self.training))


def prune(model):
    widths=[max(4,w//2) for w in model.widths]; hidden=model.hidden//2
    slim=Student(widths,hidden).to(next(model.parameters()).device)
    prev=torch.arange(3,device=next(model.parameters()).device)
    with torch.no_grad():
        for old,new in zip(model.convs,slim.convs):
            selected=old.weight.abs().sum((1,2,3)).topk(new.out_channels).indices.sort().values
            new.weight.copy_(old.weight[selected][:,prev]); new.bias.copy_(old.bias[selected]); prev=selected
        pooled=(prev[:,None]*32+torch.arange(32,device=prev.device)[None,:]).flatten()
        inputs=torch.cat([pooled,torch.arange(model.widths[-1]*32,model.widths[-1]*32+9,device=prev.device)])
        selected=model.fc1.weight.abs().sum(1).topk(hidden).indices.sort().values
        slim.fc1.weight.copy_(model.fc1.weight[selected][:,inputs]); slim.fc1.bias.copy_(model.fc1.bias[selected])
        slim.fc2.weight.copy_(model.fc2.weight[:,selected]); slim.fc2.bias.copy_(model.fc2.bias)
    return slim


def student(seed,teacher_file='teacher_predictions.jsonl',prefix=''):
    torch.set_num_threads(4)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device=torch.device('cuda:0')
    train,val,test=load()
    teacher={x['id']:x for x in map(json.loads,(OUT/teacher_file).open()) if x['success']}
    train=[r for r in train if r['id'] in teacher]
    allrows=train+val+test
    images=[]
    for r in allrows:
        with Image.open(r['image']) as im:
            images.append(np.array(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
    x=torch.tensor(np.array(images),device=device)
    f=torch.tensor([r['features'] for r in allrows],dtype=torch.float32,device=device)
    mean=f[:len(train)].mean(0); std=f[:len(train)].std(0).clamp_min(.01); f=(f-mean)/std
    y=torch.tensor([r['target'] for r in allrows],device=device)
    soft=torch.tensor([teacher[r['id']]['probabilities'] for r in train],device=device)
    val_idx=torch.arange(len(train),len(train)+len(val),device=device)
    test_idx=torch.arange(len(train)+len(val),len(allrows),device=device)
    def predict(model,idx):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x[ii],f[ii]).argmax(1) for ii in idx.split(32)]).cpu().tolist()
    def fit(model,distill,epochs,tag):
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.001)
        best=-1; checkpoint=None; trace=[]
        for epoch in range(epochs):
            model.train(); total=0
            for idx in torch.randperm(len(train),device=device).split(32):
                logits=model(x[idx],f[idx])
                loss=F.cross_entropy(logits,y[idx])
                if distill:
                    target=soft[idx].clamp_min(1e-8).pow(.5); target=target/target.sum(1,keepdim=True)
                    loss=.5*loss+.5*4*F.kl_div(F.log_softmax(logits/2,dim=1),target,reduction='batchmean')
                optimizer.zero_grad();loss.backward();optimizer.step();total+=float(loss)
            vp=predict(model,val_idx); score=metrics(val,dict(zip([r['id'] for r in val],vp)))['macro_f1']
            trace.append(dict(epoch=epoch,loss=total,validation_macro_f1=score))
            if score>best:
                best=score; checkpoint=copy.deepcopy(model.state_dict())
        model.load_state_dict(checkpoint)
        print(tag,'validation_macro_f1',best,flush=True)
        return model,trace
    result={'train_records':len(train),'training_ids':sorted(r['id'] for r in train),'teacher_file':teacher_file}
    started=time.monotonic()
    for distill in (False,True):
        tag='distilled' if distill else 'supervised'
        model,trace=fit(Student().to(device),distill,30,tag)
        def evaluate(name,m):
            predictions=dict(zip([r['id'] for r in test],predict(m,test_idx)))
            checkpoint=OUT/f'student_{prefix}{name}_seed{seed}.pt'
            torch.save(dict(state=m.state_dict(),widths=m.widths,hidden=m.hidden,feature_mean=mean.cpu(),feature_std=std.cpu()),checkpoint)
            result[name]=dict(metrics=metrics(test,predictions),predictions=predictions,
                              parameters=sum(p.numel() for p in m.parameters()),artifact_bytes=checkpoint.stat().st_size)
        evaluate(tag,model)
        slim=prune(model);evaluate(tag+'_pruned_before_recovery',slim)
        slim,recovery=fit(slim,distill,20,tag+'_pruned_recovered');evaluate(tag+'_pruned_recovered',slim)
        result[tag+'_pruned_recovered']['training_trace']=recovery
        result[tag]['training_trace']=trace
    result['wall_seconds']=time.monotonic()-started
    write(OUT/f'student_{prefix}seed{seed}.json',result)
    print('student complete',seed,result['wall_seconds'],flush=True)


def qlora(steps):
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    torch.set_num_threads(4);torch.manual_seed(20261031)
    train,val,test=load()
    processor=AutoProcessor.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',min_pixels=28*28*64,max_pixels=28*28*256)
    model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    optimize_patch_embed(model)
    letters=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD']
    assert all(len(x)==1 for x in letters),letters
    class_ids=torch.tensor([x[0] for x in letters],device='cuda:0')
    def encode(row,training=False):
        with Image.open(row['image']) as im: image=im.copy()
        messages=[{'role':'user','content':[{'type':'text','text':row['prompt']},{'type':'image','image':image}]}]
        prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=processor(text=[prompt],images=[image],return_tensors='pt')
        if training:
            answer=processor.tokenizer.encode('ABCD'[row['target']]+'<|im_end|>',add_special_tokens=False)
            old=inputs['input_ids'].shape[1]
            inputs['input_ids']=torch.cat([inputs['input_ids'],torch.tensor([answer])],dim=1)
            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
            inputs['labels']=inputs['input_ids'].clone();inputs['labels'][:,:old]=-100
        return inputs.to('cuda:0')
    def evaluate(tag):
        model.eval();model.config.use_cache=False
        predictions={};probabilities={}
        began=time.monotonic()
        def batched_eval(rows):
            with torch.inference_mode():
                for i in range(0,len(rows),8):
                    batch=rows[i:i+8];images=[];prompts=[]
                    for row in batch:
                        with Image.open(row['image']) as im:image=im.copy()
                        images.append(image)
                        messages=[{'role':'user','content':[{'type':'text','text':row['prompt']},{'type':'image','image':image}]}]
                        prompts.append(processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
                    processor.tokenizer.padding_side='left'
                    inputs=processor(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda:0')
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,class_ids].float()
                    pp=logits.argmax(1).cpu().tolist()
                    predictions.update({row['id']:p for row,p in zip(batch,pp)})
                    probabilities.update({row['id']:p for row,p in zip(batch,logits.softmax(1).cpu().tolist())})
                    if i%40==0:print(tag,'eval',i,'/',len(rows),flush=True)
        batched_eval(test)
        result=dict(metrics=metrics(test,predictions),predictions=predictions,seconds=time.monotonic()-began)
        if (OUT/'extended_test.jsonl').exists():
            extended=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
            batched_eval(extended)
            result['extended_metrics']=metrics(extended,predictions)
        if tag=='after':
            batched_eval(val)
            result['validation_metrics']=metrics(val,predictions)
        result['probabilities']=probabilities
        write(OUT/f'qlora_{tag}.json',result)
        return result
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                         gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.visual.to(dtype=torch.bfloat16)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,
                         target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none'))
    model.config.use_cache=False
    evaluate('before')
    model.train()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00015)
    order=[];rng=random.Random(20261031)
    while len(order)<steps*4:
        epoch=list(train);rng.shuffle(epoch);order.extend(epoch)
    trace=[];started=time.monotonic();optimizer.zero_grad()
    for i in range(0,steps*4,2):
        parts=[encode(row,True) for row in order[i:i+2]]
        longest=max(p['input_ids'].shape[1] for p in parts)
        batch={}
        for key in ['input_ids','attention_mask','labels']:
            fill=processor.tokenizer.pad_token_id if key=='input_ids' else -100 if key=='labels' else 0
            batch[key]=torch.cat([F.pad(p[key],(longest-p[key].shape[1],0),value=fill) for p in parts])
        for key in ['pixel_values','image_grid_thw']:
            batch[key]=torch.cat([p[key] for p in parts])
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**batch,use_cache=False).loss
        (loss/2).backward()
        if (i+2)%4==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step();optimizer.zero_grad()
            step=(i+2)//4
            trace.append(dict(step=step,loss=float(loss.detach()),elapsed=time.monotonic()-started))
            if step%5==0:print('qlora step',step,'loss',float(loss.detach()),'seconds',time.monotonic()-started,flush=True)
            if step%20==0:
                model.save_pretrained(OUT/'qlora_adapter');write(OUT/'qlora_trace.json',trace)
    model.save_pretrained(OUT/'qlora_adapter')
    write(OUT/'qlora_training.json',dict(steps=steps,microbatch=2,accumulation=2,evaluation_batch=8,train_records=len(train),
          trainable_parameters=trainable,rank=8,alpha=16,targets=['q_proj','v_proj'],
          seconds=time.monotonic()-started,trace=trace,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    evaluate('after')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['student','qlora']);parser.add_argument('--seed',type=int,default=20261031);parser.add_argument('--steps',type=int,default=80)
    parser.add_argument('--teacher-file',default='teacher_predictions.jsonl');parser.add_argument('--prefix',default='')
    args=parser.parse_args()
    if args.mode=='student':student(args.seed,args.teacher_file,args.prefix)
    else:qlora(args.steps)
