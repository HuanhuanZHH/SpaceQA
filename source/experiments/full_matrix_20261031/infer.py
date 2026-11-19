"""Resumable full-inventory inference with the frozen QLoRA and nine students."""
import hashlib
import json
import os
import sqlite3
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from common import ROOT, REPO, OUT, OLD, SEEDS, lines, dump, emit
from adapt import adapt


class Student(nn.Module):
    def __init__(self,widths=(32,64,128),hidden=128):
        super().__init__();self.widths=list(widths);self.hidden=hidden
        self.convs=nn.ModuleList([nn.Conv2d(a,b,3,padding=1) for a,b in zip([3]+list(widths[:-1]),widths)])
        self.fc1=nn.Linear(widths[-1]*32+9,hidden);self.fc2=nn.Linear(hidden,4)
    def forward(self,x,f):
        for conv in self.convs:x=F.max_pool2d(F.gelu(conv(x)),2)
        x=F.adaptive_avg_pool2d(x,(4,8)).flatten(1)
        return self.fc2(F.dropout(F.gelu(self.fc1(torch.cat([x,f],1))),.1,self.training))


def load_students():
    models={};metadata=[]
    for route,stem in [('supervised','student_supervised'),('distilled','student_finetuned_distilled'),
                       ('pruned','student_supervised_pruned_recovered')]:
        for seed in SEEDS:
            path=OLD/f'{stem}_seed{seed}.pt';data=torch.load(path,map_location='cpu',weights_only=False)
            model=Student(data['widths'],data['hidden']).cuda().eval();model.load_state_dict(data['state'])
            name=route+'_seed'+str(seed)
            models[name]=(model,data['feature_mean'].cuda(),data['feature_std'].cuda())
            metadata.append(dict(route=name,path=str(path),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parameters=sum(p.numel() for p in model.parameters())))
    dump('student_weights.json',metadata);return models


def student_batch(model_tuple,rows):
    model,mean,std=model_tuple;images=[]
    for row in rows:
        with Image.open(row['image']) as im:images.append(np.asarray(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
    x=torch.tensor(np.array(images),device='cuda');features=torch.tensor([r['features'] for r in rows],dtype=torch.float32,device='cuda')
    with torch.inference_mode():return model(x,(features-mean)/std).softmax(1).cpu().numpy()


def load_qwen():
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
    from peft import PeftModel,prepare_model_for_kbit_training
    start=time.monotonic();path=ROOT/'models/Qwen3-VL-8B-Instruct'
    processor=AutoProcessor.from_pretrained(path,min_pixels=28*28*64,max_pixels=28*28*256)
    model=Qwen3VLForConditionalGeneration.from_pretrained(path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    patch=[m for m in model.modules() if m.__class__.__name__=='Qwen3VLVisionPatchEmbed']
    assert len(patch)==1
    def forward(self,hidden_states):
        return F.linear(hidden_states.to(self.proj.weight.dtype),self.proj.weight.flatten(1),self.proj.bias)
    patch[0].forward=types.MethodType(forward,patch[0])
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False);model.model.visual.to(dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,OLD/'qlora_adapter').eval();model.config.use_cache=False
    ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCD'];assert all(len(x)==1 for x in ids)
    class_ids=torch.tensor([x[0] for x in ids],device='cuda');processor.tokenizer.padding_side='left'
    dump('qwen_weights.json',dict(base=str(path),adapter=str(OLD/'qlora_adapter'),
        adapter_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD/'qlora_adapter').glob('*') if p.is_file()},
        engine='Transformers NF4 + frozen existing PEFT; SDPA; same full-patch linear projection',startup_seconds=time.monotonic()-start))
    return processor,model,class_ids


def qwen_batch(bundle,rows):
    processor,model,class_ids=bundle;images=[];prompts=[]
    for row in rows:
        with Image.open(row['image']) as im:image=im.copy()
        images.append(image)
        prompts.append(processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':row['prompt']},
            {'type':'image','image':image}]}],tokenize=False,add_generation_prompt=True))
    inputs=processor(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,class_ids].float()
    probs=logits.softmax(1).cpu().numpy()
    assert np.isfinite(probs).all()
    return probs


def compatibility(bundle,students):
    rows=[r for r in lines(OLD/'dataset.jsonl') if r['split']=='test']
    expected=json.loads((OLD/'qlora_after.json').read_text())['predictions']
    predictions=[]
    for off in range(0,len(rows),8):predictions.extend(qwen_batch(bundle,rows[off:off+8]).argmax(1).tolist())
    differing=[r['id'] for r,p in zip(rows,predictions) if p!=expected[r['id']]]
    student_differences={}
    for name,model in students.items():
        route,seed=name.split('_seed')
        file=OLD/(('student_finetuned_seed' if route=='distilled' else 'student_seed')+seed+'.json')
        tag={'supervised':'supervised','distilled':'distilled','pruned':'supervised_pruned_recovered'}[route]
        golden=json.loads(file.read_text())[tag]['predictions'];pred=[]
        for off in range(0,len(rows),64):pred.extend(student_batch(model,rows[off:off+64]).argmax(1).tolist())
        student_differences[name]=sum(p!=golden[r['id']] for r,p in zip(rows,pred))
    result=dict(rows=len(rows),qwen_differences=len(differing),qwen_differing_ids=differing,student_differences=student_differences,
                passed=len(differing)<=2 and not any(student_differences.values()),
                note='At most two QLoRA differences tolerate batch/kernel arithmetic; original held-out inputs are used.')
    dump('compatibility.json',result);emit('compatibility',**result)
    if not result['passed']:raise RuntimeError('Frozen checkpoint compatibility failed; full inference halted for diagnosis')


def main():
    torch.set_num_threads(4);start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(OUT/'predictions.sqlite');connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE IF NOT EXISTS predictions (route TEXT,input_hash TEXT,prediction INTEGER,probabilities TEXT,status TEXT,elapsed REAL,PRIMARY KEY(route,input_hash))')
    connection.execute('CREATE TABLE IF NOT EXISTS row_inputs (id TEXT PRIMARY KEY,qwen_hash TEXT,student_hash TEXT,profile TEXT)')
    done={}
    for route,key in connection.execute('SELECT route,input_hash FROM predictions'):done.setdefault(route,set()).add(key)
    dump('inference_state.json',dict(status='loading_frozen_models',pid=os.getpid()))
    students=load_students();bundle=load_qwen()
    dump('inference_state.json',dict(status='checking_old_checkpoint_compatibility',pid=os.getpid()))
    if not (OUT/'compatibility.json').exists() or not json.loads((OUT/'compatibility.json').read_text()).get('passed'):
        compatibility(bundle,students)
    batch=32;seen_shards=set();rows_seen=0;unique_qwen=0;unique_students=0;failed=0
    def store(route,key,probs,status,elapsed):
        prediction=int(np.argmax(probs)) if probs is not None else -1
        connection.execute('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?)',
            (route,key,prediction,json.dumps([float(x) for x in probs]) if probs is not None else None,status,float(elapsed)))
        done.setdefault(route,set()).add(key)
    while time.monotonic()-start<12*3600:
        paths=sorted(p for p in (OUT/'shards').glob('*.json') if p.name not in seen_shards)
        rows=[]
        for path in paths[:100]:
            items=json.loads(path.read_text())
            if not all(r.get('preparation_version')==2 and 'student_input_sha256' in r for r in items):continue
            items=[adapt(r) for r in items]
            for r in items:connection.execute('INSERT OR REPLACE INTO row_inputs VALUES (?,?,?,?)',
                (r['id'],r['input_sha256'],r['student_input_sha256'],r['input_profile']))
            rows.extend(items);seen_shards.add(path.name)
        if not rows:
            prep=json.loads((OUT/'prepare_state.json').read_text())
            if prep['status']=='complete' and len(seen_shards)==24713:break
            dump('inference_state.json',dict(status='waiting_preparation',pid=os.getpid(),shards=len(seen_shards),rows_seen=rows_seen,
                qwen_unique_new=unique_qwen,student_inputs_new=unique_students,seconds=time.monotonic()-start))
            time.sleep(10);continue
        rows_seen+=len(rows)
        # Only identical actual model inputs may share an inference result.
        pending={r['input_sha256']:r for r in rows if r['input_sha256'] not in done.get('qlora',set())}
        missing=[r for r in pending.values() if r['input_missing']]
        for r in missing:store('qlora',r['input_sha256'],None,'input_missing',0)
        pending=sorted([r for r in pending.values() if not r['input_missing']],key=lambda r:len(r['prompt']))
        off=0
        while off<len(pending):
            rr=pending[off:off+batch];ts=time.monotonic()
            try:
                pp=qwen_batch(bundle,rr)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch>1:batch=max(1,batch//2);emit('oom_backoff',batch=batch);continue
                for r in rr:store('qlora',r['input_sha256'],None,'oom_at_batch1',time.monotonic()-ts);failed+=1
                off+=len(rr);continue
            elapsed=time.monotonic()-ts
            for r,p in zip(rr,pp):store('qlora',r['input_sha256'],p,'success',elapsed/len(rr))
            unique_qwen+=len(rr);off+=len(rr);connection.commit()
        sp={r['student_input_sha256']:r for r in rows}
        for name,model in students.items():
            rr=[r for key,r in sp.items() if key not in done.get(name,set())]
            for r in rr:
                if r['input_missing']:store(name,r['student_input_sha256'],None,'input_missing',0)
            rr=[r for r in rr if not r['input_missing']]
            for off in range(0,len(rr),64):
                sample=rr[off:off+64];ts=time.monotonic();pp=student_batch(model,sample);elapsed=time.monotonic()-ts
                for r,p in zip(sample,pp):store(name,r['student_input_sha256'],p,'success',elapsed/len(sample))
            connection.commit()
        unique_students+=len(sp)
        dump('inference_state.json',dict(status='running',pid=os.getpid(),shards=len(seen_shards),rows_seen=rows_seen,
            qwen_unique_new=unique_qwen,student_inputs_new=unique_students,failed_batches=failed,batch=batch,
            seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            route_cache_counts={n:len(v) for n,v in done.items()}))
        emit('inference',rows_seen=rows_seen,shards=len(seen_shards),qwen_unique_new=unique_qwen,seconds=time.monotonic()-start)
    else:raise RuntimeError('12-hour bounded inference budget reached; resume cache retained')
    connection.commit();connection.close()
    dump('inference_state.json',dict(status='complete',pid=os.getpid(),rows_seen=rows_seen,shards=len(seen_shards),
        qwen_unique_new=unique_qwen,seconds=time.monotonic()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        route_cache_counts={n:len(v) for n,v in done.items()}))
    emit('inference_complete',rows=rows_seen,seconds=time.monotonic()-start)


if __name__=='__main__':main()
