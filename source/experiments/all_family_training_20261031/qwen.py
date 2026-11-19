"""New all-six-family QLoRA teacher. Train/validation only until global barrier."""
import argparse
import collections
import json
import os
import random
import time
import types

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from common import *
from train import read_fit_rows,calibrate,f1_by_family,seed

BITS=torch.tensor([[(i>>j)&1 for j in range(3)] for i in range(8)],dtype=torch.float32)


def load(adapter=None,training=False):
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
    processor=AutoProcessor.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',min_pixels=28*28*64,max_pixels=28*28*256)
    processor.tokenizer.padding_side='left'
    model=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    patch=[m for m in model.modules() if m.__class__.__name__=='Qwen3VLVisionPatchEmbed'];assert len(patch)==1
    def forward(self,hidden_states):return F.linear(hidden_states.to(self.proj.weight.dtype),self.proj.weight.flatten(1),self.proj.bias)
    patch[0].forward=types.MethodType(forward,patch[0])
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=training,gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.visual.to(dtype=torch.bfloat16)
    if adapter:model=PeftModel.from_pretrained(model,adapter,is_trainable=training)
    else:model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',bias='none'))
    model.config.use_cache=False
    ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEFGH'];assert all(len(v)==1 for v in ids)
    return processor,model,torch.tensor([i[0] for i in ids],device='cuda')


def batch_logits(bundle,rows):
    processor,model,ids=bundle;images=[];prompts=[]
    for r in rows:
        with Image.open(r['image']) as im:image=im.copy()
        images.append(image)
        prompts.append(processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':r['prompt']},{'type':'image','image':image}]}],tokenize=False,add_generation_prompt=True))
    inputs=processor(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda')
    if inputs['input_ids'].shape[1]>16384:raise RuntimeError('Untruncated current input exceeds declared context budget')
    with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,ids].float()
    return logits


def predict(bundle,rows,batch=16):
    bundle[1].eval();output=[];off=0
    with torch.inference_mode():
        while off<len(rows):
            try:p=batch_logits(bundle,rows[off:off+batch]).softmax(1)@BITS.to('cuda')
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch==1:raise
                batch=max(1,batch//2);continue
            assert torch.isfinite(p).all();output.append(p.cpu().numpy());off+=len(p)
    return np.concatenate(output)


def validation_subset(rows):
    # Fixed hash selection independent of outcomes. Every family/tier is represented.
    result=[]
    for f in FAMILIES:
        for t in TIERS:
            group=sorted([r for r in rows if r['family']==f and r['tier']==t],key=lambda r:digest(['val-v1',r['id']]))
            result.extend(group[:40])
    return result


def train(smoke=False):
    torch.set_num_threads(4);seed(SEEDS[0]);tr,fullval=read_fit_rows();va=validation_subset(fullval)
    if smoke:
        tr=[next(r for r in tr if r['family']==f and r['tier']==t) for f in FAMILIES for t in TIERS]
        va=[next(r for r in va if r['family']==f and r['tier']==t) for f in FAMILIES for t in TIERS]
    info=json.loads((OUT/'preparation.json').read_text());resume=OUT/'training/qlora_resume';statepath=resume/'state.pt'
    state=torch.load(statepath,map_location='cpu',weights_only=False) if statepath.exists() and not smoke else None
    if state:assert state['train_hash']==info['split_hashes']['train']
    bundle=load(resume/'adapter' if state else None,True);model=bundle[1]
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1.5e-4,weight_decay=.01)
    if state:optimizer.load_state_dict(state['optimizer'])
    epochs=1 if smoke else 2;batch=state['batch'] if state else 4;effective=16;start=time.monotonic()
    best=state['best'] if state else -1.;best_epoch=state['best_epoch'] if state else 0;trace=state['trace'] if state else []
    seen=set(state['seen']) if state else set();coverage=collections.Counter(state['coverage']) if state else collections.Counter()
    updates=state['updates'] if state else 0
    counts=collections.Counter(r['family'] for r in tr);weights={f:len(tr)/(6*counts[f]) for f in FAMILIES}
    output=OUT/'smoke/qlora' if smoke else OUT/'checkpoints/qlora';output.mkdir(parents=True,exist_ok=True)
    def save_resume(epoch,offset,epoch_seen,epoch_cov):
        resume.mkdir(exist_ok=True);model.save_pretrained(resume/'adapter')
        tmp=statepath.with_suffix('.tmp');torch.save(dict(optimizer=optimizer.state_dict(),epoch=epoch,offset=offset,
            epoch_seen=sorted(epoch_seen),epoch_coverage=dict(epoch_cov),seen=sorted(seen),coverage=dict(coverage),
            best=best,best_epoch=best_epoch,trace=trace,updates=updates,batch=batch,train_hash=info['split_hashes']['train'],
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);tmp.replace(statepath)
    if state:torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(state['epoch'] if state else 0,epochs):
        order=list(range(len(tr)));random.Random(SEEDS[0]+epoch).shuffle(order)
        offset=state['offset'] if state and state['epoch']==epoch else 0
        epoch_seen=set(state['epoch_seen']) if state and state['epoch']==epoch else set()
        epoch_cov=collections.Counter(state['epoch_coverage']) if state and state['epoch']==epoch else collections.Counter()
        model.train();losses=[]
        while offset<len(order):
            # A complete optimizer window is retried on OOM: no partial duplicated update.
            window=order[offset:offset+effective];optimizer.zero_grad(set_to_none=True);lossvalue=0.
            try:
                for off in range(0,len(window),batch):
                    rr=[tr[i] for i in window[off:off+batch]];logits=batch_logits(bundle,rr)
                    y=torch.tensor([r['target'] for r in rr],device='cuda',dtype=torch.float32)
                    valid=torch.tensor([r['truth_valid'] for r in rr],device='cuda',dtype=torch.bool)
                    compatible=(((BITS.to('cuda')[None]-y[:,None]).abs()<.5)|~valid[:,None]).all(2)
                    loss=-(logits.masked_fill(~compatible,-float('inf')).logsumexp(1)-logits.logsumexp(1))
                    weight=torch.tensor([weights[r['family']] for r in rr],device='cuda')
                    loss=(loss*weight).sum()/len(window)
                    if not torch.isfinite(loss):raise RuntimeError('Non-finite QLoRA loss')
                    loss.backward();lossvalue+=float(loss.detach())
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                if batch==1:raise
                batch=max(1,batch//2);emit('qlora_oom_backoff',microbatch=batch,effective_batch=effective);continue
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            for i in window:
                r=tr[i];seen.add(r['id']);epoch_seen.add(r['id']);key=r['family']+'|'+r['tier'];coverage[key]+=1;epoch_cov[key]+=1
            offset+=len(window);updates+=1;losses.append(lossvalue)
            if updates%20==0:
                emit('qlora_train',epoch=epoch+1,records=offset,total=len(tr),updates=updates,loss=lossvalue,microbatch=batch,seconds=time.monotonic()-start)
                write(OUT/'training/qlora.state.json',dict(status='training',epoch=epoch+1,records_in_epoch=offset,
                    train_records=len(tr),unique_records_updated=len(seen),actual_train_coverage=dict(coverage),updates=updates,pid=os.getpid()))
            if updates%250==0 and not smoke:save_resume(epoch,offset,epoch_seen,epoch_cov)
        assert epoch_seen=={r['id'] for r in tr} and len(epoch_cov)==30
        pp=predict(bundle,va);score=f1_by_family(va,pp,np.full((6,3),.5))
        if score>best:
            best=score;best_epoch=epoch+1;model.save_pretrained(output)
        trace.append(dict(epoch=epoch+1,loss=float(np.mean(losses)) if losses else None,
            validation_family_macro_f1_at_05=score,actual_train_coverage=dict(epoch_cov),unique_train_records=len(epoch_seen)))
        emit('qlora_epoch_complete',epoch=epoch+1,records=len(epoch_seen),families=6,validation_f1=score)
        if not smoke:save_resume(epoch+1,0,set(),collections.Counter())
        state=None
    # Reload selected adapter in the same engine, then validation only.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model,load_file(str(output/'adapter_model.safetensors')))
    pp=predict(bundle,va);thresholds=calibrate(va,pp)
    report=dict(status='smoke_passed' if smoke else 'complete',route='qlora',epochs=epochs,selected_epoch=best_epoch,
        training_records=len(tr),unique_records_updated=len(seen),training_ids_sha256=digest(sorted(seen)),
        actual_train_coverage=dict(coverage),training_families=FAMILIES,train_split_sha256=info['split_hashes']['train'],
        val_split_sha256=info['split_hashes']['val'],updates=updates,trace=trace,thresholds=thresholds,
        validation_macro_f1_at_05=best,seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in parameters),
        microbatch=batch,effective_batch=effective,base='Qwen3-VL-8B-Instruct original generic base; no previous OPC adapter',
        test_read=False,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if not smoke:
        report['checkpoint_hashes']={str(p.relative_to(OUT)):sha(p) for p in checkpoint_files('qlora')}
        write(OUT/'training/qlora.json',report)
        save_lines(OUT/'training/qlora.updated_ids.jsonl',[{'id':i} for i in sorted(seen)])
        np.savez_compressed(OUT/'training/qlora.validation.npz',ids=np.array([r['id'] for r in va]),probabilities=pp)
    else:write(OUT/'smoke/qlora.json',report)


def teacher():
    tr,_=read_fit_rows();report=json.loads((OUT/'training/qlora.json').read_text());assert report['status']=='complete'
    bundle=load(OUT/'checkpoints/qlora',False);start=time.monotonic();path=OUT/'teacher_train.jsonl'
    previous={r['id']:r for r in lines(path)} if path.exists() else {}
    needed=[r for r in tr if r['id'] not in previous]
    # Cache exact inputs only, retaining all train IDs as separate provenance records.
    cache={r['input_sha256']:r['probabilities'] for r in previous.values()}
    with path.open('a') as f:
        for off in range(0,len(needed),32):
            batch=needed[off:off+32];unique={r['input_sha256']:r for r in batch if r['input_sha256'] not in cache}
            if unique:
                rr=list(unique.values());pp=predict(bundle,rr,16)
                for r,p in zip(rr,pp):cache[r['input_sha256']]=p.tolist()
            for r in batch:
                item=dict(id=r['id'],split='train',input_sha256=r['input_sha256'],probabilities=cache[r['input_sha256']])
                f.write(json.dumps(item)+'\n');previous[r['id']]=item
            f.flush()
            if off%1024==0:
                emit('teacher_train',completed=len(previous),total=len(tr),seconds=time.monotonic()-start)
                write(OUT/'teacher_state.json',dict(status='generating_train_soft_labels',records=len(previous),total=len(tr),pid=os.getpid()))
    assert set(previous)=={r['id'] for r in tr}
    write(OUT/'teacher_train.json',dict(status='complete',records=len(previous),split='train',split_hash=sha(OUT/'train.jsonl'),
        teacher_hashes=report['checkpoint_hashes'],soft_targets_sha256=sha(path),coverage=assert_coverage(tr),seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','teacher']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.mode=='train':train(a.smoke)
    else:teacher()
