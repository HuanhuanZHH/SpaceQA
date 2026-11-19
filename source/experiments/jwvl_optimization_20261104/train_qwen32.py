"""Fresh Qwen2.5-VL-32B domain QLoRA; all-family train before all30 evaluation."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
from grounding import grounded_prompt

ROOT=Path('/staff/tangjiarui/OPC');BASE=ROOT/'opc-agent/runs/all-family-training-20261031'
sys.path.insert(0,str(ROOT/'opc-agent/experiments/all_family_training_20261031'))
import qwen as original
from transformers import AutoProcessor

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));q.replace(p)
def rows(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)
def log(status,**kw):print(json.dumps(dict(status=status,**kw)),flush=True)
def coverage(rr):return dict(collections.Counter(r['family']+'|'+r['tier'] for r in rr))
def transform(rr,domain):
    # Preserve original 768x384 current-only panels. Increasing the vision budget
    # retains this detail; it does not create full-native 4096-pixel observations.
    result=[]
    for r in rr:
        s=dict(r)
        if domain:s['prompt']=grounded_prompt(r)
        result.append(s)
    return result


def fresh32_load(adapter=None,training=False):
    from transformers import Qwen2_5_VLForConditionalGeneration,BitsAndBytesConfig
    from peft import LoraConfig,get_peft_model,PeftModel,prepare_model_for_kbit_training
    modelpath=ROOT/'models/Qwen2.5-VL-32B-Instruct'
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(modelpath,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16,llm_int8_skip_modules=['visual']),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=training,gradient_checkpointing_kwargs={'use_reentrant':False})
    visual=getattr(model,'visual',None)
    if visual is None:visual=model.model.visual
    visual.to(dtype=torch.bfloat16)
    if adapter:model=PeftModel.from_pretrained(model,adapter,is_trainable=training)
    else:model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM',bias='none'))
    model.config.use_cache=False
    processor=AutoProcessor.from_pretrained(modelpath,min_pixels=28*28*64,max_pixels=28*28*1024)
    processor.tokenizer.padding_side='left'
    ids=[processor.tokenizer.encode(c,add_special_tokens=False) for c in 'ABCDEFGH'];assert all(len(i)==1 for i in ids)
    return processor,model,torch.tensor([i[0] for i in ids],device='cuda')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['control','domain'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--epochs',type=int,default=1);a=ap.parse_args()
    out=a.output.resolve();assert out.is_relative_to(ROOT/'jwvl_optimization_20261104');out.mkdir(parents=True,exist_ok=True)
    assert not (out/'evaluation_complete.json').exists(),'Already completed; do not overwrite'
    torch.set_num_threads(4);torch.manual_seed(20261104);np.random.seed(20261104);random.seed(20261104)
    info=json.loads((BASE/'preparation.json').read_text());hashes=info['split_hashes']
    assert sha(BASE/'train.jsonl')==hashes['train'] and sha(BASE/'val.jsonl')==hashes['val']
    train=transform([r for r in rows(BASE/'train.jsonl') if not r['input_missing'] and any(r['truth_valid'])],a.variant=='domain')
    val=transform([r for r in rows(BASE/'val.jsonl') if not r['input_missing'] and any(r['truth_valid'])],a.variant=='domain')
    assert len(coverage(train))==len(coverage(val))==30
    train_groups={r['training_group_id'] for r in train};val_groups={r['training_group_id'] for r in val};assert not train_groups&val_groups
    subset=original.validation_subset(val)
    config=dict(variant=a.variant,epochs=a.epochs,seed=20261104,learning_rate=3e-5,effective_batch=16,initial_microbatch=1,gradient_clip=1.,max_pixels=28*28*1024,image='existing 768x384 current-only two-panel view',initial_base='Qwen2.5-VL-32B-Instruct generic base',base_index_sha256=sha(ROOT/'models/Qwen2.5-VL-32B-Instruct/model.safetensors.index.json'),lora_rank=16,lora_targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],split_hashes=hashes,train_coverage=coverage(train),training_records=len(train),C_positive_weight=2.,P1_positive_weight=1.5,selection='fixed 40-record-per-cell validation subset at threshold .5',script_sha256=sha(__file__),grounding_sha256=sha(Path(__file__).with_name('grounding.py')),test_read=False)
    configpath=out/'config.json'
    if configpath.exists():assert json.loads(configpath.read_text())==config,'Resume config changed'
    else:write(configpath,config)
    resume=out/'resume.pt';state=torch.load(resume,map_location='cpu',weights_only=False) if resume.exists() else None
    bundle=fresh32_load(out/'resume_adapter' if state else None,True)
    bundle=(AutoProcessor.from_pretrained(ROOT/'models/Qwen2.5-VL-32B-Instruct',min_pixels=28*28*64,max_pixels=config['max_pixels']),bundle[1],bundle[2])
    bundle[0].tokenizer.padding_side='left';model=bundle[1]
    params=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01)
    if state:optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    effective=16;micro=state['micro'] if state else 1;updates=state['updates'] if state else 0
    trace=state['trace'] if state else [];best=state['best'] if state else -1.;seen=set(state['seen']) if state else set()
    counts=collections.Counter(r['family'] for r in train);fw={f:len(train)/(6*n) for f,n in counts.items()}
    start=time.time();start_updates=updates
    def save(epoch,offset):
        model.save_pretrained(out/'resume_adapter')
        p=out/'resume.pt.tmp';torch.save(dict(epoch=epoch,offset=offset,optimizer=optimizer.state_dict(),micro=micro,updates=updates,seen=sorted(seen),best=best,trace=trace,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),p);p.replace(resume)
    try:
      for epoch in range(state['epoch'] if state else 0,a.epochs):
        order=list(range(len(train)));random.Random(20261104+epoch).shuffle(order)
        offset=state['offset'] if state and state['epoch']==epoch else 0
        model.train();loss_sum=0.;windows=0
        while offset<len(order):
            window=order[offset:offset+effective];optimizer.zero_grad(set_to_none=True);lossvalue=0.
            try:
                for off in range(0,len(window),micro):
                    rr=[train[i] for i in window[off:off+micro]];logits=original.batch_logits(bundle,rr)
                    yy=torch.tensor([r['target'] for r in rr],device='cuda',dtype=torch.float32)
                    valid=torch.tensor([r['truth_valid'] for r in rr],device='cuda',dtype=torch.bool)
                    compatible=(((original.BITS.to('cuda')[None]-yy[:,None]).abs()<.5)|~valid[:,None]).all(2)
                    terms=-(logits.masked_fill(~compatible,-float('inf')).logsumexp(1)-logits.logsumexp(1))
                    weights=[fw[r['family']]*(2. if r['truth_valid'][1] and r['target'][1] else 1.)*(1.5 if r['tier']=='P1' and any(t and v for t,v in zip(r['target'],r['truth_valid'])) else 1.) for r in rr]
                    loss=(terms*torch.tensor(weights,device='cuda')).sum()/len(window)
                    assert torch.isfinite(loss);loss.backward();lossvalue+=float(loss.detach())
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
                if micro==1:raise
                micro=max(1,micro//2);continue
            torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            seen.update(train[i]['id'] for i in window);offset+=len(window);updates+=1;loss_sum+=lossvalue;windows+=1
            elapsed=time.time()-start
            status=dict(status='training',variant=a.variant,epoch=epoch+1,epochs=a.epochs,records_in_epoch=offset,train_records=len(train),updates=updates,loss=lossvalue,elapsed_seconds=elapsed,estimated_remaining_seconds=(a.epochs*len(train)/effective-updates)*elapsed/max(1,updates-start_updates),gpu_name=torch.cuda.get_device_name(),host=os.uname().nodename,microbatch=micro,effective_batch=effective)
            if updates<=3 or updates%10==0:write(out/'state.json',status);log(**{k:v for k,v in status.items() if k!='status'},status='training')
            if updates%100==0:save(epoch,offset)
        assert seen=={r['id'] for r in train}
        prob=original.predict(bundle,subset,batch=1);score=original.f1_by_family(subset,prob,np.full((6,3),.5))
        if score>best:best=score;model.save_pretrained(out/'best_adapter')
        trace.append(dict(epoch=epoch+1,loss=loss_sum/max(1,windows),validation_family_macro_f1_at_05=score,coverage=coverage(train)))
        save(epoch+1,0);state=None;log('epoch_complete',epoch=epoch+1,score=score)
      # All six-family training completes before any test is read.
      del model,bundle,optimizer;torch.cuda.empty_cache()
      bundle=fresh32_load(out/'best_adapter',False)
      bundle=(AutoProcessor.from_pretrained(ROOT/'models/Qwen2.5-VL-32B-Instruct',min_pixels=28*28*64,max_pixels=config['max_pixels']),bundle[1],bundle[2]);bundle[0].tokenizer.padding_side='left'
      val_prob=original.predict(bundle,val,batch=2);th=original.calibrate(val,val_prob)
      np.savez_compressed(out/'validation_predictions.npz',ids=np.array([r['id'] for r in val]),probabilities=val_prob)
      barrier=dict(status='pass',test_read=False,config_sha256=sha(configpath),training_families=original.FAMILIES,train_coverage=coverage(train),unique_records_updated=len(seen),thresholds=th,trace=trace,split_hashes=hashes,adapter_sha256=sha(out/'best_adapter/adapter_model.safetensors'))
      write(out/'training_barrier.json',barrier);barrierhash=sha(out/'training_barrier.json')
      assert sha(BASE/'test.jsonl')==hashes['test']
      testall=transform(list(rows(BASE/'test.jsonl')),a.variant=='domain');assert len(coverage(testall))==30
      assert not (train_groups|val_groups)&{r['training_group_id'] for r in testall}
      test=[r for r in testall if not r['input_missing']];prob=original.predict(bundle,test,batch=2)
      sys.path.insert(0,str(ROOT/'spaceqa-agent/experiments/prior_optimization_20261102'));import compare_versions as cv
      mapping={r['id']:p for r,p in zip(test,prob)};pred=[]
      with (out/'test_predictions.jsonl').open('w') as f:
        for r in testall:
            p=mapping.get(r['id']);bits=[False]*3 if p is None else (p>=np.array(th[original.FAMILIES.index(r['family'])])).tolist();pred.append(bits)
            f.write(json.dumps(dict(id=r['id'],input_sha256=r['input_sha256'],probabilities=p.tolist() if p is not None else None,status='success' if p is not None else 'input_missing'))+'\n')
      summary,packages,groups=cv.summarize(testall,pred,'qwen32b_'+a.variant)
      cv.write_csv(out/'all30_test.csv',packages);cv.write_csv(out/'groups_test.csv',groups)
      assert sha(out/'training_barrier.json')==barrierhash
      write(out/'evaluation_complete.json',dict(status='complete',summary=summary,packages=30,barrier_sha256=barrierhash,seconds=time.time()-start,metric_scope='P0-derived G/C/D technical proxy; no native localization output'))
      write(out/'state.json',dict(status='complete',variant=a.variant,summary=summary,seconds=time.time()-start));log('complete',summary=summary)
    except BaseException as ex:
      write(out/'state.json',dict(status='failed',variant=a.variant,reason=type(ex).__name__+': '+str(ex),updates=updates));raise

if __name__=='__main__':main()
