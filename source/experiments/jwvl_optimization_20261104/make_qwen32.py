"""Build a separately frozen 32B trainer from the matched continuation runner."""
from pathlib import Path

p=Path(__file__).parent
s=(p/'train_grounded.py').read_text(encoding='utf-8')
loader='''
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

'''
replacements=[
 ('initial_adapter=str(BASE/\'checkpoints/qlora\'),adapter_init_sha256=sha(BASE/\'checkpoints/qlora/adapter_model.safetensors\')',"initial_base='Qwen2.5-VL-32B-Instruct generic base',base_index_sha256=sha(ROOT/'models/Qwen2.5-VL-32B-Instruct/model.safetensors.index.json'),lora_rank=16,lora_targets=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']"),
 ("original.load(out/'resume_adapter' if state else BASE/'checkpoints/qlora',True)","fresh32_load(out/'resume_adapter' if state else None,True)"),
 ("original.load(out/'best_adapter',False)","fresh32_load(out/'best_adapter',False)"),
 ("ROOT/'models/Qwen3-VL-8B-Instruct'","ROOT/'models/Qwen2.5-VL-32B-Instruct'"),
 ("'qwen8b_'+a.variant","'qwen32b_'+a.variant"),
 ("default=2);a=ap.parse_args()","default=1);a=ap.parse_args()")]
for old,new in replacements:
    assert old in s,old
    s=s.replace(old,new)
s=s.replace('def main():',loader+'\ndef main():',1)
s=s.replace('"""Matched continuation ablation: same six-family data/weights, domain prompt on/off."""','"""Fresh Qwen2.5-VL-32B domain QLoRA; all-family train before all30 evaluation."""',1)
(p/'train_qwen32.py').write_text(s,encoding='utf-8')
print('Created separate 32B trainer')
