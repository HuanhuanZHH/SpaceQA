"""Extract soft targets from the actually fine-tuned, frozen 8B teacher."""
import json
import torch
from PIL import Image
from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
from peft import PeftModel,prepare_model_for_kbit_training
from learn_model_comparison import ROOT,OUT,load,optimize_patch_embed


def main():
    torch.set_num_threads(4)
    p=AutoProcessor.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',min_pixels=28*28*64,max_pixels=28*28*256)
    m=Qwen3VLForConditionalGeneration.from_pretrained(ROOT/'models/Qwen3-VL-8B-Instruct',
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    optimize_patch_embed(m)
    m=prepare_model_for_kbit_training(m,use_gradient_checkpointing=False);m.model.visual.to(dtype=torch.bfloat16)
    m=PeftModel.from_pretrained(m,OUT/'qlora_adapter');m.eval()
    ids=torch.tensor([p.tokenizer.encode(c,add_special_tokens=False)[0] for c in 'ABCD'],device='cuda:0')
    train,val,test=load();extra=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
    after=json.loads((OUT/'qlora_after.json').read_text());result=[]
    for r in val+test+extra:
        probs=after['probabilities'][r['id']]
        result.append(dict(id=r['id'],target=r['target'],prediction=max(range(4),key=lambda j:probs[j]),probabilities=probs,success=True))
    p.tokenizer.padding_side='left'
    with torch.inference_mode():
        for i in range(0,len(train),8):
            rows=train[i:i+8];images=[];prompts=[]
            for r in rows:
                with Image.open(r['image']) as im:image=im.copy()
                images.append(image)
                prompts.append(p.apply_chat_template([{'role':'user','content':[{'type':'text','text':r['prompt']},{'type':'image','image':image}]}],tokenize=False,add_generation_prompt=True))
            inputs=p(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda:0')
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1,ids].float()
            for r,probs in zip(rows,logits.softmax(1).cpu().tolist()):
                result.append(dict(id=r['id'],target=r['target'],prediction=max(range(4),key=lambda j:probs[j]),probabilities=probs,success=True))
            if i%80==0:print('finetuned teacher train predictions',i,flush=True)
    with (OUT/'teacher_finetuned_predictions.jsonl').open('w') as f:
        for r in result:f.write(json.dumps(r)+'\n')

if __name__=='__main__':main()
