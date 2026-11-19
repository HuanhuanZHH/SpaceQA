import faulthandler,json,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,BitsAndBytesConfig,Qwen3VLForConditionalGeneration
faulthandler.dump_traceback_later(20,repeat=True)
torch.set_num_threads(4)
root=Path('/staff/tangjiarui/OPC');out=root/'opc-agent/runs/model-comparison-20261031-v2'
t=time.monotonic()
p=AutoProcessor.from_pretrained(root/'models/Qwen3-VL-8B-Instruct',min_pixels=28*28*64,max_pixels=28*28*256)
m=Qwen3VLForConditionalGeneration.from_pretrained(root/'models/Qwen3-VL-8B-Instruct',quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
print('loaded',time.monotonic()-t,flush=True)
rows=[json.loads(s) for s in (out/'dataset.jsonl').open() if json.loads(s)['split']=='test'][:8]
images=[];prompts=[]
for r in rows:
    with Image.open(r['image']) as im:image=im.copy()
    images.append(image)
    prompts.append(p.apply_chat_template([{'role':'user','content':[{'type':'text','text':r['prompt']},{'type':'image','image':image}]}],tokenize=False,add_generation_prompt=True))
p.tokenizer.padding_side='left'
inputs=p(text=prompts,images=images,padding=True,return_tensors='pt').to('cuda:0')
print('processed',time.monotonic()-t,{k:list(v.shape) for k,v in inputs.items()},inputs['image_grid_thw'].tolist(),flush=True)
with torch.inference_mode():
    y=m(**inputs,use_cache=False,logits_to_keep=1).logits
torch.cuda.synchronize()
print('forward finished',time.monotonic()-t,list(y.shape),flush=True)
faulthandler.cancel_dump_traceback_later()
