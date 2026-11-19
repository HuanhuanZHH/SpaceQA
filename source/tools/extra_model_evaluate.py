"""Stress-set evaluation of the frozen teacher and saved small students."""
import concurrent.futures
import json
import numpy as np
import torch
from PIL import Image
from learn_model_comparison import OUT,Student,metrics,write
from batch_and_teacher import request


def main():
    torch.set_num_threads(4)
    rows=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
    if (OUT/'teacher_extended_predictions.jsonl').exists():
        results=[json.loads(s) for s in (OUT/'teacher_extended_predictions.jsonl').open()]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            results=list(pool.map(lambda row:request(row,8002),rows))
    with (OUT/'teacher_extended_predictions.jsonl').open('w') as f:
        for r in results:f.write(json.dumps(r)+'\n')
    summary={'qwen8b_original_api':metrics(rows,{r['id']:r['prediction'] for r in results if r['success']}),
             'blind_structure_rules':metrics(rows,{r['id']:r['rule_prediction'] for r in rows})}
    images=[]
    for row in rows:
        with Image.open(row['image']) as im:images.append(np.array(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
    x=torch.tensor(np.array(images),device='cuda:0');features=torch.tensor([r['features'] for r in rows],device='cuda:0')
    for path in sorted(OUT.glob('student_*_seed*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        model=Student(data['widths'],data['hidden']).to('cuda:0');model.load_state_dict(data['state']);model.eval()
        ff=(features-data['feature_mean'].to('cuda:0'))/data['feature_std'].to('cuda:0')
        with torch.inference_mode():
            pred=torch.cat([model(x[i:i+32],ff[i:i+32]).argmax(1) for i in range(0,len(rows),32)]).cpu().tolist()
        summary[path.stem]=metrics(rows,dict(zip([r['id'] for r in rows],pred)))
    write(OUT/'extended_accuracy.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
