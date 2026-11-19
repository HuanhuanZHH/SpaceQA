"""Rule + student gate, with all thresholds chosen on validation only."""
import json
import numpy as np
import torch
from PIL import Image
from learn_model_comparison import OUT,load,Student,metrics,write


def main():
    torch.set_num_threads(4)
    _,val,test=load();extra=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
    rows=val+test+extra
    images=[]
    for row in rows:
        with Image.open(row['image']) as im:images.append(np.array(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
    x=torch.tensor(np.array(images),device='cuda:0');features=torch.tensor([r['features'] for r in rows],device='cuda:0')
    summary={}
    for path in sorted(OUT.glob('student_*_seed*.pt')):
        if 'before_recovery' in path.stem:continue
        data=torch.load(path,map_location='cpu',weights_only=False)
        model=Student(data['widths'],data['hidden']).to('cuda:0');model.load_state_dict(data['state']);model.eval()
        ff=(features-data['feature_mean'].to('cuda:0'))/data['feature_std'].to('cuda:0')
        with torch.inference_mode():
            probs=torch.cat([model(x[i:i+32],ff[i:i+32]).softmax(1) for i in range(0,len(rows),32)]).cpu().numpy()
        def predict(subset,pp,tb,tc):
            result={}
            for r,p in zip(subset,pp):
                cls=int(p.argmax());threshold=tb if cls==1 else tc
                result[r['id']]=r['rule_prediction'] or (cls if cls in (1,2) and p[cls]>=threshold else 0)
            return result
        best=None
        for tb in [.4,.5,.6,.7,.8,.9,.95,.98,.99,.999,1.01]:
            for tc in [.4,.5,.6,.7,.8,.9,.95,.98,.99,.999,1.01]:
                vm=metrics(val,predict(val,probs[:len(val)],tb,tc))
                if vm['clean_false_positive_rate']>.05:continue
                score=(vm['defect_f1'],tb+tc)
                if best is None or score>best[0]:best=(score,tb,tc,vm)
        _,tb,tc,vm=best
        primary=predict(test,probs[len(val):len(val)+len(test)],tb,tc)
        extended=predict(extra,probs[len(val)+len(test):],tb,tc)
        summary[path.stem]=dict(thresholds=[tb,tc],validation_metrics=vm,
                                metrics=metrics(test,primary),extended_metrics=metrics(extra,extended))
    write(OUT/'student_hybrid_calibration.json',summary);print(json.dumps(summary))

if __name__=='__main__':main()
