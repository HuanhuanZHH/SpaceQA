"""Preserve input ablation, add independently verified raw dimensions before SFT."""
import json
import shutil
from prepare_model_comparison import ROOT,OUT,PROMPT,features,annotation_text

paths={}
for directory in ['epb_p0_p1_audit_20261027','dmsp_p0_p1_audit_20261029']:
    for r in map(json.loads,(ROOT/'phase2_visual_audit_xzj'/directory/'sample_comparisons.jsonl').open()):
        paths[r['base_sample_id']]=r['paths']['P1']['label']
archive=OUT/'pre_dimension_ablation';archive.mkdir(exist_ok=False)
shutil.copyfile(OUT/'dataset.jsonl',archive/'dataset.jsonl')
for pattern in ['teacher_*.json*','student_*.json','student_*.pt','comparison_summary.json','accuracy_comparison.csv','extended_accuracy.json','extended_test*.json*']:
    for p in OUT.glob(pattern):p.rename(archive/p.name)
rows=[]
for r in map(json.loads,(OUT/'dataset.jsonl').open()):
    if r['source']=='public_P0_assisted_real_P1':
        ann=json.load(open(paths[r['base_sample_id']]));r['features']=features(ann,r['family'])
        r['prompt']=PROMPT+annotation_text(ann,r['family'])
    else:
        r['prompt']+=' Decoded original PNG dimensions='+str([390,350] if r['family']=='DMSP_SSUSI' else [1024,1024])+'.'
    f=r['features'];r['rule_prediction']=3 if f[2] or f[8] else 1 if f[3] else 0
    rows.append(r)
with (OUT/'dataset.jsonl').open('w') as f:
    for r in rows:f.write(json.dumps(r)+'\n')
print('corrected raw-dimension inputs; all splits unchanged',len(rows))
