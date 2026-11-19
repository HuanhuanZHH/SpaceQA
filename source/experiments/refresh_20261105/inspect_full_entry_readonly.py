"""Run standard-library-only, read-only remote inventory through stdin."""
from pathlib import Path
import json
import subprocess

SOURCE=r'''
from pathlib import Path
import collections,json
R=Path('/staff/tangjiarui/OPC')
out={}
training=R/'opc-agent/runs/all-family-training-20261031'
checkpoints=training/'checkpoints'
out['checkpoint_files']=[dict(name=str(p.relative_to(checkpoints)),bytes=p.stat().st_size) for p in sorted(checkpoints.rglob('*')) if p.is_file()]
out['qwen_base']=dict(path=str(R/'models/Qwen3-VL-8B-Instruct'),exists=(R/'models/Qwen3-VL-8B-Instruct').is_dir(),bytes=sum(p.stat().st_size for p in (R/'models/Qwen3-VL-8B-Instruct').rglob('*') if p.is_file()))
road=R/'opc-agent/runs/roadshow-evaluation-20261101'
out['dispatch']={k:v for k,v in json.loads((road/'dispatch_current.json').read_text()).items() if k in {'stage','experts','expert_count'}}
pack=[];routes=collections.defaultdict(lambda:dict(seconds=0.,inferred=0,packages=0))
for folder in sorted((road/'packages').iterdir()):
    if not (folder/'inventory.jsonl').is_file():continue
    items=[]
    for marker in sorted((folder/'models').glob('*.complete.json')):
        d=json.loads(marker.read_text());route=d['route'];sec=d['model_load_and_inference_seconds'];n=d['inferred']
        items.append(dict(route=route,seconds=sec,inferred=n,hardware=d.get('hardware')))
        routes[route]['seconds']+=sec;routes[route]['inferred']+=n;routes[route]['packages']+=1
    g=R/'inference_prior_20261102/gbdt/roadshow_predictions'/(folder.name+'.complete.json')
    if g.is_file():
        d=json.loads(g.read_text());items.append(dict(route='gbdt_prior_v1',seconds=d['seconds'],inferred=d['inferred']))
        routes['gbdt_prior_v1']['seconds']+=d['seconds'];routes['gbdt_prior_v1']['inferred']+=d['inferred'];routes['gbdt_prior_v1']['packages']+=1
    prep=json.loads((folder/'preparation.json').read_text())
    pack.append(dict(package=folder.name,records=prep['records'],preparationSeconds=prep['seconds'],sumHistorical21Seconds=sum(i['seconds'] for i in items),
                     experts=len(items),expertTimes=items))
out['historicalRuntime']=dict(packages=pack,routes=dict(routes),sumSeconds=sum(r['sumHistorical21Seconds'] for r in pack))
out['latest_fusion_files']=[dict(path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else None) for p in [R/'jwvl_optimization_20261104/ensemble_v1/stack_models.pkl',R/'jwvl_optimization_20261104/ensemble_v1/frozen_policy.json',R/'eight_class_20261104/fusion_v3_missing4/models.pkl',R/'eight_class_20261104/fusion_v3_missing4/frozen_policy.json',R/'eight_class_20261104/localizer_v2/no_presence_loss.pt']]
out['complete_release_entries']=[str(p) for root in ['complete_delivery_20261103/image_context','complete_delivery_20261103/release052','complete_delivery_20261103/release051'] for p in (R/root).rglob('*') if p.is_file() and p.suffix in {'.py','.sh','.json'} and any(x in p.name for x in ['runtime','manifest','entry','Docker','scan','package','run','complete','launch'])][:100]
src=R/'opc-agent/tools/prepare_model_comparison.py'
text=src.read_text();lines=text.splitlines()
out['ulf_csv_functions']=[]
for index,line in enumerate(lines):
    if line.startswith('def ') and any(v in line for v in ['csv','raw_view','feature_view']):
        end=next((j for j in range(index+1,len(lines)) if lines[j].startswith('def ')),len(lines))
        out['ulf_csv_functions'].append(dict(line=index+1,code='\n'.join(lines[index:end])))
out['source_paths']={n:str(R/n) for n in ['opc-agent/experiments/roadshow_evaluation_20261101/roadshow.py','spaceqa-agent/experiments/eight_class_20261104/roadshow.py','research_optimization_20261103/portable_experts.py']}
print(json.dumps(out,ensure_ascii=False))
'''
result=subprocess.run(['ssh','icarus0','/staff/tangjiarui/OPC/.venv-image-models/bin/python','-B','-'],input=SOURCE,text=True,encoding='utf-8',capture_output=True,check=True)
data=json.loads(result.stdout)
out=Path('D:/OPC/outputs/chat_insights_refresh_20261105');out.mkdir(parents=True,exist_ok=True)
(out/'full_entry_readonly_inventory.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:data[k] for k in ['dispatch','historicalRuntime','latest_fusion_files','ulf_csv_functions']},ensure_ascii=False))
