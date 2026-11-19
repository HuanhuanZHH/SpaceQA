"""Frozen-expert ablations + all-30-package current-input multimodal pilot.

No expert or fusion training. Reference G/C/D stays outside prompts. This pilot
does not establish nine-class localization quality or official critical recall.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os,pickle,sys,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
import ensemble as e
from spaceqa_agent import refinement as fr


def load_validation():
    root=e.ROOT;base=root/'opc-agent/runs/all-family-training-20261031'
    sys.path.insert(0,str(root/'spaceqa-agent/experiments/prior_optimization_20261102'))
    import compare_versions as cv
    from compare_gbdt_extension import prior_from_file
    work=root/'jwvl_optimization_20261104/ensemble_v1';policy=e.read(work/'frozen_policy.json')
    assert e.sha(work/'stack_models.pkl')==policy['stack_sha256']
    hashes=e.read(base/'training_barrier.json')['split_hashes']
    assert e.sha(base/'val.jsonl')==hashes['val']
    originals=list(e.rows(base/'val.jsonl'));records=cv.load_split(base/'val.jsonl','val',hashes['val'])
    byid={r['id']:r for r in originals}
    for r in records:r['features']=byid[r['id']]['features']
    loc,_=cv.load_old_rule_locators(root/'opc-agent/runs/full-matrix-20261031/blind-rules')
    rule=np.array(cv.retained_prior(prior_from_file(records,root/'inference_prior_20261102/comparison_v1/prior_outputs/val.jsonl'),cv.old_bits(records,loc)),bool)
    probabilities={}
    for route in e.ROUTES:
        probabilities[route]=cv.load_probabilities(root/'inference_prior_20261102/gbdt' if route=='gbdt_prior_v1' else base,route,records,policy['provenance'][route])
    p=np.array([[probabilities[n][i] if probabilities[n][i] is not None else [np.nan]*3 for n in e.ROUTES] for i in range(len(records))],np.float32)
    with (work/'stack_models.pkl').open('rb') as f:models=pickle.load(f)['histgb_leaf7']
    return records,byid,p,rule,models,policy


def predict(records,p,rule,models,policy):
    x=e.features(records,p,rule);q=np.stack([m.predict_proba(x)[:,1] for m in models],axis=1)
    fi=np.array([e.FAMILIES.index(r['family']) for r in records]);th=np.array(policy['thresholds']['histgb_leaf7'])
    flags=q>=th[fi]
    for i,r in enumerate(records):
        if r['input_missing']:flags[i]=False
    return q,flags|rule


def metrics(records,pred):
    y=np.array([r['target'] for r in records],bool);v=np.array([r['truth_valid'] for r in records],bool)
    tier=np.array([r['tier'] for r in records]);family=np.array([r['family'] for r in records])
    out={"all_P1_P4":e.metric(y[tier!='P0'],v[tier!='P0'],pred[tier!='P0'])}
    out['P0_sample_FPR']=float(pred[tier=='P0'].any(1).mean()) if any(tier=='P0') else None
    for f in e.FAMILIES:
        for t in ['P0','P1','P2','P3','P4']:
            m=(family==f)&(tier==t)
            if m.any():out[f+'|'+t]=dict(records=int(m.sum()),**e.metric(y[m],v[m],pred[m]))
        for b,g in enumerate('GCD'):
            m=(family==f)&(tier!='P0')
            if m.any():out[f+'|'+g]=e.metric(y[m,b],v[m,b],pred[m,b])
    return out


def objects(annotation):
    result=[]
    def walk(v,p=''):
        if isinstance(v,dict):
            for k,x in v.items():
                ptr=p+'/'+k.replace('~','~0').replace('/','~1')
                if k in {'points','bbox_polygon_pixel'} and isinstance(x,list) and len(x)>1 and all(isinstance(t,list) and len(t)==2 and all(isinstance(a,(int,float)) for a in t) for t in x):
                    result.append({'annotation_pointer':ptr,'bbox':[min(t[0] for t in x),min(t[1] for t in x),max(t[0] for t in x),max(t[1] for t in x)],'label':v.get('label',v.get('event_type'))})
                else:walk(x,ptr)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,p+'/'+str(i))
    walk(annotation);return result


def build_case(r,p,q,out):
    sid='CASE-'+hashlib.sha256(('fine-review-20261104|'+r['id']).encode()).hexdigest()[:20]
    annotation=e.read(r['label_path']);media=out/'cases/media';media.mkdir(parents=True,exist_ok=True)
    data=Path(r['data_path']);exists=data.is_file()
    case=dict(case_id=sid,family=r['family'],sample_path='current/data'+data.suffix,label_path='current/label.json',
        annotation=annotation,data_facts={'exists':exists},fusion_probabilities=q.tolist(),
        experts={name:[round(float(x),5) for x in np.nan_to_num(p[j],nan=.5)] for j,name in enumerate(e.ROUTES)},
        structural_features=r['features'],feature_semantics='First 16 structural flags/statistics; last 240 signed hash buckets, not individually interpretable.',
        objects=objects(annotation),media={},frame_id=r.get('data_sha256'),
        domain_definitions='Check only current supported evidence. Family does not establish event class. No independent physical boundary/reference is provided.')
    if exists:
        if r['family']!='ULF_sample':
            with Image.open(data) as im:case['image_size']=list(im.size)
        else:
            import csv,datetime
            with data.open() as f:
                reader=csv.DictReader(f);cols=reader.fieldnames;rows=list(reader)
            case['data_facts'].update(csv_rows=len(rows),columns=cols)
            ts=[float(row['t']) for row in rows] if rows and 't' in rows[0] else []
            if ts and all(np.isfinite(ts)) and min(ts)<max(ts):
                case['time_axis']={'id':'current_CSV_t_relative_seconds','start':min(ts),'end':max(ts)}
            case['data_summary']={'first_rows':rows[:2],'last_row':rows[-1:]}
        im=Image.open(r['image']).convert('RGB')
        # Existing view: left current observation, right current input annotation.
        for name,box in [('observation',(0,0,384,384)),('annotation',(384,0,768,384))]:
            im.crop(box).save(media/(sid+'-'+name+'.jpg'),quality=90)
            case['media'][name]='media/'+sid+'-'+name+'.jpg'
        case['image_instruction']='Two separate 384x384 views from existing preprocessing: first current observation, second input annotation overlay. These are not predictions. Location must use original image_size pixels, not 384px preview. ULF plot x axis must be interpreted with explicit time_axis.'
    e.write(out/'cases'/(sid+'.json'),case)
    return case


def pilot(records,original,p,q,pred,rule,out,count,variants,workers):
    # Original targets are never copied into case JSON or model request.
    chosen=[]
    for f in e.FAMILIES:
        for tier in ['P0','P1','P2','P3','P4']:
            pool=[i for i,r in enumerate(records) if r['family']==f and r['tier']==tier and not r['input_missing']]
            pool.sort(key=lambda i:hashlib.sha256(('pilot20261104:'+records[i]['id']).encode()).hexdigest())
            chosen.extend(pool[:count])
    cases={};mapping=[]
    for i in chosen:
        case=build_case(original[records[i]['id']],p[i],q[i],out);cases[case['case_id']]=case
        mapping.append(dict(case_id=case['case_id'],index=i,source_id=records[i]['id'],family=records[i]['family'],tier=records[i]['tier'],target=records[i]['target'],truth_valid=records[i]['truth_valid'],rule=rule[i].astype(int).tolist(),hgb=pred[i].astype(int).tolist()))
    e.write(out/'evaluation_mapping_PRIVATE.json',mapping)
    e.write(out/'pilot_protocol.json',dict(seed=20261104,per_package=count,packages=30,cases=len(chosen),selection='deterministic pseudorandom within meta_select and each family/tier; present inputs only',reference='Existing P0-derived G/C/D proxy; no independent nine-class location truth',modes=variants,no_training=True))
    key=(e.ROOT/'.qwen3vl-api-key').read_text().strip();results=[]
    def run(mode,cid):
        case=cases[cid];path=out/'responses'/mode/(cid+'.json')
        if path.exists():return mode,cid,e.read(path)
        context=dict(case)
        if mode.endswith('vision_label'):
            context.pop('experts',None);context.pop('fusion_probabilities',None);context.pop('structural_features',None)
        images=[out/'cases'/rel for rel in case['media'].values()] if 'text_label' not in mode else []
        if mode.startswith('api_'):
            provider=e.read(Path.home()/'.config/spaceqa/providers.json')['ustc_deepseek']
            # The host's default proxy cannot reach this university gateway.
            import urllib.parse
            host=urllib.parse.urlsplit(provider['base_url']).hostname
            os.environ['no_proxy']=os.environ.get('no_proxy','')+','+host
            os.environ['NO_PROXY']=os.environ['no_proxy']
            res=fr.review_case(context,base_url=provider['base_url'],api_key=provider['api_key'],model='qwen3.8-chat',images=images,timeout=90,retry_limit=1,max_tokens=2400)
        else:
            res=fr.review_case(context,base_url='http://127.0.0.1:8001/v1',api_key=key,model='Qwen3-VL-8B-Instruct',images=images,timeout=90,retry_limit=1,max_tokens=1500,native_schema=True)
        e.write(path,res);return mode,cid,res
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(run,mode,cid) for cid in cases for mode in variants]
        for future in as_completed(pending):
            mode,cid,res=future.result();results.append((mode,cid,res))
            print(json.dumps(dict(status='review_completed',mode=mode,completed=len(results),total=len(pending),case_id=cid,result_status=res['status'],seconds=round(res['seconds'],2))),flush=True)
    summaries={};selected_records=[records[i] for i in chosen]
    summaries['rules']=metrics(selected_records,rule[chosen]);summaries['hgb7']=metrics(selected_records,pred[chosen])
    for mode in variants:
        reviews={cid:res for mo,cid,res in results if mo==mode};flags=np.zeros((len(chosen),3),bool)
        for j,m in enumerate(mapping):
            for d in reviews[m['case_id']]['defects']:flags[j,'GCD'.index(d['group'])]=True
        summaries[mode]=dict(metrics=metrics(selected_records,flags),
            metrics_rule_preserved=metrics(selected_records,flags|rule[chosen]),
            available=sum(r['status']=='reviewed' for r in reviews.values()),total=len(chosen),
            seconds=sum(r['seconds'] for r in reviews.values()),
            tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in reviews.values()),
            defect_instances=sum(len(r['defects']) for r in reviews.values()),
            aggregate=fr.aggregate_reviews(list(reviews.values()),len(chosen)))
    e.write(out/'pilot_metrics.json',dict(status='complete',cases=len(chosen),packages=30,summaries=summaries,
        official_metrics=None,localization_metrics=None,critical_recall=None,
        limitations=['Small diagnostic pilot, not full dataset metrics','Unknown/failed reviews retained as negative predictions for conservative G/C/D recall; review coverage reported','P0-derived broad proxy, not independently reviewed nine-class location truth','Model-proposed severity not scientifically validated']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--per-package',type=int,default=2);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--modes',default='vision_label_experts,vision_label');ap.add_argument('--skip-ablations',action='store_true')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    records,original,p,rule,models,policy=load_validation();select=e.partition(records)=='select'
    indices=np.where(select)[0];records=[records[i] for i in indices];p=p[select];rule=rule[select]
    q,pred=predict(records,p,rule,models,policy)
    e.write(out/'baseline_metrics.json',dict(records=len(records),hgb7=metrics(records,pred),rules=metrics(records,rule)))
    if not args.skip_ablations and not (out/'ablation_metrics.json').exists():
        base=metrics(records,pred);ablations={}
        blocks={name:[j] for j,name in enumerate(e.ROUTES)}
        blocks.update(all_expert_probabilities=list(range(21)),students=list(range(1,10)),visual_backbones=list(range(10,20)))
        # Fixed-policy neutralization is reliance analysis, not retrained ablation.
        for name,js in blocks.items():
            masked=p.copy();masked[:,js,:]=.5
            _,pp=predict(records,masked,rule,models,policy)
            ablations[name]=metrics(records,pp)
            print(json.dumps(dict(status='ablation_done',name=name,F1=ablations[name]['all_P1_P4']['F1'])),flush=True)
        # Structural features and family may remain indirectly encoded by experts.
        for name in ['structural_features','rule_input_features']:
            rr=[dict(r) for r in records];rb=rule.copy()
            if name=='structural_features':
                for r in rr:r['features']=[0.]*256
            else:rb[:]=False
            qq,_=predict(rr,p,rb,models,policy)
            fi=np.array([e.FAMILIES.index(r['family']) for r in records]);pp=rule|(qq>=np.array(policy['thresholds']['histgb_leaf7'])[fi])
            ablations[name]=metrics(records,pp)
        e.write(out/'ablation_metrics.json',dict(status='complete',records=len(records),baseline=base,ablations=ablations,
            method='Frozen HGB7 policy; selected expert probabilities replaced by neutral .5, structural block by 0; no refitting or threshold changes. Rules preserved in final predictions.',
            limitation='Sensitivity/reliance to missing signals under distribution shift; not causal isolated contribution, not performance of a retrained smaller model; correlated experts may substitute.',official_metrics=None))
    pilot(records,original,p,q,pred,rule,out,args.per_package,args.modes.split(','),args.workers)
    e.write(out/'complete.json',dict(status='complete',records=len(records),cases=30*args.per_package,modes=args.modes,no_training=True))


if __name__=='__main__':main()
