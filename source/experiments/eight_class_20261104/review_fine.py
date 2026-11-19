"""24 current-input visual API reviews with eight-class and 21-expert hints."""
from __future__ import annotations
import collections,concurrent.futures,json,os,time
from pathlib import Path
from spaceqa_agent.review_inputs import prepare_case
from spaceqa_agent import refinement as r
from spaceqa_agent.quality_ledger import dump,digest

ROOT=Path('/staff/tangjiarui/OPC');WORK=ROOT/'eight_class_20261104'


def main():
    out=WORK/'visual_review_v4';out.mkdir(parents=True,exist_ok=False)
    (out/'responses').mkdir()
    config=json.loads((Path.home()/'.config/spaceqa/providers.json').read_text())['ustc_deepseek']
    os.environ['no_proxy']=os.environ.get('no_proxy','')+',api.llm.ustc.edu.cn';os.environ['NO_PROXY']=os.environ['no_proxy']
    selected=[];base=ROOT/'opc-agent/runs/roadshow-evaluation-20261101/packages'
    for package in sorted((WORK/'roadshow_v2/packages').iterdir()):
        records={x['id']:x for x in map(json.loads,(base/package.name/'inventory.jsonl').open())}
        fine={x['id']:x for x in map(json.loads,(package/'predictions.jsonl').open())}
        broad={x['id']:x for x in map(json.loads,(ROOT/'ppt_results_20261104/hgb7_roadshow/packages'/package.name/'predictions.jsonl').open())}
        available=[s for s in fine.values() if Path(records[s['id']]['data_path']).is_file()]
        def priority(s):
            probabilities=[s['probabilities'][i] for i,c in enumerate(s['classes']) if c in s['model_candidates']]
            return -max(probabilities,default=0.),digest(s['id'])
        strong=sorted(available,key=priority)[0]
        control=sorted([s for s in available if s['id']!=strong['id']],key=lambda s:digest(s['id']))[0]
        chosen={s['id']:dict(row=records[s['id']],fine=s,broad=broad[s['id']],experts=[],role=role) for role,s in [('fine_candidate',strong),('hash_control',control)]}
        for p in sorted((base/package.name/'models').glob('*.jsonl')):
            for line in p.open():
                v=json.loads(line)
                if v['id'] in chosen:chosen[v['id']]['experts'].append(dict(name=p.stem,probabilities=v['probabilities']))
        p=ROOT/'inference_prior_20261102/gbdt/roadshow_predictions'/(package.name+'.jsonl')
        for line in p.open():
            v=json.loads(line);sid=v.get('id',v.get('sample_id'))
            if sid in chosen:chosen[sid]['experts'].append(dict(name='gbdt_prior_v1',probabilities=v['probabilities']))
        assert all(len(item['experts'])==21 for item in chosen.values());selected.extend(chosen.values())
    dump(out/'protocol.json',dict(cases=24,packages=12,selection='one high fine-candidate probability plus one deterministic hash sample per package',
        provider_model='qwen3.8-chat',max_tokens=2200,concurrency=6,retry_limit=1,timeout_seconds=90,independent_reference=False,
        scope='protocol and executable evidence pilot, not a dataset accuracy estimate',fine_policy_sha256=digest(json.loads((WORK/'fusion_v2/frozen_policy.json').read_text()))))
    def run(item):
        row=item['row'];slot=digest(row['id'])[:18]
        prepared=prepare_case(data_path=Path(row['data_path']),label_path=Path(row['label_path']),family=row['family'],output=out/'cases'/slot,
            signals=dict(fusion_probabilities=item['broad']['probabilities'],fine_probabilities=dict(zip(item['fine']['classes'],item['fine']['probabilities'])),
                experts=item['experts'],structural_features=row['features'],
                domain_definitions='Class and probability are sample-level hints. ULF time head was trained on observation-window declarations, not independent event onsets. CME time head includes weak P0 timestamp matching. Do not claim scientifically verified event jitter from either hint.'))
        case=json.loads(Path(prepared['case']).read_text())
        result=r.review_case(case,base_url=config['base_url'],api_key=config['api_key'],model='qwen3.8-chat',
            images=[Path(p) for p in prepared['images']],timeout=90,retry_limit=1,max_tokens=2200,eight_classes=True)
        result['source']=dict(sampleId=row['id'],package=row['package'],family=row['family'],role=item['role'],casePath=prepared['case'])
        dump(out/'responses'/(slot+'.json'),result)
        print(json.dumps(dict(sampleId=row['id'],status=result['status'],seconds=result['seconds'],defects=result.get('combined_defect_count'))),flush=True)
        return result
    tick=time.monotonic();results=[];failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run,item):item for item in selected}
        for future,item in futures.items():
            try:results.append(future.result())
            except Exception as exc:failed.append(dict(sampleId=item['row']['id'],error=type(exc).__name__+': '+str(exc)[:240]))
    dump(out/'receipt.json',dict(status='complete' if len(results)==24 else 'incomplete',seconds=time.monotonic()-tick,requested=24,received=len(results),
        validated=sum(x['status']=='reviewed' for x in results),failed=failed,
        fine_candidates=sum(sum(d['status']!='confirmed' for d in x.get('combined_defects',[])) for x in results),
        official_metrics=None,accuracy_improvement_established=False))


if __name__=='__main__':main()
