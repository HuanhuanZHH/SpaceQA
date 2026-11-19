"""Bounded 24-case roadshow refinement experiment, not exhaustive evaluation."""
from pathlib import Path
import concurrent.futures
import json
import os
import time
import argparse
from spaceqa_agent.review_inputs import prepare_case
from spaceqa_agent import refinement as r
from spaceqa_agent.quality_ledger import dump,digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory-root',required=True,type=Path)
    ap.add_argument('--hgb-root',required=True,type=Path)
    ap.add_argument('--gbdt-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True)
    cfg=json.loads((Path.home()/'.config/spaceqa/providers.json').read_text())['ustc_deepseek']
    # The text-only DeepSeek route was rejected on image requests. This image
    # model was capability-tested previously using this same authorized gateway.
    os.environ['no_proxy']=os.environ.get('no_proxy','')+',api.llm.ustc.edu.cn'
    os.environ['NO_PROXY']=os.environ['no_proxy']
    selected=[]
    for p in sorted(a.inventory_root.iterdir()):
        if not (p/'inventory.jsonl').is_file():continue
        rows={x['id']:x for x in (json.loads(line) for line in (p/'inventory.jsonl').open())}
        signals=[json.loads(line) for line in (a.hgb_root/p.name/'predictions.jsonl').open()]
        candidates=[s for s in signals if Path(rows[s['id']]['data_path']).is_file()]
        # One strong spatial/count signal and one fixed hash sample per package.
        strong=sorted(candidates,key=lambda s:(-max(s['probabilities'][:2]),digest(s['id'])))[0]
        control=sorted([s for s in candidates if s['id']!=strong['id']],key=lambda s:digest(s['id']))[0]
        for role,s in [('strong_GC',strong),('hash_control',control)]:
            row=rows[s['id']]
            selected.append(dict(row=row,signal=s,role=role,experts=[]))
        chosen={x['row']['id']:x for x in selected if x['row']['package']==p.name}
        for prediction_file in sorted((p/'models').glob('*.jsonl')):
            for line in prediction_file.open():
                expert=json.loads(line)
                if expert.get('id') in chosen:
                    chosen[expert['id']]['experts'].append(dict(name=prediction_file.stem,probabilities=expert['probabilities']))
        prior_file=a.gbdt_root/(p.name+'.jsonl')
        if not prior_file.is_file():raise ValueError('Missing frozen GBDT prediction artifact: '+str(prior_file))
        for line in prior_file.open():
            expert=json.loads(line);sid=expert.get('id',expert.get('sample_id'))
            if sid in chosen:chosen[sid]['experts'].append(dict(name='gbdt_prior_v1',probabilities=expert['probabilities']))
        if any(len(item['experts'])!=21 for item in chosen.values()):raise ValueError('All 21 expert probabilities are required')
    dump(a.output/'protocol.json',dict(samples=len(selected),packages=12,selection='one high G/C plus one deterministic hash sample per package',
        version='fine-review-eight-1.0',independentTruth=False,officialMetrics=None,model='qwen3.8-chat',
        concurrency=6,max_tokens=2200,timeout=90,retry_limit=1,training=False))
    (a.output/'responses').mkdir()
    def run(item):
        row=item['row'];s=item['signal'];slot=digest(row['id'])[:18]
        prepared=prepare_case(data_path=Path(row['data_path']),label_path=Path(row['label_path']),family=row['family'],
            output=a.output/'cases'/slot,signals=dict(fusion_probabilities=s['probabilities'],
                experts=item['experts'],
                structural_features=row.get('features'),
                feature_semantics='First 16 structural flags; remaining signed hash buckets have no one-to-one physical meaning.'))
        case=json.loads(Path(prepared['case']).read_text())
        result=r.review_case(case,base_url=cfg['base_url'],api_key=cfg['api_key'],model='qwen3.8-chat',
            images=[Path(x) for x in prepared['images']],timeout=90,retry_limit=1,max_tokens=2200,eight_classes=True)
        result['source']=dict(sampleId=row['id'],package=row['package'],family=row['family'],role=item['role'],casePath=prepared['case'])
        dump(a.output/'responses'/(slot+'.json'),result)
        print(json.dumps(dict(sample=row['id'],status=result['status'],defects=result.get('combined_defect_count'),seconds=result['seconds'])),flush=True)
        return result
    started=time.monotonic();results=[];failed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(run,item):item for item in selected}
        for f in concurrent.futures.as_completed(futures):
            try:results.append(f.result())
            except Exception as exc:failed.append(dict(sample=futures[f]['row']['id'],error=type(exc).__name__+': '+str(exc)[:300]))
    dump(a.output/'receipt.json',dict(status='completed',seconds=time.monotonic()-started,requested=len(selected),
        received=len(results),validated=sum(x['status']=='reviewed' for x in results),failed=failed,
        fineCandidates=sum(sum(d['status']!='confirmed' for d in x.get('combined_defects',[])) for x in results),
        officialMetrics=None,scientificAccuracyImprovementEstablished=False))


if __name__=='__main__':main()
