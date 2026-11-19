"""Isolated graph/gradient smoke fixtures; never become formal checkpoints/results."""
import collections
import json
import numpy as np
from common import *
from prepare import process

def main():
    assert OUT.name=='all-family-smoke-20261031','Smoke fixtures must be isolated'
    for name in ['rows','masks','images','training','checkpoints','smoke']: (OUT/name).mkdir(parents=True,exist_ok=True)
    rows=lines(SOURCE/'dataset.jsonl');groups=collections.defaultdict(list)
    for r in rows:groups[r['base_sample_id']].append(r)
    result=[]
    for family in FAMILIES:
        choices=[g for g in groups.values() if g[0]['family']==family and all(not r['input_missing'] for r in g)]
        choices=sorted(choices,key=lambda g:digest(g[0]['base_sample_id']))
        for split,g in zip(['train','val'],choices[:2]):
            p0=next(r for r in g if r['tier']=='P0')
            for r in g:result.append(process((r,p0,split,digest(r['base_sample_id']))))
    for split in ['train','val']:save_lines(OUT/(split+'.jsonl'),[r for r in result if r['split']==split])
    f=np.array([r['features'] for r in result if r['split']=='train'])
    write(OUT/'preparation.json',dict(smoke_fixture=True,mean=f.mean(0).tolist(),std=np.maximum(f.std(0),.01).tolist(),
        split_hashes={s:sha(OUT/(s+'.jsonl')) for s in ['train','val']}))
    emit('smoke_inputs',records=len(result),coverage=assert_coverage(result))

if __name__=='__main__':main()
