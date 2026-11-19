"""Standard-library-only feature classifier for review candidates, not scoring."""
import argparse
import gzip
import json
import math
import struct
from pathlib import Path


def predict_proba(model, features):
    if len(features)!=9 or not all(math.isfinite(float(f)) for f in features):
        raise ValueError('Nine finite verified metadata/geometry features required')
    # sklearn trees compare float32 inputs against stored double thresholds.
    features=[struct.unpack('f',struct.pack('f',float(f)))[0] for f in features]
    total=[0.0]*4
    for tree in model['trees']:
        node=0
        while tree['left'][node]>=0:
            node=tree['left'][node] if features[tree['feature'][node]]<=tree['threshold'][node] else tree['right'][node]
        values=tree['value'][node];denominator=sum(values)
        for k in range(4):total[k]+=values[k]/denominator
    return [v/len(model['trees']) for v in total]


def candidate(model,row):
    if row.get('family') not in model['families']:
        return dict(id=row.get('id'),status='unsupported_family',needs_review=True)
    f=row['features'];p=predict_proba(model,f);k=max(range(4),key=lambda j:p[j])
    structural=3 if f[2] or f[8] else 1 if f[3] else 0
    proposed=structural or (k if k in (1,2) and p[k]>=model['thresholds'][k-1] else 0)
    return dict(id=row.get('id'),base_sample_id=row.get('base_sample_id'),proposed_coarse_class=proposed,
                probabilities=p,status='candidate_only',official_defect_location=None,
                needs_review=proposed>0,may_enter_formal_score=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    model=json.loads(gzip.decompress(a.model.read_bytes())) if a.model.suffix=='.gz' else json.loads(a.model.read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with a.input.open() as source,a.output.open('x') as target:
        for line in source:
            row=json.loads(line);target.write(json.dumps(candidate(model,row))+'\n');count+=1
    print(json.dumps(dict(records=count,output=str(a.output),status='review_candidates_not_formal_findings')))


if __name__=='__main__':main()
