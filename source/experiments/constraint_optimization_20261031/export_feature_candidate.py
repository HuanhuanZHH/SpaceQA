"""Export a frozen feature learner; compare portable vs sklearn probabilities."""
import json
import gzip
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

SRC=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from portable_candidates import predict_proba,candidate
from optimize_students import gates,gate_predict
ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
sys.path.insert(0,str(ROOT/'tools'))
from learn_model_comparison import metrics
OUT=ROOT/'runs/constraint-optimization-20261031'


def main():
    rows=[json.loads(l) for l in (OUT/'expanded_dataset.jsonl').open()]
    tr=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val' and not r.get('tier')]
    model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,class_weight='balanced',random_state=20261031,n_jobs=4)
    model.fit([r['features'] for r in tr],[r['target'] for r in tr])
    th=gates(val,model.predict_proba([r['features'] for r in val]),.01)
    exported=dict(classes=[0,1,2,3],thresholds=th,families=['EPB_seg_sample','DMSP_SSUSI'],
                  input_features=9,scope='Candidate review only; no semantic object localization',trees=[])
    for est in model.estimators_:
        t=est.tree_
        exported['trees'].append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),
            feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,:].tolist()))
    bundle=OUT/'selected_candidate';dest=bundle/'feature_forest.json'
    dest.write_text(json.dumps(exported,separators=(',',':')))
    compact=dest.with_suffix('.json.gz');compact.write_bytes(gzip.compress(dest.read_bytes(),mtime=0))
    restored=json.loads(dest.read_text())
    start=time.monotonic();actual=np.array([predict_proba(restored,r['features']) for r in rows]);elapsed=time.monotonic()-start
    expected=model.predict_proba([r['features'] for r in rows])
    error=float(np.max(np.abs(actual-expected)));assert np.allclose(actual,expected,atol=1e-12)
    report=dict(model_bytes=dest.stat().st_size,compressed_model_bytes=compact.stat().st_size,trees=160,learned_image_features=False,thresholds=th,
                portable_probability_max_error=error,portable_eval_records=len(rows),portable_cpu_seconds=elapsed,
                selection='Same P1 validation gated F1/FPR as CNN; standard library deployment avoids the torch dependency. Seed fixed 20261031.',
                candidate_status='Class-only learned review candidate, no changes to formal judgments')
    for split in ('P1','P2_P4'):
        rr=[r for r in rows if r['split']=='test' and (not r.get('tier') if split=='P1' else r.get('tier'))]
        predictions={r['id']:candidate(restored,r)['proposed_coarse_class'] for r in rr}
        report[split]=metrics(rr,predictions)
    (OUT/'feature_export_validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
