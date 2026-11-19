"""Choose by validation only and test whether image content contributes."""
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
SRC=ROOT/'experiments/constraint_optimization_20261031'
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(SRC))
from learn_model_comparison import Student, metrics
from optimize_students import gates,gate_predict

OUT=ROOT/'runs/constraint-optimization-20261031'


def main():
    torch.set_num_threads(4)
    rows=[json.loads(s) for s in (OUT/'expanded_dataset.jsonl').open()]
    summary=json.loads((OUT/'student_optimization.json').read_text())
    choices=[]
    for name,r in summary.items():
        if 'selected_epoch' not in r:continue
        vm=r['variants']['0.01']['validation']
        raw=r['trace'][r['selected_epoch']-1]['validation_raw_macro_f1']
        seed=int(name.rsplit('seed',1)[-1])
        score=(vm['defect_f1'],-vm['clean_false_positive_rate'],raw,-r['parameters'],-r['weights_bytes'],-seed)
        choices.append((score,name))
    _,name=max(choices)
    selected=summary[name];th=selected['variants']['0.01']['thresholds']
    bundle=OUT/'selected_candidate';bundle.mkdir(exist_ok=True)
    shutil.copyfile(OUT/(name+'.pt'),bundle/'student.pt')
    config=dict(route=name,checkpoint='student.pt',classes=['NONE','GEOMETRY','COUNT','LABEL_OR_METADATA'],
                thresholds=th,families=['EPB_seg_sample','DMSP_SSUSI'],input_size=[256,128],parameters=selected['parameters'],
                selection='Validation-only F1/FPR/macroF1, then parameter and serialized size, then seed; no test metric used',
                status='candidate_review_only_requires_separate_localization',
                dataset_sha256=json.loads((OUT/'expanded_dataset_summary.json').read_text())['dataset_sha256'],
                P1=selected['variants']['0.01']['P1'],P2_P4=selected['variants']['0.01']['P2_P4'])
    (bundle/'config.json').write_text(json.dumps(config,indent=2))
    train=[r for r in rows if r['split']=='train']
    val=[r for r in rows if r['split']=='val' and not r.get('tier')]
    primary=[r for r in rows if r['split']=='test' and not r.get('tier')]
    heavy=[r for r in rows if r['split']=='test' and r.get('tier')]
    results={}
    for seed in (20261031,20261101,20261102):
        start=time.monotonic()
        model=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=4)
        model.fit([r['features'] for r in train],[r['target'] for r in train])
        def probs(rr):return model.predict_proba([r['features'] for r in rr])
        thresholds=gates(val,probs(val),.01)
        result=dict(thresholds=thresholds,seconds=time.monotonic()-start,
                    validation=metrics(val,gate_predict(val,probs(val),thresholds)))
        for tag,rr in [('P1',primary),('P2_P4',heavy)]:
            result[tag]=metrics(rr,gate_predict(rr,probs(rr),thresholds))
        results['features_only_seed'+str(seed)]=result
    # Counterfactual image permutation preserves all metadata. Fixed seeds and
    # family membership ensure this is a diagnostic, not a new fitted route.
    data=torch.load(bundle/'student.pt',map_location='cpu',weights_only=False)
    model=Student(data['widths'],data['hidden']).eval()
    model.load_state_dict(data['state'])
    saved=np.load(OUT/'student_probabilities.npz')
    saved_index={r:i for i,r in enumerate(saved['ids'])}
    for tag,rr in [('P1',primary),('P2_P4',heavy)]:
        ims=[]
        for r in rr:
            with Image.open(r['image']) as im:ims.append(np.array(im.resize((256,128)),dtype=np.float32).transpose(2,0,1)/255)
        x=torch.tensor(np.array(ims));f=torch.tensor([r['features'] for r in rr]);f=(f-data['feature_mean'])/data['feature_std']
        for mode in ('matched_image','permuted_image','zero_image'):
            xx=x.clone()
            if mode=='permuted_image':
                rng=np.random.RandomState(20261031)
                for family in {r['family'] for r in rr}:
                    ids=np.array([i for i,r in enumerate(rr) if r['family']==family]);perm=rng.permutation(ids);xx[ids]=x[perm]
            elif mode=='zero_image':xx.zero_()
            start=time.monotonic()
            with torch.inference_mode():
                pp=torch.cat([model(xx[i:i+32],f[i:i+32]).softmax(1) for i in range(0,len(rr),32)]).numpy()
            elapsed=time.monotonic()-start
            predictions=gate_predict(rr,pp,th)
            results[tag+'_'+mode]=dict(metrics=metrics(rr,predictions),cpu_inference_seconds=elapsed,
                 same_predictions_as_matched=None if mode=='matched_image' else sum(predictions[r['id']]==matched[r['id']] for r in rr))
            if mode=='matched_image':
                matched=predictions
                expected=np.array([saved[name][saved_index[r['id']]] for r in rr])
                results[tag+'_'+mode]['max_cpu_gpu_probability_error']=float(np.max(np.abs(pp-expected)))
                assert np.allclose(pp,expected,atol=2e-4,rtol=1e-3), 'Saved model does not reproduce training evaluation'
                assert metrics(rr,predictions)==selected['variants']['0.01'][tag]
    results['_selection']=config
    results['_caveats']=['Feature-only route is a learned model, not a deterministic structural-rule baseline.',
        'Image permutation is an intervention on this sample classifier; it does not establish scientific object recognition.',
        'Cached composite images used for this ablation, so timing excludes original raw decoding.']
    (OUT/'selection_and_ablation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps({k:v for k,v in results.items() if k!='_selection'}),flush=True)


if __name__=='__main__':main()
