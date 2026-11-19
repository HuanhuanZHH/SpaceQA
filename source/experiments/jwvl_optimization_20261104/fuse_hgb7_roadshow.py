"""Apply the validation-frozen HGB7 fusion to every roadshow record."""
from pathlib import Path
import collections, hashlib, json, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import ensemble as e

ROOT=e.ROOT;OUT=ROOT/'ppt_results_20261104/hgb7_roadshow'


def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.time()
    work=ROOT/'jwvl_optimization_20261104/ensemble_v1'
    policy=e.read(work/'frozen_policy.json');assert e.sha(work/'stack_models.pkl')==policy['stack_sha256']
    with (work/'stack_models.pkl').open('rb') as f:models=pickle.load(f)['histgb_leaf7']
    threshold=np.array(policy['thresholds']['histgb_leaf7'])
    sys.path.insert(0,str(ROOT/'spaceqa-agent/experiments/prior_optimization_20261102'))
    from fuse_roadshow_extension import load_cached
    roadshow=ROOT/'opc-agent/runs/roadshow-evaluation-20261101'
    prior=ROOT/'inference_prior_20261102';gbdt=prior/'gbdt'
    packages=[];provenance={};checks=collections.Counter()
    for folder in sorted((roadshow/'packages').iterdir()):
        if not (folder/'inventory.jsonl').exists():continue
        name=folder.name;records=list(e.rows(folder/'inventory.jsonl'))
        assert all(r.get('ground_truth_available') is False for r in records)
        prepared={r['id']:r for r in records}
        priorpath=prior/'roadshow_v2/packages'/name/'prior_observations.jsonl'
        priorrows={r['sample_id']:r for r in e.rows(priorpath)}
        currentpath=prior/'roadshow_final_v1/packages'/name/'selected_candidates.jsonl'
        current={r['sample_id']:r for r in e.rows(currentpath)}
        assert set(prepared)==set(priorrows)==set(current)
        def verify_input(r):
            label=Path(r['label_path']);assert e.sha(label)==priorrows[r['id']]['current_label_sha256']
            if not r['input_missing']:
                assert e.sha(r['data_path'])==r['data_sha256']
                payload=[e.sha(r['image']),r['prompt'],r['features']]
                digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert digest==r['input_sha256']
            return not r['input_missing']
        with ThreadPoolExecutor(max_workers=8) as pool:checks['verified_present_raw_inputs']+=sum(pool.map(verify_input,records))
        checks['verified_current_labels']+=len(records)
        probs={};sources={};cached_seconds=0.
        for route in e.ROUTES:
            if route=='gbdt_prior_v1':
                pred,mark=load_cached(gbdt/'roadshow_predictions'/(name+'.jsonl'),gbdt/'roadshow_predictions'/(name+'.complete.json'),records,inventory_hash=e.sha(folder/'inventory.jsonl'),training_hash=e.sha(gbdt/'training.json'))
                cached_seconds+=mark.get('seconds',0.)
            else:
                pred,mark=load_cached(folder/'models'/(route+'.jsonl'),folder/'models'/(route+'.complete.json'),records)
                cached_seconds+=mark.get('model_load_and_inference_seconds',0.)
            probs[route]=pred;sources[route]=mark['prediction_sha256'];checks['verified_expert_prediction_files']+=1
        p=np.array([[probs[n][r['id']].get('probabilities') or [np.nan]*3 for n in e.ROUTES] for r in records],np.float32)
        rule=np.array([current[r['id']]['current_rule_bits'] for r in records],bool)
        x=e.features(records,p,rule)
        probability=np.stack([m.predict_proba(x)[:,1] for m in models],axis=1)
        fi=e.FAMILIES.index(records[0]['family']);present=np.array([not r['input_missing'] for r in records])
        candidate=probability>=threshold[fi];candidate[~present]=False
        combined=rule|candidate
        target=OUT/'packages'/name;target.mkdir(parents=True)
        with (target/'predictions.jsonl').open('w') as f:
            for i,r in enumerate(records):
                f.write(json.dumps(dict(id=r['id'],family=r['family'],input_sha256=r['input_sha256'],label_path=r['label_path'],data_path=r['data_path'],input_missing=r['input_missing'],model='HGB7-21expert-fusion',probabilities=probability[i].tolist() if present[i] else None,thresholds=threshold[fi].tolist(),model_candidate_bits=candidate[i].astype(int).tolist(),current_rule_bits=rule[i].astype(int).tolist(),combined_flag_bits=combined[i].astype(int).tolist(),verdict='undetermined',model_findings_score_eligible=False,ground_truth_available=False))+'\n')
        summary=dict(package=name,family=records[0]['family'],records=len(records),missing_inputs=int((~present).sum()),model_candidate_samples=int(candidate.any(1).sum()),G_model_candidates=int(candidate[:,0].sum()),C_model_candidates=int(candidate[:,1].sum()),D_model_candidates=int(candidate[:,2].sum()),combined_flag_samples=int(combined.any(1).sum()),historical_all21_expert_seconds=cached_seconds,ground_truth_available=False,precision=None,recall=None,F1=None)
        packages.append(summary);provenance[name]=dict(inventory_sha256=e.sha(folder/'inventory.jsonl'),prior_sha256=e.sha(priorpath),model_prediction_sha256=sources,output_sha256=e.sha(target/'predictions.jsonl'))
        e.write(OUT/'state.json',dict(status='running',completed_packages=len(packages),records=sum(r['records'] for r in packages)))
        print(json.dumps(summary),flush=True)
    assert len(packages)==12 and sum(p['records'] for p in packages)==24782
    assert checks['verified_expert_prediction_files']==252
    from compare_versions import write_csv
    write_csv(OUT/'package_summary.csv',packages)
    e.write(OUT/'complete.json',dict(status='complete',model='HGB7-21expert-fusion',packages=12,records=24782,missing_inputs=sum(p['missing_inputs'] for p in packages),seconds=time.time()-start,checks=dict(checks),source_policy_sha256=e.sha(work/'frozen_policy.json'),stack_sha256=e.sha(work/'stack_models.pkl'),thresholds=threshold.tolist(),new_expert_forward_passes=0,scope='Fresh HGB7 forward over all verified historical expert probabilities and current features; not fresh full-expert end-to-end timing',limitations=['Original twenty expert caches have whole-file and ID checks but no per-record input hash','Current raw inputs, labels and prepared image/prompt/features are verified separately','No roadshow truth or quality tiers are used for fitting, thresholds or ranking','Model candidates retain undetermined and do not become confirmed scoring evidence'],provenance=provenance))
    e.write(OUT/'state.json',dict(status='complete',completed_packages=12,records=24782))


if __name__=='__main__':main()
