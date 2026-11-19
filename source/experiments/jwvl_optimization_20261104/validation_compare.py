"""Compare all candidates on the same held-out meta-selection validation groups."""
from pathlib import Path
import json, os, pickle, sys, time
import numpy as np
import ensemble as e

ROOT=e.ROOT
OUT=ROOT/'ppt_results_20261104/validation'


def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.time()
    sys.path.insert(0,str(ROOT/'spaceqa-agent/experiments/prior_optimization_20261102'))
    import compare_versions as cv
    from compare_gbdt_extension import prior_from_file,predictions_for
    from evaluate_roadshow_prior import selected_choices
    base=ROOT/'opc-agent/runs/all-family-training-20261031';prior=ROOT/'inference_prior_20261102'
    comp=prior/'comparison_v1';ext=prior/'gbdt_extension_v1';gbdt=prior/'gbdt'
    work=ROOT/'jwvl_optimization_20261104/ensemble_v1'
    policy=e.read(work/'frozen_policy.json')
    assert e.sha(work/'stack_models.pkl')==policy['stack_sha256']
    hashes=e.read(base/'training_barrier.json')['split_hashes']
    records=cv.load_split(base/'val.jsonl','val',hashes['val'])
    fmap={r['id']:r['features'] for r in e.rows(base/'val.jsonl')}
    for r in records:r['features']=fmap[r['id']]
    locators,_=cv.load_old_rule_locators(ROOT/'opc-agent/runs/full-matrix-20261031/blind-rules')
    rule=np.array(cv.retained_prior(prior_from_file(records,comp/'prior_outputs/val.jsonl'),cv.old_bits(records,locators)),bool)
    probabilities={}
    for route in e.ROUTES:
        prov=policy['provenance'][route]
        probabilities[route]=cv.load_probabilities(gbdt if route=='gbdt_prior_v1' else base,route,records,prov)
    choices,gc,force=selected_choices(e.read(comp/'frozen_validation_policy.json'))
    extension=e.read(ext/'frozen_validation_extension.json')
    old=predictions_for(records,rule.tolist(),probabilities,probabilities['gbdt_prior_v1'],choices,gc,force,extension['thresholds'])
    baseline=np.array([old[extension['selected'][r['family']]['variant']][i] for i,r in enumerate(records)],bool)
    p=np.array([[probabilities[n][i] if probabilities[n][i] is not None else [np.nan]*3 for n in e.ROUTES] for i in range(len(records))],np.float32)
    y=np.array([r['target'] for r in records],bool);v=np.array([r['truth_valid'] for r in records],bool)
    family=np.array([e.FAMILIES.index(r['family']) for r in records]);tier=np.array([int(r['tier'][1]) for r in records])
    present=np.array([not r['input_missing'] for r in records]);part=e.partition(records)
    cal=part=='cal';select=part=='select'
    candidates=e.aggregate(p,family,np.array(policy['top3_indices']))
    with (work/'stack_models.pkl').open('rb') as f:bundle=pickle.load(f)
    candidates.update(e.stack_predict(bundle,e.features(records,p,rule)))
    preds={'rules_only':rule,'release052':baseline};thresholds={}
    for name,prob in candidates.items():
        th=np.array(policy['thresholds'][name]);pred=e.decide(prob,rule,th,family);pred[~present]=rule[~present];preds[name]=pred
    preds['validation_selected_fusion']=np.array([preds[policy['selected'][r['family']]][i] for i,r in enumerate(records)])
    # Individual alternatives receive identical rule preservation and calibration data.
    expert_probs={route:p[:,j,:] for j,route in enumerate(e.ROUTES)}
    for kind in ['supervised','distilled','pruned']:
        indices=[j for j,route in enumerate(e.ROUTES) if route.startswith(kind+'_seed')]
        expert_probs[kind+'_3seed_mean']=np.nanmean(p[:,indices,:],axis=1)
    for route,prob in expert_probs.items():
        prob=np.nan_to_num(prob,nan=.5)
        th=e.calibrate(prob[cal],rule[cal],y[cal],v[cal]&present[cal,None],family[cal],tier[cal])
        pred=e.decide(prob,rule,th,family);pred[~present]=rule[~present]
        preds['expert_'+route]=pred;thresholds[route]=th.tolist()
    summaries=[];packages=[];groups=[]
    sub=[r for i,r in enumerate(records) if select[i]]
    assert len({r['family']+'|'+r['tier'] for r in sub})==30
    for name,pred in preds.items():
        summary,pack,gr=cv.summarize(sub,pred[select].tolist(),name)
        summaries.append(summary);packages.extend(pack);groups.extend(gr)
        assert e.metric(y[select&(tier>0)],v[select&(tier>0)],pred[select&(tier>0)])==summary['overall']
    cv.write_csv(OUT/'summary_validation.csv',[dict(variant=s['variant'],records=s['records'],macro_F1_24=s['macro_F1_24'],P1_macro_F1_6=s['P1_macro_F1_6'],P0_sample_FPR=s['P0_sample_FPR'],**s['overall']) for s in summaries])
    cv.write_csv(OUT/'all30_validation.csv',packages);cv.write_csv(OUT/'groups_validation.csv',groups)
    e.write(OUT/'validation_comparison.json',dict(status='complete',seconds=time.time()-start,original_validation_records=len(records),evaluation_records=len(sub),evaluation_groups=len({r['training_group_id'] for r in sub}),evaluation_source='same meta_select validation groups for every alternative',calibration_source='meta_cal only; no threshold fitting on meta_select',packages=30,summaries=summaries,individual_thresholds=thresholds,source_policy_sha256=e.sha(work/'frozen_policy.json'),limitations=['Selection validation, not independent blind test','Original base checkpoints selected using full original validation','P0-derived G/C/D technical proxy, not official nine-class localization']))
    print(json.dumps({'status':'complete','records':len(sub),'variants':len(summaries),'output':str(OUT)}),flush=True)


if __name__=='__main__':main()
