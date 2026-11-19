"""Apply frozen eight-class policy to all 12 unlabelled roadshow packages.

Fresh current rule checks + verified frozen-expert caches. Candidate scores stay
separate from confirmed density; no reference P0 or roadshow truth is consumed.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pickle, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import fusion as f
from fine_labels import CLASSES,NAMES
from spaceqa_agent import quality_ledger as q
from spaceqa_agent.engine import png_header

ROOT=f.ROOT;e=f.e
CONTRACTS={'AIA_AR_sample':['AR'],'HMI_AR_sample':['AR'],'CME_C2_sample':['CME'],'EPB_seg_sample':['EPB'],'DMSP_SSUSI':['PCA','TPA','unclear']}


def main(args):
    out=args.output;assert not out.exists();out.mkdir(parents=True);start=time.monotonic()
    policy=e.read(args.fusion/'frozen_policy.json');assert policy['models_sha256']==e.sha(args.fusion/'models.pkl')
    with (args.fusion/'models.pkl').open('rb') as fp:models=pickle.load(fp)
    sys.path.insert(0,str(ROOT/'spaceqa-agent/experiments/prior_optimization_20261102'))
    from fuse_roadshow_extension import load_cached
    packages=[];cards=[];provenance={}
    folders=ROOT/'opc-agent/runs/roadshow-evaluation-20261101/packages'
    for folder in sorted(folders.iterdir()):
        if not (folder/'inventory.jsonl').is_file():continue
        records=list(e.rows(folder/'inventory.jsonl'));name=folder.name
        current={r['sample_id']:r for r in e.rows(ROOT/'inference_prior_20261102/roadshow_final_v1/packages'/name/'selected_candidates.jsonl')}
        expected={r['sample_id']:r for r in e.rows(ROOT/'inference_prior_20261102/roadshow_v2/packages'/name/'prior_observations.jsonl')}
        assert {r['id'] for r in records}==set(current)==set(expected)
        def fresh(r):
            raw=Path(r['label_path']).read_bytes();label_hash=hashlib.sha256(raw).hexdigest()
            assert label_hash==expected[r['id']]['current_label_sha256']
            data=Path(r['data_path']);exists=data.is_file()
            assert exists==(not r['input_missing'])
            if exists:
                assert e.sha(data)==r['data_sha256']
                digest=hashlib.sha256(json.dumps([e.sha(r['image']),r['prompt'],r['features']],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                assert digest==r['input_sha256']
            sample=dict(id=r['id'],package=name,family=r['family'],data_path=r['data_path'],label_path=r['label_path'],
                data_exists=exists,data_sha256=r.get('data_sha256'),label_sha256=label_hash,data_hash_verification='fresh SHA256')
            if exists and data.suffix.lower()=='.png':
                m=png_header(data);sample['image_size']=[m['width'],m['height']]
            return q.analyze_sample(sample,json.loads(raw.decode('utf-8-sig')),allowed_labels=CONTRACTS.get(r['family']),
                contract_source='Participant task-family vocabulary, not independent semantic truth' if r['family'] in CONTRACTS else None)
        with ThreadPoolExecutor(max_workers=8) as pool:ledgers=list(pool.map(fresh,records))
        rule=np.zeros((len(records),8),bool)
        for i,ledger in enumerate(ledgers):
            for finding in ledger['findings']:rule[i,CLASSES.index(finding['defectType'])]=True
        probabilities={};cache_hashes={}
        for route in e.ROUTES:
            if route=='gbdt_prior_v1':
                base=ROOT/'inference_prior_20261102/gbdt'
                pred,marker=load_cached(base/'roadshow_predictions'/(name+'.jsonl'),base/'roadshow_predictions'/(name+'.complete.json'),records,
                    inventory_hash=e.sha(folder/'inventory.jsonl'),training_hash=e.sha(base/'training.json'))
            else:pred,marker=load_cached(folder/'models'/(route+'.jsonl'),folder/'models'/(route+'.complete.json'),records)
            probabilities[route]=pred;cache_hashes[route]=marker['prediction_sha256']
        p=np.array([[probabilities[n][r['id']].get('probabilities') or [np.nan]*3 for n in e.ROUTES] for r in records],np.float32)
        broad_rule=np.array([current[r['id']]['current_rule_bits'] for r in records],bool)
        x=np.concatenate([e.features(records,p,broad_rule),rule.astype(np.float32)],axis=1)
        probs={k:f.probabilities(v,f.feature_current(records,rule) if k.startswith('current') else x) for k,v in models.items()}
        fi=e.FAMILIES.index(records[0]['family']);flags={k:rule|(prob>=np.array(policy['thresholds'][k])[fi]) for k,prob in probs.items()}
        flags['rules']=rule
        for k,pred in flags.items():
            for i,r in enumerate(records):
                if r['input_missing']:pred[i]=rule[i]
        selected=np.zeros((len(records),8),bool);score=np.zeros((len(records),8),np.float32)
        for b,code in enumerate(CLASSES):
            route=policy['selected'][records[0]['family']][code]
            selected[:,b]=flags[route][:,b];score[:,b]=rule[:,b] if route=='rules' else probs[route][:,b]
        target=out/'packages'/name;target.mkdir(parents=True)
        counts=collections.Counter();unlocalized=0
        with (target/'predictions.jsonl').open('w') as fp:
            for i,r in enumerate(records):
                candidates=[CLASSES[b] for b in range(8) if selected[i,b] and not rule[i,b]]
                counts.update(candidates);unlocalized+=bool(candidates)
                fp.write(json.dumps(dict(id=r['id'],package=name,family=r['family'],classes=CLASSES,
                    prediction=selected[i].tolist(),rule=rule[i].tolist(),probabilities=score[i].tolist(),
                    model_candidates=candidates,candidate_localization_status='sample_only_pending_independent_region_or_time_evidence' if candidates else 'none',
                    label_path=r['label_path'],data_path=r['data_path'],input_sha256=r['input_sha256'],
                    ground_truth_available=False,scoreEligible=False,reference_input_used=False))+'\n')
        with (target/'samples.jsonl').open('w') as fp:
            for ledger in ledgers:fp.write(json.dumps(ledger,ensure_ascii=False)+'\n')
        card=q.aggregate(ledgers);card.update(modelCandidateSampleCount=unlocalized,modelCandidateCounts=dict(counts),
            fineClassifierVersion='eight-class-20261104',classificationDoesNotEstablishLocation=True)
        q.dump(target/'scorecard.json',card);cards.append(card)
        item=dict(package=name,family=records[0]['family'],records=len(records),confirmed_problem_samples=card['confirmedProblemSamples'],
            observed_technical_score=card['observedTechnicalScore'],model_candidate_samples=unlocalized,
            **{c+'_candidate_samples':counts[c] for c in CLASSES},precision=None,recall=None,F1=None)
        packages.append(item);e.log('roadshow_package_complete',**item)
        provenance[name]=dict(inventory_sha256=e.sha(folder/'inventory.jsonl'),verified_expert_prediction_hashes=cache_hashes,
            output_sha256=e.sha(target/'predictions.jsonl'))
    assert len(packages)==12 and sum(r['records'] for r in packages)==24782
    ranked=q.rank(cards);q.dump(out/'scorecards.json',ranked);f.csv_write(out/'package_summary.csv',packages)
    f.csv_write(out/'ranking.csv',[dict(package=c['dirId'],family=c['family'],rank=c['rankGlobalDisplay'],
        technical_score=c['observedTechnicalScore'],confirmed_problem_samples=c['confirmedProblemSamples'],samples=c['sampleCount']) for c in ranked])
    e.write(out/'complete.json',dict(status='complete',packages=12,records=24782,seconds=time.monotonic()-start,
        frozen_policy_sha256=e.sha(args.fusion/'frozen_policy.json'),new_full_expert_forward_passes=0,
        provenance=provenance,official_precision=None,official_recall=None,official_F1=None,
        scope='Fresh eight-class forward on verified expert caches plus actual current rules; not full 21-expert latency',
        caveats=['No independent roadshow fine truth','Fine model classifications are sample-level candidates until localized/adjudicated',
            'Model candidates do not deduct confirmed score','Six dimensions are broader than the eight content defect classes']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--fusion',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();main(a)
