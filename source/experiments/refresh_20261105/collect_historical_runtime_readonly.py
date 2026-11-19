"""Collect historical timer receipts only. No models/data/services are run."""
from pathlib import Path
import json
import subprocess

REMOTE=r'''
from pathlib import Path
import datetime as dt,hashlib,json
R=Path('/staff/tangjiarui/OPC')
B=R/'opc-agent/runs/roadshow-evaluation-20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def evidence(p):
    p=Path(p);raw=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),record=json.loads(raw))
packages=[]
for folder in sorted((B/'packages').iterdir()):
    prep=folder/'preparation.json'
    if not prep.is_file():continue
    pr=evidence(prep);rr=pr['record'];items=[]
    for marker in sorted((folder/'models').glob('*.complete.json')):
        proof=evidence(marker);d=proof['record']
        items.append(dict(route=d['route'],seconds=d['model_load_and_inference_seconds'],inferred=d['inferred'],hardware=d.get('hardware'),
            modelLoadIncluded=True,predictionSerializationIncluded=False,timerSource=proof))
    assert len(items)==20
    gb=evidence(R/'inference_prior_20261102/gbdt/roadshow_predictions'/(folder.name+'.complete.json'));g=gb['record']
    model20=sum(x['seconds'] for x in items);model21=model20+g['seconds'];prepseconds=rr['seconds'];n=rr['records']
    assert len({x['inferred'] for x in items})==1 and items[0]['inferred']==g['inferred']
    present=g['inferred']
    packages.append(dict(package=folder.name,family=rr['family'],registeredSamples=n,modelEligibleSamples=present,missingInputs=rr['missing_inputs'],
        rawPrepareSeconds=prepseconds,neural20Seconds=model20,gbdtSeconds=g['seconds'],expert21Seconds=model21,
        sumRecordedPrepareAnd21Seconds=prepseconds+model21,
        expert21SecondsPerRegisteredSample=model21/n,expert21SecondsPerEligibleSample=model21/present,
        prepareAnd21SecondsPerRegisteredSample=(prepseconds+model21)/n,
        expert20ModelLoadIncluded=True,gbdtModelLoadSeconds=None,gbdtModelLoadIncluded=False,
        neural20Hardware=sorted({x['hardware'] for x in items}),prepareHardware='not recorded in preparation receipt',gbdtHardware='CPU; exact host/model not recorded in per-package receipt',
        prepareTimerSource=pr,neural20TimerSources=items,gbdtTimerSource=gb))
assert len(packages)==12 and sum(p['registeredSamples'] for p in packages)==24782
sources=[R/'opc-agent/experiments/roadshow_evaluation_20261101/roadshow.py',R/'opc-agent/experiments/all_family_training_20261031/qwen.py',R/'opc-agent/experiments/all_family_training_20261031/evaluate.py',R/'opc-agent/experiments/full_matrix_20261031/prepare.py',R/'spaceqa-agent/experiments/prior_optimization_20261102/gbdt/predict_roadshow.py']
totals={key:sum(p[key] for p in packages) for key in ['registeredSamples','modelEligibleSamples','missingInputs','rawPrepareSeconds','neural20Seconds','gbdtSeconds','expert21Seconds','sumRecordedPrepareAnd21Seconds']}
totals.update(expert21WeightedSecondsPerRegisteredSample=totals['expert21Seconds']/totals['registeredSamples'],
    expert21WeightedSecondsPerEligibleSample=totals['expert21Seconds']/totals['modelEligibleSamples'],
    prepareAnd21WeightedSecondsPerRegisteredSample=totals['sumRecordedPrepareAnd21Seconds']/totals['registeredSamples'],
    qloraSeconds=sum(i['seconds'] for p in packages for i in p['neural20TimerSources'] if i['route']=='qlora'))
out=dict(scope='Historical recorded stages summed; NOT one end-to-end cold-start wall-clock run and NOT current full-pipeline benchmark',
    collectedAt=dt.datetime.now(dt.timezone.utc).isoformat(),remoteReadOnly=True,modelForwardStarted=False,
    totals=totals,packages=packages,sourceCode=[dict(path=str(p),sha256=sha(p)) for p in sources],
    missingStages=['GBDT model/vectorizer one-time loading unrecorded','Current HGB7 + eight-class fusion + current localizer + final report/audit not included in these historical totals','Neural expert prediction serialization after synchronize unrecorded','No shared parent clock across prepare/experts; orchestration/queue gaps not recoverable from per-stage elapsed values'],
    caveats=['Model elapsed times contain loading and data decoding; not pure GPU kernel times','All sample averages amortize load costs; they are throughput averages, not single-request latency','Registered-sample averages include missing-file abstention records; eligible-sample average given separately','Hardware is marker-reported icarus1 A100; no official A10 timing acceptance'])
print(json.dumps(out,ensure_ascii=False))
'''
p=subprocess.run(['ssh','icarus0','/staff/tangjiarui/OPC/.venv-image-models/bin/python','-B','-'],input=REMOTE,text=True,encoding='utf-8',capture_output=True,check=True)
data=json.loads(p.stdout)
OUT=Path('D:/OPC/outputs/runtime_benchmark_20261105');OUT.mkdir(parents=True,exist_ok=True)
(OUT/'historical_21expert_components.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data['totals'],ensure_ascii=False))
