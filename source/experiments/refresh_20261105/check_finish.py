"""Focused safety checks for the new evidence association, no remote inference."""
from pathlib import Path
import ast
import importlib.util
import sys
import tempfile
import types

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent.parent/'src'))
sys.modules['localizer']=types.ModuleType('localizer')
spec=importlib.util.spec_from_file_location('fresh_finish',HERE/'finish_refresh.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for path in HERE.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
ledger=dict(sampleId='x',dirId='p',labelPath='input.json',dataPath='input.png',imageSize=[100,100],dataPresent=True,findings=[])
signal=dict(model_candidates=['L3-07','L2-T'],classes=list(m.q.CLASSES),probabilities=[.8]*8)
geometry=[dict(index=0,pointer='/objects/0',points=[[10,10],[20,10],[20,20],[10,20]])]
region=dict(predicted_regions=[],comparison=dict(unmatched_annotations_abstain=[0]))
cases=m.spurious_candidates(ledger,signal,region,geometry,[])
assert len(cases)==1 and cases[0]['scoreEligible'] is False and cases[0]['scientific_confirmation'] is False
assert cases[0]['locator']['region_source']=='questioned_input_annotation_not_independent_predicted_defect_region'
assert not m.spurious_candidates(ledger,signal,region,geometry,[{'reason':'schema'}])
overlap=dict(predicted_regions=[dict(bbox=[38.4,38.4,76.8,76.8])],comparison=region['comparison'])
assert not m.spurious_candidates(ledger,signal,overlap,geometry,[])
label=dict(matched_events=[dict(start_dt='2020-01-01T00:00:00Z',end_dt='2020-01-01T00:00:10Z')])
domain=dict(start='2020-01-01T00:00:02+00:00',end='2020-01-01T00:00:08+00:00')
times=m.time_candidates(ledger,signal,label,domain)
assert len(times)==1 and times[0]['measurements']['starts_before_csv_seconds']==2
assert times[0]['measurements']['ends_after_csv_seconds']==2
assert times[0]['scoreEligible'] is False and not times[0]['measurements']['independent_endpoint_reference_available']
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'a.csv';path.write_text('UTC,value\n2020-01-01T00:00:02Z,1\n2020-01-01T00:00:08Z,2\n')
    actual=m.csv_domain(path)
    assert actual['start']==domain['start'] and actual['end']==domain['end']
m.q.attach_candidates(ledger,cases+times)
assert len(ledger['findings'])==2 and all(f['findingStage']=='candidate' and not f['scoreEligible'] for f in ledger['findings'])
print('PASS: geometry ambiguity abstention, questioned-input semantics, UTC context, candidate score exclusion, source syntax')
