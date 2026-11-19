"""Independent read-only checks of final refreshed sources and index totals."""
from pathlib import Path
import collections
import hashlib
import json
import math
import datetime as dt

ROOT=Path('/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105')
def rows(path):return [json.loads(line) for line in Path(path).open(encoding='utf-8-sig')]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
index=json.loads((ROOT/'report_index.json').read_text())
complete=json.loads((ROOT/'complete.json').read_text())
assert len(index['reports'])==12
grand=collections.Counter();localgrand=collections.Counter();stagegrand=collections.defaultdict(collections.Counter);packages=[];errors=[];checks=0
for report in index['reports']:
    source={Path(p).name:Path(p) for p in report['sourcefiles']};folder=source['samples.jsonl'].parent;ledgers=rows(folder/'samples.jsonl');signals={r['id']:r for r in rows(folder/'predictions.jsonl')}
    regions={r['sampleId']:r for r in rows(folder/'regions.jsonl')};card=json.loads((folder/'scorecard.json').read_text());coverage=json.loads((folder/'coverage.json').read_text())
    assert len(signals)==len(ledgers)==report['totalSamples'];checks+=1
    assert len(regions)==sum(bool(l['dataPresent'] and l.get('imageSize')) for l in ledgers)==coverage['inferenceCoverage']['actualFreshImageForwards'];checks+=1
    assert all(len(r['classes'])==len(r['probabilities'])==8 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in r['probabilities']) for r in signals.values());checks+=1
    assert all(Path(p).is_file() for p in report['sourcefiles']);checks+=1
    confirmed=set();candidates=set();sub=collections.defaultdict(lambda:dict(confirmed=set(),candidates=set(),located=set()))
    for ledger in ledgers:
        sid=ledger['sampleId'];signal=signals[sid]
        assert Path(ledger['labelPath']).is_file() and sha(ledger['labelPath'])==ledger['labelSha256'];checks+=1
        assert bool(Path(ledger['dataPath']).is_file())==ledger['dataPresent'];checks+=1
        for code in signal.get('model_candidates',[]):candidates.add(sid);sub[code]['candidates'].add(sid)
        for finding in ledger['findings']:
            code=finding['defectType'];loc=finding.get('locator') or {}
            if finding.get('findingStage')=='confirmed':
                assert finding.get('scoreEligible') is True and finding['sourceKind'] not in {'independent_RGB_model','independent_RGB_model_comparison','frozen_eight_class_plus_time_context','within_annotation_consistency'}
                confirmed.add(sid);sub[code]['confirmed'].add(sid)
            else:
                assert finding['findingStage']=='candidate' and finding.get('scoreEligible') is False
                candidates.add(sid);sub[code]['candidates'].add(sid)
                if code=='L3-07' and finding.get('candidate_trigger')=='fine_spurious_positive_and_unmatched_input_annotation':
                    assert 'questioned_input_annotation' in loc['region_source'] and finding['measurements']['negative_evidence_verified'] is False
                if code=='L2-T':
                    assert loc['scientific_endpoint_reference_available'] is False
                    assert finding['measurements']['independent_endpoint_reference_available'] is False
                checks+=1
            valid=bool(loc.get('expected_data_path') or loc.get('path')) if code=='L0-01' else bool(loc.get('annotation_pointer')) if code in {'L2-P','L2-F'} else bool(loc.get('timeInterval')) if code=='L2-T' else bool(loc.get('bbox') or loc.get('polygon'))
            if valid:sub[code]['located'].add(sid)
            if loc.get('bbox'):
                box=loc['bbox']
                assert len(box)==4 and all(math.isfinite(float(v)) for v in box) and box[0]<=box[2] and box[1]<=box[3];checks+=1
                if ledger.get('imageSize') and finding['findingStage']=='candidate':
                    w,h=ledger['imageSize'];assert 0<=box[0]<=box[2]<=w and 0<=box[1]<=box[3]<=h;checks+=1
    for code,expected in report['defectCountsByStage'].items():
        s=sub[code];assert len(s['confirmed'])==expected['confirmedSamples'];assert len(s['candidates'])==expected['candidateSamples'];assert len(s['confirmed']|s['candidates'])==expected['foundSamples'];checks+=3
        assert len(s['located'])==coverage['localizationCoverage'][code]['locatedSamples'];checks+=1
        assert len(s['candidates']-s['located'])==coverage['localizationCoverage'][code]['sampleOnlyCandidateSamples'];checks+=1
        localgrand[code]+=len(s['located'])
        stagegrand[code].update(confirmedSamples=len(s['confirmed']),candidateSamples=len(s['candidates']),foundSamples=len(s['confirmed']|s['candidates']),locatedSamples=len(s['located']),sampleOnlyCandidateSamples=len(s['candidates']-s['located']))
    assert len(confirmed)==report['confirmedSamples']==card['confirmedProblemSamples'];checks+=1
    assert len(candidates-confirmed)==report['candidateOnlySamples'];checks+=1
    assert len(confirmed|candidates)==report['foundSamples'];checks+=1
    assert abs(card['observedTechnicalScore']-(100*(1-len(confirmed)/len(ledgers))))<1e-9;checks+=1
    assert card['officialPrecision'] is None and card['officialRecall'] is None and card['officialF1'] is None;checks+=1
    grand.update(totalSamples=len(ledgers),freshImageForwards=len(regions),confirmedSamples=len(confirmed),candidateOnlySamples=len(candidates-confirmed),foundSamples=len(confirmed|candidates),missingDataSamples=sum(not l['dataPresent'] for l in ledgers),csvTimeDomainsRead=coverage['csvTimeDomainsRead'],csvNumericAxesRead=coverage.get('csvNumericAxesRead',0))
    packages.append(dict(package=report['name'],sourceHashes={Path(p).name:sha(p) for p in report['sourcefiles']},inference=coverage['inferenceCoverage'],localization=coverage['localizationCoverage']))
for field in ['totalSamples','freshImageForwards','confirmedSamples','candidateOnlySamples','foundSamples']:assert grand[field]==complete[field];checks+=1
audit=dict(status='pass',readOnly=True,auditedAt=dt.datetime.now(dt.timezone.utc).isoformat(),checks=checks,packageCount=12,totals=dict(grand),
           perClass={k:dict(v) for k,v in stagegrand.items()},packages=packages,
           limitations=['No independent official defect truth was available; these are source integrity and stage/coverage checks, not accuracy certification.'])
(ROOT/'independent_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:audit[k] for k in ['status','checks','packageCount','totals','perClass']},ensure_ascii=False))
