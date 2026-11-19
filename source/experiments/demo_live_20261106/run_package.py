"""Live, full-package demonstration: fresh rules/fusion/RGB; verified expert cache.

The runner deliberately records cache reuse. It never trains or changes thresholds.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path('/staff/tangjiarui/OPC')
BASE = ROOT / 'eight_class_20261104'

def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--package', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--baseline', required=True, type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--physical-gpu',default='')
    args = ap.parse_args()
    assert args.package.startswith(('EPB_seg_sample_', 'AIA_AR_sample_'))
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    assert baseline['name'] == args.package
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    evaluated = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    def progress(phase, percent, **more):
        value = dict(status='running', phase=phase, percent=percent,
                     elapsedSeconds=round(time.monotonic()-started, 3), **more)
        write(out/'progress.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    progress('重新读取原始文件、校验哈希并执行规则', 5)
    sys.path.insert(0, str(BASE/'code'))
    # Derive a narrow package runner from the frozen, reviewable implementation.
    source = (BASE/'code/roadshow.py').read_text(encoding='utf-8')
    assert 'assert len(packages)==12' in source
    source = source.replace("for folder in sorted(folders.iterdir()):", "for folder in sorted(folders.iterdir()):\n        if folder.name != args.package:continue")
    source = source.replace("assert len(packages)==12 and sum(r['records'] for r in packages)==24782", "assert len(packages)==1 and sum(r['records'] for r in packages)==args.expected")
    source = source.replace("status='complete',packages=12,records=24782", "status='complete',packages=len(packages),records=sum(r['records'] for r in packages)")
    (out/'classification_source.py').write_text(source, encoding='utf-8')
    module = {'__name__':'demo_classification', '__file__':str(out/'classification_source.py')}
    exec(compile(source, str(out/'classification_source.py'), 'exec'), module)
    module['main'](SimpleNamespace(fusion=BASE/'fusion_v3_missing4', output=out/'classification',
                                   package=args.package, expected=baseline['totalSamples']))
    classification_seconds = time.monotonic()-started
    progress('八类融合完成；逐张执行独立图像定位', 38)
    import torch
    torch.set_num_threads(6)
    device_label='cuda:'+args.physical_gpu if args.device=='cuda' else 'cpu'
    ground_source = (BASE/'refresh_code_20261105/ground_all_refresh.py').read_text(encoding='utf-8')
    if args.device=='cpu':
        ground_source = ground_source.replace("torch.autocast('cuda',dtype=torch.bfloat16)", 'contextlib.nullcontext()')
        ground_source = ground_source.replace('.cuda()', '.cpu()')
    ground_source = ground_source.replace("executed+=1", "executed+=1")
    (out/'localization_source.py').write_text(ground_source, encoding='utf-8')
    g = {'__name__':'demo_localization', '__file__':str(out/'localization_source.py'), 'contextlib':contextlib}
    exec(compile(ground_source, str(out/'localization_source.py'), 'exec'), g)
    original_predict = g['predict_batch']
    forwarded = 0
    expected_rgb = baseline.get('inferenceCoverage', {}).get('expectedReadableImageSamples', baseline['totalSamples'])
    def predict(*a, **kw):
        nonlocal forwarded
        result = original_predict(*a, **kw)
        forwarded += len(a[1])
        progress('逐张执行独立图像定位', min(92, 38+54*forwarded/max(1, expected_rgb)),
                 forwardedImages=forwarded, totalRecords=baseline['totalSamples'])
        return result
    g['predict_batch'] = predict
    g['main'](SimpleNamespace(roadshow=out/'classification', localizer=BASE/'localizer_v2',
                             comparison=BASE/'candidate_comparison_v1', output=out/'grounded', batch=8))
    progress('整理问题定位、六维报告和审计证据', 95, forwardedImages=forwarded)
    spec = importlib.util.spec_from_file_location('demo_finish', BASE/'refresh_code_20261105/finish_refresh.py')
    finish = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(finish)
    from spaceqa_agent import quality_ledger as q
    import localizer
    helper = localizer.helper()
    ann, _ = helper.annotation_module(ROOT)
    folder = out/'grounded/packages'/args.package
    predictions = {r['id']:r for r in finish.read_rows(out/'classification/packages'/args.package/'predictions.jsonl')}
    ledgers = finish.read_rows(folder/'samples.jsonl')
    regions = {r['sampleId']:r for r in finish.read_rows(folder/'regions.jsonl')}
    for ledger in ledgers:
        sid = ledger['sampleId']
        label = json.loads(Path(ledger['labelPath']).read_text(encoding='utf-8-sig'))
        signal = predictions[sid]
        proposed = finish.category_candidates(ledger, signal, label)
        if sid in regions:
            extracted = ann.extract_annotation(label, ledger['family'])
            geometry, errors = helper.extract_geometry(extracted['objects'], *ledger['imageSize'])
            proposed.extend(finish.spurious_candidates(ledger, signal, regions[sid], geometry, errors))
        q.attach_candidates(ledger, proposed)
    assert len(ledgers) == baseline['totalSamples'] == len(predictions)
    readable = sum(r['dataPresent'] for r in ledgers)
    assert forwarded == readable, (forwarded, readable)
    with (folder/'samples.jsonl').open('w', encoding='utf-8') as stream:
        for row in ledgers:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    shutil.copyfile(out/'classification/packages'/args.package/'predictions.jsonl', folder/'predictions.jsonl')
    card = q.aggregate(ledgers)
    write(folder/'scorecard.json', card)
    confirmed = {r['sampleId'] for r in ledgers if any(f.get('scoreEligible') for f in r['findings'])}
    positive = {code:set() for code in q.CLASSES}
    located = {code:set() for code in q.CLASSES}
    confirmed_by_code = {code:set() for code in q.CLASSES}
    cases = []
    for row in ledgers:
        for finding in row['findings']:
            code = finding['defectType']; sid = row['sampleId']
            positive[code].add(sid)
            if finish.valid_location(finding): located[code].add(sid)
            if finding.get('scoreEligible'): confirmed_by_code[code].add(sid)
            if len(cases)<8: cases.append(finish.compact_case(row, finding))
    for sid, row in predictions.items():
        for code in row.get('model_candidates', []): positive[code].add(sid)
    found = set().union(*positive.values())
    coverage = dict(baseline.get('classificationCoverage', {}))
    coverage.update(status='complete', executedSamples=len(ledgers), modelEligibleSamples=readable,
                    missingDataRuleOnlySamples=len(ledgers)-readable, freshEightHeadFusion=True,
                    freshUnderlying21Experts=False)
    localization = {code:dict(classifiedPositiveSamples=len(positive[code]), locatedSamples=len(located[code]),
                    locatedCandidateSamples=len(located[code]-confirmed_by_code[code]),
                    sampleOnlyCandidateSamples=len(positive[code]-located[code]),
                    directRuleConfirmedSamples=len(confirmed_by_code[code])) for code in q.CLASSES}
    inference = dict(status='complete', actualFreshImageForwards=forwarded,
                     expectedReadableImageSamples=readable, ratio=1.0, device=device_label, batch=8,
                     oldRegionCacheReused=False, measuredPackageSeconds=time.monotonic()-started-classification_seconds)
    write(folder/'coverage.json', dict(classificationCoverage=coverage, localizationCoverage=localization, inferenceCoverage=inference))
    report = dict(baseline)
    report.update(evaluatedAt=evaluated, reportVersion='live-demo-20261106-v1',
                  reportLabel='本次整包重评 · '+device_label+'图像定位＋八类融合', taskId=out.name,
                  totalSamples=len(ledgers), foundSamples=len(found), confirmedSamples=len(confirmed),
                  candidateOnlySamples=len(found-confirmed), detectedRatio=len(found)/len(ledgers),
                  observedTechnicalScore=card['observedTechnicalScore'], dimensions=card['dimensions'],
                  defectRecords={k:len(v) for k,v in positive.items()},
                  defectCountsByStage={c:dict(confirmedSamples=len(confirmed_by_code[c]),candidateOnlySamples=len(positive[c]-confirmed_by_code[c])) for c in positive},
                  sourcefiles=[str(folder/name) for name in ['samples.jsonl','predictions.jsonl','coverage.json','scorecard.json','regions.jsonl']],
                  classificationCoverage=coverage, localizationCoverage=localization, inferenceCoverage=inference, cases=cases,
                  elapsedSeconds=round(time.monotonic()-started,3))
    audit = dict(runId=out.name, package=args.package, records=len(ledgers), sourceHashChecks='all raw data and labels',
                 freshRgbForwards=forwarded, readableImages=readable, freshFusionSlots=len(ledgers)*8,
                 upstream21Experts='hash-verified cached predictions reused', device=device_label,
                 officialPrecision=None, officialRecall=None, officialF1=None,
                 classificationSeconds=classification_seconds, elapsedSeconds=report['elapsedSeconds'],
                 sourceSha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'classification_source.py',out/'localization_source.py']})
    report['liveAudit']=audit
    write(out/'report.json', report)
    write(out/'audit.json', audit)
    write(out/'progress.json', dict(status='completed', phase='整包重评完成', percent=100, **audit))
    print(json.dumps(audit), flush=True)

if __name__=='__main__':
    main()
