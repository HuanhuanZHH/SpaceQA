"""Complete located candidate evidence after a fresh all-image forward pass.

No model is called here. Spurious-region boxes identify questioned INPUT
annotations, not newly predicted defect masks. Time intervals are questioned
input events, never independently verified scientific endpoint truth.
"""
from __future__ import annotations
import bench_metrics as bm
bm.script_enter('finalize')
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('/staff/tangjiarui/OPC')
BASE=ROOT/'eight_class_20261104'
OUT=Path('/staff/tangjiarui/OPC/runtime_benchmark_20261105/run_20261105_singlepass')
sys.path.insert(0,str(BASE/'code'))
from spaceqa_agent import quality_ledger as q
from spaceqa_agent.engine import parse_time
import localizer


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream]


def bbox_iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    union=area(a)+area(b)-inter
    return inter/union if union else 0.


def normalized_time(value):
    result=parse_time(value)
    if result and result.tzinfo is None:result=result.replace(tzinfo=dt.timezone.utc)
    return result


def csv_domain(path):
    """Read actual CSV first/last timestamps, never infer time from a plot axis."""
    path=Path(path)
    with path.open('rb') as stream:
        head=stream.read(16384)
        stream.seek(max(0,path.stat().st_size-16384))
        tail=stream.read()
    rows=list(csv.reader(io.StringIO(head.decode('utf-8-sig'))))
    if len(rows)<2:return None
    header,first=rows[0],rows[1]
    tailrows=list(csv.reader(io.StringIO(tail.decode('utf-8',errors='ignore'))))
    last=next((r for r in reversed(tailrows) if len(r)==len(header) and r),None)
    if not last:return None
    for index,name in enumerate(header):
        if not any(t in name.lower() for t in ['time','date','utc','epoch']):continue
        start,end=normalized_time(first[index]),normalized_time(last[index])
        if start and end and start<=end:
            return dict(kind='absolute_datetime_axis',column=name,columnIndex=index,start=start.isoformat(),end=end.isoformat(),source='actual CSV first and last rows',row_axis_mapping_established=False)
    for index,name in enumerate(header):
        if name.lower() not in {'t','time','t_s','time_s','seconds','seconds_elapsed'}:continue
        try:a,b=float(first[index]),float(last[index])
        except (ValueError,IndexError):continue
        if math.isfinite(a) and math.isfinite(b) and a<=b:
            return dict(kind='numeric_axis',column=name,columnIndex=index,firstValue=a,lastValue=b,
                unit='second' if name.lower() in {'t_s','time_s','seconds','seconds_elapsed'} else 'not declared by CSV header',
                source='actual CSV first and last rows',independent_absolute_time_mapping_established=False,
                interpretation='Raw numeric axis only; no independent UTC origin is supplied by this CSV.')
    return None


def make_candidate(ledger,code,locator,detail,source,*,measurements=None,severity=None,probability=None,kind=''):
    fid='REFRESH-'+q.digest([ledger['sampleId'],code,locator,kind])[:20]
    return dict(findingId=fid,checkId=q.CHECKS[code],ruleVersion='fresh-localization-20261105-v1',dirId=ledger['dirId'],
                sampleId=ledger['sampleId'],defectType=code,sourceKind=source,detail=detail,locator=locator,
                measurements=measurements or {},severity=severity,confidence=None,
                fusion_class_probability=probability,weighted_burden=probability if q.finite(probability) else None,
                candidate_trigger=kind,scientific_confirmation=False,findingStage='candidate',verdict='undetermined',scoreEligible=False,
                evidence=dict(label_path=ledger['labelPath'],data_path=ledger['dataPath'],label_sha256=ledger.get('labelSha256')))


def spurious_candidates(ledger,signal,region,geometry,errors):
    if 'L3-07' not in signal.get('model_candidates',[]) or errors:return []
    width,height=ledger['imageSize'];size=384
    objects={g['index']:g for g in geometry}
    components=region.get('predicted_regions',[])
    result=[]
    probability=signal['probabilities'][signal['classes'].index('L3-07')]
    for index in region.get('comparison',{}).get('unmatched_annotations_abstain',[]):
        if index not in objects:continue
        g=objects[index]
        box=[min(p[0] for p in g['points']),min(p[1] for p in g['points']),max(p[0] for p in g['points']),max(p[1] for p in g['points'])]
        internal=[box[0]/width*size,box[1]/height*size,box[2]/width*size,box[3]/height*size]
        maximum=max((bbox_iou(internal,c['bbox']) for c in components),default=0.)
        # Reject ambiguous merged/overlapping predictions rather than treating
        # Hungarian assignment absence as reliable negative scientific evidence.
        if maximum>.1:continue
        locator=dict(type='imageRegion',path=ledger['labelPath'],data_path=ledger['dataPath'],annotation_pointer=g['pointer'],bbox=box,
                     coordinate_system='image_pixels',origin='top_left',axis_order='xy',image_size=[width,height],
                     region_source='questioned_input_annotation_not_independent_predicted_defect_region',
                     independent_model_evidence='fresh RGB components failed to support this input object at recorded matching tolerance')
        result.append(make_candidate(ledger,'L3-07',locator,
            '八类虚标候选阳性且该输入标注对象未匹配新一轮独立RGB组件；框标明被质疑的输入对象位置，不是模型独立预测的缺陷框。模型未检出不能确认虚标。',
            'independent_RGB_model_comparison',measurements=dict(max_predicted_component_bbox_iou=maximum,independent_component_count=len(components),
                matching_min_mask_iou=.1,negative_evidence_verified=False),severity=dict(metric='unsupported_input_annotation_count',value=1.,unit='object'),
            probability=probability,kind='fine_spurious_positive_and_unmatched_input_annotation'))
    return result


def time_candidates(ledger,signal,label,domain):
    if 'L2-T' not in signal.get('model_candidates',[]) or not ledger['dataPresent']:return []
    probability=signal['probabilities'][signal['classes'].index('L2-T')]
    result=[]
    for pointer,obj in q.objects(label):
        pair=next(((a,b) for a,b in [('start_dt','end_dt'),('event_starttime','event_endtime')] if obj.get(a) is not None and obj.get(b) is not None),None)
        if not pair:continue
        a,b=pair;start,end=normalized_time(obj[a]),normalized_time(obj[b])
        if not start or not end:continue
        interval=[start.isoformat(),end.isoformat()]
        measurements=dict(declared_event_interval=interval,interval_role='questioned_input_event_interval',independent_endpoint_reference_available=False)
        if domain and domain.get('kind','absolute_datetime_axis')=='absolute_datetime_axis':
            d0,d1=normalized_time(domain['start']),normalized_time(domain['end'])
            measurements.update(actual_csv_time_domain=[domain['start'],domain['end']],
                starts_before_csv_seconds=max(0.,(d0-start).total_seconds()),ends_after_csv_seconds=max(0.,(end-d1).total_seconds()))
        elif domain:measurements['actual_csv_numeric_axis']=domain
        loc=dict(type='timeInterval',path=ledger['labelPath'],data_path=ledger['dataPath'],annotation_pointer=pointer,
                 timeInterval=interval,time_unit='ISO8601 UTC',interval_source='questioned_input_event_record_not_independent_true_endpoint',
                 time_axis_domain=domain,scientific_endpoint_reference_available=False)
        result.append(make_candidate(ledger,'L2-T',loc,
            '新一轮八类时段候选对应此输入事件记录；区间为被质疑的输入起止时段。CSV坐标信息仅作上下文，数值t轴不提供独立UTC原点。缺独立真实端点，不能把该区间称为已证实错误时段或真实偏差量。',
            'frozen_eight_class_plus_time_context',measurements=measurements,probability=probability,kind='questioned_input_event_interval'))
    if not result:
        basic=label.get('basic_annotation') or {}
        start,end=normalized_time(basic.get('startTime')),normalized_time(basic.get('endTime'))
        if start and end:
            loc=dict(type='timeInterval',path=ledger['labelPath'],data_path=ledger['dataPath'],annotation_pointer='/basic_annotation',
                     timeInterval=[start.isoformat(),end.isoformat()],time_unit='ISO8601 UTC',
                     interval_source='questioned_observation_window_not_independent_event_endpoint',scientific_endpoint_reference_available=False)
            result.append(make_candidate(ledger,'L2-T',loc,'时段分类候选已关联到输入观测时间窗；这不是独立的事件起止定位，须结合CSV时间域和事件定义复核。',
                'frozen_eight_class_plus_time_context',measurements=dict(independent_endpoint_reference_available=False,actual_csv_time_domain=domain),
                probability=probability,kind='questioned_observation_window'))
    return result


def category_candidates(ledger,signal,label):
    if 'L2-P' not in signal.get('model_candidates',[]):return []
    probability=signal['probabilities'][signal['classes'].index('L2-P')]
    result=[]
    for pointer,obj in q.walk(label):
        expected=obj.get('event_types')
        if not isinstance(expected,list) or not expected:continue
        for index,event in enumerate(obj.get('matched_events') or []):
            if not isinstance(event,dict):continue
            key,actual=q.category(event)
            if key is not None and actual not in expected:
                loc=dict(type='annotationRecord',path=ledger['labelPath'],data_path=ledger['dataPath'],annotation_pointer=pointer+'/matched_events/'+str(index)+'/'+key)
                result.append(make_candidate(ledger,'L2-P',loc,f'事件明细类别{actual!r}与该标注声明的event_types{expected!r}不一致；保留为类别候选，不自动证明物理类别错误。',
                    'within_annotation_consistency',measurements=dict(actual=actual,allowed=expected,contract_source='within-current-label event_types summary, not independent scientific truth'),
                    probability=probability,kind='model_category_positive_and_summary_conflict'))
    return result


def valid_location(finding):
    loc=finding.get('locator') or {};code=finding['defectType']
    if code=='L0-01':return bool(loc.get('expected_data_path') or loc.get('path'))
    if code in {'L2-P','L2-F'}:return bool(loc.get('annotation_pointer'))
    if code=='L2-T':return bool(loc.get('timeInterval'))
    return bool(loc.get('bbox') or loc.get('polygon'))


def compact_case(ledger,finding):
    return dict(sampleId=ledger['sampleId'],defectType=finding['defectType'],stage=finding['findingStage'],scoreEligible=finding.get('scoreEligible',False),
                basis=finding.get('detail'),locator=finding.get('locator'),caseId=finding.get('findingId'),sourceKind=finding.get('sourceKind'),
                checkId=finding.get('checkId'),ruleVersion=finding.get('ruleVersion'),confidence=finding.get('confidence'),
                fusionClassProbability=finding.get('fusion_class_probability'),fusionGroupProbability=finding.get('fusion_group_probability'),
                severity=finding.get('severity'),measurements=finding.get('measurements'))


def main():
    started=time.monotonic()
    assert (OUT/'grounded/complete.json').is_file(),'Wait until all fresh GPU inference completes'
    target=OUT/'packages';target.mkdir(exist_ok=False)
    h=localizer.helper();ann,_=h.annotation_module(ROOT)
    oldindex=json.loads((ROOT/'chat_console_20261105/report_index.json').read_text(encoding='utf-8-sig'))
    oldreports={r['name']:r for r in oldindex['reports']}
    groundcomplete=json.loads((OUT/'grounded/complete.json').read_text())
    groundcomplete['scope']='all readable roadshow RGB images freshly forwarded; frozen classifiers only gate candidate emission; model regions remain candidates'
    groundcomplete['metadataCorrection']='Scope description corrected after completion; no inference outputs changed.'
    q.dump(OUT/'grounded/complete.json',groundcomplete)
    execution={r['package']:r for r in groundcomplete['packages']}
    classifiercomplete=json.loads((OUT/'classification/complete.json').read_text())
    generated=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
    reports=[];summaries=[];cards=[]
    bm.mark('shared_imports_annotation_module_and_report_reference')
    for folder in sorted((OUT/'grounded/packages').iterdir()):
        bm.package_begin(folder.name)
        package=folder.name;dest=target/package;dest.mkdir()
        ledgers=read_rows(folder/'samples.jsonl');signals={r['id']:r for r in read_rows(OUT/'classification/packages'/package/'predictions.jsonl')}
        regions={r['sampleId']:r for r in read_rows(folder/'regions.jsonl')}
        extras=collections.Counter();domains=0;numeric_domains=0
        bm.mark('read_current_grounded_ledgers_signals_regions')
        for ledger in ledgers:
            sid=ledger['sampleId'];signal=signals[sid]
            relevant=bool(signal.get('model_candidates'))
            if not relevant:continue
            label=json.loads(Path(ledger['labelPath']).read_text(encoding='utf-8-sig'))
            proposed=[]
            if sid in regions:
                extracted=ann.extract_annotation(label,ledger['family'])
                geometry,errors=h.extract_geometry(extracted['objects'],*ledger['imageSize'])
                if extracted['schema_status']!='ok':errors.append(dict(reason='unsupported_annotation_schema'))
                proposed.extend(spurious_candidates(ledger,signal,regions[sid],geometry,errors))
            if ledger['family']=='ULF_sample' and ledger['dataPresent']:
                domain=csv_domain(ledger['dataPath'])
                domains+=bool(domain and domain.get('kind')=='absolute_datetime_axis')
                numeric_domains+=bool(domain and domain.get('kind')=='numeric_axis')
                if domain:ledger['actualCsvNumericAxis' if domain.get('kind')=='numeric_axis' else 'actualCsvTimeDomain']=domain
                proposed.extend(time_candidates(ledger,signal,label,domain))
                proposed.extend(category_candidates(ledger,signal,label))
            if proposed:
                q.attach_candidates(ledger,proposed);extras.update(f['defectType'] for f in proposed)
                if sid in regions:
                    regions[sid]['candidates'].extend(proposed)
        bm.mark('object_and_time_context_enrichment_including_actual_csv_reads')
        card=q.aggregate(ledgers);cards.append(card)
        with (dest/'samples.jsonl').open('w',encoding='utf-8') as stream:
            for ledger in ledgers:stream.write(json.dumps(ledger,ensure_ascii=False)+'\n')
        shutil.copyfile(OUT/'classification/packages'/package/'predictions.jsonl',dest/'predictions.jsonl')
        with (dest/'regions.jsonl').open('w',encoding='utf-8') as stream:
            for region in regions.values():stream.write(json.dumps(region,ensure_ascii=False)+'\n')
        q.dump(dest/'scorecard.json',card)
        bm.mark('aggregate_and_write_final_ledgers_scorecard_regions')
        total=len(ledgers);family=ledgers[0]['family'];present=sum(l['dataPresent'] for l in ledgers)
        image_expected=sum(l['dataPresent'] and bool(l.get('imageSize')) for l in ledgers)
        assert len(regions)==image_expected, (package,len(regions),image_expected)
        assert all(len(s['classes'])==len(s['probabilities'])==len(s['prediction'])==len(s['rule'])==8 for s in signals.values())
        assert set(signals)=={l['sampleId'] for l in ledgers}
        stage_counts={};local_counts={};allconfirmed=set();allcandidate=set();examplecases=[]
        for code in q.CLASSES:
            confirmed=set();candidates=set();located=set();ci=li=0
            model={sid for sid,s in signals.items() if code in s.get('model_candidates',[])}
            for ledger in ledgers:
                for finding in ledger['findings']:
                    if finding['defectType']!=code:continue
                    sid=ledger['sampleId']
                    if finding.get('findingStage')=='confirmed' and finding.get('scoreEligible') is True:confirmed.add(sid);ci+=1
                    else:candidates.add(sid);li+=1
                    if valid_location(finding):located.add(sid)
                    if len([x for x in examplecases if x['defectType']==code])<1:
                        examplecases.append(compact_case(ledger,finding))
            candidate_union=model|candidates
            stage_counts[code]=dict(confirmedSamples=len(confirmed),eightClassCandidateSamples=len(model),ledgerCandidateSamples=len(candidates),
                candidateSamples=len(candidate_union),candidateOnlySamples=len(candidate_union-confirmed),foundSamples=len(confirmed|candidate_union),
                confirmedInstanceRecords=ci,ledgerCandidateInstanceRecords=li)
            local_counts[code]=dict(classifiedPositiveSamples=len(confirmed|candidate_union),locatedSamples=len(located),
                locatedCandidateSamples=len(located&candidate_union),sampleOnlyCandidateSamples=len(candidate_union-located),
                directRuleConfirmedSamples=len(confirmed),scientificVisualAdjudicatedSamples=0,
                locationMeaning='file/field direct checks or explicitly questioned input/model region/interval; location itself does not establish truth')
            allconfirmed|=confirmed;allcandidate|=candidate_union
        coverage=dict(version='fresh-coverage-20261105-v1',evaluatedAt=generated,
            classificationCoverage=dict(status='complete',totalSamples=total,executedSamples=total,modelEligibleSamples=present,missingDataRuleOnlySamples=total-present,
                classSlots=total*8,finiteOutputSlots=total*8,freshEightHeadFusion=True,freshUnderlying21Experts=False,
                underlyingExperts='verified frozen expert prediction cache reused',fusionPolicy=str(BASE/'fusion_v3_missing4/frozen_policy.json'),
                fusionPolicySha256=q.file_sha(BASE/'fusion_v3_missing4/frozen_policy.json')),
            inferenceCoverage=dict(status='complete' if image_expected else 'not_pixel_task',gpu=1,actualFreshImageForwards=len(regions),expectedReadableImageSamples=image_expected,
                ratio=len(regions)/image_expected if image_expected else None,modelRoute=execution[package]['route'],batch=16,
                checkpointSha256=next((r['checkpoint_sha256'] for r in regions.values()),None),
                measuredPackageSeconds=execution[package]['seconds'],
                input='raw RGB only; annotations enter comparison after neural forward; ULF is CSV and not a pixel model task',
                oldRegionCacheReused=False),localizationCoverage=local_counts,
            addedLocatedCandidateInstances=dict(extras),csvTimeDomainsRead=domains,csvNumericAxesRead=numeric_domains,
            scientificEndpointTruthAvailable=False,spuriousNegativeEvidenceVerified=False)
        q.dump(dest/'coverage.json',coverage)
        old=oldreports[package]
        row=dict(old)
        row.update(totalSamples=total,confirmedSamples=len(allconfirmed),candidateOnlySamples=len(allcandidate-allconfirmed),foundSamples=len(allconfirmed|allcandidate),
            detectedRatio=len(allconfirmed|allcandidate)/total,defectRecords={c:v['foundSamples'] for c,v in stage_counts.items() if v['foundSamples']},
            defectCountUnit='按每类去重样本',defectCountsByStage=stage_counts,cases=examplecases[:8],evaluatedAt=generated,
            reportLabel='11月5日12包重新推理与八类定位',reportVersion='refresh-20261105-v1-'+q.file_sha(dest/'samples.jsonl')[:12],
            sourcefiles=[str(dest/n) for n in ['samples.jsonl','predictions.jsonl','scorecard.json','regions.jsonl','coverage.json']],
            inferenceCoverage=coverage['inferenceCoverage'],classificationCoverage=coverage['classificationCoverage'],localizationCoverage=local_counts,
            observedTechnicalScore=card['observedTechnicalScore'],scorecard=card,score=None,formalScore=None,officialScoreApproved=False,
            softwareVersion='spaceqa-refresh-20261105-v1',versions=dict(ruleVersion=q.VERSION,localizerVersion='fresh-RGB-20261105',fusionPolicySha256=coverage['classificationCoverage']['fusionPolicySha256']),
            limitations=['Eight-head fusion and image localization were re-executed; 21 upstream expert predictions were hash-verified caches, not rerun.',
                'Spurious boxes identify questioned input annotations unsupported by model matching, not independently predicted defect regions.',
                'Time intervals identify questioned input events/windows, not independently known true scientific endpoints.',
                'Model candidates never become confirmed automatically; roadshow independent official P/R/F1 unavailable.'])
        reports.append(row)
        summaries.append(dict(package=package,family=family,totalSamples=total,confirmedSamples=len(allconfirmed),candidateOnlySamples=len(allcandidate-allconfirmed),
            foundSamples=len(allconfirmed|allcandidate),detectedRatio=len(allconfirmed|allcandidate)/total,freshImageForwards=len(regions),missingDataSamples=total-present,
            localizedByClass={c:v['locatedSamples'] for c,v in local_counts.items()},addedLocatedCandidateInstances=dict(extras)))
        print(json.dumps(dict(stage='localization_completed',**summaries[-1])),flush=True)
        bm.mark('coverage_per_class_counts_and_report_metadata')
        bm.package_end(samples=total,family=family,csv_numeric_axes_read=numeric_domains,csv_absolute_domains_read=domains)
    bm.mark('shared_loop_finalization')
    assert len(reports)==12 and sum(r['totalSamples'] for r in reports)==24782
    q.dump(OUT/'report_index.json',dict(reports=reports,generatedAt=generated,reportVersion='refresh-20261105-v1'))
    q.dump(OUT/'scorecards.json',q.rank(cards))
    ranking=sorted(summaries,key=lambda s:(s['detectedRatio'],s['package']))
    for i,row in enumerate(ranking,1):row['displayRank']=i
    q.csv_write(OUT/'ranking_detected.csv',ranking)
    complete=dict(status='complete',reportVersion='refresh-20261105-v1',completedAt=generated,packages=summaries,packageCount=12,
        totalSamples=24782,freshImageForwards=sum(r['freshImageForwards'] for r in summaries),freshEightHeadFusionSlots=24782*8,
        confirmedSamples=sum(r['confirmedSamples'] for r in summaries),candidateOnlySamples=sum(r['candidateOnlySamples'] for r in summaries),
        foundSamples=sum(r['foundSamples'] for r in summaries),underlying21Experts='verified cache reused, not freshly forwarded',
        oldImageRegionCacheReused=False,providerApiCalls=0,trainingRuns=0,officialPrecision=None,officialRecall=None,officialF1=None,
        classificationSeconds=classifiercomplete['seconds'],imageStagePackageSecondsSum=sum(r['seconds'] for r in groundcomplete['packages']),
        postprocessingSeconds=time.monotonic()-started,
        originalOutputsRetained=True)
    q.dump(OUT/'complete.json',complete);q.dump(OUT/'state.json',dict(status='complete',stage='all_packages_ready',complete=str(OUT/'complete.json')))
    print(json.dumps({k:complete[k] for k in ['status','packageCount','totalSamples','freshImageForwards','confirmedSamples','candidateOnlySamples','foundSamples']}),flush=True)
    bm.mark('shared_global_ranking_index_and_completion_write')
    bm.done()


if __name__=='__main__':main()
