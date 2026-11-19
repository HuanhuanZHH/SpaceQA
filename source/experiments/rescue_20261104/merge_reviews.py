"""Attach actually located visual reviews to fresh ledgers; recompute reports."""
from pathlib import Path
import argparse
import json
import time
from spaceqa_agent import quality_ledger as q


def merge(source, reviews, output):
    started=time.monotonic()
    if output.exists():raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    by_sample={}
    for p in reviews.glob('responses/*.json'):
        row=json.loads(p.read_text(encoding='utf-8'))
        by_sample[row['source']['sampleId']]=row
    cards=[];allfindings=[];allstates=[];allseverity=[]
    for package in sorted((source/'packages').iterdir()):
        rows=[json.loads(s) for s in (package/'samples.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        review_count=0;unavailable=0;merged_aliases=0
        for row in rows:
            review=by_sample.get(row['sampleId'])
            if not review:continue
            provenance=Path(review['source']['casePath']).parent/'provenance.json'
            prov=json.loads(provenance.read_text())
            if prov['label_sha256']!=row['labelSha256']:raise ValueError('Visual response uses a different label snapshot')
            review_count+=1;unavailable+=review['status']!='reviewed'
            row['fineReview']=dict(caseId=review['case_id'],status=review['status'],model=review['model'],
                protocol=review.get('protocol_version'),seconds=review['seconds'])
            additions=[]
            for d in review.get('combined_defects',[]):
                if d.get('status')=='confirmed':continue # Same facts already checked by fresh rules.
                code=q.canonical(d['defect_class']);loc=d['locator'];ptr=loc.get('annotation_pointer')
                if ptr and any(f['defectType']==code and f['locator'].get('annotation_pointer')==ptr for f in row['findings']):
                    merged_aliases+=1;continue
                additions.append(dict(findingId='VR-'+d['id'],checkId=q.CHECKS[code],ruleVersion=q.VERSION,
                    dirId=row['dirId'],sampleId=row['sampleId'],defectType=code,locator=loc,sourceKind='visual_api',
                    confidence=d.get('confidence'),severity=d.get('severity'),detail=d.get('explanation'),
                    fusionGroupProbability=d.get('fusion_group_probability'),weightedBurden=d.get('weighted_burden'),
                    evidence=dict(reviewCaseId=review['case_id'],reviewFile=str(reviews/'responses'),
                                  labelPath=row['labelPath'],dataPath=row['dataPath'])))
            q.attach_candidates(row,additions)
        card=q.aggregate(rows)
        old=json.loads((package/'scorecard.json').read_text())
        for key in ['hgbCoarseCandidateSamples','hgbArtifactSha256','hgbFineLocationsAvailable','freshMechanicalSeconds']:
            if key in old:card[key]=old[key]
        assert card['confirmedProblemSamples']==old['confirmedProblemSamples']
        card.update(fineReviewedSamples=review_count,fineReviewUnavailable=unavailable,mergedCandidateAliases=merged_aliases)
        target=output/'packages'/package.name;target.mkdir(parents=True)
        with (target/'samples.jsonl').open('w',encoding='utf-8') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
        cards.append(card);allfindings.extend(f for r in rows for f in r['findings']);allseverity.extend(card['severity'])
        for r in rows:
            for code,ch in r['checks'].items():allstates.append(dict(dirId=package.name,sampleId=r['sampleId'],defectType=code,
                checkId=q.CHECKS[code],ruleVersion=q.VERSION,verdict=ch['status'],scopeVerdict=ch['scopeStatus'],
                fieldPath=r['labelPath'],detail=ch['reason']))
    ranked=q.rank(cards);q.dump(output/'scorecards.json',ranked)
    for card in ranked:q.dump(output/'packages'/card['dirId']/'scorecard.json',card)
    with (output/'findings.jsonl').open('w',encoding='utf-8') as f:
        for row in allfindings:f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    q.csv_write(output/'八类缺陷比例.csv',[dict(dirId=c['dirId'],defectType=code,**d) for c in ranked for code,d in c['defects'].items()])
    q.csv_write(output/'同类缺陷严重程度排序.csv',allseverity)
    q.csv_write(output/'四状态检查记录.csv',allstates)
    q.csv_write(output/'六维评分与技术排序.csv',[dict(dirId=c['dirId'],family=c['family'],rankGlobalDisplay=c['rankGlobalDisplay'],
        rankInDatasetDisplay=c['rankInDatasetDisplay'],observedTechnicalScore=c['observedTechnicalScore'],fullQualityScore=None,
        confirmedProblemSamples=c['confirmedProblemSamples'],sampleCount=c['sampleCount'],scientificCheckCoverage=c['scientificCheckCoverage'],
        **{d['name']+'完整得分':d['score'] for d in c['dimensions']},**{d['name']+'观测技术分':d['observedTechnicalScore'] for d in c['dimensions']},
        tiedDirectoryIds=c['tiedDirectoryIds']) for c in ranked])
    # Preserve the nine mandatory roadshow evidence columns; extra fields carry
    # the actual location, independently measured extent and original source.
    q.csv_write(output/'证据表_SpaceQA_八类整改版.csv',[dict(findingId=f['findingId'],checkId=f['checkId'],ruleVersion=f['ruleVersion'],
        dirId=f['dirId'],objectId=f['sampleId'],fieldPath=f['locator'].get('annotation_pointer'),
        locatorType=f['locator'].get('type'),verdict=f['verdict'],detail=f.get('detail'),
        defectType=f['defectType'],stage=f['findingStage'],locator=f['locator'],severity=f.get('severity'),evidence=f.get('evidence')) for f in allfindings])
    # q / unique scientific ordering are still unavailable; label this export.
    q.csv_write(output/'路演结果_SpaceQA_观测技术分.csv',[dict(datasetId=c['family'],datasetVersion='v34',dirId=c['dirId'],
        rankInDataset=c['rankInDatasetDisplay'],rankGlobal=c['rankGlobalDisplay'],totalScore=round(c['observedTechnicalScore'],1),
        mainDefectTypes=';'.join(code for code,v in c['defects'].items() if v['confirmedSamples']),
        evidence=f"{c['confirmedProblemSamples']}/{c['sampleCount']}条记录有已确认契约/文件/尺寸问题；观测技术分，非官方q；同分仅显示排序") for c in ranked])
    state=json.loads((source/'run.json').read_text())
    state['freshMechanicalTotalSeconds']=state.pop('totalSeconds')
    for package in state['packages']:
        package['freshSampleLedgerSha256']=package.pop('sampleLedgerSha256')
        package['sampleLedgerSha256']=q.file_sha(output/'packages'/package['package']/'samples.jsonl')
    state.update(mergeVersion=q.VERSION,reviewReceipt=json.loads((reviews/'receipt.json').read_text()),
        outputKind='fresh_rule_ledger_with_bounded_visual_candidates',modelCandidatesScored=False,
        modelInference='Frozen HGB predictions reused; 24 new optional visual API reviews, no fresh expert training/forward',
        mergeSeconds=time.monotonic()-started,mergeScriptSha256=q.file_sha(Path(__file__)),
        endToEndFullExpertRuntimeMeasured=False)
    q.dump(output/'run.json',state)
    return ranked


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fresh',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();merge(a.fresh,a.reviews,a.output)
