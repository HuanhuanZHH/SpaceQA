from pathlib import Path
import collections
import datetime as dt
import json

BASE=Path('/staff/tangjiarui/OPC/eight_class_20261104');OUT=BASE/'refresh_20261105'
old=json.loads(Path('/staff/tangjiarui/OPC/chat_console_20261105/backup-20261105T134350/report_index.json').read_text())
new=json.loads((OUT/'report_index.json').read_text());oldby={r['name']:r for r in old['reports']}
def rows(path):return [json.loads(line) for line in Path(path).open(encoding='utf-8-sig')]
def read_ledger(report):
    paths=[p for p in report['sourcefiles'] if Path(p).name=='samples.jsonl']
    assert len(paths)==1,paths
    return rows(paths[0])
def key(ledger,f):
    loc=f.get('locator') or {}
    return json.dumps([ledger['sampleId'],f['defectType'],loc.get('annotation_pointer'),loc.get('bbox'),loc.get('polygon'),loc.get('timeInterval')],sort_keys=True)
summaries=[];totals=collections.Counter();newbyclass=collections.Counter()
for report in new['reports']:
    before=read_ledger(oldby[report['name']]);after=read_ledger(report)
    previous={key(l,f) for l in before for f in l['findings'] if f.get('findingStage')=='candidate'}
    current={key(l,f) for l in after for f in l['findings'] if f.get('findingStage')=='candidate'}
    added=current-previous;removed=previous-current;counts=collections.Counter()
    for ledger in after:
        for finding in ledger['findings']:
            if finding.get('findingStage')!='candidate':continue
            source=finding.get('sourceKind')
            if source=='independent_RGB_model':kind='fresh_rgb_geometry_candidate'
            elif source=='independent_RGB_model_comparison':kind='questioned_input_object_pointer_supported_by_fresh_comparison'
            elif source=='frozen_eight_class_plus_time_context':kind='questioned_input_time_interval_context'
            elif source=='within_annotation_consistency':kind='questioned_input_category_field_context'
            else:kind='other_'+str(source)
            counts[kind]+=1
            if key(ledger,finding) in added:
                counts['new_unique_'+kind]+=1;newbyclass[finding['defectType']]+=1
    counts.update(newUniqueCandidateLocators=len(added),removedUniqueCandidateLocators=len(removed))
    totals.update(counts)
    summaries.append(dict(package=report['name'],oldFoundSamples=oldby[report['name']].get('foundSamples'),newFoundSamples=report['foundSamples'],
        oldCandidateOnlySamples=oldby[report['name']].get('candidateOnlySamples'),newCandidateOnlySamples=report['candidateOnlySamples'],**dict(counts)))
launch=json.loads((OUT/'ground_launch.json').read_text());started=dt.datetime.fromisoformat(launch['startedAt'])
end=dt.datetime.fromtimestamp((BASE/'refresh_20261105.ground.log').stat().st_mtime,dt.timezone(dt.timedelta(hours=8)))
classification=json.loads((OUT/'classification/complete.json').read_text());complete=json.loads((OUT/'complete.json').read_text())
result=dict(oldIndex='/staff/tangjiarui/OPC/chat_console_20261105/backup-20261105T134350/report_index.json',newIndex=str(OUT/'report_index.json'),
    totals=dict(totals),newLocatorInstancesByClass=dict(newbyclass),packages=summaries,
    timing=dict(gpuStageStartedAt=started.isoformat(),gpuStageLastLogAt=end.isoformat(),gpuStageWallSeconds=(end-started).total_seconds(),
        classificationSeconds=classification['seconds'],postprocessingSeconds=complete['postprocessingSeconds'],imagePackageSecondsSum=complete['imageStagePackageSecondsSum']),
    interpretation='Different frozen fine-class policy and candidate emission scope. Locator additions are not new scientifically confirmed defects or accuracy improvements; time/object pointers do not independently localize true endpoints/false objects.')
(OUT/'comparison_to_previous.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['totals','newLocatorInstancesByClass','timing']},ensure_ascii=False))
