"""Add honest raw CSV axis context and atomically publish revised report index.

This preserves all v1 sources and all classification/confirmation decisions.
"""
from pathlib import Path
import collections
import json
import os
import shutil
import sys
import datetime as dt
import finish_refresh as f

OUT=f.OUT
index=json.loads((OUT/'report_index.json').read_text())
destroot=OUT/'numeric_context_v2/packages';destroot.mkdir(parents=True,exist_ok=False)
total=0
for report in index['reports']:
    if report['family']!='ULF_sample':continue
    src=OUT/'packages'/report['name'];dest=destroot/report['name'];dest.mkdir()
    ledgers=f.read_rows(src/'samples.jsonl');signals={r['id']:r for r in f.read_rows(src/'predictions.jsonl')}
    numeric=0;absolute=0
    for ledger in ledgers:
        signal=signals[ledger['sampleId']]
        if not ledger['dataPresent'] or not signal.get('model_candidates'):continue
        domain=f.csv_domain(ledger['dataPath'])
        if domain:
            if domain['kind']=='numeric_axis':ledger['actualCsvNumericAxis']=domain;numeric+=1
            else:ledger['actualCsvTimeDomain']=domain;absolute+=1
        for finding in ledger['findings']:
            if finding['defectType']!='L2-T' or finding['findingStage']!='candidate':continue
            finding['locator']['time_axis_domain']=domain
            finding['measurements'].pop('actual_csv_time_domain',None)
            finding['measurements']['actual_csv_numeric_axis']=domain
            finding['detail']='八类时段候选已关联到被质疑的输入事件起止记录。CSV仅含数值t轴，未提供独立UTC原点；列出其首末值供复核，不把输入区间认作真实错误时段或独立端点偏差。'
    with (dest/'samples.jsonl').open('w',encoding='utf-8') as fp:
        for ledger in ledgers:fp.write(json.dumps(ledger,ensure_ascii=False)+'\n')
    for name in ['predictions.jsonl','regions.jsonl','scorecard.json']:shutil.copyfile(src/name,dest/name)
    coverage=json.loads((src/'coverage.json').read_text());coverage['csvTimeDomainsRead']=absolute;coverage['csvNumericAxesRead']=numeric
    coverage['csvAxisInterpretation']='Numeric t-axis observed from CSV; no independent UTC origin or scientific endpoint truth.'
    f.q.dump(dest/'coverage.json',coverage)
    report['sourcefiles']=[str(dest/name) for name in ['samples.jsonl','predictions.jsonl','scorecard.json','regions.jsonl','coverage.json']]
    report['reportVersion']='refresh-20261105-v2-'+f.q.file_sha(dest/'samples.jsonl')[:12]
    report['csvNumericAxesRead']=numeric;report['independentAbsoluteTimeMappingAvailable']=False
    cases=[];seen=set()
    for ledger in ledgers:
        for finding in ledger['findings']:
            if finding['defectType'] not in seen:
                cases.append(f.compact_case(ledger,finding));seen.add(finding['defectType'])
    report['cases']=cases[:8]
    total+=numeric
index['reportVersion']='refresh-20261105-v2'
index['contextUpdatedAt']=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
f.q.dump(OUT/'report_index.v1_preserved.json',json.loads((OUT/'report_index.json').read_text()))
temp=OUT/'report_index.json.part';f.q.dump(temp,index);os.replace(temp,OUT/'report_index.json')
complete=json.loads((OUT/'complete.json').read_text());complete['reportVersion']='refresh-20261105-v2';complete['csvNumericAxesRead']=total
complete['csvIndependentAbsoluteTimeDomains']=0;complete['contextUpdatedAt']=index['contextUpdatedAt'];f.q.dump(OUT/'complete.json',complete)
print(json.dumps(dict(status='context_updated',numericAxes=total,absoluteTimeDomains=0,oldSourcesPreserved=True,index=str(OUT/'report_index.json'))))
