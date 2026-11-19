from pathlib import Path
import collections,json
root=Path('/staff/tangjiarui/OPC/chat_console_20261105')
index=json.loads((root/'report_index.json').read_text(encoding='utf-8-sig'))
results=[]
for report in index['reports']:
    if not report['name'].startswith('ULF'):continue
    path=next(Path(p) for p in report['sourcefiles'] if p.endswith('/samples.jsonl'))
    rows=[json.loads(line) for line in path.open(encoding='utf-8')]
    reasons={};counts={}
    for code in ['L0-01','L2-G','L2-P','L2-T','L2-F','L3-06','L3-07','L3-MIS']:
        counter=collections.Counter()
        reason=collections.Counter()
        for row in rows:
            check=row.get('checks',{}).get(code,{})
            counter[check.get('scopeStatus',check.get('status','absent'))]+=1
            reason[check.get('reason','unspecified')]+=1
        counts[code]=dict(counter);reasons[code]=dict(reason)
    results.append(dict(name=report['name'],total=report['totalSamples'],confirmed=report['confirmedSamples'],
                        candidateOnly=report['candidateOnlySamples'],found=report['foundSamples'],
                        foundRatio=report['detectedRatio'],technicalScore=report['observedTechnicalScore'],
                        defectCounts=report['defectRecords'],checks=counts,reasons=reasons))
print(json.dumps(results,ensure_ascii=False,indent=2))
