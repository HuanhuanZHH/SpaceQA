from pathlib import Path
import json
BASE=Path('/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105')
index=json.loads((BASE/'report_index.json').read_text());selected={}
for report in index['reports']:
    path=next(Path(p) for p in report['sourcefiles'] if Path(p).name=='samples.jsonl')
    for line in path.open():
        ledger=json.loads(line)
        for finding in ledger['findings']:
            code=finding['defectType']
            if code in selected:continue
            if code in {'L2-G','L3-06','L3-MIS'} and finding.get('sourceKind')!='independent_RGB_model':continue
            if code=='L3-07' and finding.get('sourceKind')!='independent_RGB_model_comparison':continue
            if code=='L2-T' and finding.get('sourceKind')!='frozen_eight_class_plus_time_context':continue
            selected[code]=dict(package=report['name'],family=report['family'],sampleId=ledger['sampleId'],labelPath=ledger['labelPath'],dataPath=ledger['dataPath'],
                imageSize=ledger.get('imageSize'),dataPresent=ledger['dataPresent'],finding=finding,actualCsvNumericAxis=ledger.get('actualCsvNumericAxis'))
assert len(selected)==8
(BASE/'eight_class_case_examples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:dict(package=v['package'],sampleId=v['sampleId']) for k,v in selected.items()}))
