from pathlib import Path
import json
import datetime as dt
BASE=Path('/staff/tangjiarui/OPC/eight_class_20261104')
OUT=BASE/'refresh_20261105'
launch=json.loads((OUT/'ground_launch.json').read_text())
start=dt.datetime.fromisoformat(launch['startedAt'])
end=dt.datetime.fromtimestamp((BASE/'refresh_20261105.ground.log').stat().st_mtime,dt.timezone(dt.timedelta(hours=8)))
print(json.dumps(dict(startedAt=start.isoformat(),lastInferenceLogAt=end.isoformat(),wallSeconds=(end-start).total_seconds())))
for folder in sorted((OUT/'packages').iterdir()):
    if not folder.name.startswith('ULF'):continue
    with (folder/'samples.jsonl').open() as fp:
        ledger=next(json.loads(line) for line in fp if json.loads(line)['dataPresent'])
    path=Path(ledger['dataPath'])
    with path.open('r') as fp:
        first=[next(fp).strip() for _ in range(3)]
    label=json.loads(Path(ledger['labelPath']).read_text())
    print(json.dumps(dict(package=folder.name,csvPath=str(path),csvFirstRows=first,basic=label.get('basic_annotation'),science=label.get('science_annotation')),ensure_ascii=False)[:9000])
    break
