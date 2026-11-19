from pathlib import Path
import json
OUT=Path('D:/OPC/outputs/chat_insights_refresh_20261105')
d=json.loads((OUT/'full_entry_readonly_inventory.json').read_text(encoding='utf-8'))
h=d['historicalRuntime'];packages=h['packages']
print(json.dumps(dict(preprocessingSeconds=sum(p['preparationSeconds'] for p in packages),sum21Seconds=h['sumSeconds'],sum20Seconds=h['sumSeconds']-h['routes']['gbdt_prior_v1']['seconds'],
    qwenSeconds=h['routes']['qlora']['seconds'],sampleMean21Seconds=h['sumSeconds']/24782,qwenBase=d['qwen_base'],
    checkpointTotalBytes=sum(r['bytes'] for r in d['checkpoint_files']),
    packages=[{k:p[k] for k in ['package','records','preparationSeconds','sumHistorical21Seconds','experts']} for p in packages],entries=d['complete_release_entries']),ensure_ascii=False,indent=2))
