"""Repeat deterministic rules on all 12 roadshow packages, outside frozen runs."""
from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path('/staff/tangjiarui/OPC')
OUT=ROOT/'ppt_results_20261104/rule_recheck'
INPUT=ROOT/'opc-agent/runs/roadshow-evaluation-20261101/extracted'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'spaceqa-agent-release-052-20261103/src')
    py=str(ROOT/'.venv-image-models/bin/python');state={'status':'running','runs':[]}
    (OUT/'state.json').write_text(json.dumps(state))
    for i in [1,2]:
        out=OUT/f'run_{i}';start=time.time()
        with (OUT/f'run_{i}.log').open('w') as f:
            r=subprocess.run([py,'-m','spaceqa_agent','scan','--input',str(INPUT),'--output',str(out),'--seed','20261104'],env=env,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'rule run {i} failed: {r.returncode}')
        audit=subprocess.run([py,'-m','spaceqa_agent','audit',str(out)],env=env,capture_output=True,text=True)
        (OUT/f'audit_{i}.json').write_text(audit.stdout)
        if audit.returncode:raise RuntimeError(f'audit run {i} failed')
        path=out/'06_问题清单/findings.jsonl';digest=hashlib.sha256(path.read_bytes()).hexdigest()
        cards=list((out/'04_评分卡').glob('*.json'));assert len(cards)==12
        state['runs'].append({'run':i,'elapsedSeconds':time.time()-start,'findingsSha256':digest,'datasets':len(cards),'audit':'pass'})
        (OUT/'state.json').write_text(json.dumps(state,indent=2))
    state.update(status='complete',findingBytesIdentical=state['runs'][0]['findingsSha256']==state['runs'][1]['findingsSha256'],scope='Two fresh rules-only runs; frozen expert outputs and original end-to-end timing remain separately identified')
    assert state['findingBytesIdentical']
    (OUT/'state.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
