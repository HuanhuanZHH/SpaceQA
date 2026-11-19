"""CPU-only derived six-dimension report; never relabels a fresh inference run."""
import argparse,copy,datetime,json,sys,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
from spaceqa_agent import formal_checks as f

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--protocol',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.input.resolve();source=a.source_run.resolve();out=a.output.resolve()
 if out.exists() or out==root or out.is_relative_to(root) or out==source or out.is_relative_to(source):raise ValueError('Use a new output outside original input and source run')
 card=json.loads((source/'scorecard.json').read_text());complete=json.loads((source/'complete.json').read_text());assert complete['status']=='complete' and card['runId']==complete['runId'];assert root.name==card['dirId'],'Input dirId does not match source run'
 contract=json.loads(a.protocol.read_text()) if a.protocol else {}
 if contract.get('dirId') and contract['dirId']!=root.name:raise ValueError('Protocol dirId mismatch')
 contract=dict(contract,datasetId=card['family']);out.mkdir(parents=True);runid='SIXDIM-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
 results=f.run(root,out/'formal_checks',f.IDS,contract,runid,evidence_root=a.protocol.resolve().parent if a.protocol else root);dimensions=f.apply_formal_dimensions(copy.deepcopy(card['dimensions']),results,contract);comp=f.composite({d['id']:d.get('score') for d in dimensions},contract.get('composite',{}))
 report={'runId':runid,'sourceRunId':card['runId'],'operation':'CPU-only six-dimensional rescoring of existing inference; no new model forward claimed','dirId':card['dirId'],'family':card['family'],'sampleCount':card['sampleCount'],'sourceCard':{'path':str(source/'scorecard.json'),'sha256':f.digest(source/'scorecard.json')},'sourceComplete':{'path':str(source/'complete.json'),'sha256':f.digest(source/'complete.json')},'protocol':{'path':str(a.protocol.resolve()) if a.protocol else None,'sha256':f.digest(a.protocol) if a.protocol else None,'status':contract.get('status','missing')},'dimensions':dimensions,'fullQualityScore':comp['score'],'fullQualityScoreReason':comp['reason'],'blockers':comp.get('blockers',[]),'observedTechnicalScoreUnchanged':card['observedTechnicalScore'],'version':f.VERSION}
 (out/'six_dimension_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(out/'run.log.jsonl').write_text((out/'formal_checks/run.log.jsonl').read_text());print(json.dumps({k:report[k] for k in ['runId','sourceRunId','dirId','fullQualityScore','blockers']},ensure_ascii=False))
if __name__=='__main__':main()
