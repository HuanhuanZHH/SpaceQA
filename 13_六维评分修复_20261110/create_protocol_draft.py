"""Prepare observed file evidence and an evidence-incomplete target protocol with recorded weight approval."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'src'))
from spaceqa_agent.formal_checks import DIMENSION_IDS,digest
from spaceqa_agent.protocol_tools import source_relative_path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
card=json.loads((a.source_run/'scorecard.json').read_text());assert card['dirId']==a.input.name
rows=[];decode=[]
for line in (a.source_run/'predictions.jsonl').read_text().splitlines():
 x=json.loads(line)
 for kind,pathkey,hashkey in [('label','labelPath','labelSha256'),('data','dataPath','dataSha256')]:
  name=source_relative_path(x[pathkey],card['dirId']);rows.append({'sampleId':x['sampleId'],'path':name,'sha256':x.get(hashkey),'referenceSourceRunId':card['runId']});decode.append({'sampleId':x['sampleId'],'path':name,'format':'json' if kind=='label' else 'csv' if Path(name).suffix.lower()=='.csv' else 'image'})
for name,data in [('integrity_records.json',rows),('access_records.json',decode)]: (a.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
requirements={
 'C41':'Approved required classes and minimum unique sample count per class; trusted class table',
 'C42':'Approved required time/spatial/channel cells and sampling plan; trusted measurements',
 'C43':'Independent target distribution, fixed bins and validated TV threshold',
 'C44':'Actual split assignment and source grouping, or an explicit approved no-split scope',
 'C45':'Independent publisher identity and lineage remain unverified; generated records only test consistency against the preceding real input snapshot',
 'C46':'Reviewed lineage/annotation-method evidence with reviewer and hashes',
 'C47':'Actual declared/measured values and justified numeric tolerances',
 'C48':'Authorized use/license evidence and actual reviewer; do not infer open licensing',
 'C49':'Approved intended use, mandatory conditions and verified supporting evidence',
 'C50':'Local decoding measurements only; do not infer legal permission or scientific fitness'}
checks={k:{'version':'sixdim-target-draft-20261110','basis':None,'records':None,'weight':1,'missingRequirement':v} for k,v in requirements.items()}
checks['C45'].update(basis='Local byte-consistency check against actual prior run input fingerprints; not publisher identity or independent scientific provenance',records='integrity_records.json',datasetId=card['family'],datasetVersion='v34',scope='observed_local_integrity_only')
checks['C50'].update(basis='Actual decoding of every registered data and label file in this source run; includes missing files in denominator; not a permission assessment',records='access_records.json',scope='observed_local_decoding_only')
proposal={'status':'draft_not_approved','dirId':card['dirId'],'datasetId':card['family'],'datasetVersion':'v34','intendedUse':None,'sourceRunId':card['runId'],'evidenceSource':{'path':str(a.source_run/'predictions.jsonl'),'sha256':digest(a.source_run/'predictions.jsonl')},'checks':checks,'composite':{'weights':dict.fromkeys(DIMENSION_IDS,1),'basis':'PROPOSAL ONLY: equal dimension weights, no dimension priority assumed; not literature/official prescribed weights','approvedBy':None,'approvalStatus':'draft'},'scopeWarning':'A document draft is not approval. Local integrity/decoding observations do not fill missing semantic truth, representative distribution, publisher provenance or license evidence.'}
policy_path=Path(__file__).parent/'approved_weight_policy.json'
if policy_path.exists():
 policy=json.loads(policy_path.read_text())
 assert policy.get('approvalStatus')=='approved' and policy.get('approvedBy')
 proposal['composite']=policy
 proposal['status']='evidence_incomplete_weights_approved'
(a.output/'target_protocol.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n')
print(a.output/'target_protocol.json')
