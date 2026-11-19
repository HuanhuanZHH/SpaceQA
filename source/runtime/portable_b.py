import os,json,hashlib,importlib.util,time
from pathlib import Path
def apply(records,ledgers,pred,out):
 from spaceqa_agent import quality_ledger as q
 os.environ['SPACEQA_B_OUTPUT']=str(out/'b_maps')
 path=Path(__file__).parent/'b_module/object_refinement_adapter_20261109.py'
 spec=importlib.util.spec_from_file_location('portable_b',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 receipts=[]
 for i,r in enumerate(records):
  start=time.monotonic();base=bool(pred[i,5]);applicable=r['family'] in m.m.families and not r.get('input_missing')
  if not applicable:result=dict(id=r['id'],status='not_applicable',predicted_missing=base,boxes=[])
  else:result=m.predict_current(r,base)
  result.update(base_missing=base,elapsedMs=(time.monotonic()-start)*1000,source='fresh_current_input',scientificVerdict='undetermined')
  if applicable and base:
   pred[i,5]=result['predicted_missing'];ledgers[i]['findings']=[f for f in ledgers[i]['findings'] if f['defectType']!='L3-06' or f.get('scoreEligible')]
   for j,box in enumerate(result['boxes']):
    loc=dict(type='imageRegion',path=r['data_path'],data_path=r['data_path'],bbox=box['bbox'],image_size=ledgers[i].get('imageSize'),coordinate_system='image_pixels',origin='top_left',axis_order='xy')
    f=q.finding(dict(id=r['id'],package=ledgers[i]['dirId'],data_path=r['data_path'],label_path=r['label_path'],label_sha256=r['label_sha256']), 'L3-06',loc,'B raw4 + learned4 fresh omission candidate',dict(metric='omission_candidate_count',value=1,unit='object',tolerance=None),source='B_fresh',stage='candidate',measurements={'raw_score':box['raw_score'],'learned_quality':box['quality_score']});ledgers[i]['findings'].append(f)
  receipts.append(result)
 (out/'b_receipts.json').write_text(json.dumps({'config':m.config,'weightSha256':hashlib.sha256((m.O/'quality_model.pkl').read_bytes()).hexdigest(),'records':receipts},ensure_ascii=False,indent=2))
