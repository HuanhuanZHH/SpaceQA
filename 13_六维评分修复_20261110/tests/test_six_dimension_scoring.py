import json,tempfile,unittest
from pathlib import Path
from spaceqa_agent.formal_checks import summarize_dimension,apply_formal_dimensions,composite,DIMENSION_IDS,GROUPS,measure,run,digest
class SixDimensionScoringTest(unittest.TestCase):
 def test_complete_unequal_weights_and_state(self):
  x=summarize_dimension([{'checkId':'C45','status':'pass','q':1},{'checkId':'C46','status':'fail','q':.5}],['C45','C46'],{'C45':1,'C46':3})
  self.assertEqual(x['score'],62.5);self.assertEqual(x['state'],'fail');self.assertEqual(x['denominator'],2);self.assertEqual(x['completedChecks'],2)
 def test_pending_does_not_renormalize(self):
  x=summarize_dimension([{'checkId':'C45','status':'pass','q':1}],['C45','C46']);self.assertIsNone(x['score']);self.assertEqual(x['checkCoverage'],.5);self.assertEqual(x['pendingChecks'],['C46']);self.assertEqual(x['state'],'undetermined')
 def test_explicit_na_removed(self):
  x=summarize_dimension([{'checkId':'C43','status':'pass','q':.8},{'checkId':'C44','status':'not_applicable'}],['C43','C44']);self.assertEqual(x['score'],80);self.assertEqual(x['denominator'],1)
 def test_all_na_is_not_perfect(self):
  x=summarize_dimension([{'checkId':'C44','status':'not_applicable'}],['C44']);self.assertIsNone(x['score']);self.assertIsNone(x['checkCoverage']);self.assertEqual(x['state'],'not_applicable')
 def test_invalid_utility_stays_pending(self):
  for v in [None,float('nan'),float('inf'),True,1.1,-.1]:
   x=summarize_dimension([{'checkId':'C45','status':'pass','q':v}],['C45']);self.assertIsNone(x['score']);self.assertEqual(x['state'],'undetermined')
 def test_duplicate_check_rejected(self):
  with self.assertRaises(ValueError):summarize_dimension([{'checkId':'C45','status':'pass','q':1}]*2,['C45'])
 def test_integration_updates_stale_denominators_and_state(self):
  dims=[{'id':k,'score':None,'state':'undetermined','denominator':0,'completedChecks':0} for k in DIMENSION_IDS]
  checks=[{'checkId':c,'status':'pass','q':1} for ids in GROUPS.values() for c in ids];apply_formal_dimensions(dims,checks,{})
  for d in dims[3:]:self.assertEqual(d['score'],100);self.assertEqual(d['state'],'pass');self.assertGreater(d['denominator'],0);self.assertEqual(d['completedChecks'],d['denominator'])
  self.assertIsNone(dims[1]['score'])
 def policy(self):return {'weights':dict.fromkeys(DIMENSION_IDS,1),'basis':'synthetic arithmetic fixture only','approvedBy':'TEST_FIXTURE_NOT_REAL_APPROVAL','approvalStatus':'approved'}
 def test_all_six_fixture_75(self):self.assertEqual(composite(dict(zip(DIMENSION_IDS,[100,90,80,70,60,50])),self.policy())['score'],75)
 def test_unresolved_front_dimension_blocks(self):
  d=dict.fromkeys(DIMENSION_IDS,100);d['annotationQuality']=None;r=composite(d,self.policy());self.assertIsNone(r['score']);self.assertEqual(r['pendingDimensions'],['annotationQuality'])
 def test_draft_policy_blocks_even_if_name_filled(self):
  for status in ('draft', 'unverified', None):
   with self.subTest(status=status):
    policy = self.policy()
    if status is None:
     del policy['approvalStatus']
    else:
     policy['approvalStatus'] = status
    self.assertIsNone(composite(dict.fromkeys(DIMENSION_IDS, 100), policy)['score'])
 def test_wrong_dimension_names_block(self):
  d=dict.fromkeys('abcdef',100);p=self.policy();p['weights']=dict.fromkeys(d,1);self.assertIsNone(composite(d,p)['score'])
 def test_nonfinite_weights_rejected(self):
  p=self.policy();p['weights']['completeness']=True
  with self.assertRaises(ValueError):composite(dict.fromkeys(DIMENSION_IDS,100),p)
 def test_external_evidence_does_not_modify_input(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)/'input';r.mkdir();e=Path(t)/'protocol';e.mkdir();(r/'x.json').write_text('{}');(e/'rows.json').write_text(json.dumps([{'path':'x.json','format':'json'}]));cfg={'version':'fixture','basis':'fixture','records':'rows.json'}
   self.assertEqual(measure(r,'C50',cfg,e)['status'],'pass');self.assertFalse((r/'target_protocol.json').exists())
 def test_sidecar_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);e=r/'e';e.mkdir()
   with self.assertRaises(ValueError):measure(r,'C50',{'version':'fixture','basis':'fixture','records':'../secret'},e)
 def test_integer_measurement_uses_tolerance(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'rows.json').write_text(json.dumps([{'declared':10,'measured':11,'tolerance':1,'toleranceBasis':'fixture boundary'}]));x=measure(r,'C47',{'version':'fixture','basis':'fixture','records':'rows.json'});self.assertEqual(x['status'],'pass')
 def test_event_ts_and_event_present(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)/'input';r.mkdir();out=Path(t)/'out';run(r,out,['C41'],{'datasetId':'fixture'})
   for row in (json.loads(x) for x in (out/'run.log.jsonl').read_text().splitlines()):self.assertEqual(row['ts'],row['timestamp']);self.assertIn(row['event'],['check_started','check_completed','check_failed'])

class ProtocolPathMappingTest(unittest.TestCase):
 def test_flat_and_nested_preserved(self):
  from spaceqa_agent.protocol_tools import source_relative_path as rel
  self.assertEqual(rel('/input/pkg/a.csv.json','pkg'),'a.csv.json')
  self.assertEqual(rel('/input/hash/pkg/label/a.png.json','pkg'),'label/a.png.json')
 def test_unknown_or_escaping_path_rejected(self):
  from spaceqa_agent.protocol_tools import source_relative_path as rel
  for p in ['/input/other/a','/input/pkg/../a','/input/pkg/pkg/a']:
   with self.assertRaises(ValueError):rel(p,'pkg')

class PublisherIntegrationTest(unittest.TestCase):
 def test_all_ten_checks_and_six_score_with_explicit_fixture_protocol(self):
  import importlib.util
  module_path=Path(__file__).resolve().parents[1]/'publish_run.py';spec=importlib.util.spec_from_file_location('candidate_publisher',module_path);publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
  with tempfile.TemporaryDirectory() as t:
   base=Path(t);root=base/'fixture_input';root.mkdir();side=base/'protocol';side.mkdir();out=base/'output';out.mkdir();(root/'data.json').write_text('{}');(side/'review.txt').write_text('SYNTHETIC TEST FIXTURE; NO REAL APPROVAL OR LICENSE')
   checks={}
   def put(check,rows,**config):
    path=side/(check+'.json');path.write_text(json.dumps(rows));checks[check]=dict(version='synthetic-fixture',basis='synthetic test only',records=path.name,**config)
   for check in ('C41','C42'):put(check,[{'sampleId':'x','class':'AR'}],field='class',minimumCounts={'AR':1})
   put('C43',[{'class':'AR'}],field='class',targetDistribution={'AR':1},maxTV=0)
   checks['C44']={'version':'synthetic-fixture','basis':'test only','splitTask':False}
   put('C45',[{'path':'data.json','sha256':digest(root/'data.json')}],datasetId='fixture',datasetVersion='fixture')
   for check in ('C46','C48','C49'):put(check,[{'verifiedBy':'TEST_FIXTURE','evidencePath':'review.txt','evidenceSha256':digest(side/'review.txt'),'verificationStatus':'verified','required':'fixture'}],requiredFields=['required'])
   put('C47',[{'declared':1,'measured':1,'tolerance':0,'toleranceBasis':'integer fixture equality'}]);put('C50',[{'path':'data.json','format':'json'}])
   protocol={'dirId':root.name,'datasetId':'fixture','checks':checks,'composite':{'weights':dict.fromkeys(DIMENSION_IDS,1),'basis':'Synthetic arithmetic fixture only','approvedBy':'TEST_FIXTURE','approvalStatus':'approved'}};(side/'target_protocol.json').write_text(json.dumps(protocol))
   card={'version':'fixture','dimensions':[{'id':k,'score':50 if i<3 else None,'state':'undetermined'} for i,k in enumerate(DIMENSION_IDS)]}
   publisher.finish(root,out,card,[{'family':'fixture','findings':[]}],'TEST-SYNTHETIC',side/'target_protocol.json')
   result=json.loads((out/'scorecard.json').read_text());self.assertEqual(result['fullQualityScore'],75)
   self.assertEqual([d['state'] for d in result['dimensions'][3:]],['pass']*3)
   events=[json.loads(x) for x in (out/'run.log.jsonl').read_text().splitlines()];self.assertEqual(len(events),20);self.assertTrue(all('ts' in x and 'event' in x for x in events));self.assertFalse((root/'target_protocol.json').exists())

class EvidenceSemanticsTest(unittest.TestCase):
 def test_missing_expected_hash_is_unknown_not_mismatch(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'x').write_text('x');(r/'rows').write_text(json.dumps([{'path':'x','sha256':None}]))
   x=measure(r,'C45',{'version':'fixture','basis':'fixture','records':'rows','datasetId':'fixture','datasetVersion':'fixture'});self.assertEqual(x['status'],'undetermined')
 def test_restricted_license_declaration_is_not_missing(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'e').write_text('SYNTHETIC FIXTURE');(r/'rows').write_text(json.dumps([{'redistributionAllowed':False,'verifiedBy':'TEST_FIXTURE','verificationStatus':'verified','evidencePath':'e','evidenceSha256':digest(r/'e')}]))
   cfg={'version':'fixture','basis':'fixture','records':'rows','requiredFields':['redistributionAllowed']};self.assertEqual(measure(r,'C48',cfg)['status'],'pass')
   cfg['requiredTrueFields']=['redistributionAllowed'];self.assertEqual(measure(r,'C49',cfg)['status'],'fail')
 def test_unreviewed_status_is_unknown(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'e').write_text('SYNTHETIC FIXTURE');(r/'rows').write_text(json.dumps([{'x':'x','verifiedBy':'TEST_FIXTURE','verificationStatus':'pending','evidencePath':'e','evidenceSha256':digest(r/'e')}]))
   self.assertEqual(measure(r,'C46',{'version':'fixture','basis':'fixture','records':'rows','requiredFields':['x']})['status'],'undetermined')
 def test_csv_failure_after_header_is_detected(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'x.csv').write_text('a,b\n1,2\n"unfinished,3');(r/'rows').write_text(json.dumps([{'path':'x.csv','format':'csv'}]))
   self.assertEqual(measure(r,'C50',{'version':'fixture','basis':'fixture','records':'rows'})['status'],'fail')
