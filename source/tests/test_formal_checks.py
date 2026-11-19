import unittest,tempfile,json
from pathlib import Path
from spaceqa_agent.formal_checks import measure,composite,digest
class FormalChecksTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.cfg={'version':'test','basis':'fixture frozen test protocol','records':'rows.json'}
 def tearDown(self):self.tmp.cleanup()
 def rows(self,rows): (self.root/'rows.json').write_text(json.dumps(rows))
 def test_missing_targets_not_perfect(self):
  for n in range(41,51):self.assertEqual(measure(self.root,'C'+str(n),{})['status'],'undetermined')
 def test_counts_deduplicate(self):
  self.rows([{'sampleId':'a','class':'AR'}]*2);c=dict(self.cfg,field='class',minimumCounts={'AR':2});r=measure(self.root,'C41',c);self.assertEqual(r['status'],'fail');self.assertEqual(r['q'],0)
 def test_tv_boundary(self):
  self.rows([{'class':'AR'},{'class':'B'}]);c=dict(self.cfg,field='class',targetDistribution={'AR':1.},maxTV=.5);self.assertEqual(measure(self.root,'C43',c)['status'],'pass');c['maxTV']=.49;self.assertEqual(measure(self.root,'C43',c)['status'],'fail')
 def test_leakage_by_group_or_hash(self):
  self.rows([dict(sampleId='a',split='train',sourceGroup='g',sha256='1'),dict(sampleId='b',split='test',sourceGroup='g',sha256='2')]);r=measure(self.root,'C44',dict(self.cfg,requireIndependentGroups=True));self.assertEqual(r['q'],0);self.assertEqual(r['status'],'fail')
 def test_no_split_explicit(self):self.assertEqual(measure(self.root,'C44',dict(self.cfg,splitTask=False))['status'],'not_applicable')
 def test_hash_actual_bytes(self):
  (self.root/'data').write_text('x');self.rows([{'path':'data','sha256':digest(self.root/'data')}]);c=dict(self.cfg,datasetId='d',datasetVersion='1');self.assertEqual(measure(self.root,'C45',c)['status'],'pass');(self.root/'data').write_text('y');self.assertEqual(measure(self.root,'C45',c)['status'],'fail')
 def test_license_requires_review(self):
  self.rows([{'license':'open'}]);self.assertEqual(measure(self.root,'C48',dict(self.cfg,requiredFields=['license']))['status'],'undetermined')
 def test_numeric_tolerance_missing(self):
  self.rows([{'declared':1.,'measured':1.}]);self.assertEqual(measure(self.root,'C47',self.cfg)['status'],'undetermined')
 def test_actual_decode(self):
  (self.root/'data.json').write_text('bad');self.rows([{'path':'data.json','format':'json'}]);self.assertEqual(measure(self.root,'C50',self.cfg)['status'],'fail');(self.root/'data.json').write_text('{}');self.assertEqual(measure(self.root,'C50',self.cfg)['status'],'pass')
 def test_path_traversal_rejected(self):
  self.rows([{'path':'../secret','sha256':'a'}]);self.assertRaises(ValueError,measure,self.root,'C45',dict(self.cfg,datasetId='d',datasetVersion='1'))
 def test_composite_gate_and_example(self):
  d=dict(zip('abcdef',[100,90,80,70,60,50]));p={'weights':dict.fromkeys(d,1),'basis':'test','approvedBy':'fixture'};self.assertEqual(composite(d,p)['score'],75);d['a']=None;self.assertIsNone(composite(d,p)['score']);self.assertIsNone(composite(d,{})['score'])
