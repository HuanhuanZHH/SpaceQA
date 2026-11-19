import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from spaceqa_agent.chat_scoring import evaluate,sample_checks

class SixObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.data=self.root/'example.png';self.data.write_bytes(b'\x89PNG\r\n\x1a\n'+b'\0'*8+struct.pack('>II',64,64))
        self.label=self.root/'example.json'
        self.ann=dict(label_header=dict(fileName='example.png',labelVersion='1',labelCreateTime='2026-11-06T12:00:00',annotationMethod='manual',annotator='anonymous'),basic_annotation=dict(fileName='example.png',modality='image'),hek_event_annotation=dict(event_count=1,event_types=['AR'],matched_events=[dict(event_type='AR',bbox_polygon_pixel=[[1,1],[9,1],[9,9],[1,9]])]))
        self.row=dict(sampleId='AIA:example',family='AIA_AR_sample',labelPath=str(self.label),dataPath=str(self.data),observationInterval=['2020-01-01T00:00:00','2020-01-01T00:00:00'],checks={'L2-F':{'scopeStatus':'not_applicable'}})
        self.write()
    def tearDown(self):self.temp.cleanup()
    def write(self):
        self.label.write_text(json.dumps(self.ann),encoding='utf-8');self.row['labelSha256']=hashlib.sha256(self.label.read_bytes()).hexdigest()
    def test_six_scores_are_reproducible_with_explicit_scope(self):
        profile=evaluate([self.row],self.root)
        self.assertEqual(len(profile['dimensions']),6)
        self.assertTrue(all(d['displayScore']==100 for d in profile['dimensions'].values()))
        self.assertTrue(all(d['formalScore'] is None for d in profile['dimensions'].values()))
    def test_declared_distribution_conflict_changes_actual_score(self):
        self.ann['hek_event_annotation']['event_count']=2;self.write()
        profile=evaluate([self.row],self.root)
        self.assertAlmostEqual(profile['dimensions']['coverage_distribution']['displayScore'],200/3)
        self.assertEqual(profile['dimensions']['documentation_provenance']['displayScore'],100)
    def test_unknown_is_not_pass_and_remains_in_applicable_denominator(self):
        del self.ann['basic_annotation']['modality'];self.write()
        d=evaluate([self.row],self.root)['dimensions']['usage_conditions']
        self.assertEqual(d['scopeChecksApplicable'],3);self.assertEqual(d['scopeChecksCompleted'],2)
        self.assertEqual(d['scopeCounts']['undetermined'],1);self.assertAlmostEqual(d['scoreBounds'][0],200/3)
    def test_missing_file_is_measured_failure_not_unknown_or_perfect(self):
        self.data.unlink();p=evaluate([self.row],self.root)
        self.assertEqual(p['dimensions']['completeness']['displayScore'],50)
        self.assertLess(p['dimensions']['usage_conditions']['displayScore'],100)
    def test_outside_paths_are_not_read(self):
        r=dict(self.row,labelPath=str(self.root.parent/'foreign.json'))
        self.assertTrue(all(c['status']=='undetermined' for c in sample_checks(r,self.root)))
    def test_hash_disagreement_is_evidence_failure(self):
        self.row['labelSha256']='0'*64
        d=evaluate([self.row],self.root)['dimensions']['documentation_provenance']
        self.assertEqual(d['displayScore'],80)

    def test_timeseries_has_annotation_score_without_inventing_geometry(self):
        self.row['family']='ULF_sample'
        del self.ann['hek_event_annotation']
        self.ann['ulf_event_annotation']=dict(event_count=1,event_types=['Pi2'],matched_events=[dict(kind='Pi2',start_dt='2020-01-01T00:00:00',end_dt='2020-01-01T00:10:00')])
        self.write();checks={r['checkId']:r for r in sample_checks(self.row,self.root)}
        self.assertEqual(checks['S6-A2']['status'],'not_applicable')
        self.assertEqual(checks['S6-A3']['status'],'pass')
        self.assertEqual(checks['S6-D2']['status'],'pass')
        self.assertEqual(evaluate([self.row],self.root)['dimensions']['annotation_quality']['displayScore'],100)

    def test_reversed_or_ambiguous_event_endpoints_are_not_passed(self):
        self.ann['hek_event_annotation']['matched_events'][0].update(start_dt='2020-01-01T00:10:00',end_dt='2020-01-01T00:00:00')
        self.write();check=next(r for r in sample_checks(self.row,self.root) if r['checkId']=='S6-A3')
        self.assertEqual(check['status'],'fail')
        self.ann['hek_event_annotation']['matched_events'][0].update(start_dt='2020-01-01T00:00:00Z',end_dt='2020-01-01T00:10:00')
        self.write();check=next(r for r in sample_checks(self.row,self.root) if r['checkId']=='S6-A3')
        self.assertEqual(check['status'],'undetermined')

if __name__=='__main__':unittest.main()
