"""Independent full-package evidence checks; no model or GPU dependency."""
import importlib.util
from pathlib import Path
import unittest

SOURCE=Path(__file__).resolve().parents[1]/'experiments/package_runtime_acceptance_20261103/audit_benchmark.py'
spec=importlib.util.spec_from_file_location('package_runtime_audit',SOURCE)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    routes=[dict(route='frozen',thresholds=[.6,1.01,.95])]
    record=dict(id='one',label_path='/data/one.json',data_path='/data/one.png',input_missing=False)
    candidate=dict(sample_id='one',label_path=record['label_path'],data_path=record['data_path'],
        model_findings_score_eligible=False,scientific_verdict='undetermined',
        models=[dict(**routes[0],probabilities=[.7,.99,.2],candidate_bits=[1,0,0])],
        model_candidate_bits=[1,0,0],F1=None,recall=None,precision=None)
    return candidate,record,routes


class TestWholePackageCandidateAudit(unittest.TestCase):
    def test_valid_frozen_candidates_and_disabled_head(self):
        c,r,routes=fixture();self.assertEqual(audit.check_candidate(c,r,routes),[])
        c['models'][0]['candidate_bits'][1]=1
        self.assertIn('candidate_threshold_recompute',audit.check_candidate(c,r,routes))
    def test_missing_input_cannot_have_probabilities(self):
        c,r,routes=fixture();r['input_missing']=True
        self.assertIn('missing_input_probability',audit.check_candidate(c,r,routes))
    def test_frozen_threshold_cannot_change(self):
        c,r,routes=fixture();c['models'][0]['thresholds']=[.5,1.01,.95]
        self.assertIn('frozen_threshold_changed',audit.check_candidate(c,r,routes))
    def test_no_unverified_score_or_accuracy_promotion(self):
        c,r,routes=fixture();c['F1']=.99;c['model_findings_score_eligible']=True
        errors=audit.check_candidate(c,r,routes)
        self.assertIn('candidate_promoted_to_score',errors)
        self.assertIn('unavailable_accuracy_claim',errors)


if __name__=='__main__':unittest.main()
