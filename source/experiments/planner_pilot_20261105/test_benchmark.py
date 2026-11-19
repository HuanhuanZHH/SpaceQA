import unittest
import json
from pathlib import Path
import tempfile
from benchmark import reference, tool, score


class ReferenceTests(unittest.TestCase):
    def test_missing_image_does_not_imply_dimension_failure(self):
        expected, _ = reference({'dimension_declarations': [{'width': 10, 'height': 10}]},
                                {'event_records': [], 'pixel_envelopes': []},
                                {'exists': False, 'name': 'a.png'}, 'EPB_seg_sample')
        self.assertEqual(expected['missing_data'], 'fail')
        self.assertEqual(expected['declaration_conflict'], 'not_applicable')
        self.assertEqual(expected['record_domain_violation'], 'not_applicable')

    def test_category_set_and_count_are_separate(self):
        header = {'event_summary': {'event_count': 1, 'has_events': True, 'event_types': ['AR']},
                  'dimension_declarations': []}
        records = {'event_records': [{'category': 'CH'}], 'pixel_envelopes': []}
        expected, _ = reference(header, records, {'exists': True, 'name': 'x.png'}, 'AIA_AR_sample')
        self.assertEqual(expected['declaration_conflict'], 'fail')

    def test_ulf_time_is_relative_to_observation_origin(self):
        header = {'start_time': '2026-01-01T00:00:00Z', 'sampling_rate_hz': 1,
                  'n_points': 11, 'dimension_declarations': []}
        records = {'event_records': [{'start': '2026-01-01T00:00:02', 'end': '2026-01-01T00:00:11'}],
                   'pixel_envelopes': []}
        facts = {'exists': True, 'name': 'a.csv', 'measured_rate_hz': 1, 'n_rows': 11, 'first_t': 0, 'last_t': 10}
        expected, _ = reference(header, records, facts, 'ULF_sample')
        self.assertEqual(expected['declaration_conflict'], 'pass')
        self.assertEqual(expected['record_domain_violation'], 'fail')

    def test_inspection_requires_issued_handle(self):
        task = {'task_id': 'PILOT-abc', 'facts': {'exists': True, 'name': 'x.png'}}
        result, _ = tool(task, 'inspect_data', {'data_handle': 'DATA-abc'}, False)
        self.assertIn('error', result)
        _, resolved = tool(task, 'resolve_data', {}, False)
        result, _ = tool(task, 'inspect_data', {'data_handle': 'DATA-abc'}, resolved)
        self.assertNotIn('error', result)

    def test_guessed_correct_answer_is_not_evidence_qualified(self):
        task = {'checks': ['missing_data'], 'expected': {'missing_data': 'pass'}, 'dependencies': [],
                'source': {'family': 'EPB_seg_sample'}, 'header': {}}
        result = {'answer': {'checks': {'missing_data': {'status': 'pass', 'evidence_ids': ['E99']}}}, 'tools': []}
        score(task, result)
        self.assertTrue(result['exact_correct'])
        self.assertFalse(result['evidence_qualified_correct'])

    def test_independent_cme_audit_does_not_invent_missing_class_defect(self):
        from audit_sources import independently_recompute, sha
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory)/'sample.json'
            data = Path(directory)/'absent.png'
            label.write_text(json.dumps({'hek_event_annotation': {'event_count':1, 'has_events':True,
                'event_types':['CME'], 'matched_events':[{'cme_time':'2026-01-01T00:00:00'}]}}))
            task = {'task_id':'CME-optional-type', 'expected':{'declaration_conflict':'pass'},
                    'facts':{},'source':{'label_path':str(label),'data_path':str(data),
                    'label_sha256':sha(label),'data_sha256':None,'family':'CME_C2_sample'}}
            self.assertEqual(independently_recompute(task)['independently_recomputed']['declaration_conflict'],'pass')

    def test_format_normalization_keeps_wrong_verdict_and_first_answer(self):
        from evaluate_outputs import terminal_checks
        result={'calls':[{'content':json.dumps({'type':'json_object','checks':{'missing_data':{'status':'fail'}}})},
                         {'content':json.dumps({'action':'finish','checks':{'missing_data':{'status':'pass'}}})}]}
        checks,meta=terminal_checks(result,['missing_data'])
        self.assertEqual(checks['missing_data']['status'],'fail')
        self.assertEqual(meta['call_index'],0)
        self.assertTrue(meta['syntax_normalized'])


if __name__ == '__main__':
    unittest.main()
