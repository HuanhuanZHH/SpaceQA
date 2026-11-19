import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import struct
import zlib

SCRIPT = Path(__file__).resolve().parents[1] / 'experiments/research_optimization_20261103/review_protocol.py'
SPEC = importlib.util.spec_from_file_location('review_protocol', SCRIPT)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class ReviewProtocolTests(unittest.TestCase):
    def observable_files(self, root):
        def chunk(name, data):
            return struct.pack('!I', len(data)) + name + data + struct.pack('!I', zlib.crc32(name + data) & 0xffffffff)
        image = b'\x89PNG\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 16, 12, 8, 2, 0, 0, 0))
        image += chunk(b'IDAT', zlib.compress((b'\x00' + b'\x00' * 48) * 12)) + chunk(b'IEND', b'')
        (root/'unknown.png').write_bytes(image)
        (root/'unknown.json').write_text('{}')

    def source(self, root):
        rows = []
        for family in review.FAMILIES:
            for group in range(3):
                for tier in ['P0', 'P1']:
                    rows.append(dict(id=f'{family}:{group}:{tier}', family=family, tier=tier, split='val',
                                     training_group_id=f'{family}:{group}', label_path=str(root/'unknown.json'),
                                     data_path=str(root/'unknown.png'), target=[1, 1, 1], model_score=.99))
        path = root / 'val.jsonl'
        review.write_rows(path, rows)
        return path

    def test_grouped_six_family_queue_is_blind_and_preserves_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            before = source.read_bytes()
            result = review.build_queue(source, root/'queue', 2)
            self.assertEqual(result['samples'], 12)
            mapping = review.read_rows(root/'queue/private_sampling_map.jsonl')
            self.assertEqual(len({r['source_group'] for r in mapping}), 12)
            for item in review.read_rows(root/'queue/blind_samples.jsonl'):
                self.assertNotIn('tier', item)
                self.assertNotIn('target', item)
                self.assertNotIn('model_score', item)
            merged = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(merged['unknown_samples'], 12)
            self.assertEqual(merged['consensus_samples'], 0)
            self.assertFalse(merged['scientific_metrics_computed'])
            self.assertEqual(source.read_bytes(), before)

    def test_train_test_mixture_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            rows = review.read_rows(source)
            rows[0]['split'] = 'test'
            review.write_rows(source, rows)
            with self.assertRaisesRegex(ValueError, 'all-validation'):
                review.build_queue(source, root/'queue')
            self.assertFalse((root/'queue').exists())

    def test_same_reviewer_or_disagreement_never_becomes_negative_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review.build_queue(self.source(root), root/'queue', 1)
            a = review.read_rows(root/'queue/reviewer_A.jsonl')
            for row in a:
                row.update(reviewer_id='same', review_complete=True,
                           declared_independent_of_model_predictions=True, applicable_classes=['L3-06'], defects=[])
            review.write_rows(root/'queue/reviewer_A.jsonl', a)
            review.write_rows(root/'queue/reviewer_B.jsonl', a)
            self.assertEqual(review.reconcile(root/'queue', root/'merge1')['unknown_samples'], 6)
            b = json.loads(json.dumps(a))
            for row in b:
                row['reviewer_id'] = 'different'
                row['defects'] = [dict(defect_class='L3-06', locator={'type':'bbox', 'coordinate_system':'image_pixels',
                                     'frame_id':'current-image-sha', 'image_size':[100,100], 'bbox':[0,0,10,10]})]
            review.write_rows(root/'queue/reviewer_B.jsonl', b)
            self.assertEqual(review.reconcile(root/'queue', root/'merge2')['unknown_samples'], 6)

    def test_only_complete_two_person_agreement_yields_declared_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.observable_files(root)
            review.build_queue(self.source(root), root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'], defects=[])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            result = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(result['consensus_samples'], 6)
            self.assertEqual(result['unknown_samples'], 0)
            self.assertFalse(result['official_reference'])

    def test_missing_data_cannot_create_negative_visual_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'unknown.json').write_text('{}')
            review.build_queue(self.source(root), root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'], defects=[])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            result = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(result['unknown_samples'], 6)
            pending = review.read_rows(root/'merged/requires_review_or_adjudication.jsonl')
            self.assertTrue(all(r['reason'] == 'reviewed_classes_require_missing_observation' for r in pending))

    def test_locator_dimensions_must_match_actual_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.observable_files(root)
            review.build_queue(self.source(root), root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'],
                               defects=[dict(defect_class='L3-06', locator=dict(type='bbox',
                                   coordinate_system='image_pixels', image_size=[100, 100], bbox=[0, 0, 10, 10],
                                   frame_id=review.sha(root/'unknown.png')))])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            result = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(result['unknown_samples'], 6)
            pending = review.read_rows(root/'merged/requires_review_or_adjudication.jsonl')
            self.assertTrue(all(r['reason'] == 'reviewed_locator_dimensions_do_not_match_source' for r in pending))

    def test_unreadable_image_cannot_create_negative_visual_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'unknown.png').write_bytes(b'not-a-decodable-image')
            (root/'unknown.json').write_text('{}')
            review.build_queue(self.source(root), root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'], defects=[])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            result = review.reconcile(root/'queue', root/'merged')
            # This test targets the five image families; ULF observability is
            # assessed with the reviewer's explicit time-series coverage.
            self.assertEqual(result['unknown_samples'], 5)
            pending = review.read_rows(root/'merged/requires_review_or_adjudication.jsonl')
            self.assertTrue(all(r['reason'] == 'reviewed_image_observation_unreadable' for r in pending))

    def test_agreement_on_malformed_locations_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            review.build_queue(self.source(root), root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'],
                               defects=[dict(defect_class='L3-06', locator={'bbox':[0,0,10,10]})])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            result = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(result['consensus_samples'], 0)
            self.assertEqual(result['unknown_samples'], 6)

    def test_unknown_and_critical_disagreements_are_not_silently_discarded(self):
        locator = {'type':'record', 'path':'sample.json', 'record_id':'/events/0'}
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            review.canonical_defects([dict(defect_class='L3-06', locator=locator, status='unknown')])
        first = review.canonical_defects([dict(defect_class='L3-MIS', locator=locator, critical=True, subtype='undercoverage')])
        second = review.canonical_defects([dict(defect_class='L3-MIS', locator=locator, critical=False, subtype='overcoverage')])
        self.assertNotEqual(first, second)

    def test_source_change_prevents_consensus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.source(root)
            (root/'unknown.png').write_bytes(b'original-image-bytes')
            (root/'unknown.json').write_text('{}')
            review.build_queue(source, root/'queue', 1)
            for name in ['A', 'B']:
                forms = review.read_rows(root/f'queue/reviewer_{name}.jsonl')
                for row in forms:
                    row.update(reviewer_id=name, review_complete=True,
                               declared_independent_of_model_predictions=True, applicable_classes=['L3-06'], defects=[])
                review.write_rows(root/f'queue/reviewer_{name}.jsonl', forms)
            (root/'unknown.png').write_bytes(b'different-image-bytes')
            result = review.reconcile(root/'queue', root/'merged')
            self.assertEqual(result['consensus_samples'], 0)
            self.assertEqual(result['unknown_samples'], 6)


if __name__ == '__main__':
    unittest.main()
