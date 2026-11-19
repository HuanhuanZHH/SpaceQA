import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import compare_headwise as h


class HeadwiseTests(unittest.TestCase):
    def rec(self, tier, y, valid=True):
        return dict(tier=tier, target=[y, 0, 0], truth_valid=[valid, True, True])

    def test_sweep_ties_unknown_bits_and_forced_rules(self):
        records = [self.rec('P0', 0), self.rec('P0', 0), self.rec('P1', 1),
                   self.rec('P2', 1, False), self.rec('P3', 0)]
        probs = [[.8, 0, 0], [.2, 0, 0], [.8, 0, 0], [.99, 0, 0], [.8, 0, 0]]
        options = h.candidates(records, {'m': probs}, [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], 0, {'m': 10}, .05)
        result = next(r for r in options if r['route'] == 'm' and r['threshold'] == .8)
        self.assertEqual([result[k] for k in ['TP', 'FP', 'FN', 'TN']], [1, 1, 0, 0])
        self.assertEqual(result['P0_false_positive_samples'], 2)
        self.assertEqual(result['allowed_P0_head_FPR'], .5)
        self.assertLessEqual(h.select(options, 'f1')['P0_head_FPR'], .5)

    def test_review_recall_target(self):
        records = [self.rec('P0', 0), self.rec('P1', 1), self.rec('P2', 0)]
        options = h.candidates(records, {'m': [[.1, 0, 0], [.7, 0, 0], [.2, 0, 0]]}, [[0] * 3] * 3, 0, {'m': 10}, .05)
        best = h.select(options, 'high_recall_review')
        self.assertEqual(best['recall'], 1.)
        self.assertEqual(best['F1'], 1.)
        self.assertTrue(best['recall_target_achieved'])

    def test_infeasible_recall_is_explicit(self):
        records = [self.rec('P0', 0), self.rec('P1', 1)]
        options = h.candidates(records, {'m': [[.9, 0, 0], [.8, 0, 0]]}, [[0] * 3] * 2, 0, {'m': 10}, .05)
        best = h.select(options, 'high_recall_review')
        self.assertFalse(best['recall_target_achieved'])
        self.assertIn('infeasible', best['selection_note'])

    def test_no_validation_positives_not_scored_as_perfect(self):
        records = [self.rec('P0', 0), self.rec('P1', 0)]
        options = h.candidates(records, {'m': [[.1, 0, 0], [.1, 0, 0]]}, [[0] * 3] * 2, 0, {'m': 10}, .05)
        best = h.select(options, 'high_recall_review')
        self.assertEqual(best['route'], 'rules')
        self.assertIsNone(best['recall'])
        self.assertIsNone(best['recall_target_achieved'])
        self.assertFalse(best['recall_target_applicable'])


if __name__ == '__main__':
    unittest.main()
