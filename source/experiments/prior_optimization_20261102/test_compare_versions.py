import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('comparison_candidate', Path(__file__).with_name('compare_versions.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ComparisonTests(unittest.TestCase):
    def record(self, tier, target=(0, 0, 0), valid=(True, True, True)):
        return dict(family=c.FAMILIES[0], tier=tier, target=target,
                    truth_valid=valid, input_missing=False)

    def test_unknown_truth_is_excluded(self):
        self.assertIsNone(c.count_bit(1, False, True))
        self.assertEqual([c.count_bit(y, True, p) for y, p in ((1, 1), (0, 1), (1, 0), (0, 0))], list(range(4)))

    def test_f1_and_zero_denominators(self):
        self.assertAlmostEqual(c.metrics([3, 2, 1, 4])['F1'], 2 / 3)
        self.assertIsNone(c.metrics([0, 0, 0, 0])['F1'])

    def test_deterministic_rules_missing_model_fallback(self):
        records = [self.record('P1'), self.record('P2')]
        choices = {c.FAMILIES[0]: dict(route='m', thresholds=[.5] * 3)}
        actual = c.decisions(records, {'m': [None, [.9, .1, .9]]}, [[0, 0, 1], [0, 1, 0]], choices)
        self.assertEqual(actual, [[False, False, True], [True, True, True]])

    def test_gc_gate_cannot_promote_model_d(self):
        choices = {c.FAMILIES[0]: dict(route='m', thresholds=[.5] * 3)}
        actual = c.decisions([self.record('P1')], {'m': [[.9, .1, .99]]}, [[0, 0, 0]], choices, gc_only=True)
        self.assertEqual(actual, [[True, False, False]])

    def test_replace_legacy_d_retain_gc(self):
        self.assertEqual(c.retained_prior([[0, 0, 0]], [[1, 1, 1]]), [[True, True, 0]])

    def test_false_positive_cap_prevents_bad_model(self):
        records = [self.record('P0')]
        records += [self.record(t, (1, 0, 0)) for t in c.TIERS[1:]]
        result = c.calibrate_family(records, [[.99, 0, 0]] * 5, [[False] * 3 for _ in records], gc_only=True, cap=.02)
        self.assertEqual(result['P0_sample_FPR'], 0)
        self.assertEqual(result['thresholds'][0], 1.01)

    def test_calibration_finds_predictive_signal(self):
        records = [self.record('P0')]
        records += [self.record(t, (1, 0, 0)) for t in c.TIERS[1:]]
        result = c.calibrate_family(records, [[.1, 0, 0]] + [[.8, 0, 0]] * 4,
                                    [[False] * 3 for _ in records], gc_only=True, cap=.02)
        self.assertEqual(result['macro_F1'], 1.)
        self.assertEqual(result['P0_sample_FPR'], 0)
        self.assertEqual(result['thresholds'][0], .8)

    def test_all30_and_group_counts(self):
        records, predicted = [], []
        for family in c.FAMILIES:
            for tier in c.TIERS:
                record = self.record(tier, (tier != 'P0', 0, 0))
                record['family'] = family
                records.append(record)
                predicted.append([tier != 'P0', False, False])
        records[-1]['truth_valid'] = [False, True, True]
        summary, packages, groups = c.summarize(records, predicted, 'toy')
        self.assertEqual(len(packages), 30)
        self.assertEqual(len(groups), 90)
        self.assertEqual(summary['overall']['TP'], 23)
        self.assertEqual(summary['overall']['TN'], 48)
        self.assertEqual(summary['overall']['F1'], 1.)
        self.assertEqual(summary['P0_sample_FPR'], 0)


if __name__ == '__main__':
    unittest.main()
