"""Independent guardrails for frozen-policy metric reconstruction."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "dispatch_metrics", Path(__file__).resolve().parents[1] / "scripts/recompute_dispatch.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DispatchMetricsTests(unittest.TestCase):
    def test_rule_can_recover_model_false_negative(self):
        p = MODULE.combine([0.1, 0.8, 0.2], [0.5] * 3, [True, False, False])
        self.assertEqual(p, [True, True, False])
        self.assertEqual(MODULE.count_bits([1, 0, 1], [1, 1, 1], p), [1, 1, 1, 0])

    def test_unknown_is_not_negative_and_missing_still_has_rules(self):
        p = MODULE.combine(None, [0.5] * 3, [False, True, False])
        self.assertEqual(MODULE.count_bits([0, 1, 0], [False, True, True], p), [1, 0, 0, 1])

    def test_no_positive_recall_and_f1_are_undefined(self):
        m = MODULE.metrics([0, 2, 0, 8])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["F1"])
        self.assertEqual(m["precision"], 0.0)

    def test_counts_not_mean_determine_micro_recall(self):
        self.assertAlmostEqual(MODULE.metrics([8, 4, 2, 10])["recall"], 0.8)
        self.assertAlmostEqual(MODULE.metrics([8, 4, 2, 10])["F1"], 16 / 22)

    def test_invalid_probabilities_and_path_escape_fail(self):
        with self.assertRaises(ValueError):
            MODULE.combine([float("nan"), 0, 0], [0.5] * 3, [False] * 3)
        with self.assertRaises(ValueError):
            MODULE.inside(Path("."), "../outside.pt")

    def test_rule_mapping_keeps_nonsemantic_missing_file_separate(self):
        self.assertIsNone(MODULE.rule_group("L0-03"))
        self.assertEqual(MODULE.rule_group("L2-G"), 0)
        self.assertEqual(MODULE.rule_group("L3-07"), 1)
        self.assertEqual(MODULE.rule_group("L2-T"), 2)


if __name__ == "__main__":
    unittest.main()
