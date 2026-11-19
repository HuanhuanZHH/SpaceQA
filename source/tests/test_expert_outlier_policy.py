import unittest

from spaceqa_agent.expert_robustness import robust_fuse


class ExpertOutlierTests(unittest.TestCase):
    def test_mad_audit_records_extreme_expert(self):
        result = robust_fuse({"a": 0.5, "b": 0.51, "c": 0.49, "outlier": 0.99})
        self.assertIn("outlier", [item["name"] for item in result["excludedExperts"]])
        self.assertLess(result["probability"], 0.6)

    def test_small_ensemble_is_not_silently_trimmed(self):
        result = robust_fuse({"a": 0.1, "b": 0.9})
        self.assertEqual(result["excludedExperts"], [])
        self.assertEqual(result["statistics"]["keptCount"], 2)


if __name__ == "__main__":
    unittest.main()
