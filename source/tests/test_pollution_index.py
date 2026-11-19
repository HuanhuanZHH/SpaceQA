import unittest

from spaceqa_agent.pollution_metrics import compute_pollution_index, rank_pollution


class PollutionIndexTests(unittest.TestCase):
    def test_threshold_and_overlap_are_explicit(self):
        records = [
            {"sampleId": "a", "defectProbabilities": {"L3-06": 0.8, "L3-07": 0.9}},
            {"sampleId": "b", "defectProbabilities": {"L3-06": 0.2}},
        ]
        result = compute_pollution_index(records, sample_count=2)
        self.assertAlmostEqual(result["defects"]["L3-06"]["pollutionIndexContribution"], 40.0)
        self.assertAlmostEqual(result["defects"]["L3-07"]["pollutionIndexContribution"], 45.0)
        self.assertEqual(result["pollutionIndex"], 85.0)

    def test_confirmed_finding_is_one_for_observation_only(self):
        result = compute_pollution_index([{"sampleId": "a", "confirmedCodes": ["L0-01"]}], sample_count=1)
        self.assertEqual(result["defects"]["L0-01"]["probabilitySum"], 1.0)
        self.assertIn("官方污染率", result["meaning"])

    def test_rank_keeps_ties(self):
        ranked = rank_pollution([{"datasetId": "a", "pollutionIndex": 2}, {"datasetId": "b", "pollutionIndex": 2}, {"datasetId": "c", "pollutionIndex": 4}])
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[1]["rank"], 1)
        self.assertIn("b", ranked[0]["tiedDatasetIds"])


if __name__ == "__main__":
    unittest.main()
