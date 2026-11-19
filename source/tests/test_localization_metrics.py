import unittest

from spaceqa_agent.localization_metrics import evaluate_localization, bbox_iou, interval_overlap


class LocalizationMetricTests(unittest.TestCase):
    def test_bbox_hit_uses_iou_or_center_tolerance(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        result = evaluate_localization({"bbox": [0, 0, 10, 10]}, {"bbox": [1, 1, 11, 11]}, image_size=[100, 100])
        self.assertTrue(result["hit"])
        self.assertEqual(result["status"], "confirmed")

    def test_no_reference_stays_candidate(self):
        result = evaluate_localization({"bbox": [0, 0, 10, 10]}, None, image_size=[100, 100])
        self.assertEqual(result["status"], "candidate")
        self.assertIsNone(result["hit"])

    def test_time_interval_uses_endpoint_and_overlap_tolerance(self):
        result = evaluate_localization({"timeInterval": [1, 10]}, {"timeInterval": [0, 10]}, kind="time_interval")
        self.assertTrue(result["hit"])
        self.assertAlmostEqual(interval_overlap([1, 10], [0, 10])["overlap_ratio"], 0.9)


if __name__ == "__main__":
    unittest.main()
