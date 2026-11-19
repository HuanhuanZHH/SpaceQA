import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark_temporal_routes as temporal


class TemporalRouteTests(unittest.TestCase):
    def test_timezone_normalization(self):
        self.assertEqual(temporal.timestamp("2026-01-01T08:00:00+08:00"),
                         temporal.timestamp("2026-01-01T00:00:00Z"))

    def test_regular_csv_anchors_window_but_gapped_csv_abstains(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "a.csv"
            path.write_text("t,H\n0,1\n1,2\n2,3\n", encoding="utf-8")
            value = {"basic_annotation": {"startTime": "2025-12-31T23:58:00Z",
                                           "endTime": "2026-01-01T00:02:02Z"},
                     "science_annotation": {"dateObs": "2026-01-01T00:00:00Z"}}
            row = temporal.ulf_one((root, "a.csv", value))
            self.assertTrue(row["anchor_flag"])
            self.assertEqual(row["residual_seconds"], [-120.0, 120.0])
            self.assertFalse(row["reversed"])
            path.write_text("t,H\n0,1\n1,2\n5,3\n", encoding="utf-8")
            row = temporal.ulf_one((root, "a.csv", value))
            self.assertFalse(row["anchor_flag"])
            self.assertEqual(row["state"], "irregular_time_axis_not_attributed_to_label")

    def test_cme_consensus_uses_repeated_event_not_frame_time(self):
        with tempfile.TemporaryDirectory() as temp:
            p0, p1 = Path(temp) / "P0", Path(temp) / "P1"
            p0.mkdir()
            p1.mkdir()
            for index in range(4):
                event = {"cme_time": "2026-01-01T00:00:00Z", "speed_2nd_initial": 500,
                         "speed_2nd_20R": 510, "accel_flag": "", "remarks": ""}
                value = {"basic_annotation": {"fileName": f"{index}.png", "startTime": "2026-01-01T00:15:00Z"},
                         "hek_event_annotation": {"image_time": "2026-01-01T00:00:00Z", "matched_events": [event]}}
                (p0 / f"{index}.json").write_text(json.dumps(value), encoding="utf-8")
                if index == 0:
                    event["cme_time"] = "2026-01-01T00:02:00Z"
                (p1 / f"{index}.json").write_text(json.dumps(value), encoding="utf-8")
            summary, changes = temporal.cme_routes(p0, p1)
            self.assertEqual(summary["P0"]["event_not_equal_frame_naive_flags"], 4)
            self.assertEqual(summary["P0"]["within_tier_consensus_flags"], 0)
            self.assertEqual(summary["P1"]["within_tier_consensus_flags"], 1)
            self.assertEqual(summary["P1"]["consensus_agreement_with_reference_event_changes"]["extra"], 0)
            self.assertEqual(changes[0]["delta_seconds"], 120)


if __name__ == "__main__":
    unittest.main()
