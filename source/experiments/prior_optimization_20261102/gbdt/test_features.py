import copy
import datetime as dt
import json
import unittest

import numpy as np
import train_gbdt as gbdt


def row():
    summary = dict(family="CME_C2_sample", decoded=dict(width=32, height=16), dimensions=[],
                   declared=dict(dateObs="2026-01-01T00:00:00Z"),
                   window=dict(startTime="2026-01-01T00:12:00Z", endTime="2026-01-01T00:12:00Z"),
                   events=[dict(cme_time="2026-01-01T00:01:00Z", width=120)],
                   geometry=[dict(kind="angular_sector", valid=True)], event_summary={})
    return dict(family="CME_C2_sample", features=[0.] * 256, prompt="Instructions. Current fields: " + json.dumps(summary),
                id="sample-id", target=[0,0,0], truth_valid=[True,True,True], tier="P0", input_missing=False)


def vector(record, **kwargs):
    result = gbdt.features(record, **kwargs)
    return sorted(result), np.array([result[k] for k in sorted(result)])


class FeatureTests(unittest.TestCase):
    def test_metadata_labels_paths_and_tiers_do_not_change_inputs(self):
        current = row()
        keys, before = vector(current)
        current.update(id="P4-other", tier="P4", split="test", target=[1,1,1], truth_valid=[False]*3,
                       source_key="oracle", label_path="clean/oracle.json", data_path="P0/source.png")
        current["clean_difference"] = [10,20]
        after_keys, after = vector(current)
        self.assertEqual(keys, after_keys)
        np.testing.assert_allclose(before, after, equal_nan=True)

    def test_unknown_prompt_fields_do_not_enter_whitelist(self):
        current = row()
        keys, before = vector(current)
        summary = json.loads(current["prompt"].split("Current fields: ")[1])
        summary.update(target=[1,1,1], tier="P4", source_path="clean/oracle", oracle=dict(label_error=True))
        summary["decoded"].update(verified_source_sha256="a"*64, oracle_score=1)
        summary["events"][0].update(source="P0:line1", SOL_standard="identifier", clean_error=True)
        current["prompt"] = "Current fields: " + json.dumps(summary)
        after_keys, after = vector(current)
        self.assertEqual(keys, after_keys)
        np.testing.assert_allclose(before, after, equal_nan=True)

    def test_cme_relative_time_features_are_correct(self):
        actual = gbdt.features(row(), cme_image_time="2026-01-01T00:00:00Z")
        self.assertEqual(actual["event_time.cme_time.minus_image_time.min"], 60)
        self.assertEqual(actual["event_time.cme_time.minus_window_start.max"], -660)
        self.assertEqual(actual["event_time.cme.quantization_residual_60.image_time.mean"], 0)
        self.assertEqual(actual["event_time.cme.quantization_residual_720.image_time.mean"], 60)

    def test_unknown_truth_does_not_score_as_negative(self):
        record = row()
        record.update(target=[1,0,0], truth_valid=[True,False,False])
        result = gbdt.metrics([record], np.array([[True, True, True]]))
        self.assertEqual((result["TP"],result["FP"],result["FN"],result["TN"]),(1,0,0,0))
        self.assertEqual(result["F1"],1)

    def test_no_validation_positive_suppresses_head(self):
        record = row()
        threshold, _ = gbdt.calibrate([record], np.array([[.2,.9,.95]]))
        self.assertTrue((threshold == 1.01).all())

    def test_threshold_only_uses_matching_family_known_present_rows(self):
        first = row()
        first["target"][2] = 1
        second = copy.deepcopy(first)
        second.update(target=[0,0,0], input_missing=True)
        third = copy.deepcopy(first)
        third.update(target=[0,0,0], truth_valid=[True,True,False])
        threshold, _ = gbdt.calibrate([first,second,third],np.array([[.1,.1,.4],[.1,.1,.8],[.1,.1,.9]]))
        self.assertAlmostEqual(threshold[2,2],.4)
        self.assertTrue((threshold[[0,1,3,4,5]] == 1.01).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
