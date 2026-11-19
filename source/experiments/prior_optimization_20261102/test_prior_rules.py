"""Boundary tests for the experimental rules, including actual failure schemas."""
import copy
import datetime as dt
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import prior_rules as rules


def png(path, width=32, height=16):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress((b"\0" + bytes(width)) * height)) + chunk(b"IEND", b"")
    path.write_bytes(data)


def ulf_label():
    return dict(basic_annotation=dict(fileName="a.csv", startTime="2026-01-01T00:00:00Z", endTime="2026-01-01T00:00:03Z"),
                science_annotation=dict(dateObs="2026-01-01T00:00:00Z", sampling_rate_hz=1, duration_s=4, n_points=4,
                                        components=["H", "D", "Z", "T"], units="nT (H/Z/T), deg (D)"),
                ulf_event_annotation=dict(has_events=True, event_count=0, event_types=["Pi2"], matched_events=[]))


class PriorRuleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "a.png"
        png(self.image)
        self.csv = self.root / "a.csv"
        self.csv.write_text("t,H,D,Z,T\n0,1,2,3,4\n1,2,3,4,5\n2,3,4,5,6\n3,4,5,6,7\n", encoding="utf-8")

    def evaluate(self, label=None, **kwargs):
        return rules.evaluate_sample("ULF_sample", label or ulf_label(), self.csv, label_path="a.csv.json", **kwargs)

    def test_cme_no_per_event_class_is_not_empty_class_conflict(self):
        label = dict(basic_annotation=dict(startTime="2010-10-10T22:00:05", endTime="2010-10-10T22:00:05"),
                     hek_event_annotation=dict(has_events=True, event_count=1, event_types=["CME"], image_time="2010-10-10T22:12:06",
                                               matched_events=[dict(cme_time="2010-10-10T22:12:06", width=152, central_pa=92)]))
        result = rules.evaluate_sample("CME_C2_sample", label, self.image)
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["observations"]["cme_frame_offset_seconds"], -721)

    def test_ulf_original_summary_ambiguity_is_diagnostic(self):
        result = self.evaluate()
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        conflicts = [f for f in result["findings"] if f["defectType"] == "ANNOTATION_STRUCTURE_CONFLICT"]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["certainty"], "candidate")
        self.assertFalse(conflicts[0]["score_eligible"])

    def test_ulf_supported_metadata_mismatch_is_located(self):
        for key, bad in (("n_points", 7), ("duration_s", 400), ("sampling_rate_hz", 2), ("components", ["H", "D", "Z", "F"])):
            with self.subTest(key=key):
                label = ulf_label()
                label["science_annotation"][key] = bad
                result = self.evaluate(label)
                self.assertEqual(result["confirmed_bits"], [0, 0, 1])
                match = [f for f in result["findings"] if f["score_eligible"]]
                self.assertEqual(len(match), 1)
                self.assertTrue(match[0]["locator"].endswith("#/science_annotation/" + key))

    def test_regular_truncated_data_does_not_become_parameter_pollution(self):
        self.csv.write_text("t,H,D,Z,T\n0,1,2,3,4\n1,2,3,4,5\n", encoding="utf-8")
        result = self.evaluate()
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        mismatches = [f for f in result["findings"] if f["defectType"] == "L2-M"]
        self.assertEqual(len(mismatches), 2)
        self.assertTrue(all(f["certainty"] == "candidate" for f in mismatches))

    def test_entire_csv_axis_is_checked_including_last_row(self):
        self.csv.write_text("t,H,D,Z,T\n0,1,2,3,4\n1,2,3,4,5\n2,3,4,5,6\n99,4,5,6,7\n", encoding="utf-8")
        result = self.evaluate(policy=rules.Policy(trust_ulf_dateobs=True))
        self.assertFalse(result["observations"]["csv_statistics"]["regular"])
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        self.assertTrue(any(f["defectType"] == "CSV_AXIS_DIAGNOSTIC" for f in result["findings"]))

    def test_nonfinite_duplicate_and_nonmonotonic_axes_abstain(self):
        for axis in ([0, 1, "nan", 3], [0, 1, 1, 2], [0, 2, 1, 3]):
            with self.subTest(axis=axis):
                self.csv.write_text("t,H\n" + "".join(str(t) + ",1\n" for t in axis), encoding="utf-8")
                label = ulf_label()
                label["science_annotation"].pop("components")
                result = self.evaluate(label, policy=rules.Policy(trust_ulf_dateobs=True))
                self.assertEqual(result["confirmed_bits"], [0, 0, 0])
                self.assertFalse(result["observations"]["csv_statistics"]["regular"])

    def test_inclusive_and_elapsed_duration_semantics(self):
        label = ulf_label()
        label["science_annotation"]["duration_s"] = 3
        self.assertEqual(self.evaluate(label)["confirmed_bits"], [0, 0, 0])
        inclusive = self.evaluate(label, policy=rules.Policy(ulf_duration_convention="inclusive"))
        self.assertEqual(inclusive["confirmed_bits"], [0, 0, 1])
        label["science_annotation"]["duration_s"] = 4
        self.assertEqual(self.evaluate(label, policy=rules.Policy(ulf_duration_convention="inclusive"))["confirmed_bits"], [0, 0, 0])

    def test_time_anchor_requires_independent_provenance(self):
        label = ulf_label()
        label["basic_annotation"].update(startTime="2025-12-31T23:58:00Z", endTime="2026-01-01T00:02:03Z")
        weak = self.evaluate(label)
        self.assertEqual(weak["confirmed_bits"], [0, 0, 0])
        strong = self.evaluate(label, policy=rules.Policy(trust_ulf_dateobs=True))
        self.assertEqual(strong["confirmed_bits"], [0, 0, 1])
        offsets = [f["observation"]["offset_seconds"] for f in strong["findings"] if f["defectType"] == "L2-T"]
        self.assertEqual(offsets, [-120, 120])

    def test_basic_start_cannot_self_validate_time_even_if_marked_independent(self):
        label = ulf_label()
        label["basic_annotation"]["endTime"] = "2026-01-01T00:02:03Z"
        result = self.evaluate(label, time_origin=dict(value=label["basic_annotation"]["startTime"], source="basic_annotation.startTime", independent=True))
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        findings = [f for f in result["findings"] if f["defectType"] == "L2-T"]
        self.assertFalse(findings[0]["observation"]["independent_origin"])

    def test_absolute_csv_cropped_window_overrides_hourly_prior(self):
        self.csv.write_text("datetime,H,D,Z,T\n2026-01-01T08:07:30+08:00,1,2,3,4\n2026-01-01T08:07:31+08:00,1,2,3,4\n", encoding="utf-8")
        label = ulf_label()
        label["basic_annotation"].update(startTime="2026-01-01T00:07:30Z", endTime="2026-01-01T00:07:31Z")
        label["science_annotation"].update(n_points=2, duration_s=2)
        result = self.evaluate(label)
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        self.assertFalse(any(f["defectType"] == "L2-T" for f in result["findings"]))

    def test_units_and_ulf_physical_class_are_not_inferred_from_naked_csv(self):
        label = ulf_label()
        label["science_annotation"]["units"] = "pT (H/Z/T), deg (D)"
        label["ulf_event_annotation"].update(event_count=1, matched_events=[dict(kind="Pc3")])
        result = self.evaluate(label)
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        self.assertTrue(any(f["defectType"] == "ULF_UNIT_PROVENANCE_REQUIRED" for f in result["findings"]))
        self.assertTrue(any(f["defectType"] == "L2-P" and f["certainty"] == "candidate" for f in result["findings"]))

    def test_actual_image_dimensions_override_fixed_1024_prior(self):
        label = dict(basic_annotation=dict(imgSize="32x16 pixels"), science_annotation=dict(mask=dict(imageWidth=32, imageHeight=16, shapes=[])))
        result = rules.evaluate_sample("EPB_seg_sample", label, self.image)
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        label["science_annotation"]["mask"]["imageWidth"] = 16
        result = rules.evaluate_sample("EPB_seg_sample", label, self.image)
        self.assertEqual(result["confirmed_bits"], [0, 0, 1])
        self.assertTrue(any(f["pointer"] == "/science_annotation/mask/imageWidth" for f in result["findings"]))

    def test_known_categories_all_six_families_and_text_alias(self):
        for family, allowed in rules.ALLOWED.items():
            for category in allowed:
                with self.subTest(family=family, category=category):
                    label = dict(hek_event_annotation=dict(event_types=[category], matched_events=[dict(event_type=category)]))
                    path = self.csv if family == "ULF_sample" else self.image
                    self.assertFalse(any(f["defectType"] == "L2-P" for f in rules.evaluate_sample(family, label, path)["findings"]))
        label = dict(hek_event_annotation=dict(matched_events=[dict(concept="Active Region")]))
        self.assertEqual(rules.evaluate_sample("HMI_AR_sample", label, self.image)["confirmed_bits"], [0, 0, 0])

    def test_missing_data_retains_location_without_semantic_bits(self):
        result = rules.evaluate_sample("CME_C2_sample", {}, self.root / "missing.png", label_path="missing.json")
        self.assertEqual(result["confirmed_bits"], [0, 0, 0])
        finding = next(f for f in result["findings"] if f["defectType"] == "L0-01")
        self.assertTrue(finding["score_eligible"])
        self.assertEqual(finding["locator"], "missing.json#/basic_annotation/fileName")
        self.assertIsNone(finding["group"])

    def test_cross_frame_consensus_is_candidate_and_never_uses_12_minute_truth(self):
        records = []
        anchor = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        for index, offset in enumerate((-900, 0, 800)):
            event = dict(speed_2nd_initial=500, speed_2nd_20R=510, accel_flag="", remarks="",
                         cme_time=(anchor + dt.timedelta(seconds=60 if index == 0 else 0)).isoformat())
            label = dict(basic_annotation=dict(startTime=(anchor + dt.timedelta(seconds=offset)).isoformat()),
                         hek_event_annotation=dict(image_time=anchor.isoformat(), event_types=["CME"], matched_events=[event]))
            records.append(dict(family="CME_C2_sample", label=label, data_path=self.image, label_path=f"a/{index}.json", sample_id=f"old-id-{index}"))
        results = rules.evaluate_package(records)
        self.assertTrue(all(result["confirmed_bits"] == [0, 0, 0] for result in results))
        candidates = [f for result in results for f in result["findings"] if f["defectType"] == "L2-T"]
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["certainty"], "candidate")
        self.assertEqual(candidates[0]["observation"]["offset_seconds"], 60)
        self.assertEqual(results[0]["sample_id"], "old-id-0")
        # No pooling of the same physical observation across different packages.
        for index, record in enumerate(records):
            record["package"] = f"different-{index}"
        results = rules.evaluate_package(records)
        self.assertFalse(any(f["defectType"] == "L2-T" for result in results for f in result["findings"]))

    def test_cme_duplicate_identity_cannot_create_false_consensus(self):
        label = dict(basic_annotation=dict(startTime="2026-01-01T00:00:00Z"), hek_event_annotation=dict(image_time="2026-01-01T00:00:00Z"))
        event = dict(cme_time="2026-01-01T00:00:00Z", speed_2nd_initial=1, speed_2nd_20R=2, accel_flag="", remarks="")
        label["hek_event_annotation"]["matched_events"] = [event, copy.deepcopy(event)]
        records = [dict(family="CME_C2_sample", label=copy.deepcopy(label), data_path=self.image, label_path=f"a/{i}.json") for i in range(3)]
        results = rules.evaluate_package(records)
        self.assertFalse(any(f["defectType"] == "L2-T" for result in results for f in result["findings"]))

    def test_no_tier_target_or_clean_fields_influence_verdict(self):
        label = ulf_label()
        before = self.evaluate(label)
        label.update(tier="P4", clean=True, target=[1, 1, 1], official_answer="polluted")
        after = self.evaluate(label)
        self.assertEqual(before["confirmed_bits"], after["confirmed_bits"])
        self.assertEqual(before["findings"], after["findings"])

    def test_legacy_epb_relative_path_resolves_current_data_file(self):
        package = self.root / "epb"
        (package / "data").mkdir(parents=True)
        (package / "label").mkdir()
        png(package / "data" / "a.png")
        label = dict(basic_annotation=dict(fileName="old\\obsolete\\a.png", relativePath="old/path/"))
        (package / "label" / "a.json").write_text(json.dumps(label), encoding="utf-8")
        records = rules.load_package(package, "EPB_seg_sample")
        self.assertEqual(Path(records[0]["data_path"]), package / "data" / "a.png")
        self.assertFalse(any(f["defectType"] == "L0-01" for f in rules.evaluate_package(records)[0]["findings"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
