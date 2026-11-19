import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opc_agent
from fixture_factory import build_fixture, write_json, write_png


class AgentTests(unittest.TestCase):
    def test_embedded_annex_catalog_is_available(self):
        tiers = opc_agent.load_tiers(ROOT)
        self.assertEqual(len({key for key in tiers if key in {
            "HMI_AR_sample_6yaatkycjv", "AIA_AR_sample_zar6xq3d6q", "ULF_sample_tsp8ilm7h4"
        }}), 3)
        self.assertEqual(tiers["HMI_AR_sample_6yaatkycjv"]["tier"], "clean")

    def test_reference_aware_pipeline_detects_injected_families(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, corrupt = build_fixture(base)
            output = base / "output"
            summary = opc_agent.run(publish, output, 7, set(opc_agent.ALL_CHECKS))
            self.assertEqual(summary["datasetCount"], 2)
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            actual = {row["defectType"] for row in rows if row["datasetId"] == corrupt.name and row["status"] == opc_agent.STATUS_FAIL}
            expected = {"L0-03", "L2-F", "L2-P", "L2-T", "L2-G", "L3-MIS", "L3-07", "SAMPLE_DUPLICATE"}
            self.assertTrue(expected <= actual, expected - actual)
            orphan = next(row for row in rows if row["datasetId"] == corrupt.name and row["defectType"] == "L0-03")
            self.assertEqual(orphan["candidateDefectTypes"], ["L0-01", "L0-03"])
            self.assertEqual(orphan["classificationStatus"], "ambiguous_official_definition")
            # The fixture removes b.png as well as its event annotation. With
            # no current image, the semantic omission cannot be confirmed.
            pending = [row for row in rows if row["datasetId"] == corrupt.name
                       and row["defectType"] == "REFERENCE_IMAGE_ALIGNMENT_REQUIRED"]
            self.assertTrue(pending)
            self.assertNotIn("L3-06", actual)

    def test_official_visualization_schemas_are_compared(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish = base / "publish"
            ulf_clean = publish / "ULF_sample_aaaaaaaaaa"
            ulf_corrupt = publish / "ULF_sample_bbbbbbbbbb"
            epb_clean = publish / "EPB_seg_sample_aaaaaaaaaa"
            epb_corrupt = publish / "EPB_seg_sample_bbbbbbbbbb"
            for dataset in (ulf_clean, ulf_corrupt, epb_clean, epb_corrupt):
                (dataset / "data").mkdir(parents=True)
                (dataset / "label").mkdir(parents=True)

            ulf_event = {
                "start_dt": "2026-01-01T00:10:00",
                "end_dt": "2026-01-01T00:20:00",
                "kind": "Pi2",
                "source": "events.jsonl:1",
            }
            for dataset, event in (
                (ulf_clean, ulf_event),
                (ulf_corrupt, {**ulf_event, "start_dt": "2026-01-01T00:11:00", "kind": "WRONG"}),
            ):
                (dataset / "data" / "a.csv").write_text("t,H,D,Z,T\n0,1,2,3,4\n", encoding="utf-8")
                value = {
                    "label_header": {"fileName": "a.csv"},
                    "basic_annotation": {"fileName": "a.csv", "startTime": "2026-01-01T00:00:00", "endTime": "2026-01-01T00:59:59"},
                    "science_annotation": {"sampling_rate_hz": 1, "n_points": 1},
                    "ulf_event_annotation": {"has_events": True, "event_count": 1, "event_types": [event["kind"]], "matched_events": [event]},
                }
                write_json(dataset / "label" / "a.csv.json", value)

            clean_shape = {"label": "EPB", "shape_type": "polygon", "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}
            corrupt_shape = {"label": "WRONG", "shape_type": "polygon", "points": [[2, 2], [5, 2], [5, 5], [2, 5]]}
            for dataset, shape in ((epb_clean, clean_shape), (epb_corrupt, corrupt_shape)):
                write_png(dataset / "data" / "a.png")
                value = {
                    "label_header": {"fileName": "a.png"},
                    "basic_annotation": {"fileName": "a.png", "startTime": "2026-01-01T00:00:00", "endTime": "2026-01-01T00:00:00"},
                    "science_annotation": {"mask": {"imageWidth": 8, "imageHeight": 8, "shapes": [shape]}},
                }
                write_json(dataset / "label" / "a.png.json", value)

            with (publish / "publish_tier_labels.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("dataset_id", "tier", "randomized_dir", "defect_class_ids"))
                writer.writeheader()
                for family, clean, corrupt in (
                    ("ULF_sample", ulf_clean, ulf_corrupt),
                    ("EPB_seg_sample", epb_clean, epb_corrupt),
                ):
                    writer.writerow({"dataset_id": family, "tier": "clean", "randomized_dir": clean.name, "defect_class_ids": ""})
                    writer.writerow({"dataset_id": family, "tier": "P2", "randomized_dir": corrupt.name, "defect_class_ids": "L2-P L2-T L2-G"})

            output = base / "output"
            opc_agent.run(publish, output, 41, set(opc_agent.ALL_CHECKS))
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            ulf_types = {row["defectType"] for row in rows if row["datasetId"] == ulf_corrupt.name and row["checkId"] == "C10" and row["status"] == opc_agent.STATUS_FAIL}
            epb_types = {row["defectType"] for row in rows if row["datasetId"] == epb_corrupt.name and row["checkId"] == "C10" and row["status"] == opc_agent.STATUS_FAIL}
            self.assertTrue({"L2-P", "L2-T"} <= ulf_types)
            self.assertTrue({"L2-P", "L2-G"} <= epb_types)
            geometry_failures = [row for row in rows if row["datasetId"] in {epb_clean.name, epb_corrupt.name} and row["checkId"] == "C05" and row["status"] == opc_agent.STATUS_FAIL]
            self.assertEqual(geometry_failures, [])

    def test_outputs_have_official_audit_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 11, set(opc_agent.ALL_CHECKS))
            first = json.loads((output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()[0])
            required = {"findingId", "checkId", "ruleVersion", "datasetId", "datasetVersion", "defectType", "status", "evidenceRef", "severity", "affectedScope", "criticality", "basis", "confirmedOrPending"}
            self.assertTrue(required <= set(first))
            evidence_header = (output / "07_证据索引" / "evidence_index.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertEqual(evidence_header, "evidenceId,findingId,locatorType,locator,contentHash,capturedBy")

    def test_null_polygon_is_not_a_geometry_defect(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, clean, _ = build_fixture(base)
            label_path = clean / "label" / "a.png.json"
            value = json.loads(label_path.read_text(encoding="utf-8"))
            value["hek_event_annotation"]["matched_events"][0]["bbox_polygon_pixel"] = None
            label_path.write_text(json.dumps(value), encoding="utf-8")
            output = base / "output"
            opc_agent.run(publish, output, 17, set(opc_agent.ALL_CHECKS))
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            invalid = [row for row in rows if row["datasetId"] == clean.name and row["checkId"] == "C05" and row["status"] == opc_agent.STATUS_FAIL]
            self.assertEqual(invalid, [])

    def test_generic_event_annotation_and_bbox_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, clean, corrupt = build_fixture(base)
            for dataset in (clean, corrupt):
                label_path = dataset / "label" / "a.png.json"
                value = json.loads(label_path.read_text(encoding="utf-8"))
                annotation = value.pop("hek_event_annotation")
                events = annotation.pop("matched_events")
                annotation["events"] = events
                value["event_annotation"] = annotation
                for event in events:
                    polygon = event.pop("bbox_polygon_pixel")
                    left = min(point[0] for point in polygon)
                    top = min(point[1] for point in polygon)
                    right = max(point[0] for point in polygon)
                    bottom = max(point[1] for point in polygon)
                    event["bbox"] = [left, top, right - left, bottom - top]
                    event["width"] = right - left
                    event["height"] = bottom - top
                label_path.write_text(json.dumps(value), encoding="utf-8")
            output = base / "output"
            opc_agent.run(publish, output, 29, set(opc_agent.ALL_CHECKS))
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            actual = {row["defectType"] for row in rows if row["datasetId"] == corrupt.name and row["status"] == opc_agent.STATUS_FAIL}
            self.assertTrue({"L2-G", "L3-MIS"} <= actual)
            dimension_failures = [row for row in rows if row["datasetId"] == corrupt.name and row["checkId"] == "C06" and row["status"] == opc_agent.STATUS_FAIL]
            self.assertEqual(len(dimension_failures), 1)

    def test_conclusions_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            first_output = base / "first"
            second_output = base / "second"
            opc_agent.run(publish, first_output, 23, set(opc_agent.ALL_CHECKS))
            opc_agent.run(publish, second_output, 23, set(opc_agent.ALL_CHECKS))
            first = (first_output / "06_问题清单" / "findings.jsonl").read_bytes()
            second = (second_output / "06_问题清单" / "findings.jsonl").read_bytes()
            self.assertEqual(first, second)

    def test_run_id_identifies_enabled_check_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            minimal = opc_agent.run(publish, base / "minimal", 31, {"C01", "C02", "C03"})
            complete = opc_agent.run(publish, base / "complete", 31, set(opc_agent.ALL_CHECKS))
            repeat = opc_agent.run(publish, base / "repeat", 31, {"C03", "C01", "C02"})
            self.assertNotEqual(minimal["runId"], complete["runId"])
            self.assertEqual(minimal["runId"], repeat["runId"])

    def test_unalignable_siblings_are_pending_not_passed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, corrupt = build_fixture(base)
            for label_path in (corrupt / "label").glob("*.json"):
                value = json.loads(label_path.read_text(encoding="utf-8"))
                value["basic_annotation"]["startTime"] = "2030-01-01T00:00:00"
                value["basic_annotation"]["fileName"] = f"other-{label_path.name}.png"
                label_path.write_text(json.dumps(value), encoding="utf-8")
            output = base / "output"
            opc_agent.run(publish, output, 37, set(opc_agent.ALL_CHECKS))
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            alignment = [
                row for row in rows
                if row["datasetId"] == corrupt.name and row["defectType"] == "REFERENCE_ALIGNMENT_UNAVAILABLE"
            ]
            self.assertEqual(len(alignment), 1)
            self.assertEqual(alignment[0]["status"], opc_agent.STATUS_PENDING)

    def test_shape_matching_is_invariant_to_list_order(self):
        left = {"label": "EPB", "shape_type": "polygon", "points": [[1, 1], [4, 1], [4, 4], [1, 4]]}
        right = {"label": "EPB", "shape_type": "polygon", "points": [[10, 10], [14, 10], [14, 14], [10, 14]]}
        reference = [("$.shapes[0]", left), ("$.shapes[1]", right)]
        actual = [("$.shapes[0]", right), ("$.shapes[1]", left)]
        matches, missing, extra = opc_agent.match_science_shapes(reference, actual)
        self.assertEqual([(ref, current) for ref, current, _, _ in matches], [(0, 1), (1, 0)])
        self.assertEqual(missing, [])
        self.assertEqual(extra, [])

    def test_event_source_key_is_invariant_to_list_order(self):
        first = {"source": "events.jsonl:17", "kind": "Pi2"}
        second = {"source": "events.jsonl:23", "kind": "Pi2"}
        self.assertEqual(opc_agent.event_key(first, 0), opc_agent.event_key(first, 1))
        self.assertNotEqual(opc_agent.event_key(first, 0), opc_agent.event_key(second, 0))


if __name__ == "__main__":
    unittest.main()
