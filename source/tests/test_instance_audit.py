"""Behavioral checks for independent instance matching and localization scoring."""

import copy
import itertools
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from spaceqa_agent.instance_audit import (
    LocatorError, audit_annotations, evaluate_defects, locator_metrics, maximum_weight_matching,
)


def box(values=(0.1, 0.1, 0.5, 0.5), **kwargs):
    return {"type": "bbox", "bbox": list(values), "coordinate_system": "image_normalized",
            "frame_id": "current-image-sha", **kwargs}


def obj(identity, locator=None, label="event", **kwargs):
    return {"id": identity, "label": label, "sample_id": "sample1",
            "locator": locator or box(), "confidence": 0.99, **kwargs}


def defect(identity, locator=None, label="boundary_position_deviation", **kwargs):
    raw = obj(identity, locator, label, **kwargs)
    raw["defect_class"] = raw.pop("label")
    return raw


def candidate_request(refs=None, annotations=None, **kwargs):
    return {"sample_id": "sample1", "data_available": True,
            "reference": {"status": "ok", "source": "model", "independent_of_annotations": True,
                          "complete": False, "objects": refs or []},
            "annotations": annotations or [], **kwargs}


def evaluation_request(truth=None, predictions=None):
    return {"rule_version": "test-policy-1", "tolerance_source": "explicit_test_fixture_only",
            "evaluation_thresholds": {"iou_min": 0.5},
            "truth": {"independent": True, "exhaustive": True,
                      "source": "declared human review; not official truth",
                      "reviewed_samples": ["sample1"], "defects": truth or []},
            "predictions": predictions or []}


class GeometryTests(unittest.TestCase):
    def test_bbox_iou_and_directional_coverage(self):
        metric = locator_metrics(box((0, 0, 1, 1)), box((0, 0, 0.5, 1)))
        self.assertAlmostEqual(metric["iou"], 0.5)
        self.assertAlmostEqual(metric["coverage_reference"], 0.5)
        self.assertAlmostEqual(metric["coverage_annotation"], 1)
        self.assertAlmostEqual(metric["normalized_distance"], 0.25 / 2 ** 0.5)

    def test_pixels_normalization_and_explicit_bottom_origin(self):
        pixels = box((10, 20, 50, 60), coordinate_system="image_pixels", image_size=[100, 100])
        normalized = box((0.1, 0.2, 0.5, 0.6))
        flipped = box((0.1, 0.4, 0.5, 0.8), origin="bottom_left")
        self.assertAlmostEqual(locator_metrics(pixels, normalized)["iou"], 1)
        self.assertAlmostEqual(locator_metrics(flipped, normalized)["iou"], 1)

    def test_missing_or_different_coordinate_frame_is_not_guessed(self):
        missing = box()
        missing.pop("frame_id")
        for invalid in [missing, box(frame_id="rotated-image"), box(coordinate_system="heliographic")]:
            with self.assertRaises(LocatorError):
                locator_metrics(box(), invalid)
        with self.assertRaises(LocatorError):
            locator_metrics(box(), box((1, 1, 20, 20)))

    def test_concave_polygon_uses_true_region_not_bounding_box(self):
        l_shape = {"type": "polygon", "polygon": [[0, 0], [1, 0], [1, .25], [.25, .25], [.25, 1], [0, 1]],
                   "coordinate_system": "image_normalized", "frame_id": "current-image-sha"}
        self.assertAlmostEqual(locator_metrics(l_shape, box((0, 0, 1, 1)))["iou"], 7 / 16)
        self.assertAlmostEqual(locator_metrics(l_shape, box((.5, .5, 1, 1)))["iou"], 0)
        reverse = {**l_shape, "polygon": list(reversed(l_shape["polygon"]))}
        self.assertAlmostEqual(locator_metrics(l_shape, reverse)["iou"], 1)

    def test_invalid_polygon_abstains_instead_of_crude_bbox(self):
        invalid = {"type": "polygon", "polygon": [[0, 0], [1, 1], [0, 1], [1, 0]],
                   "coordinate_system": "image_normalized", "frame_id": "current-image-sha"}
        with self.assertRaises(LocatorError):
            locator_metrics(invalid, box())

    def test_time_axes_units_interval_iou_and_endpoint_errors(self):
        a = {"type": "timeInterval", "coordinate_system": "time_seconds", "time_axis_id": "event-start-UTC",
             "timeInterval": [10, 20], "axis_range": [0, 100]}
        b = {**a, "timeInterval": [12, 22]}
        m = locator_metrics(a, b)
        self.assertAlmostEqual(m["iou"], 8 / 12)
        self.assertAlmostEqual(m["normalized_distance"], .02)
        self.assertEqual(m["start_error_seconds"], 2)
        with self.assertRaises(LocatorError):
            locator_metrics(a, {**b, "time_axis_id": "different-origin"})
        with self.assertRaises(LocatorError):
            locator_metrics(a, {**b, "coordinate_system": "milliseconds"})


class AssignmentTests(unittest.TestCase):
    def test_many_to_many_greedy_trap_preserves_both_instances(self):
        self.assertEqual(maximum_weight_matching([[.95, .8], [.9, None]]), [(0, 1), (1, 0)])

    def test_global_weight_and_forbidden_edges(self):
        self.assertEqual(maximum_weight_matching([[.9, .85], [.8, .1]]), [(0, 1), (1, 0)])
        self.assertEqual(maximum_weight_matching([[None], [None]]), [])
        self.assertEqual(maximum_weight_matching([[], []]), [])

    def test_small_matrices_agree_with_exhaustive_assignment(self):
        rng = random.Random(19)
        for _ in range(20):
            scores = [[None if rng.random() < .3 else round(rng.random(), 2) for _ in range(3)] for _ in range(3)]
            best = (-1, -1)
            for columns in itertools.product([None, 0, 1, 2], repeat=3):
                chosen = [j for j in columns if j is not None]
                if len(set(chosen)) != len(chosen) or any(j is not None and scores[i][j] is None for i, j in enumerate(columns)):
                    continue
                objective = (len(chosen), sum(scores[i][j] for i, j in enumerate(columns) if j is not None))
                best = max(best, objective)
            pairs = maximum_weight_matching(scores)
            self.assertEqual(len(pairs), best[0])
            self.assertAlmostEqual(sum(scores[i][j] for i, j in pairs), best[1])


class CandidateTests(unittest.TestCase):
    def test_no_annotations_is_missing_annotation_not_spurious(self):
        result = audit_annotations(candidate_request([obj("det1")]))
        self.assertEqual([x["candidate_kind"] for x in result["candidates"]], ["missing_annotation"])
        self.assertIsNone(result["official_metrics"])

    def test_missing_collections_or_failed_parse_are_unknown_not_empty(self):
        missing_annotations = candidate_request([obj("det1")])
        missing_annotations.pop("annotations")
        missing_reference_objects = candidate_request([], [obj("ann1")])
        missing_reference_objects["reference"].update(complete=True, negative_evidence_verified=True)
        missing_reference_objects["reference"].pop("objects")
        unsupported = candidate_request([obj("det1")])
        unsupported["annotation_parse_status"] = "unsupported_structure"
        for request in [missing_annotations, missing_reference_objects, unsupported]:
            result = audit_annotations(request)
            self.assertEqual(result["status"], "abstained")
            self.assertEqual(result["candidates"], [])

    def test_detector_empty_or_incomplete_is_not_spurious_annotation(self):
        request = candidate_request([], [obj("ann1")])
        request["reference"]["complete"] = True
        result = audit_annotations(request)
        self.assertEqual(result["candidates"], [])
        self.assertIn("not_verified_negative_evidence", result["abstentions"][0]["reason"])

    def test_verified_exhaustive_negative_can_propose_spurious_candidate(self):
        request = candidate_request([], [obj("ann1")])
        request["reference"].update(source="independent_reference", complete=True, negative_evidence_verified=True)
        result = audit_annotations(request)
        self.assertEqual(result["candidates"][0]["defect_class"], "L3-07")
        self.assertEqual(result["candidates"][0]["status"], "candidate")

    def test_failed_missing_input_prompted_or_low_confidence_abstains(self):
        variants = []
        base = candidate_request([obj("det1")], [obj("ann1")])
        failed = copy.deepcopy(base)
        failed["reference"]["status"] = "failed"
        variants.append(failed)
        missing = copy.deepcopy(base)
        missing["data_available"] = False
        variants.append(missing)
        prompted = copy.deepcopy(base)
        prompted["reference"]["independent_of_annotations"] = False
        variants.append(prompted)
        low = copy.deepcopy(base)
        low["reference"].update(complete=True, negative_evidence_verified=True)
        low["reference"]["objects"][0]["confidence"] = .2
        variants.append(low)
        for request in variants:
            result = audit_annotations(request)
            self.assertEqual(result["candidates"], [])
            self.assertEqual(result["status"], "abstained")

    def test_class_error_matches_without_class_gate(self):
        result = audit_annotations(candidate_request([obj("det1", label="spot")], [obj("ann1", label="filament")]))
        self.assertEqual([x["candidate_kind"] for x in result["candidates"]], ["class_error"])
        self.assertEqual(result["candidates"][0]["defect_class"], "L2-P")

    def test_class_agnostic_localizer_cannot_report_class_error(self):
        for reference in [obj("mask", label=None), obj("mask", label="AIA", class_known=False)]:
            result = audit_annotations(candidate_request([reference], [obj("ann1", label="sunspot")]))
            self.assertEqual(len(result["matches"]), 1)
            self.assertEqual(result["candidates"], [])
            self.assertTrue(any(a.get("dimension") == "class" for a in result["abstentions"]))

    def test_coverage_preserves_subtype_under_one_official_class(self):
        result = audit_annotations(candidate_request([obj("det", box((.1, .1, .9, .9)))],
                                                      [obj("ann", box((.2, .2, .8, .8)))]))
        candidate = result["candidates"][0]
        self.assertEqual((candidate["defect_class"], candidate["subtype"]), ("L3-MIS", "undercoverage"))

    def test_boundary_and_coverage_hypotheses_are_distinguished(self):
        ref = obj("det1", box((.2, .2, .8, .8)))
        cases = [(box((.3, .3, .7, .7)), "undercoverage"),
                 (box((.1, .1, .9, .9)), "overcoverage"),
                 (box((.3, .2, .9, .8)), "boundary_position_deviation")]
        for locator, expected in cases:
            result = audit_annotations(candidate_request([ref], [obj("ann1", locator)]))
            self.assertEqual([x["candidate_kind"] for x in result["candidates"]], [expected])

    def test_wrong_frame_invalid_annotation_prevents_false_absence(self):
        result = audit_annotations(candidate_request([obj("det1")], [obj("ann1", box(frame_id="other-image"))]))
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["status"], "abstained")
        invalid = obj("bad", box((0, 0, 50, 50)))
        result = audit_annotations(candidate_request([obj("det1")], [invalid]))
        self.assertEqual(result["candidates"], [])

    def test_duplicate_detector_objects_do_not_create_false_missing_annotations(self):
        result = audit_annotations(candidate_request([obj("det1"), obj("det2")], [obj("ann1")]))
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(len(result["reference_duplicates"]), 1)

    def test_time_shift_and_explicit_tolerance(self):
        loc = {"type": "timeInterval", "coordinate_system": "time_seconds", "time_axis_id": "UTC",
               "timeInterval": [10, 20], "axis_range": [0, 100]}
        request = candidate_request([obj("det1", loc)], [obj("ann1", {**loc, "timeInterval": [11, 21]})],
                                    thresholds={"time_endpoint_tolerance_seconds": .5})
        self.assertEqual(audit_annotations(request)["candidates"][0]["defect_class"], "L2-T")
        request["thresholds"]["time_endpoint_tolerance_seconds"] = 2
        self.assertEqual(audit_annotations(request)["candidates"], [])


class EvaluatorTests(unittest.TestCase):
    def test_candidate_output_uses_official_codes_and_can_be_scored_directly(self):
        missing = audit_annotations(candidate_request([obj("det1")]))["candidates"]
        request = evaluation_request([defect("t", label="L3-06")], missing)
        self.assertEqual(evaluate_defects(request)["metrics"]["TP"], 1)
        negatives = candidate_request([], [obj("ann1")])
        negatives["reference"].update(source="independent_reference", complete=True, negative_evidence_verified=True)
        spurious = audit_annotations(negatives)["candidates"]
        request = evaluation_request([defect("t", label="L3-07")], spurious)
        self.assertEqual(evaluate_defects(request)["metrics"]["TP"], 1)

    def test_truth_independence_scope_and_tolerances_are_required(self):
        for patch in ["independence", "scope", "tolerance"]:
            request = evaluation_request([defect("t")], [defect("p")])
            if patch == "independence":
                request["truth"]["independent"] = False
            elif patch == "scope":
                request["truth"].pop("reviewed_samples")
            else:
                request.pop("evaluation_thresholds")
            result = evaluate_defects(request)
            self.assertIsNone(result["metrics"])
            self.assertIsNone(result["official_metrics"])

    def test_class_and_location_must_both_match(self):
        for prediction in [defect("p", label="class_error"), defect("p", box((.7, .7, 1, 1)))]:
            metrics = evaluate_defects(evaluation_request([defect("t")], [prediction]))["metrics"]
            self.assertEqual((metrics["TP"], metrics["FP"], metrics["FN"]), (0, 1, 1))

    def test_evaluator_many_to_many_global_matching(self):
        truth = [defect("tA", box((0, 0, .6, 1))), defect("tB", box((.4, 0, 1, 1)))]
        predictions = [defect("pX", box((.2, 0, .8, 1))), defect("pY", box((0, 0, .3, 1)))]
        request = evaluation_request(truth, predictions)
        request["evaluation_thresholds"]["iou_min"] = .45
        result = evaluate_defects(request)
        self.assertEqual(result["metrics"]["TP"], 2)
        self.assertEqual({(x["truth_id"], x["prediction_id"]) for x in result["matches"]}, {("tA", "pY"), ("tB", "pX")})

    def test_duplicate_reports_do_not_inflate_tp_or_fp(self):
        request = evaluation_request([defect("t", critical=True)], [defect("p1"), defect("p2")])
        result = evaluate_defects(request)
        self.assertEqual(result["metrics"]["TP"], 1)
        self.assertEqual(result["metrics"]["FP"], 0)
        self.assertEqual(len(result["duplicates"]), 1)
        self.assertEqual(result["metrics"]["critical_defect_recall"], 1)
        self.assertFalse(result["official_qualification_established"])

    def test_unknown_critical_flags_cannot_inflate_key_defect_recall(self):
        truth = [defect("known-critical", critical=True),
                 defect("unresolved-critical-status", box((.6, .6, 1, 1)), critical=None)]
        result = evaluate_defects(evaluation_request(truth, [defect("p")]))
        metrics = result["metrics"]
        self.assertEqual(metrics["recall"], .5)
        self.assertIsNone(metrics["critical_defect_recall"])
        self.assertFalse(metrics["critical_annotations_complete"])
        self.assertEqual(metrics["declared_known_critical_subset_recall"], 1)
        self.assertEqual(metrics["unknown_critical_truth_ids"], ["unresolved-critical-status"])

    def test_nonboolean_critical_flags_are_not_treated_as_false(self):
        result = evaluate_defects(evaluation_request([defect("t", critical="false")], [defect("p")]))
        self.assertIsNone(result["metrics"])

    def test_unknown_excluded_and_unreviewed_are_never_negatives(self):
        truth = [defect("unknown", status="unknown"),
                 defect("excluded", box((.6, .6, 1, 1)), status="excluded")]
        predictions = [defect("p1"), defect("p2", box((.6, .6, 1, 1))), defect("p3", sample_id="unreviewed")]
        result = evaluate_defects(evaluation_request(truth, predictions))
        self.assertEqual(result["metrics"]["FP"], 0)
        self.assertIsNone(result["metrics"]["recall"])
        self.assertEqual(len(result["ignored_predictions"]), 3)

    def test_unlocalized_unknown_masks_sample_without_masking_known_true_match(self):
        truth = [defect("known"), {"id": "u", "sample_id": "sample1", "status": "unknown", "defect_class": "*"}]
        predictions = [defect("p1"), defect("p2", box((.6, .6, 1, 1)))]
        result = evaluate_defects(evaluation_request(truth, predictions))
        self.assertEqual((result["metrics"]["TP"], result["metrics"]["FP"]), (1, 0))

    def test_unknown_null_class_or_incompatible_frame_cannot_create_false_negatives(self):
        for unknown in [
            {"id": "u", "sample_id": "sample1", "status": "unknown", "defect_class": None},
            defect("u", box(frame_id="unverified-other-frame"), status="unknown"),
        ]:
            result = evaluate_defects(evaluation_request([unknown], [defect("p")]))
            self.assertEqual(result["metrics"]["FP"], 0)
            self.assertEqual(len(result["ignored_predictions"]), 1)

    def test_omitted_evaluation_collections_are_not_verified_empty_truth(self):
        request = evaluation_request([], [defect("p")])
        request["truth"].pop("defects")
        self.assertIsNone(evaluate_defects(request)["metrics"])

    def test_reviewed_clean_sample_false_positive_and_no_fake_tn(self):
        result = evaluate_defects(evaluation_request([], [defect("p")]))
        self.assertEqual((result["metrics"]["TP"], result["metrics"]["FP"], result["metrics"]["FN"]), (0, 1, 0))
        self.assertNotIn("TN", result["metrics"])
        self.assertIsNone(result["metrics"]["recall"])

    def test_partial_sample_class_review_does_not_create_unreviewed_negatives(self):
        request = evaluation_request([], [defect("reviewed"), defect("not-reviewed", label="class_error")])
        request["truth"]["reviewed_sample_classes"] = [
            {"sample_id": "sample1", "defect_classes": ["boundary_position_deviation"]}]
        result = evaluate_defects(request)
        self.assertEqual(result["metrics"]["FP"], 1)
        self.assertEqual(result["ignored_predictions"], [{"id": "not-reviewed", "reason": "unreviewed_sample_class"}])
        self.assertEqual(result["exhaustive_scope_meaning"], "only_explicit_sample_class_pairs")
        request["truth"]["defects"] = [defect("outside-truth", label="class_error")]
        self.assertIsNone(evaluate_defects(request)["metrics"])

    def test_time_endpoint_tolerance_is_explicit(self):
        a = {"type": "timeInterval", "coordinate_system": "time_seconds", "time_axis_id": "UTC", "timeInterval": [10, 20]}
        request = evaluation_request([defect("t", a, "time_boundary_jitter")],
                                     [defect("p", {**a, "timeInterval": [11, 21]}, "time_boundary_jitter")])
        request["evaluation_thresholds"] = {"time_tolerance_seconds": 1}
        result = evaluate_defects(request)
        self.assertEqual(result["metrics"]["TP"], 1)
        self.assertEqual(result["evaluation_thresholds"]["time_tolerance_seconds"], 1)

    def test_file_record_locators_and_per_class_tolerance(self):
        loc = {"type": "record", "path": "labels/example.json", "record_id": "annotation-7"}
        result = evaluate_defects(evaluation_request([defect("t", loc, "orphan_annotation")], [defect("p", loc, "orphan_annotation")]))
        self.assertEqual(result["metrics"]["TP"], 1)
        request = evaluation_request([defect("t")], [defect("p", box((.2, .1, .6, .5)))])
        request["evaluation_thresholds"] = {"iou_min": .9, "per_class": {"boundary_position_deviation": {"iou_min": .5}}}
        self.assertEqual(evaluate_defects(request)["metrics"]["TP"], 1)

    def test_invalid_truth_or_prediction_cannot_disappear_from_denominator(self):
        for source in ["truth", "prediction"]:
            request = evaluation_request([defect("t")], [defect("p")])
            if source == "truth":
                request["truth"]["defects"][0].pop("locator")
            else:
                request["predictions"][0].pop("locator")
            result = evaluate_defects(request)
            self.assertEqual(result["status"], "unavailable")
            self.assertIsNone(result["metrics"])

    def test_a_class_without_applicable_tolerance_is_not_silently_failed(self):
        request = evaluation_request([defect("t")], [defect("p")])
        request["evaluation_thresholds"] = {"time_tolerance_seconds": 1}
        self.assertIsNone(evaluate_defects(request)["metrics"])

    def test_json_cli_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "input.json", Path(tmp) / "output.json"
            source.write_text(json.dumps(evaluation_request([defect("t")], [defect("p")])) , encoding="utf-8")
            completed = subprocess.run([sys.executable, "-m", "spaceqa_agent.instance_audit", "evaluate", "--input", str(source), "--output", str(target)], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["metrics"]["TP"], 1)


if __name__ == "__main__":
    unittest.main()
