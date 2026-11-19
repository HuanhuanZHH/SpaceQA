"""End-to-end export tests: old false summary conflict and coarse model flag."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from spaceqa_agent import engine, priors
from fixture_factory import write_png


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spaceqa_integration_export_test", ROOT / "scripts/export_integrated.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class IntegratedExportTests(unittest.TestCase):
    def prepare(self, root):
        package = "CME_C2_sample_abcdefghij"
        raw = root / package
        (raw / "data").mkdir(parents=True)
        (raw / "label").mkdir()
        data = raw / "data/a.png"
        write_png(data, 32, 32)
        label_path = raw / "label/a.png.json"
        label = {"basic_annotation": {"fileName": "a.png", "startTime": "2020-01-01T00:00:00", "endTime": "2020-01-01T00:00:00"},
                 "hek_event_annotation": {"image_time": "2020-01-01T00:12:00", "has_events": True,
                     "event_count": 1, "event_types": ["CME"],
                     "matched_events": [{"cme_time": "2020-01-01T00:12:00", "width": 30, "central_pa": 100}]}}
        label_path.write_text(json.dumps(label), encoding="utf-8")
        old, fresh = root / "old", root / "fresh"
        old_folder, fresh_folder = old / "packages" / package, fresh / "packages" / package
        old_folder.mkdir(parents=True)
        fresh_folder.mkdir(parents=True)
        sid = package + ":a.png.json"
        record = {"id": sid, "family": "CME_C2_sample", "label_path": str(label_path),
                  "data_path": str(data), "input_missing": False}
        dataset = engine.inventory_dataset(raw, None)
        engine.base_analysis(dataset, "old-run", set(engine.ALL_CHECKS) - {"C10"})
        self.assertTrue(any(f["defectType"] == "ANNOTATION_STRUCTURE_CONFLICT" for f in dataset.findings))
        engine.jsonl_dump(old_folder / "inventory.jsonl", [record])
        engine.jsonl_dump(old_folder / "rule_observations.jsonl", dataset.findings)
        prior = priors.evaluate_sample("CME_C2_sample", label, data, label_path=label_path, sample_id=sid)
        prior["current_label_sha256"] = exporter.sha(label_path)
        engine.jsonl_dump(fresh_folder / "prior_observations.jsonl", [prior])
        candidate = dict(sample_id=sid, label_path=str(label_path), route="test-model",
                         probabilities=[.1, .1, .9], thresholds=[.8, .8, .8],
                         model_candidate_bits=[0, 0, 1], frozen_validation_policy_sha256="f" * 64)
        engine.json_dump(fresh / "policy.json", {"status": "frozen", "test_used_for_selection": False})
        candidate["frozen_validation_policy_sha256"] = exporter.sha(fresh / "policy.json")
        engine.jsonl_dump(fresh_folder / "selected_candidates.jsonl", [candidate])
        engine.json_dump(fresh / "complete.json", dict(status="complete", ground_truth_available=False,
            selected_variant="synthetic-policy", packages=1, records=1,
            frozen_validation_policy_sha256=exporter.sha(fresh / "policy.json")))
        return old, fresh, label_path

    def test_model_flag_stays_pending_and_export_is_auditable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, fresh, _ = self.prepare(root)
            before = engine.RULE_VERSION
            result = exporter.export(old, fresh, root / "output", frozen_policy=fresh / "policy.json")
            findings = list(exporter.lines(root / "output/06_问题清单/findings.jsonl"))
            self.assertEqual(result["samples"], 1)
            self.assertIsNone(result["F1"])
            self.assertEqual(engine.RULE_VERSION, before)
            self.assertFalse(any(f["defectType"] == "ANNOTATION_STRUCTURE_CONFLICT" and f["status"] == engine.STATUS_FAIL for f in findings))
            candidates = [f for f in findings if f["defectType"] == "MODEL_D_CANDIDATE"]
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["status"], engine.STATUS_PENDING)
            self.assertFalse(candidates[0]["scoreEligible"])
            card = json.loads(next((root / "output/04_评分卡").glob("*.json")).read_text(encoding="utf-8"))
            self.assertEqual(card["scoringInputs"]["labelFileCount"], 1)
            checked = json.loads((root / "output/integration_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(checked["status"], "pass", checked)

    def test_changed_label_is_rejected_and_global_version_restored(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, fresh, label_path = self.prepare(root)
            label_path.write_text("{}", encoding="utf-8")
            before = engine.RULE_VERSION
            with self.assertRaisesRegex(ValueError, "Source label changed"):
                exporter.export(old, fresh, root / "output", frozen_policy=fresh / "policy.json")
            self.assertEqual(engine.RULE_VERSION, before)

    def test_truncated_probability_or_mixed_policy_is_rejected(self):
        record = {"data_path": "/data/a.png"}
        candidate = dict(frozen_validation_policy_sha256="a" * 64,
                         probabilities=[.9], thresholds=[.8] * 3, model_candidate_bits=[1, 0, 0])
        with self.assertRaisesRegex(ValueError, "three finite probabilities"):
            exporter.validate_candidate(candidate, record, "a" * 64)
        candidate["probabilities"] = [.9, .1, .1]
        with self.assertRaisesRegex(ValueError, "policy hash"):
            exporter.validate_candidate(candidate, record, "b" * 64)

    def test_output_cannot_be_inside_raw_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old, fresh, label_path = self.prepare(root)
            output = label_path.parents[1] / "new-output"
            with self.assertRaisesRegex(ValueError, "outside the original data"):
                exporter.export(old, fresh, output, frozen_policy=fresh / "policy.json")
            self.assertFalse(output.exists())

    def test_endpoint_fields_and_category_aliases_are_not_double_counted(self):
        findings = [{"defectType": "L2-T", "pointer": "/basic_annotation/" + field,
                     "certainty": "confirmed", "score_eligible": True,
                     "basis": "measured time contradiction", "observation": {"offset_seconds": value}}
                    for field, value in (("startTime", -120), ("endTime", 120))]
        merged = exporter.coalesce_prior_findings(findings)
        self.assertEqual(len(merged), 1)
        self.assertEqual(exporter.measured_degree(merged[0]), (120.0, "timeDeviation", "second"))
        self.assertEqual(len(merged[0]["observation"]["sources"]), 2)


if __name__ == "__main__":
    unittest.main()
