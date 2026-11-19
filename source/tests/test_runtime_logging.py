import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fixture_factory import build_fixture
from spaceqa_agent import engine
from spaceqa_agent.audit import audit


class RuntimeLoggingTests(unittest.TestCase):
    def test_live_events_are_written_before_check_executes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "evaluation"
            original = engine.check_c01
            observed = []

            def check(dataset):
                events = [json.loads(line) for line in (output / "08_运行日志/run.log.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events[-1]["event"], "检查开始")
                self.assertEqual(events[-1]["datasetId"], dataset.dataset_id)
                self.assertEqual(events[-1]["seed"], 778)
                observed.append(dataset.dataset_id)
                return original(dataset)

            with patch.object(engine, "check_c01", check):
                engine.run(publish, output, 778, {"C01"})
            self.assertEqual(len(observed), 2)
            result = audit(output)
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertEqual(result["startedCheckCount"], 2)
            self.assertEqual(result["completedCheckCount"], 2)
            metrics = json.loads((output / "08_运行日志/runtime_metrics.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(metrics["elapsedSeconds"], 0)
            self.assertGreaterEqual(metrics["processUsage"]["cpuDeltaSeconds"], 0)
            self.assertEqual(metrics["memoryScope"], "process_lifetime_high_water")
            self.assertEqual(metrics["modelForwardPasses"], 0)

    def test_missing_start_or_mismatched_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "evaluation"
            engine.run(publish, output, 779, {"C01"})
            log = output / "08_运行日志/run.log.jsonl"
            events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            events.pop(0)
            events[0]["seed"] = 0
            engine.jsonl_dump(log, events)
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("expected one start/end" in error for error in result["errors"]))
            self.assertTrue(any("log seed differs" in error for error in result["errors"]))

    def test_check_failure_is_retained_and_cannot_pass_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "evaluation"
            with patch.object(engine, "check_c01", side_effect=RuntimeError("controlled failure")):
                engine.run(publish, output, 780, {"C01"})
            log = [json.loads(line) for line in (output / "08_运行日志/run.log.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(row["event"] == "失败" for row in log), 2)
            self.assertTrue(all("controlled failure" in row["failureReason"] for row in log if row["event"] == "失败"))
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("missing completed log" in error for error in result["errors"]))

    def test_low_level_run_rejects_overwrite_and_input_output_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "existing"
            output.mkdir()
            marker = output / "sentinel.txt"
            marker.write_text("preserve", encoding="utf-8")
            for target in [output, publish / "derived", publish]:
                with self.assertRaises(ValueError):
                    engine.run(publish, target, 781, {"C01"})
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertFalse((publish / "derived").exists())

    def test_scorecard_version_and_evidence_coverage_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "evaluation"
            engine.run(publish, output, 782, {"C01"})
            path = next((output / "04_评分卡").glob("*.json"))
            card = json.loads(path.read_text(encoding="utf-8"))
            card["ruleVersion"] = "different-version"
            card["evidenceRefs"] = []
            engine.json_dump(path, card)
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("scorecard ruleVersion differs" in e for e in result["errors"]))
            self.assertTrue(any("evidence references do not cover" in e for e in result["errors"]))

    def test_runtime_summary_version_and_cost_declaration_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            publish, _, _ = build_fixture(root)
            output = root / "evaluation"
            engine.run(publish, output, 783, {"C01"})
            path = output / "08_运行日志/runtime_metrics.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["seed"] = -1
            data["externalApiCalls"] = 1
            engine.json_dump(path, data)
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("runtime metrics seed differs" in e for e in result["errors"]))
            self.assertTrue(any("rules-only runtime has nonzero" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
