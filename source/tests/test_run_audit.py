import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opc_agent
from fixture_factory import build_fixture
from tools.audit_run import audit


class RunAuditTests(unittest.TestCase):
    def test_generated_run_passes_integrity_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 31, set(opc_agent.ALL_CHECKS))
            result = audit(output)
            self.assertEqual(result["status"], "pass", result["errors"])
            self.assertEqual(result["findingCount"], result["evidenceCount"])

    def test_modified_evidence_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 37, set(opc_agent.ALL_CHECKS))
            finding = json.loads(
                (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            evidence = output / "07_证据索引" / "evidence" / f"{finding['evidenceRef']}.json"
            evidence.write_text("{}\n", encoding="utf-8")
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("content hash mismatch" in error for error in result["errors"]))

    def test_modified_scorecard_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 43, set(opc_agent.ALL_CHECKS))
            scorecard = next((output / "04_评分卡").glob("*.json"))
            value = json.loads(scorecard.read_text(encoding="utf-8"))
            value["totalScore"] = 99.9999
            scorecard.write_text(json.dumps(value), encoding="utf-8")
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("total score mismatch" in error for error in result["errors"]))

    def test_missing_runtime_metadata_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 47, set(opc_agent.ALL_CHECKS))
            (output / "08_运行日志/seeds.yaml").unlink()
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("seeds.yaml" in error for error in result["errors"]))

    def test_runtime_version_and_finding_rule_version_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            publish, _, _ = build_fixture(base)
            output = base / "output"
            opc_agent.run(publish, output, 53, set(opc_agent.ALL_CHECKS))
            environment = output / "08_运行日志/system_env.txt"
            environment.write_text("opc_agent_version=other\nrule_version=other\npython=3.10\nplatform=test\n", encoding="utf-8")
            findings = output / "06_问题清单/findings.jsonl"
            rows = [json.loads(line) for line in findings.read_text(encoding="utf-8").splitlines()]
            rows[0]["ruleVersion"] = "other"
            findings.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            result = audit(output)
            self.assertEqual(result["status"], "fail")
            self.assertTrue(any("system version differs" in error for error in result["errors"]))
            self.assertTrue(any("finding rule version differs" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
