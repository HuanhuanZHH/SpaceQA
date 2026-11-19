"""Contract and API failure tests; no real network or API key is used."""
import io
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch, Mock

from spaceqa_agent.explain import endpoint_url, explain, load_case, validate_answer


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.case = {"findings": [{"finding_id": "F-1", "status": "待复核"}]}
        self.answer = {"summary": "需要复核", "explanations": [
            {"finding_id": "F-1", "explanation": "模型候选尚未确证"}],
            "review_items": ["核验原图边界"], "limitations": ["没有独立真值"]}

    def test_endpoint_rejects_credentials_and_insecure_remote_api(self):
        self.assertEqual(endpoint_url("https://api.example/v1/"), "https://api.example/v1/chat/completions")
        self.assertEqual(endpoint_url("http://127.0.0.1:8000/v1"), "http://127.0.0.1:8000/v1/chat/completions")
        for url in ("http://remote.example/v1", "https://name:pass@api.example/v1", "https://api.example/v1?key=x"):
            with self.assertRaises(ValueError):
                endpoint_url(url)

    def test_unknown_and_duplicate_finding_citations_are_rejected(self):
        self.assertEqual(validate_answer(self.answer, self.case), self.answer)
        bad = dict(self.answer, explanations=[{"finding_id": "invented", "explanation": "x"}])
        with self.assertRaises(ValueError):
            validate_answer(bad, self.case)
        bad["explanations"] = self.answer["explanations"] * 2
        with self.assertRaises(ValueError):
            validate_answer(bad, self.case)

    def test_reading_run_preserves_original_and_marks_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "06_问题清单/findings.jsonl"
            target.parent.mkdir()
            original = '\n'.join(json.dumps({"findingId": f"F-{i}"}) for i in range(3)) + '\n'
            target.write_text(original, encoding="utf-8")
            result = load_case(None, root, ["F-1"])
            self.assertEqual(result["selection"], {"selected": 1, "total": 3, "complete_run": False})
            self.assertEqual(result["findings"], [{"findingId": "F-1"}])
            self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_api_success_records_usage_and_does_not_mutate_case(self):
        response = {"choices": [{"message": {"content": json.dumps(self.answer)}}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20}}
        opener = Mock()
        opener.open.return_value = io.BytesIO(json.dumps(response).encode())
        with patch("spaceqa_agent.explain.urllib.request.build_opener", return_value=opener):
            result = explain(self.case, base_url="https://api.example/v1", api_key="test-token", model="test-model")
        self.assertFalse(result["changes_scores"])
        self.assertEqual(result["usage"]["prompt_tokens"], 30)
        self.assertEqual(self.case["findings"][0]["status"], "待复核")
        self.assertNotIn("test-token", json.dumps(result))

    def test_api_errors_hide_provider_body_and_do_not_retry_unauthorized(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://api.example", 401, "secret-provider-body", {}, None)
        with patch("spaceqa_agent.explain.urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(ValueError, "HTTP 401") as ctx:
                explain(self.case, base_url="https://api.example/v1", api_key="test-token", model="test-model")
        self.assertNotIn("secret-provider-body", str(ctx.exception))
        self.assertEqual(opener.open.call_count, 1)

    def test_case_includes_verified_evidence_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            finding = root / "06_问题清单/findings.jsonl"
            finding.parent.mkdir()
            finding.write_text(json.dumps({"findingId": "F-1", "evidenceRef": "EV-F-1"}) + "\n", encoding="utf-8")
            evidence = root / "07_证据索引/evidence/EV-F-1.json"
            evidence.parent.mkdir(parents=True)
            evidence.write_text(json.dumps({"findingId": "F-1", "evidenceId": "EV-F-1", "observation": {"actual": 1024}}), encoding="utf-8")
            index = root / "07_证据索引/evidence_index.csv"
            with index.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["evidenceId", "contentHash"])
                writer.writeheader()
                writer.writerow({"evidenceId": "EV-F-1", "contentHash": hashlib.sha256(evidence.read_bytes()).hexdigest()})
            result = load_case(None, root, ["F-1"])
            self.assertEqual(result["findings"][0]["evidence"]["observation"]["actual"], 1024)
            evidence.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence failed"):
                load_case(None, root, ["F-1"])


if __name__ == "__main__":
    unittest.main()
