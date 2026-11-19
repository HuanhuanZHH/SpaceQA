"""HTTP, data isolation, immutable inputs, and real background job regression tests."""
import contextlib
import hashlib
import http.client
import json
import shutil
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_factory import build_fixture, write_json, write_png
from spaceqa_agent import engine
from spaceqa_agent.web_service import Application, ApiError, create_server


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


class BrowserServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.publish, self.clean, self.corrupt = build_fixture(self.root)
        self.history = self.root / "history"
        self.history.mkdir()
        self.existing = self.history / "existing-run"
        engine.run(self.publish, self.existing, 20261021, set(engine.ALL_CHECKS))
        self.initial = hashes(self.publish)
        self.app = Application([self.publish], [self.history], self.root / "web-state", start_worker=False)

    def tearDown(self):
        self.app.close()
        self.temp.cleanup()

    @contextlib.contextmanager
    def server(self):
        server = create_server("127.0.0.1", 0, data_roots=[self.publish], run_roots=[self.history],
                               state_dir=self.root / "http-state", start_worker=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def request(self, server, method, path, payload=None, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        body = None if payload is None else json.dumps(payload)
        fields = {"Content-Type": "application/json"} if payload is not None else {}
        fields.update(headers or {})
        client.request(method, path, body=body, headers=fields)
        response = client.getresponse()
        raw = response.read()
        result = json.loads(raw) if "application/json" in response.getheader("Content-Type", "") else raw
        status = response.status
        client.close()
        return status, result

    def test_dataset_sample_image_and_original_annotation_are_distinct(self):
        datasets = self.app.list_datasets()
        self.assertEqual(len(datasets), 2)
        ds = next(d for d in datasets if d["name"] == self.clean.name)
        samples = self.app.list_samples(ds["id"], {})
        self.assertEqual(samples["total"], 2)
        sample = self.app.sample_detail(ds["id"], samples["items"][0]["id"])
        self.assertEqual((sample["width"], sample["height"]), (8, 8))
        self.assertEqual(len(sample["annotations"]), 1)
        self.assertIn("Original annotation", sample["annotationMeaning"])
        self.assertTrue(self.app.sample_image(ds["id"], sample["id"]).is_file())
        self.assertEqual(hashes(self.publish), self.initial)

    def test_missing_data_label_is_retained(self):
        ds = next(d for d in self.app.list_datasets() if d["name"] == self.corrupt.name)
        samples = self.app.list_samples(ds["id"], {})["items"]
        orphan = next(x for x in samples if x["labelName"] == "orphan.png.json")
        detail = self.app.sample_detail(ds["id"], orphan["id"])
        self.assertTrue(detail["missingData"])
        self.assertTrue(detail["label"])
        self.assertIsNone(detail["imageUrl"])
        with self.assertRaises(ApiError):
            self.app.sample_image(ds["id"], orphan["id"])

    def test_history_findings_filters_and_hash_verified_evidence(self):
        runs = self.app.list_runs()
        self.assertEqual(len(runs), 1)
        identity = runs[0]["id"]
        detail = self.app.run_detail(identity)
        self.assertEqual(len(detail["scorecards"]), 2)
        self.assertEqual(sum(detail["statusCounts"].values()), detail["findingCount"])
        page = self.app.filter_findings(identity, {"status": ["不通过"], "limit": ["2"]})
        self.assertEqual(len(page["items"]), 2)
        self.assertTrue(all(row["status"] == "不通过" for row in page["items"]))
        finding = page["items"][0]
        evidence = self.app.evidence(identity, finding["findingId"])
        self.assertTrue(evidence["sourceVerified"])
        payload = self.existing / "07_证据索引/evidence" / (finding["evidenceRef"] + ".json")
        changed = json.loads(payload.read_text(encoding="utf-8"))
        changed["observation"] = {"altered": True}
        payload.write_text(json.dumps(changed), encoding="utf-8")
        self.assertFalse(self.app.evidence(identity, finding["findingId"])["sourceVerified"])

    def test_http_health_security_origin_and_no_arbitrary_paths(self):
        with self.server() as server:
            code, body = self.request(server, "GET", "/api/health")
            self.assertEqual(code, 200)
            self.assertEqual(body["service"], "spaceqa-web")
            self.assertFalse(body["capabilities"]["liveExpertInference"])
            code, _ = self.request(server, "GET", "/api/health", headers={"Host": "malicious.example"})
            self.assertEqual(code, 403)
            code, _ = self.request(server, "POST", "/api/jobs", {}, {"Origin": "https://malicious.example"})
            self.assertEqual(code, 403)
            code, _ = self.request(server, "POST", "/api/jobs", {}, {"Content-Type": "text/plain"})
            self.assertEqual(code, 415)
            code, _ = self.request(server, "GET", "/%2e%2e/pyproject.toml")
            self.assertEqual(code, 400)
            code, _ = self.request(server, "GET", "/api/files?path=/etc/passwd")
            self.assertEqual(code, 404)
            code, _ = self.request(server, "POST", "/api/jobs", {"datasetId": "fake", "input": "/etc"})
            self.assertEqual(code, 400)

    def test_legacy_evaluation_and_demo_entries_use_current_console(self):
        with self.server() as server:
            expected = {"/evaluation.html": b"CASE REVIEW DEMO", "/demo.html": b"CASE REVIEW DEMO", "/evaluation.js": b"loadRuns"}
            for path in ("/evaluation.html", "/demo.html", "/evaluation.js"):
                code, body = self.request(server, "GET", path)
                self.assertEqual(code, 200, path)
                self.assertIn(expected[path], body)

    def test_queue_cancel_restart_and_invalid_requests(self):
        ds = self.app.list_datasets()[0]
        job = self.app.submit({"datasetId": ds["id"], "seed": 19})
        self.assertEqual(job["status"], "queued")
        self.assertNotIn("_input", job)
        self.assertNotIn("_output", job)
        self.assertEqual(self.app.cancel(job["id"])["status"], "cancelled")
        another = self.app.submit({"datasetId": ds["id"]})
        recovered = Application([self.publish], [self.history], self.root / "web-state", start_worker=False)
        try:
            self.assertEqual(recovered.job(another["id"])["status"], "interrupted")
            self.assertEqual(recovered.job(job["id"])["status"], "cancelled")
        finally:
            recovered.close()
        for payload in ({"datasetId": ds["id"], "mode": "expert; rm -rf"},
                        {"datasetId": ds["id"], "seed": True}, {"datasetId": []}):
            with self.assertRaises(ApiError):
                self.app.submit(payload)

    def test_worker_runs_real_scan_audit_priors_and_preserves_inputs(self):
        worker = Application([self.publish], [self.history], self.root / "worker-state")
        try:
            ds = next(d for d in worker.list_datasets() if d["name"] == self.clean.name)
            job = worker.submit({"datasetId": ds["id"], "mode": "rules_priors", "seed": 23})
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                state = worker.job(job["id"])
                if state["status"] in {"succeeded", "failed", "interrupted"}:
                    break
                time.sleep(0.05)
            self.assertEqual(state["status"], "succeeded", state)
            self.assertEqual(state["audit"]["status"], "pass")
            self.assertEqual(state["priorSummary"]["samples"], 2)
            self.assertIn("unchanged", state["priorSummary"]["scoringEffect"])
            detail = worker.run_detail(state["runId"])
            self.assertEqual(detail["audit"]["status"], "pass")
            self.assertEqual(detail["manifest"]["seed"], 23)
            self.assertEqual(hashes(self.publish), self.initial)
        finally:
            worker.close()

    def test_csv_preview_numeric_channels_and_bounded_subsampling(self):
        package = self.publish / "ULF_sample_cccccccccc"
        (package / "data").mkdir(parents=True)
        (package / "label").mkdir()
        data = package / "data/trace.csv"
        data.write_text("t_s,amplitude\n" + "".join(f"{i},{i % 11}\n" for i in range(6000)), encoding="utf-8")
        write_json(package / "label/trace.csv.json", {"basic_annotation": {"fileName": "trace.csv"}})
        self.app.discover(force=True)
        ds = next(d for d in self.app.list_datasets() if d["family"] == "ULF_sample")
        sample_id = self.app.list_samples(ds["id"], {})["items"][0]["id"]
        detail = self.app.sample_detail(ds["id"], sample_id)
        self.assertEqual(detail["totalRows"], 6000)
        self.assertEqual(detail["series"][0]["name"], "amplitude")
        self.assertLessEqual(len(detail["series"][0]["points"]), 2400)
        self.assertIsNone(detail["imageUrl"])
        data.write_text("T,X\n100,2\n101,3\n", encoding="utf-8")
        physical, _, _ = self.app._series(data)
        self.assertEqual([r["name"] for r in physical], ["T", "X"])

    def test_evidence_same_name_packages_use_actual_source_path(self):
        second = self.root / "another-release"
        second.mkdir()
        shutil.copytree(self.corrupt, second / self.corrupt.name)
        app = Application([self.publish, second], [self.history], self.root / "distinct-state", start_worker=False)
        try:
            run_id = app.list_runs()[0]["id"]
            finding = next(row for row in app.findings(run_id)
                           if row["datasetId"] == self.corrupt.name and row["status"] == "不通过")
            evidence = app.evidence(run_id, finding["findingId"])
            linked = app.dataset(evidence["sample"]["datasetId"])
            self.assertEqual(Path(linked["path"]), self.corrupt.resolve())
            self.assertEqual(evidence["sample"]["matchMethod"], "canonical-source-path")
            no_path = {"locator": "/input/label/a.png.json"}
            self.assertIsNone(app._evidence_sample(finding, no_path))
        finally:
            app.close()

    def test_reject_writable_state_inside_input_and_public_bind(self):
        with self.assertRaises(ValueError):
            Application([self.publish], [], self.publish / "web", start_worker=False)
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0, data_roots=[self.publish], state_dir=self.root / "public-state")

    def test_symlink_to_outside_data_is_not_served(self):
        outside = self.root / "outside.png"
        write_png(outside)
        linked = self.clean / "data/escape.png"
        try:
            linked.symlink_to(outside)
        except OSError:
            self.skipTest("Symlink creation unavailable")
        write_json(self.clean / "label/escape.png.json", {"basic_annotation": {"fileName": "escape.png"}})
        ds = next(d for d in self.app.list_datasets() if d["name"] == self.clean.name)
        catalog = self.app.sample_catalog(ds["id"])
        sample = next(row for row in catalog.values() if row["labelName"] == "escape.png.json")
        with self.assertRaises(ApiError):
            self.app.sample_image(ds["id"], sample["id"])


if __name__ == "__main__":
    unittest.main()
