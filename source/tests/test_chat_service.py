"""Conversation persistence, real adapter contracts, and archive boundaries."""
import contextlib
import http.client
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spaceqa_agent.chat_service import ChatApplication, ChatError, create_server


class UpstreamFixture:
    def __init__(self, root):
        self.root = root
        self.uploads = root / "uploads"
        self.uploads.mkdir()
        self.package = root / "data" / "AIA_AR_sample_demo"
        (self.package / "data").mkdir(parents=True)
        (self.package / "label").mkdir()
        self.ds = {"id": "ds-first", "name": self.package.name, "family": "AIA_AR_sample", "path": str(self.package), "labelCount": 3, "dataCount": 3}
        self.requests = []
        self.count = 0
        self.case_rows = [
            self.case("s1", "confirmed", "fail", True, "rule", "L0-01"),
            self.case("s1", "candidate", "undetermined", False, "model", "L2-G"),
            self.case("s2", "candidate", "undetermined", False, "model", "L3-06"),
            self.case("s3", "confirmed", "fail", True, "model", "L2-G"),
            self.case("s3", "observation", "undetermined", False, "rule", "L2-T"),
        ]

    def case(self, sample, stage, verdict, eligible, source, code):
        return {"caseId": "case-" + sample + "-" + stage + "-" + code, "packageId": "ds-first", "sampleId": sample, "findingStage": stage, "verdict": verdict, "scoreEligible": eligible, "sourceKind": source, "finding": {"defectType": code, "basis": "Checked current file against its registered annotation", "locator": {"type": "sample", "value": sample}}}

    def datasets(self):
        rows = [self.ds]
        for label in self.uploads.rglob("label"):
            if (label.parent / "data").is_dir():
                p = label.parent
                rows.append({"id": "ds-" + p.parent.name, "name": p.name, "family": "AIA_AR_sample", "path": str(p), "labelCount": len(list(label.glob("*.json"))), "dataCount": len(list((p / "data").iterdir()))})
        return rows

    def __call__(self, path, payload=None):
        self.requests.append((path, payload))
        if path.startswith("/api/datasets?"):
            rows = self.datasets()
            return {"items": rows, "total": len(rows)}
        if path == "/api/evaluations" and payload:
            self.count += 1
            return {"taskId": "eval-" + str(self.count), "status": "queued", "phase": "queued", "createdAt": "2026-11-05T12:00:00+00:00"}
        if path.startswith("/api/evaluations/"):
            if "/cases?" in path:
                return {"items": self.case_rows, "total": len(self.case_rows)}
            if path.endswith("/coverage"):
                return {"ds-first": {"checks": [{"tool": "inventory", "executionStatus": "succeeded", "coverage": {"unit": "samples", "processed": 3, "requested": 3}}], "dimensions": {"completeness": [{"conclusionStatus": "fail"}]}}}
            return {"taskId": path.rsplit("/", 1)[-1], "status": "completed", "phase": "completed", "actualMode": "workflow", "request": {"packages": [{"packageId": "ds-first", "inputPath": str(self.package)}]}, "usage": {"elapsedSeconds": 1.25}, "packageStates": {"ds-first": {"result": {"score": {"score": 66.67, "officialScoreApproved": False}, "limitations": ["Semantic candidates need independent review"]}}}}
        raise AssertionError(path)


class ChatServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake = UpstreamFixture(self.root)
        self.env = mock.patch.dict(os.environ, {"SPACEQA_CHAT_LLM_BASE_URL": "", "SPACEQA_CHAT_LLM_API_KEY": "", "SPACEQA_CHAT_LLM_MODEL": "", "SPACEQA_CHAT_LLM_TRUST_ENV": ""})
        self.env.start()
        self.app = self.make_app()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def make_app(self, **kwargs):
        return ChatApplication("http://127.0.0.1:8781", self.root / "state", self.fake.uploads, transport=self.fake, **kwargs)

    def test_one_dataset_session_submits_real_workflow_once_and_persists(self):
        first = self.app.create({"datasetId": "ds-first"})
        self.assertEqual(self.fake.count, 1)
        request = next(body for path, body in self.fake.requests if path == "/api/evaluations")
        self.assertEqual(request["mode"], "workflow")
        self.assertNotIn("expertLimit", request)
        again = self.app.create({"datasetId": "ds-first"})
        self.assertEqual(again["id"], first["id"])
        self.assertEqual(self.fake.count, 1)
        loaded = self.make_app().get(first["id"], refresh=False)
        self.assertEqual(loaded["dataset"]["id"], "ds-first")

    def test_confirmed_and_candidates_are_deduplicated_and_never_promoted(self):
        initial = self.app.create({"datasetId": "ds-first"})
        completed = self.app.get(initial["id"])
        result = completed["analysis"]["result"]
        self.assertEqual(result["confirmedSamples"], 1)
        self.assertEqual(result["candidateOnlySamples"], 1)
        self.assertEqual(result["foundSamples"], 2)
        self.assertAlmostEqual(result["detectedRatio"], 2 / 3)
        self.assertEqual(completed["analysis"]["progress"]["processed"], 3)
        self.assertEqual(sum(m.get("kind") == "analysis_summary" for m in completed["messages"]), 1)
        repeat = self.app.get(initial["id"])
        self.assertEqual(len(repeat["messages"]), len(completed["messages"]))
        reply = self.app.answer(initial["id"], {"message": "F1是多少"})
        self.assertIn("不能", reply["messages"][-1]["content"])
        self.assertEqual(reply["messages"][-1]["replySource"], "structured_evidence")
        self.assertIn("/index.html#datasets/", reply["messages"][-1]["evidence"][-1]["url"])
        self.assertEqual(self.app.report(initial["id"])["result"]["foundSamples"], 2)
        details = self.app._answer(self.app.sessions[initial["id"]], "主要有哪些问题，请举一个样本")
        self.assertEqual(details.count("•"), 1)
        self.assertIn("确认1条", details)
        self.assertIn("待复核", details)
        missing = self.app._answer(self.app.sessions[initial["id"]], "漏标的样本")
        self.assertIn("s2", missing)
        self.assertNotIn("• s1", missing)
        short = self.app._answer(self.app.sessions[initial["id"]], "请用150字说明主要问题并举一个样本")
        self.assertLessEqual(len(short), 150)

    def test_saved_report_immediate_answer_requires_no_new_evaluation(self):
        cache = self.root / "reports.json"
        cache.write_text(json.dumps({"reports": [{"inputPath": str(self.fake.package), "totalSamples": 3, "confirmedSamples": 1, "candidateOnlySamples": 1, "foundSamples": 2, "defectRecords": {"L0-01": 1, "L3-06": 1}, "dimensions": [{"id": "annotationQuality", "state": "undetermined", "checkCoverage": 0.5, "score": None}], "evaluatedAt": "2026-11-05", "reportLabel": "Frozen test evidence", "sourcefiles": ["private-source-file"], "cases": [{"sampleId": "s1", "defectType": "L0-01", "stage": "confirmed", "basis": "Missing file", "sourcefile": "private-source-file", "measurements": {"rule_checks": "private-rule-dump" * 10000}}]}]}), encoding="utf-8")
        app = self.make_app(report_index=cache)
        session = app.create({"datasetId": "ds-first", "preferCached": True})
        self.assertEqual(self.fake.count, 0)
        self.assertEqual(session["analysis"]["source"], "saved_report")
        self.assertEqual(session["status"], "completed")
        self.assertTrue(app.demo()["available"])
        summary = session["messages"][-1]
        self.assertIn("真实历史评价", summary["content"])
        self.assertIn("按每类去重样本", summary["content"])
        self.assertIn("数据文件缺失（L0-01）", summary["content"])
        self.assertNotIn("/evaluation.html#None", json.dumps(summary))
        self.assertTrue(any(e["url"].endswith("/report") for e in summary["evidence"]))
        reply = app.answer(session["id"], {"message": "具体哪个样本有问题"})
        self.assertIn("s1", reply["messages"][-1]["content"])
        self.assertNotIn("同一样本可重复出现", reply["messages"][-1]["content"])
        self.assertIn("annotation_quality", session["analysis"]["result"]["coverage"]["dimensions"])
        self.assertIn("50.00%", app._answer(app.sessions[session["id"]], "六维覆盖如何"))
        app.llm_base, app.llm_key, app.llm_model = "http://example.invalid/v1", "test-placeholder", "test-model"
        with mock.patch("spaceqa_agent.chat_service.urllib.request.build_opener") as builder, self.assertLogs("spaceqa_agent.chat_service", level="WARNING") as logs:
            request = builder.return_value.open
            request.side_effect = OSError("sensitive response should not be logged")
            _, reply_source = app._llm_answer(app.sessions[session["id"]], "解释一下", "实际记录")
        self.assertEqual(reply_source, "structured_fallback")
        self.assertIn("error_type=OSError", logs.output[0])
        self.assertNotIn("sensitive response", logs.output[0])
        self.assertEqual(builder.call_args[0][0].proxies, {})
        body = json.loads(request.call_args[0][0].data)
        context = body["messages"][1]["content"].split("当前包结构化事实：\n", 1)[1].split("\n\n本包问题：", 1)[0]
        self.assertLessEqual(len(context), 16000)
        self.assertIsInstance(json.loads(context), dict)
        self.assertNotIn("private-source-file", context)
        self.assertNotIn("private-rule-dump", context)
        app.start(session["id"])
        self.assertEqual(self.fake.count, 1)
        self.assertEqual(app.get(session["id"], refresh=False)["analysis"]["source"], "live")

    def test_browse_roots_constrain_paths_and_only_registered_dataset_can_select(self):
        app = self.make_app(browse_roots=[self.root / "data"])
        roots = app.folders()
        self.assertFalse(roots["canSelect"])
        folder = app.folders(str(self.root / "data"))
        self.assertTrue(folder["folders"][0]["isDataset"])
        selected = app.folders(str(self.fake.package))
        self.assertTrue(selected["canSelect"])
        self.assertEqual(selected["selectedDataset"]["id"], "ds-first")
        with self.assertRaises(ChatError) as exc:
            app.folders(str(self.root.parent))
        self.assertEqual(exc.exception.status, 403)
        extracted = self.root / "extracted"
        public = self.root / "newpublish"
        public.mkdir()
        registrations = []
        for wrapper, names in [("a" * 64, ["EPB_seg_sample_demo"]), ("b" * 64, ["AIA_one", "AIA_two"]), ("normal-wrapper", ["ULF_demo"]), ("c" * 64, ["deep/ULF_nested"])]:
            for name in names:
                package = extracted / wrapper / name
                package.mkdir(parents=True)
                registrations.append({"id": "ds-" + package.name, "name": package.name, "family": "EPB_seg_sample" if name.startswith("EPB") else "ULF", "path": str(package), "labelCount": 3, "dataCount": 3})
        picker = self.make_app(browse_roots=[extracted, public, self.fake.uploads])
        with mock.patch.object(picker, "datasets", return_value=registrations):
            roots = picker.folders()
            self.assertEqual([r["name"] for r in roots["folders"]], ["路演测试数据", "公开训练与验证数据", "浏览器上传"])
            listing = picker.folders(str(extracted))
            by_name = {r["name"]: r for r in listing["folders"]}
            shown = by_name["EPB_seg_sample_demo"]
            self.assertTrue(shown["isDataset"])
            self.assertEqual(shown["family"], "EPB_seg_sample")
            self.assertEqual(Path(shown["path"]), extracted / ("a" * 64) / "EPB_seg_sample_demo")
            self.assertEqual(picker.folders(shown["path"])["selectedDataset"]["id"], shown["datasetId"])
            self.assertFalse(by_name["b" * 64]["isDataset"])
            self.assertFalse(by_name["normal-wrapper"]["isDataset"])
            self.assertFalse(by_name["c" * 64]["isDataset"])
            self.assertEqual(listing["breadcrumbs"][1]["name"], "路演测试数据")
            self.assertEqual(picker.list()["items"], [])
            self.assertEqual(self.fake.count, 0)

    def archive(self, files):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            for name, content in files.items():
                z.writestr(name, content)
        buffer.seek(0)
        return buffer

    def test_zip_upload_registers_exactly_one_new_dataset_and_starts(self):
        stream = self.archive({"AIA_AR_sample_new/data/a.png": b"PNG", "AIA_AR_sample_new/label/a.png.json": b"{}"})
        result = self.app.upload(stream, len(stream.getvalue()), "new.zip")
        self.assertEqual(result["dataset"]["name"], "AIA_AR_sample_new")
        self.assertEqual(self.fake.count, 1)
        self.assertEqual(len(list(self.fake.uploads.glob(".upload*.zip"))), 0)

    def test_zip_traversal_and_multi_package_rejected_without_execution(self):
        stream = self.archive({"../escape.txt": "bad"})
        with self.assertRaises(ChatError):
            self.app.upload(stream, len(stream.getvalue()), "bad.zip")
        self.assertFalse((self.root / "escape.txt").exists())
        self.assertEqual(list(self.fake.uploads.iterdir()), [])
        stream = self.archive({"AIA_AR_sample_one/data/a.png": b"PNG", "AIA_AR_sample_one/label/a.png.json": b"{}", "AIA_AR_sample_two/data/b.png": b"PNG", "AIA_AR_sample_two/label/b.png.json": b"{}"})
        with self.assertRaises(ChatError) as exc:
            self.app.upload(stream, len(stream.getvalue()), "two.zip")
        self.assertEqual(exc.exception.status, 422)
        self.assertEqual(self.fake.count, 0)
        self.assertEqual(list(self.fake.uploads.iterdir()), [])

    def test_archive_declared_size_limit_is_checked_before_extract(self):
        stream = self.archive({"AIA/data/a.txt": "123456789"})
        with mock.patch("spaceqa_agent.chat_service.MAX_EXPANDED", 4):
            with self.assertRaises(ChatError) as exc:
                self.app.upload(stream, len(stream.getvalue()), "too-large.zip")
        self.assertEqual(exc.exception.status, 413)
        self.assertEqual(list(self.fake.uploads.iterdir()), [])

    @contextlib.contextmanager
    def server(self):
        server = create_server("127.0.0.1", 0, upstream="http://127.0.0.1:8781", state_dir=self.root / "http-state", upload_root=self.fake.uploads, transport=self.fake, browse_roots=[self.root / "data"])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)

    def request(self, server, method, path, payload=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        body = None if payload is None else json.dumps(payload)
        fields = {"Content-Type": "application/json"} if body is not None else {}
        fields.update(headers or {})
        conn.request(method, path, body, fields)
        response = conn.getresponse()
        result = json.loads(response.read())
        code = response.status
        conn.close()
        return code, result

    def test_http_session_report_and_origin_boundaries(self):
        with self.server() as server:
            status, created = self.request(server, "POST", "/api/chat/sessions", {"datasetId": "ds-first"})
            self.assertEqual(status, 201)
            identity = created["id"]
            status, complete = self.request(server, "GET", "/api/chat/sessions/" + identity)
            self.assertEqual(status, 200)
            self.assertEqual(complete["status"], "completed")
            status, exported = self.request(server, "GET", "/api/chat/sessions/" + identity + "/report")
            self.assertEqual(status, 200)
            self.assertEqual(exported["result"]["confirmedSamples"], 1)
            status, body = self.request(server, "POST", "/api/chat/sessions", {"datasetId": "ds-first"}, {"Origin": "https://untrusted.example"})
            self.assertEqual(status, 403)
            status, body = self.request(server, "GET", "/api/chat/sessions", headers={"Host": "untrusted.example"})
            self.assertEqual(status, 403)

    def test_rejected_post_bodies_preserve_403_without_executing_requests(self):
        with self.server() as server:
            for index in range(24):
                status, body = self.request(server, "POST", "/api/chat/sessions", {"datasetId": "ds-first", "padding": "x" * (index * 1000)}, {"Origin": "https://untrusted.example"})
                self.assertEqual(status, 403)
                self.assertEqual(body["error"], "不接受跨站请求。")
            self.assertEqual(self.fake.count, 0)
            self.assertEqual(server.application.list()["items"], [])
            # Rejection consumes bytes only; valid same-origin requests still
            # follow the original route and start exactly one real workflow.
            status, body = self.request(server, "POST", "/api/chat/sessions", {"datasetId": "ds-first"}, {"Origin": f"http://127.0.0.1:{server.server_port}"})
            self.assertEqual(status, 201)
            self.assertEqual(self.fake.count, 1)

    def test_rejection_does_not_wait_for_large_or_incomplete_declared_bodies(self):
        with self.server() as server:
            for length in [1024, 10 * 1024**2]:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                started = time.monotonic()
                conn.request("POST", "/api/chat/sessions", None, {"Origin": "https://untrusted.example", "Content-Type": "application/json", "Content-Length": str(length)})
                response = conn.getresponse()
                self.assertEqual(response.status, 403)
                self.assertEqual(json.loads(response.read())["error"], "不接受跨站请求。")
                self.assertLess(time.monotonic() - started, 1.5)
                conn.close()
            self.assertEqual(self.fake.count, 0)

    def test_public_bind_and_nonloopback_upstream_rejected(self):
        with self.assertRaises(ValueError):
            self.make_app_from_public()
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0, upstream="http://127.0.0.1:8781", state_dir=self.root / "public", upload_root=self.fake.uploads)

    def make_app_from_public(self):
        return ChatApplication("https://example.com", self.root / "bad", self.fake.uploads)


if __name__ == "__main__":
    unittest.main()
