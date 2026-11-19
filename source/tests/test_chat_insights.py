"""Real ledger contracts, safe case browsing and non-mutating explanations."""
import copy
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spaceqa_agent.chat_service import ChatApplication, ChatError, create_server
from spaceqa_agent.chat_insights import canonical_case, case_analysis


class InsightFixture:
    def __init__(self, root):
        self.name = "EPB_seg_sample_demo"
        self.package = root / "data" / self.name
        self.package.mkdir(parents=True)
        self.ds = {"id": "ds-epb", "name": self.name, "family": "EPB_seg_sample", "path": str(self.package), "labelCount": 4, "dataCount": 3}
        self.samples = root / "ledger" / self.name / "samples.jsonl"
        self.predictions = root / "predictions" / self.name / "predictions.jsonl"
        self.samples.parent.mkdir(parents=True)
        self.predictions.parent.mkdir(parents=True)
        self.requests = []
        self.rows = []
        for i in range(4):
            sid = self.name + f":s{i}.json"
            checks = {code: {"status": "undetermined", "scopeStatus": "undetermined", "reason": "Needs independent reference"} for code in ["L0-01", "L2-G", "L2-P", "L2-T", "L2-F", "L3-06", "L3-07", "L3-MIS"]}
            checks["L0-01"] = {"status": "fail" if i == 0 else "pass", "scopeStatus": "fail" if i == 0 else "pass"}
            checks["L2-G"] = {"status": "not_applicable", "scopeStatus": "not_applicable"}
            checks["L2-F"] = {"status": "undetermined" if i == 0 else "fail" if i == 1 else "pass", "scopeStatus": "undetermined" if i == 0 else "fail" if i == 1 else "pass"}
            findings = []
            if i in {0, 1}:
                code = "L0-01" if i == 0 else "L2-F"
                findings.append({"findingId": f"case-{i}", "sampleId": sid, "defectType": code, "findingStage": "confirmed", "scoreEligible": True, "sourceKind": "rule", "detail": "Missing file" if i == 0 else "Declared 1024×512, actual 1024×1024", "locator": {"path": str(self.package / f"s{i}.json"), "annotation_pointer": "/basic_annotation/imgSize"} if i else {"path": str(self.package / "s0.json")}, "checkId": "C35", "ruleVersion": "ledger-v1", "measurements": {"declared": [1024, 512], "actual": [1024, 1024], "max_pixel_error": 512} if i else {}, "severity": {"metric": "max_relative_dimension_error", "value": .5, "unit": "ratio"} if i else None})
            self.rows.append({"sampleId": sid, "dirId": self.name, "checks": checks, "findings": findings, "labelPath": str(self.package / f"s{i}.json")})
        self.pred_rows = [{"id": self.name + ":s1.json", "package": self.name, "classes": ["L3-06", "L3-07"], "model_candidates": ["L3-06"], "probabilities": [.9, .2], "label_path": str(self.package / "s1.json")}, {"id": self.name + ":s2.json", "package": self.name, "classes": ["L3-06", "L3-07"], "model_candidates": ["L3-06", "L3-07"], "probabilities": [.8, .7], "label_path": str(self.package / "s2.json")}]
        self.write()
        self.report = root / "report.json"
        self.report.write_text(json.dumps({"reports": [{"inputPath": str(self.package), "name": self.name, "totalSamples": 4, "confirmedSamples": 2, "candidateOnlySamples": 1, "foundSamples": 3, "sourcefiles": [str(self.samples), str(self.predictions)], "defectRecords": {"L0-01": 1, "L2-F": 1, "L3-06": 2, "L3-07": 1}, "evaluatedAt": "2026-11-05T00:00:00+00:00", "dimensions": [{"id": "annotationQuality", "score": None, "state": "undetermined"}]}]}), encoding="utf-8")

    def write(self):
        self.samples.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n", encoding="utf-8")
        self.predictions.write_text("\n".join(json.dumps(r) for r in self.pred_rows) + "\n", encoding="utf-8")

    def __call__(self, path, payload=None):
        self.requests.append((path, payload))
        if path.startswith("/api/datasets?"):
            return {"items": [self.ds], "total": 1}
        if path.startswith("/api/datasets/ds-epb/samples?"):
            query = parse_qs(urlsplit(path).query)["q"][0]
            return {"items": [{"id": query.removesuffix(".json"), "name": query, "labelName": query}], "total": 1}
        if path.startswith("/api/datasets/ds-epb/samples/"):
            return {"id": path.rsplit("/", 1)[-1], "kind": "image", "imageUrl": path + "/image", "width": 1024, "height": 1024, "annotations": [], "label": {"basic_annotation": {"imgSize": [1024, 512]}}, "missingData": path.endswith("s0")}
        raise AssertionError(path)


class ChatInsightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fake = InsightFixture(self.root)
        self.env = mock.patch.dict(os.environ, {"SPACEQA_CHAT_LLM_BASE_URL": "", "SPACEQA_CHAT_LLM_API_KEY": "", "SPACEQA_CHAT_LLM_MODEL": ""})
        self.env.start()
        self.app = ChatApplication("http://127.0.0.1:8781", self.root / "state", self.root / "uploads", transport=self.fake, report_index=self.fake.report)
        self.session = self.app.create({"datasetId": "ds-epb"})
        self.sid = self.session["id"]

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_coverage_unknown_and_not_applicable_never_become_full_marks(self):
        insight = self.app.insights(self.sid)
        axes = {a["code"]: a for a in insight["axes"]}
        self.assertEqual(len(axes), 8)
        self.assertTrue(insight["catalogComplete"])
        self.assertEqual(axes["L0-01"]["qualityScore"], 75)
        self.assertEqual(axes["L0-01"]["state"], "completed")
        self.assertEqual(axes["L2-F"]["state"], "partial")
        self.assertEqual(axes["L2-F"]["coverage"]["ratio"], .75)
        self.assertIsNone(axes["L2-P"]["qualityScore"])
        self.assertEqual(axes["L2-G"]["state"], "not_applicable")
        self.assertIsNone(axes["L2-G"]["qualityScore"])
        self.assertEqual(axes["L3-06"]["foundSamples"], 2)
        self.assertIsNone(axes["L3-06"]["qualityScore"])
        self.assertEqual(axes["L3-06"]["candidateProfileScore"], 50)
        self.assertEqual(insight["summary"]["foundSamples"], 3)
        self.assertIn("annotation_quality", insight["sixDimensions"])

    def test_full_catalog_filter_page_and_deduplicate_findings(self):
        self.fake.rows[0]["findings"].append(copy.deepcopy(self.fake.rows[0]["findings"][0]))
        self.fake.write()
        result = self.app.case_list(self.sid, {"defectType": ["L3-06"], "stage": ["candidate"], "limit": ["1"], "offset": ["1"]})
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["items"]), 1)
        all_rows = self.app.case_list(self.sid, {})
        self.assertEqual(len(all_rows["items"]), 5)
        self.assertEqual(sum(c["caseId"] == "case-0" for c in all_rows["items"]), 1)
        self.assertEqual(self.app.case_list(self.sid, {"q": ["s1"]})["total"], 2)
        with self.assertRaises(ChatError):
            self.app.case_list(self.sid, {"limit": ["1000"]})
        with self.assertRaises(ChatError):
            self.app.case_list(self.sid, {"stage": ["all-files"]})

    def test_nine_tier_archive_is_sample_level_and_allows_cross_membership(self):
        # Make the last sample a genuinely complete, clean record.
        for code in self.fake.rows[3]["checks"]:
            if code != "L2-G":
                self.fake.rows[3]["checks"][code] = {"status": "pass", "scopeStatus": "pass"}
        self.fake.write()
        insight = self.app.insights(self.sid)
        self.assertEqual([row["id"] for row in insight["tiers"]], ["clean", "L0-01", "L2-G", "L2-P", "L2-T", "L2-F", "L3-06", "L3-07", "L3-MIS"])
        clean = self.app.tiers(self.sid, {"tier": ["clean"]})
        self.assertEqual(clean["total"], 1)
        self.assertEqual(clean["items"][0]["stage"], "clean")
        # The same sample can belong to both a confirmed and a candidate tier.
        missing = self.app.tiers(self.sid, {"tier": ["L2-F"]})
        self.assertEqual(missing["total"], 1)
        candidates = self.app.tiers(self.sid, {"tier": ["L3-06"]})
        self.assertEqual(candidates["total"], 2)

    def test_foreign_package_rows_and_unregistered_sources_are_excluded(self):
        foreign = copy.deepcopy(self.fake.rows[0]); foreign["dirId"] = "OTHER"
        foreign["sampleId"] = "OTHER:s0.json"
        foreign["findings"][0]["sampleId"] = foreign["sampleId"]
        foreign["findings"][0]["findingId"] = "foreign-case"
        self.fake.rows.append(foreign)
        self.fake.pred_rows.append({"id": "OTHER:s5.json", "package": "OTHER", "model_candidates": ["L3-06"]})
        self.fake.write()
        result = self.app.case_list(self.sid, {})
        self.assertFalse(result["catalogComplete"])
        self.assertNotIn("foreign-case", {c["caseId"] for c in result["items"]})
        with self.assertRaises(ChatError):
            self.app.case_detail(self.sid, "foreign-case")

    def test_case_detail_preserves_label_box_and_concrete_measurements(self):
        detail = self.app.case_detail(self.sid, "case-1")
        self.assertEqual(detail["sample"]["width"], 1024)
        self.assertIn("512", " ".join(detail["analysis"]["reasoning"]))
        self.assertIn("/basic_annotation/imgSize", detail["analysis"]["location"])
        self.assertIn("最大相对尺寸误差 50.00%", detail["analysis"]["severity"])
        self.assertEqual(detail["case"]["stage"], "confirmed")
        self.assertTrue(detail["case"]["scoreEligible"])
        self.assertEqual(detail["analysis"]["expertContributions"][0]["name"], "确定性规则")
        missing = self.app.case_detail(self.sid, "case-0")
        self.assertTrue(missing["sample"]["missingData"])

    def test_case_explanation_is_selected_case_only_cached_and_non_mutating(self):
        original = copy.deepcopy(self.app.get(self.sid, refresh=False))
        self.app.llm_model = "test-model"
        captured = []
        def llm(session, query, fallback):
            captured.append(copy.deepcopy(session))
            return "只解释当前记录的尺寸矛盾。", "grounded_llm"
        with mock.patch.object(self.app, "_llm_answer", side_effect=llm) as call:
            first = self.app.case_explain(self.sid, "case-1", {})
            repeat = self.app.case_explain(self.sid, "case-1", {})
        self.assertEqual(call.call_count, 1)
        self.assertFalse(first["cached"])
        self.assertTrue(repeat["cached"])
        self.assertTrue(repeat["statusPreserved"])
        self.assertEqual([c["caseId"] for c in captured[0]["analysis"]["result"]["cases"]], ["case-1"])
        self.assertEqual(self.app.get(self.sid, refresh=False), original)
        with self.assertRaises(ChatError):
            self.app.case_explain(self.sid, "case-1", {"path": "/etc/passwd"})

    def test_fusion_probabilities_and_class_candidates_keep_distinct_meanings(self):
        raw = {"sampleId": self.fake.name + ":s3.json", "defectType": "L2-P", "findingStage": "candidate", "scoreEligible": False,
               "sourceKind": "hgb8", "detail": "Sample-level classifier candidate", "fusion_class_probability": .8, "fusion_group_probability": .95}
        case = canonical_case(raw, self.fake.name)
        analysis = case_analysis(case)
        self.assertEqual(case["fusionClassProbability"], .8)
        self.assertEqual(case["fusionGroupProbability"], .95)
        self.assertIn("可能", analysis["impact"])
        self.assertNotIn("不满足本次类别契约", analysis["impact"])
        self.assertEqual([r["probability"] for r in analysis["expertContributions"]], [None, .8, .95])
        self.assertFalse(case["scoreEligible"])

    def test_timeseries_preview_never_claims_row_index_is_physical_time(self):
        original_transport = self.app.transport
        def transport(path, payload=None):
            result = original_transport(path, payload)
            if "/samples/" in path and "?" not in path:
                result["kind"] = "timeseries"
                result["series"] = [{"name": "value", "points": [[0, 1], [1, 2]]}]
            return result
        self.app.transport = transport
        detail = self.app.case_detail(self.sid, "case-1")
        self.assertEqual(detail["sample"]["seriesAxis"]["kind"], "row_index")
        self.assertFalse(detail["sample"]["seriesAxis"]["eventOverlayAllowed"])

    def full_execution_records(self):
        codes = ["L0-01", "L2-G", "L2-P", "L2-T", "L2-F", "L3-06", "L3-07", "L3-MIS"]
        predictions = []
        for index, ledger in enumerate(self.fake.rows):
            ledger["dataPresent"] = index != 0
            candidates = ["L3-06"] if index == 1 else ["L3-06", "L3-07"] if index == 2 else []
            rules = [code == "L0-01" and index == 0 or code == "L2-F" and index == 1 for code in codes]
            predictions.append({"id": ledger["sampleId"], "package": self.fake.name, "classes": codes,
                                "prediction": [rules[i] or code in candidates for i, code in enumerate(codes)], "rule": rules,
                                "probabilities": [.9 if code in candidates else 1.0 if rules[i] else 0.0 for i, code in enumerate(codes)],
                                "model_candidates": candidates, "input_sha256": f"actual-input-hash-{index}" if index else None,
                                "label_path": ledger["labelPath"]})
        self.fake.pred_rows = predictions
        self.fake.write()

    def test_executed_zero_detection_gets_display_100_without_confirming_scientific_quality(self):
        self.full_execution_records()
        axes = {a["code"]: a for a in self.app.insights(self.sid)["axes"]}
        self.assertEqual(axes["L2-P"]["displayScore"], 100)
        self.assertEqual(axes["L2-P"]["stateLabel"], "已执行")
        self.assertEqual(axes["L2-P"]["classificationCoverage"]["modelEvaluatedSamples"], 3)
        self.assertEqual(axes["L2-P"]["classificationCoverage"]["ruleOnlyMissingDataSamples"], 1)
        self.assertTrue(axes["L2-P"]["classificationCoverage"]["complete"])
        self.assertIsNone(axes["L2-P"]["confirmedQualityScore"])
        self.assertEqual(axes["L2-P"]["state"], "undetermined")
        self.assertEqual(axes["L3-06"]["displayScore"], 50)
        self.assertIsNone(axes["L2-G"]["displayScore"])
        self.assertEqual(axes["L2-G"]["stateLabel"], "不适用")
        candidates = self.app.case_list(self.sid, {"defectType": ["L3-06"]})["items"]
        self.assertTrue(all(c["stage"] == "candidate" and not c["scoreEligible"] for c in candidates))

    def test_missing_or_incomplete_execution_never_fills_zero_detection_with_100(self):
        first = {a["code"]: a for a in self.app.insights(self.sid)["axes"]}
        self.assertIsNone(first["L2-P"]["displayScore"])
        self.assertEqual(first["L2-P"]["stateLabel"], "未执行")
        self.full_execution_records()
        self.fake.pred_rows[3]["probabilities"][2] = None
        self.fake.write()
        second = {a["code"]: a for a in self.app.insights(self.sid)["axes"]}
        self.assertIsNone(second["L2-P"]["displayScore"])
        self.assertEqual(second["L2-P"]["stateLabel"], "部分覆盖")
        self.assertFalse(second["L2-P"]["classificationCoverage"]["complete"])

    def test_global_localization_coverage_is_not_assigned_to_every_defect_axis(self):
        self.app.saved_reports[0]["localizationCoverage"] = {"ratio": 1.0, "level": "region", "L2-G": {"ratio": .5, "level": "region"}}
        result = self.app.insights(self.sid)
        axes = {a["code"]: a for a in result["axes"]}
        self.assertEqual(axes["L2-G"]["localizationCoverage"], {"ratio": .5, "level": "region"})
        self.assertEqual(axes["L0-01"]["localizationCoverage"], {})
        self.assertEqual(axes["L3-06"]["localizationCoverage"], {})
        self.assertEqual(result["localizationCoverage"]["ratio"], 1.0)

    def test_bbox_analysis_does_not_invent_pixel_coordinate_units(self):
        base = {"sampleId": self.fake.name + ":s2.json", "defectType": "L2-G", "stage": "candidate", "scoreEligible": False, "locator": {"bbox": [.1, .2, .3, .4]}}
        unknown = case_analysis(canonical_case(base, self.fake.name))
        self.assertIn("未声明坐标单位", unknown["location"])
        self.assertNotIn("原图像素", unknown["location"])
        base["locator"]["coordinate_system"] = "image_normalized"
        normalized = case_analysis(canonical_case(base, self.fake.name))
        self.assertIn("归一化坐标", normalized["location"])
        base["locator"]["coordinate_system"] = "image_pixels"
        pixels = case_analysis(canonical_case(base, self.fake.name))
        self.assertIn("原图像素", pixels["location"])

    def test_new_completed_report_refreshes_existing_conversation_without_duplicate_summary(self):
        self.app.answer(self.sid, {"message": "保留这条用户追问"})
        old = self.app.get(self.sid, refresh=False)
        old_summary = next(m for m in old["messages"] if m.get("kind") == "analysis_summary")
        report = json.loads(self.fake.report.read_text(encoding="utf-8"))
        report["reports"][0].update(reportVersion="fresh-v2", reportLabel="新完成的真实推理", evaluatedAt="2026-11-05T08:30:00+00:00", foundSamples=4, candidateOnlySamples=2)
        self.fake.report.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        updated = self.app.get(self.sid)
        self.assertEqual(updated["id"], old["id"])
        self.assertEqual(updated["createdAt"], old["createdAt"])
        self.assertEqual(len(updated["messages"]), len(old["messages"]))
        self.assertEqual([m for m in updated["messages"] if m["role"] == "user"], [m for m in old["messages"] if m["role"] == "user"])
        summaries = [m for m in updated["messages"] if m.get("kind") == "analysis_summary"]
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["id"], old_summary["id"])
        self.assertIn("4 条", summaries[0]["content"])
        self.assertEqual(updated["analysis"]["result"]["foundSamples"], 4)
        self.assertEqual(updated["analysis"]["result"]["reportVersion"], "fresh-v2")
        self.assertEqual(self.app.create({"datasetId": "ds-epb"})["id"], self.sid)
        # Existing candidate status is unchanged by loading a newer report.
        self.assertTrue(all(not c["scoreEligible"] for c in self.app.case_list(self.sid, {"stage": ["candidate"]})["items"]))
        restarted = ChatApplication("http://127.0.0.1:8781", self.root / "state", self.root / "uploads", transport=self.fake, report_index=self.fake.report)
        self.assertEqual(restarted.get(self.sid)["analysis"]["result"]["reportVersion"], "fresh-v2")
        self.assertEqual(len(restarted.get(self.sid)["messages"]), len(old["messages"]))

    def test_language_model_does_not_block_progress_reads(self):
        for action in ['case', 'chat']:
            entered, release = threading.Event(), threading.Event()
            def slow_model(*args):
                entered.set()
                release.wait(3)
                return '依据当前记录解释。', 'grounded_llm'
            with mock.patch.object(self.app, '_llm_answer', side_effect=slow_model), ThreadPoolExecutor(max_workers=2) as pool:
                future=pool.submit(self.app.case_explain,self.sid,'case-1',{'message':'并发核验-'+action}) if action=='case' else pool.submit(self.app.answer,self.sid,{'message':'并发核验-'+action})
                try:
                    self.assertTrue(entered.wait(1))
                    read=pool.submit(self.app.get,self.sid)
                    self.assertEqual(read.result(timeout=.5)['id'],self.sid)
                finally:
                    release.set()
                    future.result(timeout=2)

    def test_http_endpoints_are_integrated(self):
        server = create_server(upstream="http://127.0.0.1:8781", port=0, state_dir=self.root / "state", upload_root=self.root / "uploads", transport=self.fake, report_index=self.fake.report)
        thread = threading.Thread(target=server.serve_forever, daemon=True);thread.start()
        try:
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            for suffix in ["insights", "cases?limit=1", "tiers?tier=L3-06&limit=1", "cases/case-1"]:
                connection.request("GET", "/api/chat/sessions/" + self.sid + "/" + suffix)
                response = connection.getresponse();body = json.loads(response.read())
                self.assertEqual(response.status, 200, body)
            connection.request("POST", "/api/chat/sessions/" + self.sid + "/cases/case-1/explain", body="{}", headers={"Content-Type": "application/json"})
            response = connection.getresponse();body = json.loads(response.read())
            self.assertEqual(response.status, 200, body)
            self.assertEqual(body["replySource"], "structured_evidence")
            connection.close()
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
