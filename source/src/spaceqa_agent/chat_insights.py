"""Evidence-backed dataset portraits and case-by-case demo explanations.

This module is intentionally stdlib-only so the isolated chat deployment can
load it next to chat_service.py without importing the scientific worker.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from urllib.parse import quote

DEFECTS = (
    ("L0-01", "数据文件缺失"), ("L2-G", "边界与位置偏差"),
    ("L2-P", "类别标注错误"), ("L2-T", "时段起止时刻抖动"),
    ("L2-F", "元数据尺寸矛盾"), ("L3-06", "漏标"),
    ("L3-07", "虚标"), ("L3-MIS", "覆盖不足或覆盖过度"),
)
NAMES = dict(DEFECTS)
TIER_DEFS = (("clean", "干净样本"),) + DEFECTS
SOURCE_NAMES = {"rule": "确定性规则", "independent_RGB_model": "独立RGB定位模型", "hgb8": "冻结八类融合候选"}


def canonical(code):
    return "L0-01" if code == "L0-03" else code


def short_name(value):
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sample_identity(row):
    return str(row.get("sampleId") or row.get("id") or "")


def canonical_case(raw, package, sample=None, source=""):
    finding = raw.get("finding") if isinstance(raw.get("finding"), dict) else raw
    code = canonical(finding.get("defectType", finding.get("defect_type")))
    if code not in NAMES or raw.get("findingStage") == "observation":
        return None
    identity = str(raw.get("sampleId") or finding.get("sampleId") or (sample_identity(sample) if sample else ""))
    if ":" in identity and identity.split(":", 1)[0] != package:
        return None
    label_name = short_name(raw.get("labelName") or (sample or {}).get("labelPath") or identity.rsplit(":", 1)[-1])
    stage = raw.get("stage", raw.get("findingStage", finding.get("findingStage")))
    eligible = raw.get("scoreEligible", finding.get("scoreEligible", False)) is True
    confirmed = stage == "confirmed" and eligible
    locator = finding.get("locator") or raw.get("locator") or {}
    if not isinstance(locator, dict):
        locator = {"description": str(locator)[:1000]}
    key = raw.get("caseId") or raw.get("findingId") or finding.get("findingId")
    if not key:
        token = json.dumps([identity, code, stage, locator], ensure_ascii=False, sort_keys=True)
        key = "case-" + hashlib.sha256(token.encode()).hexdigest()[:24]
    case = {"caseId": str(key), "sampleId": identity, "sampleName": label_name,
            "labelName": label_name, "defectType": code, "defectName": NAMES[code],
            "stage": "confirmed" if confirmed else "candidate", "scoreEligible": confirmed,
            "basis": finding.get("basis") or finding.get("detail") or raw.get("basis") or "见原始检查记录",
            "locator": copy.deepcopy(locator), "checkId": finding.get("checkId", raw.get("checkId")),
            "ruleVersion": finding.get("ruleVersion", raw.get("ruleVersion")),
            "confidence": finding.get("confidence", raw.get("confidence")),
            "severity": copy.deepcopy(finding.get("severity", raw.get("severity"))),
            "measurements": copy.deepcopy(finding.get("measurements", raw.get("measurements"))) or {},
            "sourceKind": finding.get("sourceKind", raw.get("sourceKind")) or "unknown",
            "sourcefile": source or raw.get("sourcefile"),
            "fusionClassProbability": raw.get("fusionClassProbability", raw.get("fusion_class_probability")),
            "fusionGroupProbability": raw.get("fusionGroupProbability", raw.get("fusion_group_probability"))}
    return case


def case_analysis(case):
    if case.get("defectType") == "CLEAN":
        return {"summary": f"{case.get('sampleName') or case.get('sampleId')}：未发现已记录问题。",
                "reasoning": ["该样本已完成本次登记的检查，且没有对应的已确认问题或模型候选。"],
                "location": "整份样本；没有问题定位区域。", "severity": "无已记录缺陷。",
                "impact": "在当前检查范围内作为干净样本展示。此状态不替代独立科学真值。",
                "recommendation": "如需确认科学正确性，请结合独立参考数据和领域规则复核。",
                "statusMeaning": "干净档案表示当前运行没有记录到问题，不能推断未运行或未覆盖的检查通过。",
                "expertContributions": []}
    locator = case.get("locator") or {}
    measure = case.get("measurements") or {}
    location = []
    if case.get("labelName"):
        location.append(case["labelName"])
    pointer = locator.get("annotation_pointer") or locator.get("annotationPointer")
    if pointer:
        location.append("标注字段 " + str(pointer))
    if locator.get("bbox") is not None:
        coordinate = locator.get("coordinate_system")
        if coordinate == "image_pixels":
            unit = "原图像素"
        elif coordinate in {"normalized", "image_normalized", "normalized_0_1"}:
            unit = "归一化坐标"
        else:
            unit = str(coordinate) if coordinate else "原记录未声明坐标单位"
        location.append("图像框 " + json.dumps(locator["bbox"], ensure_ascii=False) + "，坐标制：" + unit)
    interval = locator.get("timeInterval", locator.get("time_interval"))
    if interval is not None:
        location.append("时间区间 " + json.dumps(interval, ensure_ascii=False) + "（采用原记录单位）")
    if not pointer and not locator.get("bbox") and interval is None:
        location.append("定位到样本/文件；原记录未提供可核验的区域或时间定位")
    severity = case.get("severity")
    severity_text = "原记录未量化严重程度。"
    if isinstance(severity, dict):
        value = severity.get("value", severity.get("normalized"))
        metric = severity.get("metric", "原记录程度指标")
        ratio_names = {"max_relative_dimension_error": "最大相对尺寸误差", "affected_object_fraction": "受影响标注对象比例"}
        count_names = {"unmatched_predicted_target": "未匹配预测目标数", "unmatched_predicted_target_count": "未匹配预测目标数",
                       "unmatched_annotation_object": "未匹配标注对象数", "unmatched_annotation_count": "未匹配标注对象数",
                       "missing_data_count": "缺失数据文件数", "invalid_object_count": "无效类别标注数"}
        if metric in ratio_names and finite(value):
            severity_text = f"{ratio_names[metric]} {value:.2%}（原始指标 {value}）"
        elif metric in count_names and finite(value):
            number = int(value) if int(value) == value else value
            severity_text = f"{count_names[metric]}：{number} 个" + ("（候选计数）" if case["stage"] != "confirmed" else "")
        elif metric in {"mask_iou", "bbox_iou", "event_iou"}:
            severity_text = f"{ {'mask_iou': '掩膜', 'bbox_iou': '边界框', 'event_iou': '事件时段'}[metric]}交并比 IoU：{value}（原始度量，不能单独证明缺陷或错误程度）"
        else:
            severity_text = f"{metric}：{value} {severity.get('unit', '')}".strip()
    elif severity is not None:
        severity_text = str(severity)
    confirmed = case["stage"] == "confirmed"
    impacts = {
        "L0-01": ("数据与标注不能完整配对，相关样本无法执行依赖原始数据的检查。", "核对原始包、引用文件名与传输清单；本系统保留缺失记录，不自动补图。"),
        "L2-F": ("标注声明尺寸与真实图像不一致，可能影响坐标解释和后续缩放。", "对照PNG头及尺寸字段人工核验，确认采用的坐标空间。"),
        "L2-P": ("类别字段不满足本次类别契约；集合内类别的科学语义仍需独立参考。", "核查该条类别字段与允许类别表，不将契约异常自动解释为科学物理错误。"),
        "L2-G": ("边界或位置可能偏离目标，影响目标定位与区域统计。", "对照当前图像、输入标注和独立区域证据核验边界。"),
        "L2-T": ("事件区间可能偏移，影响事件覆盖与时段统计。", "使用原始时间轴、事件定义与独立事件参考核验起止时刻。"),
        "L3-06": ("可能存在未被当前标注覆盖的目标或事件。", "检查候选区域/时段是否确为任务目标，再与当前标注逐一匹配。"),
        "L3-07": ("当前标注可能缺少相应真实目标支持。", "逐条核对图像/曲线中的证据；没有标注不能单独证明虚标。"),
        "L3-MIS": ("当前标注可能覆盖过少或过多，影响区域或事件范围统计。", "结合目标定义核查输入标注与独立区域参考的覆盖关系。"),
    }
    impact, recommendation = impacts[case["defectType"]]
    if case["defectType"] == "L2-P" and not (measure.get("actual") is not None and measure.get("allowed") is not None):
        impact = "模型提示类别可能不一致，尚需结合当前数据、任务类别定义和独立参考核验。" if not confirmed else "原检查记录确认类别异常，具体依据以该条原始证据为准。"
    reasoning = [str(case["basis"])]
    if measure.get("declared") is not None and measure.get("actual") is not None:
        reasoning.append("声明值 " + json.dumps(measure["declared"], ensure_ascii=False) + "；实测值 " + json.dumps(measure["actual"], ensure_ascii=False) + "。")
    if measure.get("max_pixel_error") is not None:
        reasoning.append(f"最大尺寸差为 {measure['max_pixel_error']} 像素。")
    if measure.get("actual") is not None and measure.get("allowed") is not None:
        reasoning.append("当前类别 " + str(measure["actual"]) + "；允许类别 " + json.dumps(measure["allowed"], ensure_ascii=False) + "。")
    contributions = [{"name": SOURCE_NAMES.get(case.get("sourceKind"), str(case.get("sourceKind") or "原检查工具")),
                      "role": "提供直接核验证据" if confirmed else "提出待复核候选",
                      "probability": case.get("confidence") if finite(case.get("confidence")) else None,
                      "evidence": f"检查编号 {case.get('checkId') or '未提供'}；规则版本 {case.get('ruleVersion') or '未提供'}"}]
    if finite(case.get("fusionClassProbability")):
        contributions.append({"name": "冻结八类融合输出", "role": "样本级类别候选概率",
                              "probability": case["fusionClassProbability"], "evidence": "概率不是正确率或F1，也不能单独确认缺陷。"})
    if finite(case.get("fusionGroupProbability")):
        contributions.append({"name": "G/C/D粗粒度融合输出", "role": "触发进一步定位调查的分组概率",
                              "probability": case["fusionGroupProbability"], "evidence": "分组概率与八类细分类概率不同，不能替代独立定位证据。"})
    return {"summary": f"{case['sampleName']}：{case['defectName']}，{'已确认' if confirmed else '待复核候选'}。",
            "reasoning": reasoning, "location": "；".join(location), "severity": severity_text,
            "impact": impact, "recommendation": recommendation,
            "statusMeaning": "本条有可核验证据，保留原记录的可计分确认状态。" if confirmed else "本条是候选，未被自动提升为确认问题，不参与确认缺陷扣分。",
            "expertContributions": contributions}


class InsightsMixin:
    """Expects ChatApplication's error, API, session and LLM adapters."""

    def _insight_state(self):
        with self.lock:
            if not hasattr(self, "_insight_cache"):
                self._insight_cache = {}
                self._insight_catalog_lock = threading.RLock()
        return self._insight_catalog_lock

    def _report_for_session(self, session):
        # Match the exact registered input path, never names alone.
        ds = next((r for r in self.datasets() if r["id"] == session["dataset"]["id"]), None)
        if not ds:
            raise self.insight_error(404, "当前数据集未登记。")
        path = Path(ds["path"]).resolve()
        report = session.get("_liveReport") if session.get("analysis", {}).get("source") == "live_recompute" else None
        report = report or next((r for r in self.saved_reports if r.get("inputPath") and Path(r["inputPath"]).resolve() == path), None)
        return ds, report

    def _case_catalog(self, identity):
        session = self.get(identity)
        facts = session.get("analysis", {}).get("result") or {}
        ds, report = self._report_for_session(session)
        source_files = []
        if session.get("analysis", {}).get("source") in {"saved_report", "live_recompute"} and report:
            for name in report.get("sourcefiles", []):
                p = Path(name)
                if p.name not in {"samples.jsonl", "predictions.jsonl", "coverage.json"} or p.parent.name != ds["name"] or p.is_symlink():
                    continue
                try:
                    st = p.stat()
                    if st.st_size > 256 * 1024**2:
                        continue
                    source_files.append((p, st.st_mtime_ns, st.st_size))
                except OSError:
                    continue
        stamp = (identity, facts.get("taskId"), facts.get("evaluatedAt"), facts.get("reportSignature"), tuple((str(p), m, n) for p, m, n in source_files))
        lock = self._insight_state()
        with lock:
            if stamp in self._insight_cache:
                return self._insight_cache[stamp]
            cases, checks, seen_ids, prediction_rows, data_present = [], {}, set(), {}, {}
            execution_metadata = {k: copy.deepcopy(facts.get(k) or {}) for k in ("classificationCoverage", "localizationCoverage", "inferenceCoverage")}
            ledger_rows, skipped_rows = 0, 0
            input_records = []
            for p, _, _ in sorted(source_files, key=lambda v: v[0].name != "samples.jsonl"):
                if p.name == "coverage.json":
                    metadata = json.loads(p.read_text(encoding="utf-8-sig"))
                    if isinstance(metadata, dict):
                        execution_metadata.update({k: copy.deepcopy(metadata[k]) for k in execution_metadata if k in metadata})
                    continue
                with p.open(encoding="utf-8-sig") as stream:
                    for index, line in enumerate(stream):
                        if index >= 300_000 or len(line) > 16 * 1024**2:
                            raise self.insight_error(413, "案例来源超出演示读取上限。")
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeError):
                            skipped_rows += 1
                            continue
                        sid = sample_identity(row)
                        owner = row.get("dirId", row.get("package"))
                        if owner != ds["name"] or not sid.startswith(ds["name"] + ":"):
                            skipped_rows += 1
                            continue
                        if p.name == "samples.jsonl":
                            ledger_rows += 1
                            input_records.append({k: row.get(k) for k in ('sampleId','family','labelPath','dataPath','labelSha256','checks','imageSize','observationInterval')})
                            checks[sid] = row.get("checks") if isinstance(row.get("checks"), dict) else {}
                            data_present[sid] = row.get("dataPresent")
                            for finding in row.get("findings", []):
                                case = canonical_case(finding, ds["name"], row, str(p))
                                if case and case["caseId"] not in seen_ids:
                                    seen_ids.add(case["caseId"]); cases.append(case)
                        else:
                            prediction_rows[sid] = row
            # Classifier candidates remain sample-level even when another
            # independently localized candidate exists for the same sample.
            for sid, row in prediction_rows.items():
                classes = row.get("classes") or []
                probabilities = row.get("probabilities") or []
                for code in row.get("model_candidates") or []:
                    if canonical(code) not in NAMES:
                        continue
                    probability = probabilities[classes.index(code)] if code in classes and classes.index(code) < len(probabilities) else None
                    raw = {"sampleId": sid, "defectType": code, "stage": "candidate", "scoreEligible": False,
                           "sourceKind": "hgb8", "checkId": "eight-class-candidate", "ruleVersion": report.get("reportLabel", "frozen-eight-class") if report else "frozen-eight-class",
                           "basis": "冻结八类分类器输出的样本级候选；原记录尚无独立区域或时间真值确认。",
                           "locator": {"type": "sample", "path": row.get("label_path"), "data_path": row.get("data_path"), "localization_status": row.get("candidate_localization_status")},
                           "fusionClassProbability": probability, "confidence": probability}
                    case = canonical_case(raw, ds["name"], source="predictions.jsonl")
                    if case and case["caseId"] not in seen_ids:
                        seen_ids.add(case["caseId"]); cases.append(case)
            complete = ledger_rows == len(checks) == facts.get("totalSamples") and bool(source_files) and skipped_rows == 0
            if not source_files:
                if session.get("analysis", {}).get("source") == "live" and facts.get("taskId"):
                    raw_cases, complete = self._cases(facts["taskId"])
                else:
                    raw_cases, complete = facts.get("cases", []), False
                for raw in raw_cases:
                    if raw.get("packageId") and raw["packageId"] not in {ds["id"], ds["name"]}:
                        continue
                    case = canonical_case(raw, ds["name"])
                    if case and case["caseId"] not in seen_ids:
                        seen_ids.add(case["caseId"]); cases.append(case)
            cases.sort(key=lambda c: (c["stage"] != "confirmed", list(NAMES).index(c["defectType"]), c["sampleName"], c["caseId"]))
            result = {"session": session, "facts": facts, "dataset": ds, "cases": cases, "checks": checks,
                      "predictionRows": prediction_rows, "dataPresent": data_present, "executionMetadata": execution_metadata,
                      "catalogComplete": complete, "sourceFilesRead": len(source_files), "ledgerRows": ledger_rows,
                      "skippedSourceRows": skipped_rows, "inputRecords": input_records, "stamp": hashlib.sha256(repr(stamp).encode()).hexdigest()}
            self._insight_cache[stamp] = result
            return result

    def insights(self, identity):
        cat = self._case_catalog(identity)
        facts = cat["facts"]
        total = facts.get("totalSamples")
        by_code = {code: {"confirmed": set(), "candidate": set()} for code in NAMES}
        for case in cat["cases"]:
            by_code[case["defectType"]][case["stage"]].add(case["sampleId"])
        axes = []
        for code, name in DEFECTS:
            observed = collections.Counter()
            raw_counts = collections.Counter()
            reasons = collections.Counter()
            for checks in cat["checks"].values():
                check = checks.get(code) or (checks.get("L0-03") if code == "L0-01" else None)
                if not isinstance(check, dict):
                    continue
                status = check.get("scopeStatus", check.get("status", "undetermined"))
                observed[status] += 1
                raw_counts[check.get("status", "undetermined")] += 1
                reasons[str(check.get("reason") or "原记录未说明")] += 1
            completed = observed["pass"] + observed["fail"]
            na = observed["not_applicable"]
            unknown = max(0, (total or 0) - completed - na)
            applicable = max(0, (total or 0) - na)
            cset, pset = by_code[code]["confirmed"], by_code[code]["candidate"]
            confirmed, candidate, found = len(cset), len(pset - cset), len(cset | pset)
            stage_counts = (facts.get("defectCountsByStage") or {}).get(code)
            if not cat["catalogComplete"] and isinstance(stage_counts, dict):
                confirmed = stage_counts.get("confirmedSamples", confirmed)
                candidate = stage_counts.get("candidateOnlySamples", candidate)
                found = stage_counts.get("foundSamples", confirmed + candidate)
            if total and na == total:
                state = "not_applicable"
            elif completed and completed == applicable and cat["ledgerRows"] == total:
                state = "completed"
            elif completed:
                state = "partial"
            else:
                state = "undetermined"
            can_score = state in {"completed", "partial"} and bool(total) and 0 <= found <= total
            detected_risk = 100 * found / total if total and (found or completed) else None
            confirmed_risk = 100 * confirmed / total if total and (confirmed or completed) else None
            profile = 100 - detected_risk if detected_risk is not None and state != "not_applicable" else None
            # A scientific verdict may need independent reference even though
            # the classifier really ran. Keep the old verdict/quality fields,
            # but derive the display portrait from actual execution records.
            valid_records, model_evaluated, missing_rule_only = set(), set(), set()
            missing_data = {sid for sid, present in cat["dataPresent"].items() if present is False}
            for sid, row in cat["predictionRows"].items():
                classes = row.get("classes") or []
                probabilities = row.get("probabilities") or []
                prediction = row.get("prediction") or []
                rule = row.get("rule") or []
                if classes.count(code) != 1:
                    continue
                slot = classes.index(code)
                if slot >= len(probabilities) or slot >= len(prediction) or slot >= len(rule):
                    continue
                if not finite(probabilities[slot]) or not 0 <= probabilities[slot] <= 1 or type(prediction[slot]) is not bool or type(rule[slot]) is not bool:
                    continue
                valid_records.add(sid)
                if row.get("input_sha256") and sid not in missing_data:
                    model_evaluated.add(sid)
                elif sid in missing_data:
                    missing_rule_only.add(sid)
            eligible = max(0, (total or 0) - len(missing_data))
            full_model_execution = bool(total) and eligible > 0 and len(valid_records) == total and len(model_evaluated) == eligible and missing_rule_only == missing_data
            full_rule_execution = state == "completed"
            execution_complete = full_model_execution or full_rule_execution
            if state == "not_applicable":
                display_state, state_label, display_score = "not_applicable", "不适用", None
            elif execution_complete:
                display_state, state_label = "completed", "已执行"
                display_score = 100 - 100 * found / total if total and 0 <= found <= total else None
            elif completed or model_evaluated or found:
                display_state, state_label = "partial", "部分覆盖"
                # A positive record supports a partial portrait. Zero findings
                # in an incompletely executed pack do not justify 100 points.
                display_score = 100 - 100 * found / total if total and 0 < found <= total else None
            else:
                display_state, state_label, display_score = "not_executed", "未执行", None
            classification_coverage = {"recordSamples": len(valid_records), "modelEvaluatedSamples": len(model_evaluated),
                                       "ruleOnlyMissingDataSamples": len(missing_rule_only), "eligibleModelSamples": eligible,
                                       "totalSamples": total, "complete": full_model_execution,
                                       "ratio": len(model_evaluated) / eligible if eligible else None,
                                       "sourceKind": "persisted_eight_class_execution" if valid_records else "none",
                                       "evaluatedAt": facts.get("evaluatedAt"), "timestampBasis": "report_evaluated_at",
                                       "meaning": "预测记录包含八类输出与输入哈希字段；缺数据样本单列为规则处理，不冒充图像模型前向。"}
            localization_metadata = cat["executionMetadata"].get("localizationCoverage") or {}
            axis_localization = localization_metadata.get(code, {}) if isinstance(localization_metadata, dict) else {}
            if not isinstance(axis_localization, dict):
                axis_localization = {}
            axes.append({"code": code, "name": name, "applicable": False if state == "not_applicable" else True if applicable and cat["checks"] else None,
                         "state": state, "confirmedSamples": confirmed, "candidateOnlySamples": candidate, "foundSamples": found,
                         "denominator": total, "confirmedRisk": confirmed_risk, "detectedRisk": detected_risk,
                         "qualityScore": 100 - 100 * found / total if can_score else None,
                         "confirmedQualityScore": 100 - 100 * confirmed / total if can_score else None,
                         "candidateProfileScore": profile, "coverage": {"completedSamples": completed, "applicableSamples": applicable,
                         "undeterminedSamples": unknown, "notApplicableSamples": na, "ratio": completed / applicable if applicable else None},
                         "checkCounts": dict(raw_counts), "scopeCounts": dict(observed), "reasons": [r for r, _ in reasons.most_common(3)],
                         "scopeMismatch": state == "not_applicable" and found > 0,
                         "displayScore": display_score, "displayState": display_state, "stateLabel": state_label,
                         "classificationCoverage": classification_coverage,
                         "localizationCoverage": copy.deepcopy(axis_localization),
                         "displayScoreMeaning": "八类检出画像分；100只表示本次已执行范围未检出该类，不证明科学上无缺陷。",
                         "scoreMeaning": "当前检查范围的检出画像展示分；包含候选时不代表确认扣分，缺项不记满分。"})
        try:
            from .chat_reporting import enrich_insights
        except ImportError:
            from chat_reporting import enrich_insights
        profile = self._scoped_six_dimensions(cat)
        previous = (facts.get('coverage') or {}).get('dimensions', {})
        previous = {r['id']:r for r in previous} if isinstance(previous,list) else previous
        previous = {k:r for k,r in previous.items() if isinstance(r,dict)}
        dimensions = {k:dict(previous.get(k,{}),**r) for k,r in profile.get('dimensions',{}).items()} or previous
        tiers = self._tier_summary(cat)
        # Build the formal eight-class pollution diagnostic from the same
        # sample records that feed the UI.  It is kept separate from the
        # six-dimensional technical observations and from official P/R/F1.
        try:
            from .pollution_metrics import compute_pollution_index
            pollution_rows = {str(row.get("sampleId")): {"sampleId": row.get("sampleId"), "defectProbabilities": {}, "confirmedCodes": []}
                              for row in cat.get("inputRecords", []) if row.get("sampleId")}
            for case in cat.get("cases", []):
                sid = str(case.get("sampleId", ""))
                item = pollution_rows.setdefault(sid, {"sampleId": sid, "defectProbabilities": {}, "confirmedCodes": []})
                code = canonical(case.get("defectType"))
                probability = case.get("fusionClassProbability", case.get("confidence"))
                if finite(probability):
                    item["defectProbabilities"][code] = max(item["defectProbabilities"].get(code, 0.0), float(probability))
                if case.get("stage") == "confirmed" and case.get("scoreEligible") is True and code not in item["confirmedCodes"]:
                    item["confirmedCodes"].append(code)
            for sid, row in cat.get("predictionRows", {}).items():
                item = pollution_rows.setdefault(str(sid), {"sampleId": sid, "defectProbabilities": {}, "confirmedCodes": []})
                classes, probabilities = row.get("classes") or [], row.get("probabilities") or []
                if isinstance(classes, list) and isinstance(probabilities, list):
                    for code, probability in zip(classes, probabilities):
                        code = canonical(code)
                        if code in NAMES and finite(probability):
                            item["defectProbabilities"][code] = max(item["defectProbabilities"].get(code, 0.0), float(probability))
            pollution = compute_pollution_index(list(pollution_rows.values()), sample_count=facts.get("totalSamples") or len(pollution_rows))
        except Exception as exc:
            pollution = {"pollutionIndex": None, "error": str(exc), "meaning": "无法从当前报告复算概率污染指数"}
        expert_audit = self._expert_fusion_audit(cat)
        return enrich_insights({"schemaVersion": "spaceqa-insights-v1", "sessionId": identity,
                "datasetName": cat["dataset"]["name"], "evaluatedAt": facts.get("evaluatedAt"),
                "summary": {k: facts.get(k) for k in ("totalSamples", "confirmedSamples", "candidateOnlySamples", "foundSamples", "detectedRatio", "observedTechnicalScore")},
                "reportVersion": facts.get("reportVersion"), "reportSignature": facts.get("reportSignature"),
                "classificationCoverage": copy.deepcopy(cat["executionMetadata"].get("classificationCoverage") or {}),
                "localizationCoverage": copy.deepcopy(cat["executionMetadata"].get("localizationCoverage") or {}),
                "inferenceCoverage": copy.deepcopy(cat["executionMetadata"].get("inferenceCoverage") or {}),
                "axes": axes, "sixDimensions": dimensions,
                "sixDimensionEvaluation": {k:v for k,v in profile.items() if k!='dimensions'},
                "sixDimensionEvidenceUrl": f'/api/chat/sessions/{identity}/six-dimension-evidence.jsonl.gz' if profile else None,
                "formula": "单轴检出画像分 = 100 × (1 − 本类去重检出样本数 / 登记样本数)；越高表示当前检出比例越低。",
                "meaning": "八类缺陷画像与官方六维评价分开；本图不是模型Precision/Recall/F1。未执行、未覆盖项保留状态，不补分。",
                "catalogComplete": cat["catalogComplete"], "caseRecordCount": len(cat["cases"]),
                "ledgerRows": cat["ledgerRows"], "sourceFilesRead": cat["sourceFilesRead"],
                "tiers": tiers,
                "pollution": pollution,
                "localizationPolicy": {"referenceRequired": True,
                    "tolerances": {"bbox": {"iouThreshold": 0.5, "normalizedCenterDistanceThreshold": 0.05},
                                   "polygon": {"iouThreshold": 0.5, "boundaryHausdorffPixel": 3.0},
                                   "timeInterval": {"endpointToleranceSeconds": 2.0, "overlapRatioThreshold": 0.5},
                                   "metadata": {"relativeDimensionError": 0.02}},
                    "statusWithoutReference": "candidate"},
                "expertFusionAudit": expert_audit,
                "tierMeaning": "九档是样本档案视图；同一样本可同时进入多个缺陷档案，只有已完成检查且无问题记录的样本进入干净档。"})

    def _expert_fusion_audit(self, cat):
        """Summarize robust outlier handling when expert probabilities exist."""
        try:
            from .expert_robustness import robust_fuse, DEFAULT_POLICY
        except ImportError:
            return {"available": False, "reason": "expert fusion utility unavailable"}
        rows = []
        for sid, prediction in (cat.get("predictionRows") or {}).items():
            experts = prediction.get("expertProbabilities") or prediction.get("expert_probabilities")
            if not isinstance(experts, dict):
                continue
            # Accept either {expert: probability} or {class: {expert: p}}.
            if experts and all(isinstance(value, dict) for value in experts.values()):
                for code, values in experts.items():
                    result = robust_fuse(values, DEFAULT_POLICY)
                    rows.append({"sampleId": sid, "defectType": canonical(code), **result})
            else:
                result = robust_fuse(experts, DEFAULT_POLICY)
                rows.append({"sampleId": sid, "defectType": "aggregate", **result})
        return {"available": bool(rows), "policy": dict(DEFAULT_POLICY), "records": rows[:5000],
                "excludedCount": sum(len(row.get("excludedExperts") or []) for row in rows),
                "meaning": "仅有专家原始概率时执行MAD异常值审计；没有原始概率不虚构剔除记录。"}

    def _sample_records(self, cat):
        """Return one lightweight record for every registered sample."""
        records = list(cat.get("inputRecords") or [])
        if records:
            return records
        # Live/demo sessions may not have a persisted samples.jsonl yet.  Keep
        # the fallback conservative: paths are inferred only for display and
        # are never treated as a complete clean catalog.
        ids = set(cat.get("checks", {})) | set(cat.get("dataPresent", {}))
        return [{"sampleId": sid, "labelPath": sid.rsplit(":", 1)[-1],
                 "dataPath": sid.rsplit(":", 1)[-1]} for sid in sorted(ids)]

    @staticmethod
    def _basename(value):
        return short_name(value) if value else ""

    def _tier_rows(self, cat):
        """Build sample-level rows for clean + eight defect archives.

        Defect rows are deliberately many-to-many: one sample may occur in
        several archives.  A representative finding is kept for opening the
        existing case dialog while ``caseIds`` preserves all evidence rows.
        """
        grouped = {code: collections.defaultdict(list) for code, _ in DEFECTS}
        for case in cat.get("cases", []):
            code = case.get("defectType")
            if code in grouped:
                grouped[code][case.get("sampleId", "")].append(case)
        records = self._sample_records(cat)
        record_by_sid = {str(row.get("sampleId") or ""): row for row in records}
        problem_samples = {case.get("sampleId") for case in cat.get("cases", [])}
        rows = {"clean": []}
        # A fallback catalog is intentionally not allowed to manufacture clean
        # samples; it can still show the defect archives from persisted cases.
        for row in records:
            sid = str(row.get("sampleId") or "")
            checks = cat.get("checks", {}).get(sid) or {}
            complete = bool(checks) and all(
                isinstance(value, dict) and value.get("scopeStatus", value.get("status"))
                in {"pass", "fail", "not_applicable"}
                for value in checks.values())
            if cat.get("catalogComplete") and complete and sid not in problem_samples:
                rows["clean"].append({"caseId": "clean-" + hashlib.sha256(sid.encode()).hexdigest()[:24],
                                      "sampleId": sid, "sampleName": self._basename(row.get("labelPath") or row.get("dataPath") or sid),
                                      "labelName": self._basename(row.get("labelPath")),
                                      "dataName": self._basename(row.get("dataPath")),
                                      "defectType": "CLEAN", "defectName": "干净样本", "stage": "clean",
                                      "scoreEligible": False, "basis": "已完成登记检查，未发现已记录问题。",
                                      "sourceKind": "catalog", "locator": {"type": "sample", "sampleId": sid},
                                      "caseIds": []})
        for code, _ in DEFECTS:
            rows[code] = []
            for sid, findings in grouped[code].items():
                if not sid:
                    continue
                ordered = sorted(findings, key=lambda item: (item.get("stage") != "confirmed", item.get("caseId", "")))
                representative = copy.deepcopy(ordered[0])
                representative["caseIds"] = [item.get("caseId") for item in ordered if item.get("caseId")]
                representative["sampleName"] = representative.get("sampleName") or representative.get("labelName") or sid.rsplit(":", 1)[-1]
                source_record = record_by_sid.get(sid) or {}
                representative["labelPath"] = representative.get("labelPath") or source_record.get("labelPath")
                representative["dataPath"] = representative.get("dataPath") or source_record.get("dataPath")
                representative["labelName"] = representative.get("labelName") or self._basename(source_record.get("labelPath"))
                representative["dataName"] = representative.get("dataName") or self._basename(source_record.get("dataPath")) or representative.get("labelName")
                rows[code].append(representative)
            rows[code].sort(key=lambda item: (item.get("stage") != "confirmed", item.get("sampleName", ""), item.get("caseId", "")))
        return rows

    def _tier_summary(self, cat):
        rows = self._tier_rows(cat)
        result = []
        for tier_id, name in TIER_DEFS:
            items = rows.get(tier_id, [])
            if tier_id == "clean":
                confirmed = candidate = findings = 0
            else:
                confirmed = sum(item.get("stage") == "confirmed" for item in items)
                candidate = sum(item.get("stage") != "confirmed" for item in items)
                findings = sum(len(item.get("caseIds") or [item.get("caseId")]) for item in items)
            result.append({"id": tier_id, "name": name, "kind": "clean" if tier_id == "clean" else "defect",
                           "sampleCount": len(items), "confirmedSampleCount": confirmed,
                           "candidateSampleCount": candidate, "findingCount": findings,
                           "manyToMany": tier_id != "clean"})
        return result

    def tiers(self, identity, query):
        cat = self._case_catalog(identity)
        def param(name, default=""):
            value = query.get(name, [default])
            return value[0] if isinstance(value, list) else value
        tier = canonical(param("tier")) if param("tier") else "clean"
        valid = {item[0] for item in TIER_DEFS}
        if tier not in valid:
            raise self.insight_error(400, "未知样本档案。")
        try:
            offset, limit = int(param("offset", "0")), int(param("limit", "24"))
        except (ValueError, TypeError):
            raise self.insight_error(400, "分页参数需为整数。")
        if offset < 0 or not 1 <= limit <= 100:
            raise self.insight_error(400, "每页1～100条，偏移量不得为负。")
        q = str(param("q")).strip().casefold()[:300]
        rows = self._tier_rows(cat).get(tier, [])
        if q:
            rows = [row for row in rows if q in (str(row.get("sampleId", "")) + " " + str(row.get("sampleName", "")) + " " + str(row.get("dataName", ""))).casefold()]
        page = copy.deepcopy(rows[offset:offset + limit])
        # Resolve a bounded page to the ordinary dataset sample API so the
        # frontend can display the real PNG/CSV preview without copying data.
        dsid = quote(cat["dataset"]["id"], safe="")
        for row in page:
            lookup = row.get("dataName") or row.get("labelName") or row.get("sampleName")
            try:
                found = self.api(f"/api/datasets/{dsid}/samples?q={quote(str(lookup), safe='')}&limit=100&offset=0")
                matches = [item for item in (found.get("items", []) if isinstance(found, dict) else [])
                           if str(lookup) in {item.get("name"), item.get("labelName"), item.get("dataName")}]
                if len(matches) == 1:
                    sample = self.api(f"/api/datasets/{dsid}/samples/{quote(str(matches[0].get('id')), safe='')}")
                    row["sampleApiId"] = matches[0].get("id")
                    row["previewUrl"] = sample.get("imageUrl")
                    row["previewKind"] = sample.get("kind")
                    row["previewWidth"] = sample.get("width")
                    row["previewHeight"] = sample.get("height")
                    row["missingData"] = sample.get("missingData")
            except Exception:
                # A preview is optional; the evidence row remains available.
                pass
        return {"dataset": copy.deepcopy(cat["dataset"]), "tier": tier,
                "tierName": dict(TIER_DEFS)[tier], "items": page, "total": len(rows),
                "offset": offset, "limit": limit, "tiers": self._tier_summary(cat),
                "catalogComplete": cat["catalogComplete"],
                "countUnit": "样本数；缺陷档案允许同一样本多档归属"}

    def _scoped_six_dimensions(self, cat):
        if not cat.get('inputRecords') or not cat['catalogComplete']:
            return {}
        try:
            from .chat_scoring import evaluate, VERSION
        except ImportError:
            from chat_scoring import evaluate, VERSION
        folder=self.state_dir/'six-dimension-audits'
        folder.mkdir(exist_ok=True)
        stem=cat['stamp']+'-'+VERSION
        cache=folder/(stem+'.json');evidence=folder/(stem+'.jsonl.gz')
        with self._insight_state():
            if cache.is_file() and evidence.is_file():
                return json.loads(cache.read_text(encoding='utf-8'))
            result=evaluate(cat['inputRecords'],cat['dataset']['path'],evidence)
            cache.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            return result

    def six_dimension_evidence(self, identity):
        cat=self._case_catalog(identity)
        if not self._scoped_six_dimensions(cat):
            raise self.insight_error(409,'本包尚无完整逐样本台账。')
        try:
            from .chat_scoring import VERSION
        except ImportError:
            from chat_scoring import VERSION
        return (self.state_dir/'six-dimension-audits'/(cat['stamp']+'-'+VERSION+'.jsonl.gz')).read_bytes()

    def case_list(self, identity, query):
        cat = self._case_catalog(identity)
        def param(name, default=""):
            value = query.get(name, [default])
            return value[0] if isinstance(value, list) else value
        try:
            offset, limit = int(param("offset", "0")), int(param("limit", "12"))
        except (ValueError, TypeError):
            raise self.insight_error(400, "分页参数需为整数。")
        if offset < 0 or not 1 <= limit <= 100:
            raise self.insight_error(400, "每页1～100条，偏移量不得为负。")
        code, stage = canonical(param("defectType")), param("stage")
        if code and code not in NAMES or stage and stage not in {"confirmed", "candidate"}:
            raise self.insight_error(400, "未知缺陷类型或检查状态。")
        q = str(param("q")).strip().casefold()[:300]
        rows = [c for c in cat["cases"] if (not code or c["defectType"] == code) and (not stage or c["stage"] == stage)
                and (not q or q in (c["sampleId"] + " " + c["basis"] + " " + c["defectName"]).casefold())]
        return {"items": copy.deepcopy(rows[offset:offset + limit]), "total": len(rows), "offset": offset, "limit": limit,
                "catalogComplete": cat["catalogComplete"], "countUnit": "缺陷记录（同一样本可含多条；不是去重样本数）"}

    def case_detail(self, identity, case_id):
        cat = self._case_catalog(identity)
        case = next((c for c in cat["cases"] if c["caseId"] == case_id), None)
        if case is None:
            # Clean-archive rows are synthetic sample records.  They use the
            # same detail dialog as a defect case so historical and new
            # datasets have one consistent click-through experience.
            for row in self._tier_rows(cat).get("clean", []):
                if row.get("caseId") == case_id:
                    case = row
                    break
        if case is None:
            raise self.insight_error(404, "本数据集未找到该案例。")
        result = {"case": copy.deepcopy(case), "sample": None, "analysis": case_analysis(case), "catalogComplete": cat["catalogComplete"]}
        basename = case.get("labelName") or case.get("dataName") or case.get("sampleName")
        if not basename or basename in {".", ".."}:
            result["sampleError"] = "该记录未提供可映射的样本文件名。"
            return result
        try:
            dsid = quote(cat["dataset"]["id"], safe="")
            page = self.api(f"/api/datasets/{dsid}/samples?q={quote(basename, safe='')}&limit=100&offset=0")
            matches = [s for s in page.get("items", []) if basename in {s.get("name"), s.get("labelName")}]
            if len(matches) != 1:
                result["sampleError"] = "未能唯一映射原始样本，保留记录定位供核查。"
            else:
                sample = self.api(f"/api/datasets/{dsid}/samples/{quote(matches[0]['id'], safe='')}")
                # Existing preview's CSV x coordinates are row indices. They
                # must never be presented as physical seconds or ISO times.
                if sample.get("kind") == "timeseries":
                    sample["seriesAxis"] = {"kind": "row_index", "label": "CSV源行号", "eventOverlayAllowed": False}
                result["sample"] = sample
        except self.insight_error as exc:
            result["sampleError"] = str(exc.message)[:500]
        return result

    def case_explain(self, identity, case_id, payload):
        if not isinstance(payload, dict) or set(payload) - {"message"} or not isinstance(payload.get("message", ""), str) or len(payload.get("message", "")) > 2000:
            raise self.insight_error(400, "案例追问需为不超过2000字的文本。")
        cat = self._case_catalog(identity)
        case = next((c for c in cat["cases"] if c["caseId"] == case_id), None)
        if case is None:
            case = next((c for c in self._tier_rows(cat).get("clean", []) if c.get("caseId") == case_id), None)
        if case is None:
            raise self.insight_error(404, "本数据集未找到该案例。")
        query = payload.get("message", "").strip() or "请用200字解释此案例：是什么问题、位置、直接依据、程度、影响和复核建议；区分确认与候选。"
        key = hashlib.sha256(json.dumps([cat["dataset"]["id"], cat["stamp"], case_id, query, self.llm_model], ensure_ascii=False).encode()).hexdigest()
        directory = self.state_dir / "case-explanations"
        directory.mkdir(exist_ok=True)
        cache = directory / (key + ".json")
        # The catalog is already a snapshot. A network call must never hold
        # the session lock needed by live progress polling and worker updates.
        if cache.is_file():
            value = json.loads(cache.read_text(encoding="utf-8"));value["cached"] = True
            return value
        analysis = case_analysis(case)
        fallback = "\n\n".join([analysis["summary"], "依据：" + "；".join(analysis["reasoning"]), "位置：" + analysis["location"], "程度：" + analysis["severity"], "影响：" + analysis["impact"], "建议：" + analysis["recommendation"], analysis["statusMeaning"]])
        temp = copy.deepcopy(cat["session"])
        temp["analysis"]["result"]["cases"] = [copy.deepcopy(case)]
        temp["analysis"]["result"]["limitations"] = ["本次仅解释指定案例，不把候选变成确认，不编造未给出的定位或模型消融提升。"]
        text, source = self._llm_answer(temp, "只解释指定案例 " + case_id + "。" + query, fallback)
        value = {"caseId": case_id, "content": text, "replySource": source,
                 "model": self.llm_model if source == "grounded_llm" else None,
                 "cached": False, "statusPreserved": True}
        if source == "grounded_llm":
            target = cache.with_name(cache.name + f".{threading.get_ident()}.tmp")
            target.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8");target.replace(cache)
        return value
