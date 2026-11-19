#!/usr/bin/env python3
"""Export fresh prior findings + frozen cached model candidates into an audited run.

The external roadshow runner must have verified current labels and prediction
files. This exporter does not perform new neural inference, infer roadshow truth,
or promote coarse model flags to confirmed/localized scientific defects.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spaceqa_agent import engine
from spaceqa_agent.audit import audit

INTEGRATION_VERSION = "2026.11.02-prior-integration-1"


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_d(code):
    return code in {"L2-P", "L2-F", "L2-M", "L2-T", "TIME_INVALID", "ANNOTATION_STRUCTURE_CONFLICT"}


def prior_check(code):
    if code == "L2-P":
        return "C09"
    if code == "L2-F":
        return "C06"
    if code in {"L2-M", "L2-T", "TIME_INVALID"}:
        return "C07"
    return "C04"


def measured_degree(item):
    """Preserve mechanical degree; never use model probability as severity."""
    observation = item.get("observation", {})
    if "merged_degree" in observation:
        return tuple(observation["merged_degree"])
    if "offset_seconds" in observation or "reversal_seconds" in observation:
        return abs(float(observation.get("offset_seconds", observation.get("reversal_seconds")))), "timeDeviation", "second"
    if "png_dimension" in observation:
        try:
            return abs(float(observation["actual"]) - float(observation["png_dimension"])), "dimensionDeviation", "pixel"
        except (TypeError, ValueError):
            pass
    if "png_dimensions" in observation and isinstance(observation.get("declared"), list):
        return max(abs(float(a) - float(b)) for a, b in zip(observation["declared"], observation["png_dimensions"])), "dimensionDeviation", "pixel"
    measured, declared = observation.get("measured"), observation.get("declared")
    if type(declared) in (int, float):
        values = measured if isinstance(measured, list) else [measured]
        values = [v for v in values if type(v) in (int, float)]
        if values:
            reference = min(values, key=lambda x: abs(declared - x))
            return abs(declared - reference) / max(abs(reference), 1e-12), "relativeDeviation", "ratio"
    return 1, "count", "item"


def coalesce_prior_findings(findings):
    """A width/height pair or two window endpoints are one defect instance."""
    grouped = {}
    for item in findings:
        code, pointer = item["defectType"], item["pointer"]
        parent, _, field = pointer.rpartition("/")
        if code == "L2-F":
            locus = "current-image-dimension-declaration"
        elif code == "L2-T" and field in {"startTime", "endTime", "start_dt", "end_dt"}:
            locus = parent
        elif code == "L2-P" and field in {"event_type", "concept", "kind", "label"}:
            locus = parent
        else:
            locus = pointer
        grouped.setdefault((code, locus, item["certainty"], item["score_eligible"]), []).append(item)
    result = []
    for items in grouped.values():
        item = copy.deepcopy(items[0])
        if len(items) > 1:
            degrees = [measured_degree(v) for v in items]
            # Instances above are grouped only within the same defect family.
            # Keep every source field even when the scalar degree uses a max.
            degree = max(degrees, key=lambda v: v[0])
            item["observation"] = {"sources": [{"pointer": v["pointer"], "measurement": v["observation"]} for v in items],
                "source_field_count": len(items), "merged_degree": list(degree)}
            item["basis"] = "; ".join(dict.fromkeys(v["basis"] for v in items))
        result.append(item)
    return result


def source_root(records, package):
    for parent in Path(records[0]["label_path"]).parents:
        if parent.name == package:
            return parent
    raise ValueError("Cannot identify package root from its current inventory")


def validate_candidate(candidate, record, policy_hash):
    if candidate.get("frozen_validation_policy_sha256") != policy_hash:
        raise ValueError("Candidate frozen policy hash differs from completion marker")
    probabilities, thresholds = candidate.get("probabilities"), candidate.get("thresholds")
    bits = candidate.get("model_candidate_bits")
    if not isinstance(bits, list) or len(bits) != 3 or any(type(b) not in (int, bool) or b not in (0, 1) for b in bits):
        raise ValueError("Expected exactly three boolean model candidate bits")
    if (not isinstance(thresholds, list) or len(thresholds) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1.01 for v in thresholds)):
        raise ValueError("Expected exactly three finite model thresholds")
    if probabilities is None:
        if any(bits):
            raise ValueError("A missing model input cannot yield positive model candidate bits")
    elif (not isinstance(probabilities, list) or len(probabilities) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)):
        raise ValueError("Expected exactly three finite probabilities in [0,1]")
    elif any(flag and p < t for flag, p, t in zip(bits, probabilities, thresholds)):
        raise ValueError("Positive model flag is below its declared frozen threshold")
    if candidate.get("data_path", record["data_path"]) != record["data_path"]:
        raise ValueError("Current data locator mismatch")


def export(old_run: Path, fresh_run: Path, output: Path, seed: int = 20261021,
           frozen_policy: Path | None = None):
    if output.exists():
        raise ValueError("Refusing to overwrite an integrated run")
    complete = json.loads((fresh_run / "complete.json").read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or complete.get("ground_truth_available") is not False:
        raise ValueError("A completed, truth-free roadshow replay is required")
    if frozen_policy is None or sha(frozen_policy) != complete.get("frozen_validation_policy_sha256"):
        raise ValueError("Provide the exact frozen policy used by the completed roadshow run")
    package_folders = sorted(p for p in (fresh_run / "packages").iterdir() if p.is_dir())
    if not package_folders or len(package_folders) != complete.get("packages"):
        raise ValueError("Package coverage differs from the completion marker")
    if "summaries" in complete and {r["package"] for r in complete["summaries"]} != {p.name for p in package_folders}:
        raise ValueError("Package names differ from the completion marker")
    for source in (old_run, fresh_run):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Keep integrated outputs outside the source runs")
    run_id = "RUN-" + hashlib.sha256((sha(fresh_run / "complete.json") + str(seed) + INTEGRATION_VERSION).encode()).hexdigest()[:12].upper()
    datasets, package_summaries, source_hashes = [], [], {}
    old_version = engine.RULE_VERSION
    started_all = time.monotonic()
    engine.RULE_VERSION = INTEGRATION_VERSION
    try:
        for folder in package_folders:
            package = folder.name
            started = time.perf_counter()
            prior_path = folder / "prior_observations.jsonl"
            candidate_path = folder / "selected_candidates.jsonl"
            if not prior_path.is_file() or not candidate_path.is_file():
                raise ValueError("Missing fresh observations or selected candidates: " + package)
            inventory_path = old_run / "packages" / package / "inventory.jsonl"
            legacy_path = old_run / "packages" / package / "rule_observations.jsonl"
            inventory = list(lines(inventory_path))
            priors = list(lines(prior_path))
            candidates = list(lines(candidate_path))
            if not inventory or len(priors) != len(inventory) or len(candidates) != len(inventory):
                raise ValueError("Record coverage mismatch: " + package)
            if len({r["id"] for r in inventory}) != len(inventory):
                raise ValueError("Duplicate inventory identifier")
            raw_root = source_root(inventory, package)
            if output.resolve().is_relative_to(raw_root.resolve()):
                raise ValueError("Integrated outputs must remain outside the original data package")
            dataset = engine.inventory_dataset(raw_root, None)
            # Score denominators count parsed labels. Their bytes are checked
            # again below; no scientific parser needs their bodies here.
            dataset.labels = {name: {} for name in dataset.label_files}
            legacy = list(lines(legacy_path))
            source_hashes[package] = {p.name: sha(p) for p in (inventory_path, legacy_path, prior_path, candidate_path)}
            for finding in legacy:
                # D is replaced by the scoped current-data checks. Pass and
                # pending summaries are rebuilt below rather than replayed as
                # misleading passes after new contradictions were detected.
                if (legacy_d(finding["defectType"]) or finding["status"] != engine.STATUS_FAIL
                        or finding["checkId"] == "C10"):
                    continue
                retained = copy.deepcopy(finding)
                source_rule = retained["ruleVersion"]
                retained["ruleVersion"] = INTEGRATION_VERSION
                retained["_observation"] = {"sourceRuleVersion": source_rule,
                    "sourceFindingId": retained.get("findingId"),
                    "sourceObservation": retained.get("_observation"),
                    "sourceFindingsSha256": source_hashes[package][legacy_path.name]}
                dataset.findings.append(retained)
            missing_labels = {f["_locator"].split("#")[0] for f in dataset.findings
                              if f["defectType"] in {"L0-01", "L0-03"}}
            model_samples = 0
            for record, prior, candidate in zip(inventory, priors, candidates):
                if record["id"] != prior["sample_id"] or record["id"] != candidate["sample_id"]:
                    raise ValueError("ID or order mismatch: " + package)
                if prior["label_path"] != record["label_path"] or candidate["label_path"] != record["label_path"]:
                    raise ValueError("Current label locator mismatch")
                validate_candidate(candidate, record, complete["frozen_validation_policy_sha256"])
                current_label = Path(record["label_path"])
                if sha(current_label) != prior["current_label_sha256"]:
                    raise ValueError("Source label changed after fresh prior checking")
                for item in coalesce_prior_findings(prior["findings"]):
                    code = item["defectType"]
                    if code == "L0-01" and record["label_path"] in missing_labels:
                        continue
                    confirmed = bool(item["score_eligible"] and item["certainty"] == "confirmed")
                    if code == "L0-01":
                        check = "C02"
                    else:
                        check = prior_check(code)
                    degree, metric, unit = measured_degree(item)
                    made = engine.make_finding(dataset, check, code,
                        engine.STATUS_FAIL if confirmed else engine.STATUS_PENDING,
                        record["label_path"] + "#" + item["pointer"], item["basis"],
                        severity_value=degree, severity_metric=metric, severity_unit=unit, affected_count=1,
                        affected_ratio=1 / max(1, len(dataset.label_files)),
                        criticality="关键" if code == "L0-01" else "一般",
                        locator_type="annotationRecord", observation={
                            "sampleId": record["id"], "sourceRuleVersion": item.get("rule_version"),
                            "certainty": item["certainty"], "scoreEligible": confirmed,
                            "measurement": item["observation"], "sourcePriorSha256": source_hashes[package][prior_path.name]})
                    made["scoreEligible"] = confirmed
                    made["reviewSuggestion"] = "核对所列原始字段与数据证据；证据不足时请求数据提供方补充定义或可信参照"
                    dataset.findings.append(made)
                if any(candidate["model_candidate_bits"]):
                    model_samples += 1
                for group, flag in zip(("G", "C", "D"), candidate["model_candidate_bits"]):
                    if not flag:
                        continue
                    index = ("G", "C", "D").index(group)
                    made = engine.make_finding(dataset, "C10", "MODEL_" + group + "_CANDIDATE",
                        engine.STATUS_PENDING, record["label_path"],
                        "冻结视觉/字段专家提出粗粒度候选，尚不具备九类科学缺陷的独立定位确证",
                        locator_type="sample", observation={
                            "sampleId": record["id"], "dataPath": record["data_path"],
                            "route": candidate["route"], "group": group,
                            "probability": candidate["probabilities"][index],
                            "threshold": candidate["thresholds"][index],
                            "sourceCandidatesSha256": source_hashes[package][candidate_path.name],
                            "frozenPolicySha256": candidate["frozen_validation_policy_sha256"],
                            "coarsePredictionOnly": True, "scoreEligible": False})
                    made["scoreEligible"] = False
                    made["modelResult"] = {"route": candidate["route"], "group": group,
                        "probability": candidate["probabilities"][index], "threshold": candidate["thresholds"][index]}
                    made["reviewSuggestion"] = "检查当前图像和对应标注；取得独立位置真值或人工审核后再决定是否为科学错误"
                    dataset.findings.append(made)
            completed_old = {r["checkId"] for r in legacy}
            for check in engine.ALL_CHECKS:
                current = [r for r in dataset.findings if r["checkId"] == check]
                if not current:
                    original = [r for r in legacy if r["checkId"] == check]
                    if check == "C10" or not original or any(r["status"] == engine.STATUS_PENDING for r in original):
                        status, basis = engine.STATUS_PENDING, "已执行检查编排；缺少独立领域证据或可信参照，保留待判定"
                    elif all(r["status"] == engine.STATUS_NA for r in original):
                        status, basis = engine.STATUS_NA, original[0]["basis"]
                    elif check in {"C04", "C06", "C07", "C09"}:
                        status, basis = engine.STATUS_PASS, "当前输入先验检查未发现可确证矛盾；不表示科学语义全部正确"
                    elif check in completed_old:
                        status, basis = engine.STATUS_PASS, "复用已验证当前标签对应的结构检查结果，未发现确认失败"
                    else:
                        status, basis = engine.STATUS_PENDING, "未实现或缺少证据，保留在适用检查范围"
                    dataset.findings.append(engine.make_finding(dataset, check, "NONE", status, str(dataset.root), basis))
                engine.log_event(dataset, run_id, check, started, "实际完成")
                dataset.run_events[-1].update({"ruleVersion": INTEGRATION_VERSION,
                    "systemVersion": engine.VERSION, "seed": seed,
                    "executionMode": "fresh-prior-and-verified-cached-candidate-integration",
                    "modelForwardPasses": 0})
            engine.finalize_ids(dataset)
            package_summaries.append({"package": package, "family": inventory[0]["family"],
                "samples": len(inventory), "missing_inputs": sum(bool(r["input_missing"]) for r in inventory),
                "model_candidate_samples": model_samples,
                "confirmed_findings": sum(r["status"] == engine.STATUS_FAIL for r in dataset.findings),
                "pending_findings": sum(r["status"] == engine.STATUS_PENDING for r in dataset.findings),
                "official_F1": None})
            datasets.append(dataset)
        if sum(r["samples"] for r in package_summaries) != complete.get("records"):
            raise ValueError("Sample coverage differs from the completion marker")
        enabled = set(engine.ALL_CHECKS)
        cards = [engine.score_dataset(d, enabled, run_id) for d in datasets]
        engine.write_outputs(output.resolve(), datasets, cards, run_id, seed, enabled)
        spec_path = output / "03_评价规范/quality_spec.yaml"
        specification = json.loads(spec_path.read_text(encoding="utf-8"))
        specification["ruleVersion"] = INTEGRATION_VERSION
        specification["integration"] = {"confirmed": "current-input mechanical contradictions and retained structural checks",
            "modelCandidates": "pending, sample/coarse-group only, excluded from penalties",
            "DPolicy": "legacy D conclusions replaced by new scoped current-data checks",
            "scope": "cached model replay integration; not new GPU inference or official hidden-set scoring",
            "deduplication": "one dimension declaration per sample; paired window endpoints merged; category aliases within one event merged; all source pointers retained",
            "scoreRanges": "legacy check-coverage bookkeeping intervals, not scientific confidence intervals"}
        engine.json_dump(spec_path, specification)
        (output / "09_未完成事项说明.md").write_text(
            "# 未完成事项说明\n\n"
            "- 视觉/字段专家只产生粗粒度候选，不作为九类科学缺陷的独立定位确证，不计入确认缺陷罚分。\n"
            "- 路演缺少独立科学真值，F1、精确率、召回率和官方分数均不可计算。\n"
            "- 本次使用已核验的历史模型概率，并重新执行当前输入先验检查，不是新神经网络推理耗时测试。\n"
            "- 文档/使用条件证据不足保留未知；不凭包名或档位判定污染。\n"
            "- 完整专家镜像与指定A10端到端时限仍需单独验收。\n", encoding="utf-8")
        summary = {"status": "complete", "runId": run_id, "packages": len(datasets),
            "samples": sum(r["samples"] for r in package_summaries),
            "integrationVersion": INTEGRATION_VERSION, "selectedVariant": complete["selected_variant"],
            "frozenPolicySha256": complete["frozen_validation_policy_sha256"],
            "sourceHashes": source_hashes, "packageSummaries": package_summaries,
            "groundTruthAvailable": False, "F1": None, "officialScore": None,
            "gpuForwardPasses": 0, "exportSeconds": time.monotonic() - started_all}
        engine.json_dump(output / "integration_summary.json", summary)
        checked = audit(output)
        engine.json_dump(output / "integration_audit.json", checked)
        if checked["status"] != "pass":
            raise ValueError("Integrated run failed its independent audit: " + str(checked["errors"][:5]))
        return {k: v for k, v in summary.items() if k not in {"sourceHashes", "packageSummaries"}}
    finally:
        engine.RULE_VERSION = old_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-roadshow-run", required=True, type=Path)
    parser.add_argument("--fresh-roadshow-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-policy", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    result = export(args.original_roadshow_run, args.fresh_roadshow_run, args.output, args.seed, args.frozen_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
