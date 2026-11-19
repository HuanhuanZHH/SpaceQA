#!/usr/bin/env python3
"""Validate an OPC run's finding, evidence, ranking, and log integrity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


STATUS_FAIL = "不通过"
STATUS_PASS = "通过"
STATUS_PENDING = "待判定"
CHECK_DIMENSIONS = {
    "C01": ("completeness",),
    "C02": ("completeness", "sampleCorrespondence"),
    "C03": ("completeness", "sampleCorrespondence"),
    "C04": ("annotationQuality",),
    "C05": ("annotationQuality", "sampleCorrespondence"),
    "C06": ("annotationQuality", "sampleCorrespondence"),
    "C07": ("annotationQuality",),
    "C08": ("coverageDistribution",),
    "C09": ("annotationQuality",),
    "C10": ("annotationQuality", "coverageDistribution"),
    "C11": ("traceability",),
    "C12": ("usageConditions",),
}
DIMENSION_IDS = (
    "completeness",
    "annotationQuality",
    "sampleCorrespondence",
    "coverageDistribution",
    "traceability",
    "usageConditions",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(left: Any, right: Any, tolerance: float = 1e-4) -> bool:
    if left is None or right is None:
        return left is right
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and abs(float(left) - float(right)) <= tolerance


def recompute_scorecard(
    card: dict[str, Any],
    findings: list[dict[str, Any]],
    enabled_checks: set[str],
    aggregation: dict[str, Any],
) -> dict[str, Any]:
    default_denominator = card.get("scoringInputs", {}).get("defaultAffectedDenominator")
    if not isinstance(default_denominator, int) or default_denominator < 1:
        raise ValueError("scorecard has invalid defaultAffectedDenominator")
    finding_cap = float(aggregation["singleFindingPenaltyCap"])
    critical_multiplier = float(aggregation["criticalMultiplier"])
    critical_cap = float(aggregation["criticalCap"])
    critical_ratio_slope = float(aggregation["criticalRatioSlope"])
    dimensions = []
    for dim_id in DIMENSION_IDS:
        relevant = {check for check, dims in CHECK_DIMENSIONS.items() if dim_id in dims and check in enabled_checks}
        rows = [row for row in findings if row.get("checkId") in relevant]
        failures = [row for row in rows if row.get("status") == STATUS_FAIL]
        completed = {row.get("checkId") for row in rows if row.get("status") in {STATUS_PASS, STATUS_FAIL, STATUS_PENDING}}
        determined = {row.get("checkId") for row in rows if row.get("status") in {STATUS_PASS, STATUS_FAIL}}
        if not completed or not determined:
            dimensions.append({"dimId": dim_id, "score": None, "range": None})
            continue
        penalty = 0.0
        for finding in failures:
            ratio = finding.get("affectedScope", {}).get("ratio")
            if not isinstance(ratio, (int, float)):
                ratio = 1 / default_denominator
            multiplier = critical_multiplier if finding.get("criticality") == "关键" else 1.0
            penalty += min(finding_cap, 100.0 * float(ratio) * multiplier)
        score = round(max(0.0, 100.0 - penalty), 4)
        pending_count = sum(1 for row in rows if row.get("status") == STATUS_PENDING)
        lower = round(score * len(completed) / max(1, len(relevant)), 4)
        dimensions.append({"dimId": dim_id, "score": score, "range": [lower, score] if pending_count else [score, score]})
    values = [row["score"] for row in dimensions if isinstance(row["score"], (int, float))]
    total = sum(values) / len(values) if values else None
    critical = [row for row in findings if row.get("status") == STATUS_FAIL and row.get("criticality") == "关键"]
    if total is not None and critical:
        critical_ratio = sum(
            float(row.get("affectedScope", {}).get("ratio"))
            if isinstance(row.get("affectedScope", {}).get("ratio"), (int, float))
            else 1 / default_denominator
            for row in critical
        )
        critical_ceiling = max(0.0, critical_cap - critical_ratio_slope * min(1.0, critical_ratio))
        total = min(total, critical_ceiling)
    return {
        "dimensions": dimensions,
        "totalScore": round(total, 4) if total is not None else None,
        "criticalFindingIds": [row.get("findingId") for row in critical],
    }


def audit(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = run_dir / "00_MANIFEST.yaml"
    findings_path = run_dir / "06_问题清单" / "findings.jsonl"
    evidence_index_path = run_dir / "07_证据索引" / "evidence_index.csv"
    logs_path = run_dir / "08_运行日志" / "run.log.jsonl"
    ranking_path = run_dir / "05_排序表" / "ranking_global.csv"
    spec_path = run_dir / "03_评价规范" / "quality_spec.yaml"
    score_dir = run_dir / "04_评分卡"
    environment_path = run_dir / "08_运行日志/system_env.txt"
    seeds_path = run_dir / "08_运行日志/seeds.yaml"

    required = [manifest_path, findings_path, evidence_index_path, logs_path, ranking_path,
                spec_path, environment_path, seeds_path]
    for path in required:
        if not path.is_file():
            errors.append(f"missing required artifact: {path.relative_to(run_dir)}")
    if not score_dir.is_dir():
        errors.append(f"missing required artifact: {score_dir.relative_to(run_dir)}")
    if errors:
        return {"runDirectory": str(run_dir), "status": "fail", "errors": errors}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    for field in ("ruleVersion", "specVersion"):
        if specification.get(field) != manifest.get(field):
            errors.append(f"quality specification {field} differs from manifest")
    aggregation = specification.get("aggregation", {})
    for field in ("singleFindingPenaltyCap", "criticalMultiplier", "criticalCap", "criticalRatioSlope"):
        if not isinstance(aggregation.get(field), (int, float)):
            errors.append(f"quality specification lacks numeric aggregation.{field}")
    findings = load_jsonl(findings_path)
    logs = load_jsonl(logs_path)
    environment = dict(line.split("=", 1) for line in environment_path.read_text(encoding="utf-8").splitlines()
                       if "=" in line)
    for field in ("opc_agent_version", "python", "platform", "rule_version"):
        if not environment.get(field):
            errors.append(f"system environment lacks {field}")
    if environment.get("opc_agent_version") != manifest.get("agentVersion"):
        errors.append("system version differs from manifest")
    if environment.get("rule_version") != manifest.get("ruleVersion"):
        errors.append("rule version differs from manifest")
    seeds = {}
    try:
        seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
        if not isinstance(seeds, dict) or any(type(seeds.get(k)) is not int for k in ("globalSeed", "samplingSeed")):
            errors.append("random seed list is incomplete or invalid")
            seeds = {}
    except (ValueError, TypeError):
        errors.append("random seed list is not valid JSON")
    for row in findings:
        if row.get("datasetId") not in manifest.get("datasets", []):
            errors.append(f"finding refers to an unknown dataset: {row.get('findingId')}")
        if not row.get("checkId") or row.get("checkId") not in manifest.get("enabledChecks", []):
            errors.append(f"finding lacks an enabled check ID: {row.get('findingId')}")
        if row.get("ruleVersion") != manifest.get("ruleVersion"):
            errors.append(f"finding rule version differs from manifest: {row.get('findingId')}")
        if row.get("status") not in {STATUS_PASS, STATUS_FAIL, STATUS_PENDING, "不适用"}:
            errors.append(f"finding has unknown check status: {row.get('findingId')}")
    with evidence_index_path.open(encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    with ranking_path.open(encoding="utf-8-sig", newline="") as handle:
        ranking_rows = list(csv.DictReader(handle))

    finding_ids = [row.get("findingId") for row in findings]
    evidence_refs = [row.get("evidenceRef") for row in findings]
    if len(set(finding_ids)) != len(finding_ids):
        errors.append("findingId values are not unique")
    if len(set(evidence_refs)) != len(evidence_refs):
        errors.append("evidenceRef values are not unique")
    if len(evidence_rows) != len(findings):
        errors.append(f"evidence row count {len(evidence_rows)} != finding count {len(findings)}")

    findings_by_id = {row.get("findingId"): row for row in findings}
    indexed_ids = set()
    for row in evidence_rows:
        evidence_id = row.get("evidenceId", "")
        finding_id = row.get("findingId", "")
        indexed_ids.add(evidence_id)
        finding = findings_by_id.get(finding_id)
        if finding is None:
            errors.append(f"evidence {evidence_id} refers to unknown finding {finding_id}")
            continue
        if finding.get("evidenceRef") != evidence_id:
            errors.append(f"evidence reference mismatch for finding {finding_id}")
        evidence_path = run_dir / "07_证据索引" / "evidence" / f"{evidence_id}.json"
        if not evidence_path.is_file():
            errors.append(f"missing evidence payload: {evidence_id}")
            continue
        if sha256(evidence_path) != row.get("contentHash"):
            errors.append(f"content hash mismatch: {evidence_id}")
            continue
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("evidenceId", evidence_id),
            ("findingId", finding_id),
            ("datasetId", finding.get("datasetId")),
            ("checkId", finding.get("checkId")),
            ("ruleVersion", finding.get("ruleVersion")),
            ("locator", row.get("locator")),
            ("locatorType", row.get("locatorType")),
        ):
            if payload.get(key) != expected:
                errors.append(f"evidence payload mismatch {evidence_id}: {key}")
    for evidence_ref in set(evidence_refs) - indexed_ids:
        errors.append(f"finding has no indexed evidence: {evidence_ref}")

    datasets = manifest.get("datasets", [])
    checks = manifest.get("enabledChecks", [])
    completed = {
        (row.get("datasetId"), row.get("checkId"))
        for row in logs
        if row.get("status") == "实际完成"
    }
    logging_version = manifest.get("loggingVersion", 1)
    if type(logging_version) is not int or logging_version < 1:
        errors.append("manifest has an invalid loggingVersion")
        logging_version = 1
    strict_logging = logging_version >= 2
    start_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    terminal_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in logs:
        key = (row.get("datasetId"), row.get("checkId"))
        if key[0] not in datasets or key[1] not in checks:
            errors.append(f"log refers to an unknown dataset/check: {key[0]}/{key[1]}")
        expected_fields = {"runId": manifest.get("runId"), "ruleVersion": manifest.get("ruleVersion"),
                           "systemVersion": manifest.get("agentVersion"), "seed": seeds.get("globalSeed")}
        for field, expected in expected_fields.items():
            if field not in row and not strict_logging:
                continue
            if row.get(field) != expected:
                errors.append(f"log {field} differs from run metadata: {key[0]}/{key[1]}")
        if row.get("event") == "检查开始":
            start_records.setdefault(key, []).append(row)
        elif row.get("event") in {"检查结束", "失败"}:
            terminal_records.setdefault(key, []).append(row)
        if strict_logging:
            elapsed = row.get("elapsedMs")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                errors.append(f"log has invalid elapsedMs: {key[0]}/{key[1]}")
            if not row.get("ts"):
                errors.append(f"log lacks timestamp: {key[0]}/{key[1]}")
            if row.get("event") in {"检查结束", "失败"}:
                usage = row.get("processUsage", {})
                cpu = usage.get("cpuSeconds")
                if type(cpu) not in (int, float) or not math.isfinite(cpu) or cpu < 0:
                    errors.append(f"log lacks measured process CPU: {key[0]}/{key[1]}")
                peak = usage.get("peakRssBytes")
                if peak is not None and (type(peak) is not int or peak < 0):
                    errors.append(f"log has invalid process peak RSS: {key[0]}/{key[1]}")
                if usage.get("peakRssScope") != "process_lifetime_high_water":
                    errors.append(f"log has no explicit memory measurement scope: {key[0]}/{key[1]}")
    if strict_logging:
        for dataset_id in datasets:
            for check_id in checks:
                key = (dataset_id, check_id)
                starts = start_records.get(key, [])
                ends = terminal_records.get(key, [])
                if len(starts) != 1 or len(ends) != 1:
                    errors.append(f"expected one start/end log for {dataset_id}/{check_id}")
                elif logs.index(starts[0]) >= logs.index(ends[0]):
                    errors.append(f"start log follows terminal log: {dataset_id}/{check_id}")
    elif not start_records:
        warnings.append("Legacy logging: completion records only; no start/end timeline claimed")
    if strict_logging:
        runtime_path = run_dir / "08_运行日志/runtime_metrics.json"
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            for field, expected in (("runId", manifest.get("runId")), ("systemVersion", manifest.get("agentVersion")),
                                    ("ruleVersion", manifest.get("ruleVersion")), ("seed", seeds.get("globalSeed"))):
                if runtime.get(field) != expected:
                    errors.append(f"runtime metrics {field} differs from manifest")
            elapsed = runtime.get("elapsedSeconds")
            if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
                errors.append("runtime metrics has invalid elapsedSeconds")
            usage = runtime.get("processUsage", {})
            for field in ("cpuSeconds", "cpuDeltaSeconds"):
                value = usage.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    errors.append(f"runtime metrics has invalid processUsage.{field}")
            if runtime.get("memoryScope") != "process_lifetime_high_water":
                errors.append("runtime metrics lacks process lifetime memory scope")
            if runtime.get("executionMode") == "rules-only":
                for field in ("modelForwardPasses", "externalApiCalls", "externalModelTokens"):
                    if runtime.get(field) != 0:
                        errors.append(f"rules-only runtime has nonzero or missing {field}")
        except (OSError, ValueError, TypeError, AttributeError):
            errors.append("runtime metrics is missing or invalid JSON")
    for dataset_id in datasets:
        for check_id in checks:
            if (dataset_id, check_id) not in completed:
                errors.append(f"missing completed log: {dataset_id}/{check_id}")

    ranked_datasets = [row.get("datasetId") for row in ranking_rows]
    if sorted(ranked_datasets) != sorted(datasets):
        errors.append("ranking datasets do not match manifest datasets")
    try:
        ranks = [int(row["rank"]) for row in ranking_rows]
        if ranks != list(range(1, len(ranks) + 1)):
            errors.append("ranking positions are not consecutive from 1")
    except (KeyError, TypeError, ValueError):
        errors.append("ranking contains an invalid rank")

    findings_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        findings_by_dataset.setdefault(str(finding.get("datasetId")), []).append(finding)
    cards = []
    for path in sorted(score_dir.glob("*.json")):
        try:
            cards.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid scorecard {path.name}: {exc}")
    cards_by_dataset = {str(card.get("datasetId")): card for card in cards}
    if sorted(cards_by_dataset) != sorted(datasets):
        errors.append("scorecard datasets do not match manifest datasets")
    for card in cards:
        for field in ("runId", "ruleVersion", "specVersion"):
            if card.get(field) != manifest.get(field):
                errors.append(f"scorecard {field} differs from manifest: {card.get('datasetId')}")
        expected_refs = [row.get("evidenceRef") for row in findings_by_dataset.get(str(card.get("datasetId")), [])]
        if sorted(card.get("evidenceRefs", [])) != sorted(expected_refs):
            errors.append(f"scorecard evidence references do not cover its findings: {card.get('datasetId')}")

    formula_ready = all(
        isinstance(aggregation.get(field), (int, float))
        for field in ("singleFindingPenaltyCap", "criticalMultiplier", "criticalCap", "criticalRatioSlope")
    )
    if formula_ready:
        for dataset_id in datasets:
            card = cards_by_dataset.get(dataset_id)
            if card is None:
                continue
            try:
                recomputed = recompute_scorecard(card, findings_by_dataset.get(dataset_id, []), set(checks), aggregation)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"cannot recompute scorecard {dataset_id}: {exc}")
                continue
            submitted_dimensions = {row.get("dimId"): row for row in card.get("dimensions", [])}
            for expected in recomputed["dimensions"]:
                submitted = submitted_dimensions.get(expected["dimId"], {})
                if not _close(submitted.get("score"), expected["score"]):
                    errors.append(f"dimension score mismatch: {dataset_id}/{expected['dimId']}")
                submitted_range = submitted.get("range")
                expected_range = expected["range"]
                if expected_range is None:
                    if submitted_range is not None:
                        errors.append(f"dimension range mismatch: {dataset_id}/{expected['dimId']}")
                elif (
                    not isinstance(submitted_range, list)
                    or len(submitted_range) != 2
                    or not all(_close(actual, target) for actual, target in zip(submitted_range, expected_range))
                ):
                    errors.append(f"dimension range mismatch: {dataset_id}/{expected['dimId']}")
            if not _close(card.get("totalScore"), recomputed["totalScore"]):
                errors.append(f"total score mismatch: {dataset_id}")
            if card.get("criticalDefects") != recomputed["criticalFindingIds"]:
                errors.append(f"critical defect list mismatch: {dataset_id}")

    expected_order = sorted(
        cards,
        key=lambda card: (
            -(card.get("totalScore") if card.get("totalScore") is not None else -1),
            len(card.get("criticalDefects", [])),
            str(card.get("datasetId")),
        ),
    )
    if [card.get("datasetId") for card in expected_order] != ranked_datasets:
        errors.append("ranking order cannot be recomputed from scorecards")
    for row in ranking_rows:
        card = cards_by_dataset.get(str(row.get("datasetId")))
        if card is None:
            continue
        try:
            submitted_total = float(row["totalScore"]) if row.get("totalScore") not in (None, "") else None
        except (TypeError, ValueError):
            submitted_total = "invalid"
        if not _close(submitted_total, card.get("totalScore")):
            errors.append(f"ranking total score mismatch: {row.get('datasetId')}")

    for artifact in manifest.get("artifacts", []):
        path = (run_dir / artifact).resolve()
        if not path.is_relative_to(run_dir):
            errors.append(f"manifest artifact escapes run directory: {artifact}")
        elif not path.exists():
            errors.append(f"manifest artifact is missing: {artifact}")

    return {
        "runDirectory": str(run_dir),
        "runId": manifest.get("runId"),
        "status": "pass" if not errors else "fail",
        "findingCount": len(findings),
        "evidenceCount": len(evidence_rows),
        "datasetCount": len(datasets),
        "enabledCheckCount": len(checks),
        "completedLogPairs": len(completed),
        "completedCheckCount": len(completed),
        "startedCheckCount": len(start_records),
        "loggingVersion": manifest.get("loggingVersion", 1),
        "runtimeMetadataChecked": True,
        "warnings": warnings,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
