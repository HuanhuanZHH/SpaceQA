#!/usr/bin/env python3
"""Compare OPC detection routes on a controlled multi-family benchmark.

Reported precision/recall are package-by-defect-class metrics.  The public
competition labels do not disclose hidden defect locations, so this script does
not present them as official location-level D or S_Q scores.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import opc_agent
from multifamily_fixture import CANONICAL_MISSING, FAMILIES, TIERS, build_multifamily_fixture
from tools.audit_run import audit


CANDIDATES = {
    "A_schema_pairing": {"C01", "C02", "C03"},
    "B_single_package_rules": {f"C{index:02d}" for index in range(1, 10)},
    "C_rules_plus_sibling_diff": set(opc_agent.ALL_CHECKS),
}
TIER_ORDER = {tier: index for index, tier in enumerate(TIERS)}
SCORABLE_CLASSES = {
    CANONICAL_MISSING,
    "L2-G",
    "L2-P",
    "L2-T",
    "L2-F",
    "L2-M",
    "L3-06",
    "L3-07",
    "L3-MIS",
}
INTENSITY_CLASSES = {"L2-G", "L2-T", "L2-F", "L3-MIS"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _canonical_class(value: str) -> str:
    return CANONICAL_MISSING if value in {"L0-01", "L0-03"} else value


def _predicted_classes(findings: list[dict[str, Any]]) -> dict[str, set[str]]:
    predicted: dict[str, set[str]] = collections.defaultdict(set)
    for row in findings:
        if row.get("status") != opc_agent.STATUS_FAIL:
            continue
        defect_type = _canonical_class(str(row.get("defectType", "")))
        if defect_type in SCORABLE_CLASSES:
            predicted[row["datasetId"]].add(defect_type)
    return predicted


def _class_metrics(predicted: dict[str, set[str]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = fp = fn = 0
    by_family: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    packages = []
    for dataset_id, item in sorted(truth.items()):
        expected = set(item["expectedClasses"])
        actual = predicted.get(dataset_id, set())
        package_tp = expected & actual
        package_fp = actual - expected
        package_fn = expected - actual
        tp += len(package_tp)
        fp += len(package_fp)
        fn += len(package_fn)
        family_counts = by_family[item["family"]]
        family_counts["tp"] += len(package_tp)
        family_counts["fp"] += len(package_fp)
        family_counts["fn"] += len(package_fn)
        packages.append(
            {
                "datasetId": dataset_id,
                "family": item["family"],
                "tier": item["tier"],
                "expected": sorted(expected),
                "predicted": sorted(actual),
                "falsePositive": sorted(package_fp),
                "missed": sorted(package_fn),
            }
        )
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall and precision + recall else 0.0
    family_metrics = {}
    for family, counts in sorted(by_family.items()):
        denominator = counts["tp"] + counts["fn"]
        family_metrics[family] = {
            **counts,
            "recall": counts["tp"] / denominator if denominator else None,
        }
    return {
        "unit": "package x canonical defect class",
        "canonicalization": "L0-01 and L0-03 collapse to L0-MISSING because their published observations are indistinguishable",
        "truePositive": tp,
        "falsePositive": fp,
        "falseNegative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "byFamily": family_metrics,
        "packages": packages,
    }


def _read_scores(output: Path) -> dict[str, float | None]:
    scores = {}
    for path in (output / "04_评分卡").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        scores[value["datasetId"]] = value.get("totalScore")
    return scores


def _tier_pairwise(values: dict[str, float | None], truth: dict[str, dict[str, Any]], higher_is_better: bool) -> dict[str, Any]:
    comparable = correct = ties = 0
    by_family: dict[str, list[str]] = collections.defaultdict(list)
    for dataset_id, item in truth.items():
        if values.get(dataset_id) is not None:
            by_family[item["family"]].append(dataset_id)
    for datasets in by_family.values():
        for left_index, left in enumerate(datasets):
            for right in datasets[left_index + 1 :]:
                left_tier = TIER_ORDER[truth[left]["tier"]]
                right_tier = TIER_ORDER[truth[right]["tier"]]
                if left_tier == right_tier:
                    continue
                comparable += 1
                left_value = float(values[left])
                right_value = float(values[right])
                if left_value == right_value:
                    ties += 1
                    continue
                expected_left_better = left_tier < right_tier
                predicted_left_better = left_value > right_value if higher_is_better else left_value < right_value
                correct += int(expected_left_better == predicted_left_better)
    return {
        "comparablePairs": comparable,
        "correctPairs": correct,
        "ties": ties,
        "accuracyWithTiesWrong": correct / comparable if comparable else None,
    }


def _monotonicity(findings: list[dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: dict[tuple[str, str, str], int] = collections.Counter()
    severities: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for row in findings:
        if row.get("status") != opc_agent.STATUS_FAIL:
            continue
        dataset_id = row["datasetId"]
        if dataset_id not in truth:
            continue
        defect_type = _canonical_class(str(row.get("defectType", "")))
        if defect_type not in SCORABLE_CLASSES:
            continue
        key = (truth[dataset_id]["family"], truth[dataset_id]["tier"], defect_type)
        counts[key] += 1
        severity = row.get("severity", {}).get("value")
        if isinstance(severity, (int, float)) and math.isfinite(severity):
            severities[key].append(float(severity))

    coverage_pairs = coverage_correct = 0
    intensity_pairs = intensity_correct = 0
    for family in FAMILIES:
        present_by_tier = {
            tier: set(next(item["expectedClasses"] for item in truth.values() if item["family"] == family and item["tier"] == tier))
            for tier in TIERS
        }
        for left, right in zip(TIERS[1:], TIERS[2:]):
            shared = present_by_tier[left] & present_by_tier[right]
            for defect_type in shared:
                coverage_pairs += 1
                coverage_correct += int(counts[(family, left, defect_type)] < counts[(family, right, defect_type)])
                if defect_type not in INTENSITY_CLASSES:
                    continue
                left_values = severities[(family, left, defect_type)]
                right_values = severities[(family, right, defect_type)]
                if left_values and right_values:
                    intensity_pairs += 1
                    left_mean = sum(left_values) / len(left_values)
                    right_mean = sum(right_values) / len(right_values)
                    intensity_correct += int(left_mean < right_mean)
    return {
        "coverageStrictIncrease": {
            "comparableConsecutivePairs": coverage_pairs,
            "correctPairs": coverage_correct,
            "accuracy": coverage_correct / coverage_pairs if coverage_pairs else None,
        },
        "meanSeverityStrictIncrease": {
            "classes": sorted(INTENSITY_CLASSES),
            "comparableConsecutivePairs": intensity_pairs,
            "correctPairs": intensity_correct,
            "accuracy": intensity_correct / intensity_pairs if intensity_pairs else None,
        },
    }


def _clean_failures(findings: list[dict[str, Any]], truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for row in findings
        if truth[row["datasetId"]]["tier"] == "clean" and row.get("status") == opc_agent.STATUS_FAIL
    ]
    return {
        "count": len(rows),
        "byType": dict(sorted(collections.Counter(row["defectType"] for row in rows).items())),
        "note": "Includes non-official structural findings; useful for schema compatibility regression testing.",
    }


def _run_candidate(name: str, checks: set[str], publish: Path, truth: dict[str, dict[str, Any]], output: Path) -> dict[str, Any]:
    tracemalloc.start()
    started = time.perf_counter()
    summary = opc_agent.run(publish, output, 20261023, checks)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    findings_path = output / "06_问题清单" / "findings.jsonl"
    findings = _load_jsonl(findings_path)
    predicted = _predicted_classes(findings)
    return {
        "candidate": name,
        "checks": sorted(checks),
        "elapsedSeconds": round(elapsed, 6),
        "peakPythonMemoryBytes": peak_bytes,
        "runSummary": summary,
        "packageClassPresence": _class_metrics(predicted, truth),
        "tierPairwise": _tier_pairwise(_read_scores(output), truth, higher_is_better=True),
        "severityMonotonicity": _monotonicity(findings, truth),
        "cleanFailures": _clean_failures(findings, truth),
        "pendingCount": sum(row.get("status") == opc_agent.STATUS_PENDING for row in findings),
        "audit": audit(output),
        "findingsSha256": _sha256(findings_path),
    }


def _label_payloads(dataset: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((dataset / "label").iterdir())
        if path.is_file()
    }


def _weak_reference_scores(publish: Path, truth: dict[str, dict[str, Any]]) -> tuple[dict[str, float], float]:
    started = time.perf_counter()
    by_family: dict[str, dict[str, Path]] = collections.defaultdict(dict)
    for dataset_id, item in truth.items():
        by_family[item["family"]][item["tier"]] = publish / dataset_id
    scores: dict[str, float] = {}
    for tiers in by_family.values():
        clean = tiers["clean"]
        clean_labels = _label_payloads(clean)
        clean_data_names = {path.name for path in (clean / "data").iterdir() if path.is_file()}
        denominator = max(1, len(clean_labels))
        for tier, dataset in tiers.items():
            labels = _label_payloads(dataset)
            changed = sum(labels.get(name) != digest for name, digest in clean_labels.items())
            label_count_delta = abs(len(labels) - len(clean_labels))
            data_names = {path.name for path in (dataset / "data").iterdir() if path.is_file()}
            data_symmetric_difference = len(clean_data_names ^ data_names)
            scores[dataset.name] = (changed + label_count_delta + data_symmetric_difference) / denominator
    return scores, time.perf_counter() - started


def _auc_clean_vs_polluted(scores: dict[str, float], truth: dict[str, dict[str, Any]]) -> float | None:
    clean = [score for dataset, score in scores.items() if truth[dataset]["tier"] == "clean"]
    polluted = [score for dataset, score in scores.items() if truth[dataset]["tier"] != "clean"]
    if not clean or not polluted:
        return None
    wins = 0.0
    for positive in polluted:
        for negative in clean:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(clean) * len(polluted))


def _weak_baseline(publish: Path, truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores, elapsed = _weak_reference_scores(publish, truth)
    return {
        "candidate": "D_weak_clean_reference_anomaly",
        "scope": "package anomaly and ordinal tier screening only; no defect class or location output",
        "supervision": "one clean package identity per family; no defect-class or location labels",
        "features": ["changed label payload ratio", "label-count delta", "data filename-set delta"],
        "elapsedSeconds": round(elapsed, 6),
        "packageClassPresence": None,
        "cleanVsPollutedAuc": _auc_clean_vs_polluted(scores, truth),
        "tierPairwise": _tier_pairwise(scores, truth, higher_is_better=False),
        "scores": dict(sorted(scores.items())),
        "limitation": "A high anomaly score cannot identify L2/L3 semantics and payload hashes are sensitive to benign metadata changes.",
    }


def run_benchmark(work_directory: Path | None = None) -> dict[str, Any]:
    if work_directory is None:
        manager = tempfile.TemporaryDirectory()
        base = Path(manager.name)
    else:
        manager = None
        base = work_directory
        base.mkdir(parents=True, exist_ok=True)
    try:
        publish, truth = build_multifamily_fixture(base)
        results = []
        for name, checks in CANDIDATES.items():
            results.append(_run_candidate(name, checks, publish, truth, base / f"run-{name}"))
        results.append(_weak_baseline(publish, truth))

        repeat_output = base / "run-C-repeat"
        opc_agent.run(publish, repeat_output, 20261023, CANDIDATES["C_rules_plus_sibling_diff"])
        first_hash = results[2]["findingsSha256"]
        second_hash = _sha256(repeat_output / "06_问题清单" / "findings.jsonl")
        return {
            "benchmark": "controlled OPC multi-family ablation",
            "fixture": {
                "families": list(FAMILIES),
                "tiers": list(TIERS),
                "packageCount": len(truth),
                "samplesPerPackage": 20,
                "isHiddenSetEstimate": False,
            },
            "metricBoundary": {
                "classMetric": "package x canonical defect class, not hidden location-level D",
                "tierMetric": "all 10 pairwise tier comparisons within each of 6 families",
                "severityMetric": "synthetic monotonic injection only, not official S_Q",
            },
            "candidates": results,
            "determinism": {
                "candidate": "C_rules_plus_sibling_diff",
                "repeatFindingsIdentical": first_hash == second_hash,
                "firstSha256": first_hash,
                "secondSha256": second_hash,
            },
        }
    finally:
        if manager is not None:
            manager.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark" / "multifamily_results.json")
    parser.add_argument("--keep-workdir", type=Path)
    args = parser.parse_args()
    result = run_benchmark(args.keep_workdir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

