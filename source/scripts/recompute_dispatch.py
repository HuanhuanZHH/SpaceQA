#!/usr/bin/env python3
"""Recompute a frozen validation-selected policy without model fitting.

Only aggregate counts are exported. Raw observations and sample identifiers stay
in the external run directory. The policy is selected expert OR blind rules.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

FAMILIES = (
    "AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample", "EPB_seg_sample",
    "DMSP_SSUSI", "ULF_sample",
)
TIERS = ("P0", "P1", "P2", "P3", "P4")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes run directory")
    return path


def metrics(counts: list[int]) -> dict[str, int | float | None]:
    tp, fp, fn, tn = counts
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "F1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
    }


def count_bits(target: list, valid: list, predicted: list) -> list[int]:
    """Unknown truth bits never count as positives or negatives."""
    counts = [0, 0, 0, 0]
    require(len(target) == len(valid) == len(predicted) == 3, "Expected three G/C/D bits")
    for y, known, p in zip(target, valid, predicted):
        if known:
            index = 0 if y and p else 1 if p else 2 if y else 3
            counts[index] += 1
    return counts


def combine(probabilities: list | None, thresholds: list, rule_bits: list) -> list[bool]:
    if probabilities is None:
        return [bool(x) for x in rule_bits]
    require(len(probabilities) == len(thresholds) == len(rule_bits) == 3,
            "Expected three probabilities, thresholds and rule bits")
    require(all(isinstance(p, (float, int)) and math.isfinite(p) and 0 <= p <= 1
                for p in probabilities), "Invalid prediction probability")
    return [bool(rule) or p >= threshold
            for p, threshold, rule in zip(probabilities, thresholds, rule_bits)]


def rule_group(code: str) -> int | None:
    if code.startswith("GEOMETRY_") or code in {"L2-G", "L3-MIS"}:
        return 0
    if code in {"L3-06", "L3-07"}:
        return 1
    if code in {"L2-P", "L2-F", "L2-M", "L2-T", "ANNOTATION_STRUCTURE_CONFLICT"}:
        return 2
    return None


def recompute(run_dir: Path, dispatch_path: Path, rule_dir: Path) -> dict:
    frozen = read_json(dispatch_path)
    require(frozen.get("status") == "frozen", "Dispatch must already be frozen")
    require(frozen.get("selection_source") ==
            "Original A validation only; no B data or B outcomes used",
            "Expected original-validation-only expert selection")
    selected = frozen["selected"]
    require(set(selected) == set(FAMILIES), "Expected all six selected families")
    manifest = read_json(run_dir / "code_manifest.json")
    global_barrier = read_json(run_dir / "training_barrier.json")
    require(global_barrier.get("status") == "pass", "Training barrier has not passed")
    require(global_barrier["code_manifest"] == manifest, "Source manifests disagree")
    split_hashes = global_barrier["split_hashes"]
    for split in ("train", "val", "test"):
        require(sha256(run_dir / (split + ".jsonl")) == split_hashes[split],
                "Split hash mismatch: " + split)
    require(frozen["original_val_sha256"] == split_hashes["val"],
            "Dispatch was selected on a different validation split")

    records = {}
    locators = {}
    with (run_dir / "test.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            require(r["split"] == "test", "Non-test record in test split")
            require(r["id"] not in records, "Duplicate held-out identifier")
            require(r["family"] in FAMILIES and r["tier"] in TIERS, "Unknown family/tier")
            records[r["id"]] = {k: r[k] for k in (
                "family", "tier", "target", "truth_valid", "input_missing", "input_sha256")}
            for field in ("label_path", "data_path"):
                locator = r.get(field)
                if locator:
                    require(locator not in locators or locators[locator] == r["id"],
                            "Ambiguous test file locator")
                    locators[locator] = r["id"]
    require(records, "Empty held-out test split")

    rule_bits = {sid: [False] * 3 for sid in records}
    evidence_path = rule_dir / "07_证据索引/evidence_index.csv"
    findings_path = rule_dir / "06_问题清单/findings.jsonl"
    with evidence_path.open(encoding="utf-8-sig", newline="") as stream:
        evidence = {r["evidenceId"]: r["locator"].split("#")[0]
                    for r in csv.DictReader(stream)}
    with findings_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            require(r["checkId"] != "C10", "Reference-assisted rules are forbidden")
            if r["status"] != "不通过":
                continue
            group = rule_group(r["defectType"])
            if group is None:
                continue
            sid = locators.get(evidence[r["evidenceRef"]])
            if sid is not None:
                rule_bits[sid][group] = True

    predictions = {}
    provenance = {}
    unique_checkpoints = {}
    routes = sorted({v["route"] for v in selected.values()} - {"rules"})
    for route in routes:
        require(bool(re.fullmatch(r"[A-Za-z0-9_]+", route)), "Invalid route name")
        report_path = run_dir / "training" / (route + ".json")
        report = read_json(report_path)
        require(report.get("status") == "complete" and report.get("test_read") is False,
                "Training not independently complete: " + route)
        require(set(report["training_families"]) == set(FAMILIES)
                and len(report["actual_train_coverage"]) == 30,
                "Training did not cover all families and tiers: " + route)
        barrier_path = run_dir / "route_barriers" / (route + ".json")
        barrier = read_json(barrier_path)
        require(barrier.get("status") == "pass" and barrier.get("route") == route,
                "Invalid per-model training barrier")
        require(barrier["split_hashes"] == split_hashes, "Per-model split hashes disagree")
        require(barrier["routes"][route]["report_sha256"] == sha256(report_path),
                "Training report hash mismatch")
        require(barrier["checkpoint_hashes"] == report["checkpoint_hashes"],
                "Training/checkpoint provenance disagree")
        for relative, expected_hash in report["checkpoint_hashes"].items():
            require(frozen["checkpoint_hashes"][relative] == expected_hash,
                    "Dispatch checkpoint hash differs from trained checkpoint")
            path = inside(run_dir, relative)
            require(sha256(path) == expected_hash, "Checkpoint hash mismatch: " + relative)
            unique_checkpoints[relative] = {"sha256": expected_hash, "bytes": path.stat().st_size}
        for family, choice in selected.items():
            if choice["route"] == route:
                require(choice["thresholds"] == report["thresholds"][FAMILIES.index(family)],
                        "Frozen thresholds differ from training validation calibration")

        complete_path = run_dir / "predictions" / (route + ".complete.json")
        complete = read_json(complete_path)
        require(complete.get("status") == "complete" and complete["route"] == route,
                "Incomplete prediction file")
        require(complete["barrier_hash"] == sha256(barrier_path), "Prediction barrier hash mismatch")
        required_ids = {sid for sid, r in records.items()
                        if selected[r["family"]]["route"] == route}
        seen = set()
        digest = hashlib.sha256()
        line_count = 0
        path = run_dir / "predictions" / (route + ".jsonl")
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                r = json.loads(line)
                line_count += 1
                sid = r["id"]
                if sid not in required_ids:
                    continue
                require(sid not in seen, "Duplicate selected-model prediction")
                require(r["input_sha256"] == records[sid]["input_sha256"],
                        "Prediction input hash mismatch")
                require(r["status"] in {"success", "input_missing"},
                        "Unexpected prediction status")
                require((r["status"] == "input_missing") == bool(records[sid]["input_missing"]),
                        "Missing-input state differs between test split and prediction")
                predictions[sid] = r["probabilities"] if r["status"] == "success" else None
                seen.add(sid)
        require(seen == required_ids, "Selected-model test predictions are incomplete")
        require(line_count == complete["records"], "Prediction record count mismatch")
        require(digest.hexdigest() == complete["prediction_hash"], "Prediction hash mismatch")
        provenance[route] = {
            "prediction_sha256": digest.hexdigest(), "prediction_records": line_count,
            "report_sha256": sha256(report_path), "barrier_sha256": sha256(barrier_path),
        }

    counts = collections.defaultdict(lambda: [0] * 4)
    samples = collections.Counter()
    missing = collections.Counter()
    false_positive_samples = collections.Counter()
    unknown_bits = collections.Counter()
    for sid, r in records.items():
        choice = selected[r["family"]]
        p = rule_bits[sid] if choice["route"] == "rules" else combine(
            predictions[sid], choice["thresholds"], rule_bits[sid])
        key = (r["family"], r["tier"])
        increment = count_bits(r["target"], r["truth_valid"], p)
        counts[key] = [x + y for x, y in zip(counts[key], increment)]
        samples[key] += 1
        missing[key] += bool(r["input_missing"])
        unknown_bits[key] += sum(not v for v in r["truth_valid"])
        if r["tier"] == "P0" and any(p):
            false_positive_samples[key] += 1

    packages = []
    families = []
    total = [0] * 4
    recalls = []
    for family in FAMILIES:
        merged = [0] * 4
        for tier in TIERS:
            key = (family, tier)
            require(samples[key] > 0, "Missing held-out family/tier package")
            m = metrics(counts[key])
            p0 = tier == "P0"
            packages.append({
                "family": family, "tier": tier, "route": selected[family]["route"],
                "n": samples[key], "missing_inputs": missing[key],
                "unknown_truth_bits": unknown_bits[key], **m,
                "P0_sample_FPR": false_positive_samples[key] / samples[key] if p0 else None,
            })
            if not p0:
                merged = [x + y for x, y in zip(merged, counts[key])]
                if m["recall"] is not None:
                    recalls.append(m["recall"])
        total = [x + y for x, y in zip(total, merged)]
        families.append({"family": family, "route": selected[family]["route"], **metrics(merged)})
    return {
        "schema_version": 1, "policy": "validation-selected expert OR blind rules",
        "scope": "heldout_test", "positive_tiers": list(TIERS[1:]),
        "selection_source": frozen["selection_source"], "test_records": len(records),
        "truth": "Public P0-derived technical proxy; unknown bits masked; not official scientific truth",
        "dispatch_sha256": sha256(dispatch_path), "test_split_sha256": split_hashes["test"],
        "source_code_manifest_sha256": sha256(run_dir / "code_manifest.json"),
        "rule_evidence_sha256": sha256(evidence_path), "rule_findings_sha256": sha256(findings_path),
        "rule_hash_note": "Captured at recomputation; original training barrier did not record rule-evidence hashes",
        "verified_model_artifacts": provenance, "unique_selected_checkpoints": unique_checkpoints,
        "checkpoint_bytes": sum(x["bytes"] for x in unique_checkpoints.values()),
        "checkpoint_size_note": "Checkpoints only; excludes external backbone initialization files, dependencies and container",
        "overall": metrics(total),
        "overall_note": "P1-P4 pooled known truth bits; P0 false positives are reported separately",
        "overall_all_tiers": metrics([sum(c[k] for c in counts.values()) for k in range(4)]),
        "package_macro_recall": sum(recalls) / len(recalls),
        "families": families, "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dispatch", required=True, type=Path)
    parser.add_argument("--rule-run-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite output: " + str(args.output))
    rule_dir = args.rule_run_dir or args.run_dir.parent / "full-matrix-20261031/blind-rules"
    result = recompute(args.run_dir, args.dispatch, rule_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "packages": len(result["packages"]),
                      "overall": result["overall"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
