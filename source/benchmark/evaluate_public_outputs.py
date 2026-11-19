#!/usr/bin/env python3
"""Evaluate only the public labels disclosed by the official annex.

The annex exposes package-level defect-class presence and tier order, not defect
locations or counts. This script deliberately does not report official D/S_Q.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path


TIERS = {"clean": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def load_findings(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(findings_path: Path, ranking_path: Path, catalog_path: Path) -> dict:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    packages = {row["randomizedDir"]: row for row in catalog["packages"]}
    findings = load_findings(findings_path)
    by_dataset = collections.defaultdict(list)
    for finding in findings:
        by_dataset[finding["datasetId"]].append(finding)

    ranks = {}
    with ranking_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ranks[row["datasetId"]] = int(row["rank"])

    package_results = []
    totals = collections.Counter()
    evaluated_dataset_ids = sorted(set(by_dataset) | set(ranks))
    for dataset_id in evaluated_dataset_ids:
        rows = by_dataset[dataset_id]
        if dataset_id not in packages:
            package_results.append({"datasetId": dataset_id, "catalogStatus": "unknown"})
            continue
        expected = set(packages[dataset_id]["defectClassIds"])
        predicted = {
            row["defectType"] for row in rows
            if row.get("status") == "不通过" and isinstance(row.get("defectType"), str) and row["defectType"].startswith("L")
        }
        tp = expected & predicted
        fp = predicted - expected
        fn = expected - predicted
        totals.update({"tp": len(tp), "fp": len(fp), "fn": len(fn)})
        package_results.append(
            {
                "datasetId": dataset_id,
                "family": packages[dataset_id]["datasetId"],
                "tier": packages[dataset_id]["tier"],
                "expectedDefectClasses": sorted(expected),
                "predictedDefectClasses": sorted(predicted),
                "truePositiveClasses": sorted(tp),
                "falsePositiveClasses": sorted(fp),
                "missedClasses": sorted(fn),
                "classPresenceExactMatch": not fp and not fn,
            }
        )

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None
    comparable_pairs = 0
    correct_pairs = 0
    by_family = collections.defaultdict(list)
    for dataset_id in ranks:
        if dataset_id in packages:
            by_family[packages[dataset_id]["datasetId"]].append(dataset_id)
    for family_rows in by_family.values():
        for left_index, left in enumerate(family_rows):
            for right in family_rows[left_index + 1 :]:
                left_tier = TIERS[packages[left]["tier"]]
                right_tier = TIERS[packages[right]["tier"]]
                if left_tier == right_tier:
                    continue
                comparable_pairs += 1
                expected_left_better = left_tier < right_tier
                predicted_left_better = ranks[left] < ranks[right]
                correct_pairs += int(expected_left_better == predicted_left_better)
    return {
        "scope": "public package-level defect-class presence and within-family tier ordering only",
        "notMeasured": ["defect location precision/recall", "defect counts", "severity numeric error", "official D", "official S_Q"],
        "catalogProvenance": catalog["provenance"],
        "datasetsEvaluated": len(evaluated_dataset_ids),
        "classPresenceMicro": {
            "truePositive": totals["tp"],
            "falsePositive": totals["fp"],
            "falseNegative": totals["fn"],
            "precision": precision,
            "recall": recall,
        },
        "tierPairwise": {
            "comparablePairs": comparable_pairs,
            "correctPairs": correct_pairs,
            "accuracy": correct_pairs / comparable_pairs if comparable_pairs else None,
        },
        "packages": package_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--findings", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.findings, args.ranking, args.catalog)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
