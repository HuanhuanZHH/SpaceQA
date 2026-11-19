#!/usr/bin/env python3
import json
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import opc_agent
from fixture_factory import build_fixture


EXPECTED = {"L0-03", "L2-F", "L2-P", "L2-T", "L2-G", "L3-MIS", "L3-06", "L3-07", "SAMPLE_DUPLICATE"}
TRACKED = EXPECTED | {"L0-01"}
CANDIDATES = {
    "A_schema_pairing": {"C01", "C02", "C03"},
    "B_single_package_rules": {"C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09"},
    "C_rules_plus_sibling_diff": set(opc_agent.ALL_CHECKS),
}


def main():
    results = []
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        publish, _, corrupt = build_fixture(base)
        for name, checks in CANDIDATES.items():
            output = base / name
            started = time.perf_counter()
            opc_agent.run(publish, output, 20261021, checks)
            elapsed = time.perf_counter() - started
            rows = [json.loads(line) for line in (output / "06_问题清单" / "findings.jsonl").read_text(encoding="utf-8").splitlines()]
            predicted = {row["defectType"] for row in rows if row["datasetId"] == corrupt.name and row["status"] == opc_agent.STATUS_FAIL and row["defectType"] in TRACKED}
            tp = len(predicted & EXPECTED)
            precision = tp / len(predicted) if predicted else 0.0
            recall = tp / len(EXPECTED)
            results.append({
                "candidate": name,
                "checks": sorted(checks),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                "elapsedSeconds": round(elapsed, 4),
                "detected": sorted(predicted),
                "missed": sorted(EXPECTED - predicted),
            })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
