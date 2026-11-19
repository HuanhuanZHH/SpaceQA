#!/usr/bin/env python3
"""Check a frozen submission directory against the v1.1 delivery layout.

This is a packaging audit, not a scientific accuracy test. It intentionally
reports missing external evidence instead of inferring it from README text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REQUIRED_FINDING = {
    "findingId", "checkId", "ruleVersion", "datasetId", "datasetVersion",
    "defectType", "status", "evidenceRef", "severity", "affectedScope",
}
REQUIRED_SCORECARD = {
    "datasetId", "datasetVersion", "ruleVersion", "specVersion", "runId",
    "dimensions", "coverage", "checkStatusSummary", "topDefects", "evidenceRefs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="frozen submission root")
    parser.add_argument("--json", dest="json_path", type=Path, help="write machine-readable result")
    args = parser.parse_args()
    root = args.root.resolve()
    checks = []

    def file_check(name: str, relative: str, *, required: bool = True):
        path = root / relative
        ok = path.is_file() if path.suffix else path.is_dir()
        checks.append({"id": name, "status": "PASS" if ok else ("MISSING" if required else "PARTIAL"), "path": relative})
        return path if ok else None

    manifest = file_check("manifest", "00_MANIFEST.yaml")
    for name, rel in [
        ("plan_pdf", "01_方案规划.pdf"), ("deploy_pdf", "02_部署与使用说明.pdf"),
        ("spec_pdf", "03_评价规范.pdf"), ("spec_yaml", "03_评价规范.yaml"),
        ("scorecards", "04_评分卡"), ("ranking", "05_排序表/ranking_global.csv"),
        ("findings", "06_问题清单/findings.jsonl"), ("evidence_index", "07_证据索引/evidence_index.csv"),
        ("run_log", "08_运行日志/run.log.jsonl"), ("environment", "08_运行日志/system_env.txt"),
        ("seeds", "08_运行日志/seeds.yaml"), ("unfinished_pdf", "09_未完成事项说明.pdf"),
        ("video", "10_演示视频.mp4"), ("image_tar", "image/spaceqa-agent-image.tar"),
    ]:
        file_check(name, rel)

    finding_path = root / "06_问题清单/findings.jsonl"
    finding_errors = []
    if finding_path.is_file():
        for line_no, line in enumerate(finding_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                finding_errors.append(f"line {line_no}: invalid JSON")
                continue
            missing = sorted(REQUIRED_FINDING - row.keys())
            if missing:
                finding_errors.append(f"line {line_no}: missing {','.join(missing)}")
        checks.append({"id": "finding_schema", "status": "PASS" if not finding_errors else "ERROR", "errors": finding_errors[:20]})

    score_errors = []
    score_dir = root / "04_评分卡"
    if score_dir.is_dir():
        cards = sorted(score_dir.glob("*.json"))
        if not cards:
            score_errors.append("no scorecard JSON files")
        for path in cards:
            row = read_json(path)
            if not isinstance(row, dict):
                score_errors.append(f"{path.name}: invalid JSON")
                continue
            missing = sorted(REQUIRED_SCORECARD - row.keys())
            if missing:
                score_errors.append(f"{path.name}: missing {','.join(missing)}")
        checks.append({"id": "scorecard_schema", "status": "PASS" if not score_errors else "ERROR", "errors": score_errors[:20], "count": len(cards)})

    ranking_path = root / "05_排序表/ranking_global.csv"
    ranking_errors = []
    if ranking_path.is_file():
        with ranking_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        required = {"rank", "datasetId", "datasetVersion", "totalScore", "comparisonBasis"}
        if not required.issubset(rows[0].keys() if rows else set()):
            ranking_errors.append("required ranking columns are missing")
        ranks = [int(row["rank"]) for row in rows if str(row.get("rank", "")).isdigit()]
        if ranks != list(range(1, len(ranks) + 1)):
            ranking_errors.append("rank is not continuous from 1")
        if len({row.get("datasetId") for row in rows}) != len(rows):
            ranking_errors.append("datasetId is duplicated")
    if ranking_path.is_file():
        checks.append({"id": "ranking_schema", "status": "PASS" if not ranking_errors else "ERROR", "errors": ranking_errors})

    log_path = root / "08_运行日志/run.log.jsonl"
    log_errors = []
    if log_path.is_file():
        for line_no, line in enumerate(log_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                log_errors.append(f"line {line_no}: invalid JSON")
                continue
            required = {"ts", "runId", "event", "checkId", "status", "elapsedMs", "retryCount"}
            missing = sorted(required - row.keys())
            if missing:
                log_errors.append(f"line {line_no}: missing {','.join(missing)}")
        checks.append({"id": "log_schema", "status": "PASS" if not log_errors else "ERROR", "errors": log_errors[:20]})

    result = {"root": str(root), "checks": checks, "sha256": {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file() and p.stat().st_size < 1024 * 1024 * 1024}}
    result["status"] = "PASS" if all(item["status"] == "PASS" for item in checks) else "INCOMPLETE"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_path:
        args.json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
