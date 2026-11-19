#!/usr/bin/env python3
"""Compile the official v1.1 Markdown annex into runtime JSON catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


DEFECTS = {
    "L0-01": ("数据文件缺失", True),
    "L0-03": ("孤儿标注", True),
    "L2-G": ("边界与位置偏差", True),
    "L2-P": ("类别标注错误", True),
    "L2-T": ("时段起止时刻抖动", True),
    "L2-F": ("元数据尺寸矛盾", True),
    "L3-06": ("漏标", True),
    "L3-07": ("虚标", True),
    "L3-MIS": ("覆盖不足/覆盖过度", True),
    "L2-A": ("标量数值扰动", False),
    "L2-B": ("结构参数失配", False),
    "L2-C": ("角度循环抖动", False),
    "L2-M": ("元数据/参数失配", False),
    "L1-TS": ("时序断裂/重复", False),
}


def parse_rows(markdown: str) -> list[dict]:
    rows = []
    pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)%\s*\|\s*(.*?)\s*\|\s*$")
    for line in markdown.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        index, directory, family, tier, coverage, defects_text = match.groups()
        defect_ids = re.findall(r"L\d(?:-[A-Z0-9]+)+", defects_text)
        non_scoring = []
        if family.strip() == "CME_C2_sample" and tier.strip() != "clean":
            non_scoring.append("L2-B")
        if family.strip() == "ULF_sample" and tier.strip() != "clean":
            non_scoring.append("L1-TS")
        rows.append(
            {
                "index": int(index),
                "randomizedDir": directory.strip(),
                "datasetId": family.strip(),
                "tier": tier.strip(),
                "coveragePercent": int(coverage),
                "defectClassIds": defect_ids,
                "nonScoringDefectClassIds": non_scoring,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    markdown = source_bytes.decode("utf-8-sig")
    rows = parse_rows(markdown)
    if len(rows) != 30 or [row["index"] for row in rows] != list(range(1, 31)):
        raise SystemExit(f"Expected annex rows 1..30, parsed {len(rows)} rows")
    unknown = sorted({item for row in rows for item in row["defectClassIds"] + row["nonScoringDefectClassIds"] if item not in DEFECTS})
    if unknown:
        raise SystemExit(f"Unknown defect IDs: {unknown}")
    provenance = {
        "sourceFile": args.source.name,
        "sourceSha256": hashlib.sha256(source_bytes).hexdigest(),
        "sourceSection": "附件一 A3/A4",
        "annexVersion": "v2.0",
        "annexDate": "2026-10-19",
        "generation": "deterministic Markdown table compilation",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "publish_tier_catalog.json").write_text(
        json.dumps({"provenance": provenance, "packages": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    defect_rows = [
        {"defectClassId": defect_id, "name": name, "findableWithoutTruth": findable}
        for defect_id, (name, findable) in DEFECTS.items()
    ]
    (args.output_dir / "defect_class_catalog.json").write_text(
        json.dumps({"provenance": provenance, "defectClasses": defect_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"packages": len(rows), "defectClasses": len(defect_rows), "sourceSha256": provenance["sourceSha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
