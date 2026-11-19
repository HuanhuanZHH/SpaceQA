"""Expose a B-module result as a versioned, read-only SpaceQA module record.

The adapter deliberately does not alter formal findings or scores.  It is useful
for report builders and the web service to display B metrics with their evidence
and truth limitations attached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_record(root: Path) -> dict:
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "指标" / "metrics.json").read_text(encoding="utf-8"))
    return {
        "moduleName": "B_omission_and_eight_class",
        "version": verification.get("detectionVersion"),
        "status": "validated_offline_compatible",
        "formalScoreMutation": False,
        "metrics": {
            "all8_micro": metrics.get("micro"),
            "omission": next((row for row in metrics.get("current", []) if row.get("code") == "L3-06"), None),
        },
        "verification": verification,
        "artifacts": [
            "指标/metrics.json",
            "指标/eight_class_metrics.csv",
            "交付文件/03_评价规范.md",
            "交付文件/03_评价规范.yaml",
            "source/verify_b.py",
        ],
        "limitations": [
            "P0/规则派生技术参考，不是独立科学真值",
            "B 结果默认作为离线复评层，不替换正式确认项",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("b_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_record(args.b_root), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

