#!/usr/bin/env python3
"""Evidence-first OPC dataset quality evaluator.

The scoring and evidence paths are deterministic. Optional model-backed checks are
intentionally excluded from the competition path until they have golden fixtures.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import struct
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import __version__, backend_strategy
from .resources import asset
from .runtime_metrics import process_usage


VERSION = __version__
RULE_VERSION = "2026.10.31-7"
SPEC_VERSION = "1.0.0"
SCORE_FINDING_CAP = 25.0
SCORE_CRITICAL_MULTIPLIER = 2.0
SCORE_CRITICAL_CAP = 59.0
SCORE_CRITICAL_RATIO_SLOPE = 10.0
STATUS_PASS = "通过"
STATUS_FAIL = "不通过"
STATUS_PENDING = "待判定"
STATUS_NA = "不适用"
ALL_CHECKS = ("C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12")

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

DIMENSION_LABELS = {
    "completeness": "完整性",
    "annotationQuality": "标注质量",
    "sampleCorrespondence": "样本对应关系",
    "coverageDistribution": "覆盖与分布",
    "traceability": "文档与来源追溯",
    "usageConditions": "使用条件",
}

EXPECTED_EVENT_TYPES = {
    "AIA_AR_sample": {"AR"},
    "HMI_AR_sample": {"AR"},
    "CME_C2_sample": {"CME"},
    "EPB_seg_sample": {"EPB"},
    "DMSP_SSUSI": {"PCA", "TPA", "unclear"},
}

TIME_FIELD_PAIRS = (
    ("startTime", "endTime"),
    ("event_starttime", "event_endtime"),
    ("start_dt", "end_dt"),
    ("dateObs", "dateEnd"),
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_dump(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def png_header(path: Path) -> dict[str, int]:
    # The server filesystem advertises 1 MiB blocks. Buffered reads would fetch
    # a full block per PNG even though this check only needs the 29-byte header.
    with path.open("rb", buffering=0) as handle:
        header = handle.read(29)
    if len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG signature or IHDR")
    width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", header[16:29]
    )
    if compression != 0 or filtering != 0:
        raise ValueError("unsupported PNG compression or filtering method")
    return {
        "width": width,
        "height": height,
        "bitDepth": bit_depth,
        "colorType": color_type,
        "interlace": interlace,
    }


def walk_dicts(value: Any, path: str = "$") -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        yield path, value
        for key, nested in value.items():
            yield from walk_dicts(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from walk_dicts(nested, f"{path}[{index}]")


def family_name(name: str) -> str:
    return re.sub(r"_[a-z0-9]{10}$", "", name)


def label_reference(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    basic = value.get("basic_annotation")
    header = value.get("label_header")
    if isinstance(basic, dict) and isinstance(basic.get("fileName"), str):
        return basic["fileName"]
    if isinstance(header, dict) and isinstance(header.get("fileName"), str):
        return header["fileName"]
    return None


def event_annotation(value: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    for key in ("hek_event_annotation", "event_annotation"):
        annotation = value.get(key)
        if isinstance(annotation, dict):
            return key, annotation
    for key, annotation in value.items():
        if key.endswith("event_annotation") and isinstance(annotation, dict):
            return key, annotation
    return None, None


def event_records(annotation: dict[str, Any] | None) -> tuple[str | None, list[dict[str, Any]]]:
    if not annotation:
        return None, []
    for key in ("matched_events", "events", "annotations", "objects"):
        records = annotation.get(key)
        if isinstance(records, list):
            return key, [record for record in records if isinstance(record, dict)]
    return None, []


def event_category(event: dict[str, Any]) -> Any:
    for key in ("event_type", "class", "kind", "label"):
        if event.get(key) not in (None, ""):
            return event[key]
    return None


def sample_key(label_path: Path, value: dict[str, Any]) -> str:
    basic = value.get("basic_annotation") or {}
    if isinstance(basic, dict):
        # Window times can themselves be polluted; retain stable file identity.
        for key in ("fileName", "startTime", "timestamp"):
            if basic.get(key):
                return f"{key}:{basic[key]}"
    return f"label:{label_path.name}"


def event_key(event: dict[str, Any], index: int) -> str:
    for key in ("SOL_standard", "solar_object_locator", "kb_archivid", "event_id", "id"):
        if event.get(key):
            return f"{key}:{event[key]}"
    if event.get("source"):
        return f"source:{event['source']}"
    return f"index:{index}"


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
    ) / 2.0


def numeric_polygon(value: Any) -> list[tuple[float, float]] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, list) or len(point) != 2:
            return None
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in point):
            return None
        points.append((float(point[0]), float(point[1])))
    return points


def science_shapes(value: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    science = value.get("science_annotation")
    if not isinstance(science, dict):
        return []
    shapes = []
    for json_path, nested in walk_dicts(science, "$.science_annotation"):
        if isinstance(nested.get("shape_type"), str) and "points" in nested:
            shapes.append((json_path, nested))
    return shapes


def shape_polygon(shape: dict[str, Any]) -> list[tuple[float, float]] | None:
    raw = shape.get("points")
    if not isinstance(raw, list):
        return None
    points = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            return None
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in point):
            return None
        points.append((float(point[0]), float(point[1])))
    shape_type = str(shape.get("shape_type", "")).lower()
    if shape_type == "rectangle":
        if len(points) != 2:
            return None
        (x1, y1), (x2, y2) = points
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    if shape_type == "polygon" and len(points) >= 3:
        return points
    return None


def _shape_stable_key(shape: dict[str, Any]) -> str | None:
    for key in ("shape_id", "shapeId", "object_id", "objectId", "id", "source"):
        value = shape.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    return None


def _shape_geometry_cost(reference: dict[str, Any], actual: dict[str, Any]) -> float:
    ref_points = shape_polygon(reference)
    actual_points = shape_polygon(actual)
    if not ref_points or not actual_points:
        return math.inf
    ref_center = (
        sum(point[0] for point in ref_points) / len(ref_points),
        sum(point[1] for point in ref_points) / len(ref_points),
    )
    actual_center = (
        sum(point[0] for point in actual_points) / len(actual_points),
        sum(point[1] for point in actual_points) / len(actual_points),
    )
    center_distance = math.hypot(actual_center[0] - ref_center[0], actual_center[1] - ref_center[1])
    scale_distance = abs(math.sqrt(polygon_area(actual_points)) - math.sqrt(polygon_area(ref_points)))
    return center_distance + scale_distance


def match_science_shapes(
    reference: list[tuple[str, dict[str, Any]]],
    actual: list[tuple[str, dict[str, Any]]],
) -> tuple[list[tuple[int, int, str, float]], list[int], list[int]]:
    """Deterministically pair shapes without relying on mutable list order."""
    matches: list[tuple[int, int, str, float]] = []
    unmatched_ref = set(range(len(reference)))
    unmatched_actual = set(range(len(actual)))
    ref_keys: dict[str, list[int]] = collections.defaultdict(list)
    actual_keys: dict[str, list[int]] = collections.defaultdict(list)
    for index, (_, shape) in enumerate(reference):
        key = _shape_stable_key(shape)
        if key:
            ref_keys[key].append(index)
    for index, (_, shape) in enumerate(actual):
        key = _shape_stable_key(shape)
        if key:
            actual_keys[key].append(index)
    for key in sorted(set(ref_keys) & set(actual_keys)):
        if len(ref_keys[key]) != 1 or len(actual_keys[key]) != 1:
            continue
        ref_index = ref_keys[key][0]
        actual_index = actual_keys[key][0]
        matches.append((ref_index, actual_index, "stableKey", 0.0))
        unmatched_ref.discard(ref_index)
        unmatched_actual.discard(actual_index)

    candidates = []
    for ref_index in unmatched_ref:
        for actual_index in unmatched_actual:
            cost = _shape_geometry_cost(reference[ref_index][1], actual[actual_index][1])
            candidates.append((cost, ref_index, actual_index))
    for cost, ref_index, actual_index in sorted(candidates):
        if ref_index not in unmatched_ref or actual_index not in unmatched_actual:
            continue
        matches.append((ref_index, actual_index, "geometryMinCost", cost))
        unmatched_ref.remove(ref_index)
        unmatched_actual.remove(actual_index)
    return sorted(matches), sorted(unmatched_ref), sorted(unmatched_actual)


@dataclass
class Dataset:
    root: Path
    dataset_id: str
    dataset_version: str = "unknown"
    tier: str | None = None
    data_files: dict[str, Path] = field(default_factory=dict)
    label_files: dict[str, Path] = field(default_factory=dict)
    png_meta: dict[str, dict[str, int]] = field(default_factory=dict)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    parse_errors: dict[str, str] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    run_events: list[dict[str, Any]] = field(default_factory=list)
    seed: int | None = None
    event_log_path: Path | None = field(default=None, repr=False)
    logging_version: int = 1
    backend_profile: str = "legacy"
    backend_observations: list[dict[str, Any]] = field(default_factory=list)
    backend_formal_signal_count: int = 0


def make_finding(
    dataset: Dataset,
    check_id: str,
    defect_type: str,
    status: str,
    locator: str,
    basis: str,
    severity_value: float = 0.0,
    severity_metric: str = "count",
    severity_unit: str = "item",
    tolerance: float | None = 0.0,
    affected_count: int | None = None,
    affected_ratio: float | None = None,
    criticality: str = "一般",
    confirmed: str = "已确认错误",
    observation: Any = None,
    locator_type: str = "file",
) -> dict[str, Any]:
    return {
        "findingId": "",
        "checkId": check_id,
        "ruleVersion": RULE_VERSION,
        "datasetId": dataset.dataset_id,
        "datasetVersion": dataset.dataset_version,
        "defectType": defect_type,
        "status": status,
        "evidenceRef": "",
        "severity": {
            "metric": severity_metric,
            "value": severity_value,
            "unit": severity_unit,
            "tolerance": tolerance,
        },
        "affectedScope": {
            "sampleCount": affected_count,
            "ratio": affected_ratio,
        },
        "criticality": criticality,
        "basis": basis,
        "notApplicableReason": basis if status == STATUS_NA else None,
        "confirmedOrPending": "待复核事项" if status == STATUS_PENDING else confirmed,
        "_locator": locator,
        "_locatorType": locator_type,
        "_observation": observation,
    }


def add_check_result(dataset: Dataset, check_id: str, failures: list[dict[str, Any]], pass_basis: str) -> None:
    if failures:
        dataset.findings.extend(failures)
    else:
        dataset.findings.append(
            make_finding(dataset, check_id, "NONE", STATUS_PASS, str(dataset.root), pass_basis, observation={"failureCount": 0})
        )


def _record_event(dataset: Dataset, event: dict[str, Any]) -> None:
    dataset.run_events.append(event)
    if dataset.event_log_path is not None:
        # Flush every actual event so the web worker can show progress and a
        # terminated process leaves its partial history available for review.
        with dataset.event_log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()


def start_check(dataset: Dataset, run_id: str, check_id: str) -> tuple[float, float]:
    started = time.perf_counter()
    cpu = time.process_time()
    _record_event(dataset, {
        "ts": utc_now(), "runId": run_id, "datasetId": dataset.dataset_id,
        "event": "检查开始", "checkId": check_id, "status": "运行中",
        "failureReason": None, "elapsedMs": 0.0,
        "ruleVersion": RULE_VERSION, "systemVersion": VERSION, "seed": dataset.seed,
        "executionMode": "rules-only", "modelForwardPasses": 0,
        "tokenUsage": 0, "retryCount": 0,
    })
    return started, cpu


def log_event(dataset: Dataset, run_id: str, check_id: str, started: float, status: str,
              error: str | None = None, started_cpu: float | None = None) -> None:
    usage = process_usage()
    if started_cpu is not None:
        usage["cpuDeltaSeconds"] = round(max(0.0, usage["cpuSeconds"] - started_cpu), 6)
    _record_event(dataset,
        {
            "ts": utc_now(),
            "runId": run_id,
            "datasetId": dataset.dataset_id,
            "event": "检查结束" if status == "实际完成" else "失败",
            "checkId": check_id,
            "status": status,
            "failureReason": error,
            "elapsedMs": round((time.perf_counter() - started) * 1000, 3),
            "resourcePeak": {"cpu": None, "memoryBytes": usage["peakRssBytes"], "accelerator": None},
            "resourceMeasurementScope": "process_lifetime_high_water_not_check_exclusive",
            "processUsage": usage,
            "ruleVersion": RULE_VERSION,
            "systemVersion": VERSION,
            "seed": dataset.seed,
            "executionMode": "rules-only",
            "modelForwardPasses": 0,
            "tokenUsage": 0,
            "retryCount": 0,
        }
    )


def discover_datasets(input_path: Path) -> list[Path]:
    input_path = input_path.resolve()
    roots = []
    known_families = set(EXPECTED_EVENT_TYPES) | {"ULF_sample"}
    for directory, subdirs, filenames in os.walk(input_path):
        root = Path(directory)
        nested = "data" in subdirs and "label" in subdirs
        suffixes = {Path(name).suffix.lower() for name in filenames}
        flat = ".json" in suffixes and (
            bool(suffixes & {".png", ".csv"}) or family_name(root.name) in known_families
        )
        if nested or flat:
            roots.append(root)
            # A package is an input unit; its data and label directories are not.
            subdirs[:] = []
    return sorted(roots, key=lambda path: str(path))


def load_tiers(input_path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    embedded_catalog = asset("spec", "publish_tier_catalog.json")
    if embedded_catalog.is_file():
        catalog = json.loads(embedded_catalog.read_text(encoding="utf-8"))
        for row in catalog.get("packages", []):
            directory = row.get("randomizedDir")
            if directory:
                result[directory] = {
                    "dataset_id": row.get("datasetId"),
                    "tier": row.get("tier"),
                    "randomized_dir": directory,
                    "defect_class_ids": " ".join(row.get("defectClassIds", [])),
                    "coverage_percent": str(row.get("coveragePercent", "")),
                    "catalog_source": "embedded official v1.1 annex",
                }
    candidates = list(input_path.rglob("publish_tier_labels.csv")) if input_path.is_dir() else []
    for path in candidates:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                directory = row.get("randomized_dir") or row.get("directory")
                if directory:
                    result[directory] = row
    return result


def inventory_dataset(root: Path, tier_info: dict[str, str] | None) -> Dataset:
    tier = tier_info.get("tier") if tier_info else None
    dataset_id = root.name
    version = tier or "unknown"
    dataset = Dataset(root=root, dataset_id=dataset_id, dataset_version=version, tier=tier)
    if (root / "data").is_dir() and (root / "label").is_dir():
        dataset.data_files = {path.name: path for path in sorted((root / "data").iterdir()) if path.is_file()}
        dataset.label_files = {path.name: path for path in sorted((root / "label").iterdir()) if path.is_file()}
    else:
        files = [path for path in sorted(root.iterdir()) if path.is_file()]
        dataset.data_files = {path.name: path for path in files if path.suffix.lower() in {".png", ".csv"}}
        dataset.label_files = {path.name: path for path in files if path.suffix.lower() == ".json"}
    return dataset


def check_c01(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    for name, path in dataset.data_files.items():
        if path.suffix.lower() == ".png":
            try:
                dataset.png_meta[name] = png_header(path)
            except Exception as exc:
                failures.append(make_finding(dataset, "C01", "FILE_CORRUPT", STATUS_FAIL, str(path), f"PNG 无法解析: {exc}", 1, criticality="关键", observation={"error": str(exc)}))
    for name, path in dataset.label_files.items():
        if path.suffix.lower() != ".json":
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level JSON must be an object")
            dataset.labels[name] = value
        except Exception as exc:
            dataset.parse_errors[name] = str(exc)
            failures.append(make_finding(dataset, "C01", "JSON_INVALID", STATUS_FAIL, str(path), f"JSON 无法解析: {exc}", 1, criticality="关键", observation={"error": str(exc)}))
    add_check_result(dataset, "C01", failures, "全部 PNG 头与 JSON 对象均可确定性解析")
    return failures


def check_c02(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    referenced = collections.defaultdict(list)
    for label_name, value in dataset.labels.items():
        ref = label_reference(value)
        if ref:
            referenced[ref].append(label_name)
            if ref not in dataset.data_files:
                finding = make_finding(
                    dataset,
                    "C02",
                    "L0-03",
                    STATUS_FAIL,
                    str(dataset.label_files[label_name]),
                    "标注引用的数据文件不存在；该观测同时符合官方 L0-01 与 L0-03 文字定义，暂按孤儿标注 L0-03 输出",
                    1,
                    criticality="关键",
                    observation={
                        "observableCondition": "label_references_absent_data_file",
                        "referencedFile": ref,
                        "candidateDefectTypes": ["L0-01", "L0-03"],
                    },
                )
                finding["candidateDefectTypes"] = ["L0-01", "L0-03"]
                finding["classificationStatus"] = "ambiguous_official_definition"
                failures.append(finding)
        else:
            failures.append(make_finding(dataset, "C02", "LABEL_REFERENCE_MISSING", STATUS_FAIL, str(dataset.label_files[label_name]), "标注缺少可解析的 fileName 引用", 1, criticality="关键"))
    for data_name, path in dataset.data_files.items():
        if data_name not in referenced:
            failures.append(make_finding(dataset, "C02", "DATA_WITHOUT_LABEL", STATUS_FAIL, str(path), "数据文件没有任何标签引用", 1, observation={"dataFile": data_name}))
    denominator = max(1, len(dataset.data_files) + len(dataset.labels))
    for finding in failures:
        finding["affectedScope"]["ratio"] = 1 / denominator
    add_check_result(dataset, "C02", failures, "数据文件与标签引用形成完整对应关系")
    return failures


def check_c03(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    for label_name, value in dataset.labels.items():
        header = value.get("label_header") or {}
        basic = value.get("basic_annotation") or {}
        header_ref = header.get("fileName") if isinstance(header, dict) else None
        basic_ref = basic.get("fileName") if isinstance(basic, dict) else None
        header_name = Path(header_ref).name if isinstance(header_ref, str) else None
        basic_name = Path(basic_ref).name if isinstance(basic_ref, str) else None
        header_targets_label = header_name == label_name
        header_targets_data = header_name in dataset.data_files if header_name else False
        basic_targets_data = basic_name in dataset.data_files if basic_name else False

        # Organizer examples use two valid conventions: header points to the
        # data file, or header points to the JSON while basic points to data.
        if header_ref and basic_ref and not header_targets_label and header_name != basic_name:
            failures.append(make_finding(
                dataset,
                "C03",
                "LABEL_REFERENCE_CONFLICT",
                STATUS_FAIL,
                str(dataset.label_files[label_name]),
                "label_header.fileName 与 basic_annotation.fileName 既非同一数据引用，也非 JSON/数据双引用约定",
                1,
                observation={
                    "header": header_ref,
                    "basic": basic_ref,
                    "headerTargetsLabel": header_targets_label,
                    "headerTargetsData": header_targets_data,
                    "basicTargetsData": basic_targets_data,
                },
            ))

        expected_from_data = {f"{basic_name}.json", f"{Path(basic_name).stem}.json"} if basic_name else set()
        if basic_name and not header_targets_label and label_name not in expected_from_data:
            failures.append(make_finding(
                dataset,
                "C03",
                "LABEL_FILENAME_MISMATCH",
                STATUS_FAIL,
                str(dataset.label_files[label_name]),
                "JSON 文件名既不等于 label_header.fileName，也不符合 <data file>.json 或 <data stem>.json 约定",
                1,
                observation={"labelFile": label_name, "header": header_ref, "basic": basic_ref, "expectedFromData": sorted(expected_from_data)},
            ))
    add_check_result(dataset, "C03", failures, "标签文件名及 fileName 字段符合数据引用或 JSON/数据双引用约定")
    return failures


def check_c04(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    for label_name, value in dataset.labels.items():
        _, annotation = event_annotation(value)
        if annotation is None:
            continue
        _, event_list = event_records(annotation)
        events = event_list
        declared = annotation.get("event_count")
        if declared is None:
            declared = annotation.get("n_events")
        actual = len(events)
        has_events = annotation.get("has_events")
        event_types = annotation.get("event_types")
        observed_types = sorted({str(event_category(event)) for event in events or [] if event_category(event) is not None})
        reasons = []
        if declared is not None and (not isinstance(declared, int) or declared != actual):
            reasons.append("event_count 与 matched_events 长度不一致")
        if has_events is not None and bool(actual) != bool(has_events):
            reasons.append("has_events 与 matched_events 是否为空不一致")
        if isinstance(event_types, list) and sorted(map(str, event_types)) != observed_types:
            reasons.append("event_types 与事件记录中的类别集合不一致")
        if reasons:
            failures.append(make_finding(dataset, "C04", "ANNOTATION_STRUCTURE_CONFLICT", STATUS_FAIL, str(dataset.label_files[label_name]), "；".join(reasons), 1, observation={"declared": declared, "actual": actual, "hasEvents": has_events, "declaredTypes": event_types, "observedTypes": observed_types}))
    add_check_result(dataset, "C04", failures, "事件计数、存在标志与类别集合结构自洽")
    return failures


def check_c05(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    for label_name, value in dataset.labels.items():
        ref = label_reference(value)
        image_meta = dataset.png_meta.get(ref or "")
        for json_path, nested in walk_dicts(value):
            if "bbox" in nested and "bbox_polygon_pixel" not in nested:
                bbox = nested.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in bbox):
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_INVALID", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}.bbox", "bbox 不是四个有限数值 [left, top, width, height]", 1, locator_type="annotationRecord"))
                else:
                    left, top, width, height = map(float, bbox)
                    if width <= 0 or height <= 0:
                        failures.append(make_finding(dataset, "C05", "GEOMETRY_ZERO_AREA", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}.bbox", "bbox 宽高必须为正", min(width, height), "minimumExtent", "pixel", 0, locator_type="imageRegion", observation={"bbox": bbox}))
                    elif image_meta and (left < 0 or top < 0 or left + width > image_meta["width"] or top + height > image_meta["height"]):
                        overflow = max(-left, -top, left + width - image_meta["width"], top + height - image_meta["height"], 0)
                        failures.append(make_finding(dataset, "C05", "GEOMETRY_OUT_OF_BOUNDS", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}.bbox", "bbox 超出实际图像边界", overflow, "maxOverflow", "pixel", 0, locator_type="imageRegion", observation={"bbox": bbox, "image": image_meta}))
            if "bbox_polygon_pixel" not in nested:
                if "shape_type" not in nested or "points" not in nested:
                    continue
                points = shape_polygon(nested)
                locator = f"{dataset.label_files[label_name]}#{json_path}.points"
                if points is None:
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_INVALID", STATUS_FAIL, locator, "shape_type/points 不是合法 rectangle 或 polygon", 1, locator_type="annotationRecord", observation={"shapeType": nested.get("shape_type")}))
                    continue
                if polygon_area(points) <= 0:
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_ZERO_AREA", STATUS_FAIL, locator, "标注形状面积不为正", 1, locator_type="imageRegion"))
            else:
                raw_polygon = nested.get("bbox_polygon_pixel")
                if raw_polygon is None:
                    continue
                points = numeric_polygon(raw_polygon)
                locator = f"{dataset.label_files[label_name]}#{json_path}.bbox_polygon_pixel"
                if points is None:
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_INVALID", STATUS_FAIL, locator, "多边形不是至少三个有限二维点", 1, locator_type="annotationRecord"))
                    continue
                if points[0] != points[-1]:
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_OPEN", STATUS_FAIL, locator, "多边形首尾点不闭合", 1, locator_type="imageRegion", observation={"first": points[0], "last": points[-1]}))
                if polygon_area(points) <= 0:
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_ZERO_AREA", STATUS_FAIL, locator, "多边形面积不为正", 1, locator_type="imageRegion"))
            if image_meta:
                bad = [(x, y) for x, y in points if x < 0 or y < 0 or x >= image_meta["width"] or y >= image_meta["height"]]
                if bad:
                    max_overflow = max(max(-x, -y, x - image_meta["width"] + 1, y - image_meta["height"] + 1, 0) for x, y in bad)
                    failures.append(make_finding(dataset, "C05", "GEOMETRY_OUT_OF_BOUNDS", STATUS_FAIL, locator, "标注区域超出实际图像边界", max_overflow, "maxOverflow", "pixel", 0, locator_type="imageRegion", observation={"badPoints": bad[:10], "image": image_meta}))
    add_check_result(dataset, "C05", failures, "全部可核验标注多边形合法、闭合且位于图像边界内")
    return failures


def check_c11(dataset: Dataset) -> list[dict[str, Any]]:
    raw_data_paths = set(dataset.data_files.values())
    metadata_files = [
        path for path in dataset.root.iterdir()
        if path.is_file() and path not in raw_data_paths
        and path.suffix.lower() in {".md", ".txt", ".pdf", ".yaml", ".yml", ".csv"}
    ]
    if not metadata_files:
        dataset.findings.append(make_finding(dataset, "C11", "PROVENANCE_DOCUMENT_MISSING", STATUS_PENDING, str(dataset.root), "未发现可复核的数据来源、版本、处理链或字段字典文档；需补充机器可读元数据与原始说明", observation={"metadataFiles": []}))
        return []
    dataset.findings.append(make_finding(dataset, "C11", "NONE", STATUS_PASS, str(dataset.root), "发现可用于来源与版本追溯的随包文档", observation={"metadataFiles": [path.name for path in metadata_files]}))
    return []


def check_c12(dataset: Dataset) -> list[dict[str, Any]]:
    usage_keys = {"license", "licence", "usage", "purpose", "permission", "restrictions", "declaredusage", "useconditions"}
    matched = []
    for label_name, value in dataset.labels.items():
        for json_path, nested in walk_dicts(value):
            for key in nested:
                if key.lower() in usage_keys and nested.get(key) not in (None, "", []):
                    matched.append({"label": label_name, "path": f"{json_path}.{key}"})
                    break
            if matched:
                break
        if matched:
            break
    usage_documents = [
        path.name for path in dataset.root.iterdir()
        if path.is_file() and re.search(r"(license|licence|usage|permission|readme|使用|许可)", path.name, re.IGNORECASE)
    ]
    if not matched and not usage_documents:
        dataset.findings.append(make_finding(dataset, "C12", "USAGE_CONDITION_MISSING", STATUS_PENDING, str(dataset.root), "未发现用途、权限、许可或限制条件声明；需补充使用条件后才能评价该维度", observation={"usageFields": [], "usageDocuments": []}))
        return []
    dataset.findings.append(make_finding(dataset, "C12", "NONE", STATUS_PASS, str(dataset.root), "发现非空使用条件字段或随包许可文档", observation={"usageFields": matched, "usageDocuments": usage_documents}))
    return []


def check_c06(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    dimension_pairs = (("width", "height"), ("imageWidth", "imageHeight"), ("imgWidth", "imgHeight"), ("image_width", "image_height"))
    for label_name, value in dataset.labels.items():
        ref = label_reference(value)
        image_meta = dataset.png_meta.get(ref or "")
        data_path = dataset.data_files.get(ref or "")
        if not image_meta and not (data_path and data_path.suffix.lower() == ".csv"):
            continue
        if image_meta:
            for json_path, nested in walk_dicts(value):
                path_lower = json_path.lower()
                if any(token in path_lower for token in ("matched_events", ".events", ".annotations", ".objects", "bbox")):
                    continue
                if not any(token in path_lower for token in ("basic_annotation", "science_annotation", "metadata", "image", "header", "size")):
                    continue
                size_text = nested.get("imgSize")
                size_match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*(?:pixels?|px|像素)?\s*", size_text, re.IGNORECASE) if isinstance(size_text, str) else None
                if size_match:
                    declared_size = tuple(map(int, size_match.groups()))
                    actual_size = (image_meta["width"], image_meta["height"])
                    if declared_size != actual_size:
                        failures.append(make_finding(
                            dataset, "C06", "L2-F", STATUS_FAIL,
                            f"{dataset.label_files[label_name]}#{json_path}.imgSize",
                            "imgSize 字符串声明的尺寸与 PNG IHDR 不一致",
                            max(abs(a-b) for a,b in zip(declared_size, actual_size)),
                            "maxDimensionError", "pixel", 0,
                            locator_type="annotationRecord",
                            observation={"declared": declared_size, "actual": actual_size},
                        ))
                for width_key, height_key in dimension_pairs:
                    if width_key not in nested and height_key not in nested:
                        continue
                    declared = (nested.get(width_key), nested.get(height_key))
                    actual = (image_meta["width"], image_meta["height"])
                    if declared != actual:
                        failures.append(make_finding(dataset, "C06", "L2-F", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}", "标注声明的图像尺寸与 PNG IHDR 不一致", max(abs((declared[0] or 0) - actual[0]), abs((declared[1] or 0) - actual[1])) if all(isinstance(x, (int, float)) for x in declared) else 1, "maxDimensionError", "pixel", 0, locator_type="annotationRecord", observation={"declared": declared, "actual": actual}))
            continue

        science = value.get("science_annotation")
        if not isinstance(science, dict) or data_path is None:
            continue
        try:
            with data_path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                row_count = 0
                time_values = []
                for row in reader:
                    if not row:
                        continue
                    row_count += 1
                    if header and header[0].strip().lower() == "t":
                        try:
                            t_value = float(row[0])
                            if math.isfinite(t_value):
                                time_values.append(t_value)
                        except (ValueError, IndexError):
                            pass
        except (OSError, UnicodeError, csv.Error, StopIteration):
            continue

        reasons = []
        relative_errors = []
        declared_points = science.get("n_points")
        if isinstance(declared_points, (int, float)) and declared_points != row_count:
            reasons.append("n_points 与 CSV 数据行数不一致")
            relative_errors.append(abs(float(declared_points) - row_count) / max(1, row_count))

        declared_components = science.get("components")
        time_columns = {"t", "time", "timestamp", "datetime", "date"}
        actual_components = header[1:] if header and header[0].strip().lower() in time_columns else header
        if isinstance(declared_components, list) and list(map(str, declared_components)) != actual_components:
            reasons.append("components 与 CSV 数值列不一致")
            expected_set = set(map(str, declared_components))
            actual_set = set(actual_components)
            relative_errors.append(len(expected_set ^ actual_set) / max(1, len(expected_set | actual_set)))

        rate = science.get("sampling_rate_hz")
        duration = science.get("duration_s")
        deltas = [b-a for a,b in zip(time_values, time_values[1:])]
        positive_deltas = [x for x in deltas if x > 0]
        cadence = statistics.median(positive_deltas) if positive_deltas else None
        cadence_supported = cadence is not None and len(time_values) == row_count and sum(abs(x-cadence) <= 1e-6 for x in deltas) >= .9*len(deltas)
        actual_rate = 1/cadence if cadence_supported else None
        coverage = max(time_values)-min(time_values)+cadence if cadence_supported else None
        if actual_rate is not None and isinstance(rate, (int, float)) and abs(rate-actual_rate) > 1e-6:
            reasons.append("sampling_rate_hz 与 CSV 时间轴典型采样间隔不一致")
            relative_errors.append(abs(rate-actual_rate)/actual_rate)
        if coverage is not None and isinstance(duration, (int, float)) and abs(duration-coverage) > 1e-6:
            reasons.append("duration_s 与 CSV 时间轴包含末采样间隔的覆盖时长不一致")
            relative_errors.append(abs(duration-coverage)/max(coverage, 1e-6))

        # Published file names are anonymized. Their arbitrary prefix cannot
        # establish the observatory's scientific identity.
        observatory = science.get("observatory")

        if reasons:
            failures.append(make_finding(
                dataset,
                "C06",
                "L2-M",
                STATUS_FAIL,
                f"{dataset.label_files[label_name]}#$.science_annotation",
                "；".join(reasons),
                max(relative_errors or [1.0]),
                "maxRelativeMetadataError",
                "ratio",
                0,
                affected_count=1,
                affected_ratio=1 / max(1, len(dataset.labels)),
                locator_type="annotationRecord",
                observation={
                    "csvFile": data_path.name,
                    "csvRowCount": row_count,
                    "csvColumns": header,
                    "declaredNPoints": declared_points,
                    "declaredComponents": declared_components,
                    "samplingRateHz": rate,
                    "durationSeconds": duration,
                    "actualSamplingRateHz": actual_rate,
                    "actualCoverageSeconds": coverage,
                    "cadenceSupported": cadence_supported,
                    "observatory": observatory,
                    "violations": reasons,
                    "relativeErrors": relative_errors,
                },
            ))
    add_check_result(dataset, "C06", failures, "图像尺寸或 ULF CSV 行数、分量、采样率、时长和台站元数据与数据体一致")
    return failures


def check_c07(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    for label_name, value in dataset.labels.items():
        for json_path, nested in walk_dicts(value):
            for start_key, end_key in TIME_FIELD_PAIRS:
                if not nested.get(start_key) or not nested.get(end_key):
                    continue
                start = parse_time(nested[start_key])
                end = parse_time(nested[end_key])
                if start is None or end is None:
                    failures.append(make_finding(dataset, "C07", "TIME_INVALID", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}", "时间字段无法按 ISO 时间解析", 1, locator_type="timeInterval", observation={"start": nested[start_key], "end": nested[end_key]}))
                elif start > end:
                    failures.append(make_finding(dataset, "C07", "L2-T", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}", "起始时间晚于结束时间", (start - end).total_seconds(), "reversal", "second", 0, locator_type="timeInterval", observation={"start": nested[start_key], "end": nested[end_key]}))
    add_check_result(dataset, "C07", failures, "全部可核验起止时间可解析且顺序有效")
    return failures


def exact_duplicate_groups(paths: Iterable[Path]) -> list[list[Path]]:
    by_size: dict[int, list[Path]] = collections.defaultdict(list)
    for path in paths:
        by_size[path.stat().st_size].append(path)
    by_hash: dict[str, list[Path]] = collections.defaultdict(list)
    for candidates in by_size.values():
        if len(candidates) < 2:
            continue
        for path in candidates:
            by_hash[file_hash(path)].append(path)
    return [group for group in by_hash.values() if len(group) > 1]


def check_c08(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    groups = exact_duplicate_groups(dataset.data_files.values())
    for group in groups:
        failures.append(make_finding(dataset, "C08", "SAMPLE_DUPLICATE", STATUS_FAIL, str(group[0]), "多个数据文件内容的 SHA-256 完全相同", len(group), "duplicateGroupSize", "file", 1, affected_count=len(group), affected_ratio=len(group) / max(1, len(dataset.data_files)), observation={"files": [str(path) for path in group]}))
    add_check_result(dataset, "C08", failures, "未发现数据文件内容完全重复")
    return failures


def check_c09(dataset: Dataset) -> list[dict[str, Any]]:
    failures = []
    expected = EXPECTED_EVENT_TYPES.get(family_name(dataset.dataset_id))
    if not expected:
        dataset.findings.append(make_finding(dataset, "C09", "L2-P", STATUS_PENDING, str(dataset.root), "当前数据类型缺少可机械复核的类别字典；需补充标注定义或同源 clean 版本", observation={"family": family_name(dataset.dataset_id)}))
        return failures
    for label_name, value in dataset.labels.items():
        annotation_key, annotation = event_annotation(value)
        _, events = event_records(annotation)
        summary_types = annotation.get("event_types") if annotation else None
        if isinstance(summary_types, list):
            unexpected = sorted({str(t) for t in summary_types} - expected)
            if unexpected:
                failures.append(make_finding(
                    dataset, "C09", "L2-P", STATUS_FAIL,
                    f"{dataset.label_files[label_name]}#$.{annotation_key}.event_types",
                    "事件汇总中的类别不在该数据类型声明的合法集合内", 1,
                    locator_type="annotationRecord",
                    observation={"actual":summary_types, "unexpected":unexpected, "expected":sorted(expected)},
                ))
        for index, event in enumerate(events or []):
            actual = event_category(event)
            if actual and actual not in expected:
                failures.append(make_finding(dataset, "C09", "L2-P", STATUS_FAIL, f"{dataset.label_files[label_name]}#event[{index}]", "类别不在该数据集类型的已声明类别集合内", 1, locator_type="annotationRecord", observation={"actual": actual, "expected": sorted(expected)}))
        for json_path, shape in science_shapes(value):
            actual = shape.get("label")
            if actual and actual not in expected:
                failures.append(make_finding(dataset, "C09", "L2-P", STATUS_FAIL, f"{dataset.label_files[label_name]}#{json_path}.label", "形状类别不在该数据集类型的已声明类别集合内", 1, locator_type="annotationRecord", observation={"actual": actual, "expected": sorted(expected)}))
    add_check_result(dataset, "C09", failures, f"事件类别均属于声明集合 {sorted(expected)}")
    return failures


def base_analysis(dataset: Dataset, run_id: str, enabled_checks: set[str]) -> None:
    checks = {
        "C01": check_c01,
        "C02": check_c02,
        "C03": check_c03,
        "C04": check_c04,
        "C05": check_c05,
        "C06": check_c06,
        "C07": check_c07,
        "C08": check_c08,
        "C09": check_c09,
        "C11": check_c11,
        "C12": check_c12,
    }
    for check_id, function in checks.items():
        if check_id not in enabled_checks:
            continue
        started, started_cpu = start_check(dataset, run_id, check_id)
        try:
            function(dataset)
            log_event(dataset, run_id, check_id, started, "实际完成", started_cpu=started_cpu)
        except Exception as exc:
            dataset.findings.append(make_finding(dataset, check_id, "CHECK_EXECUTION_FAILURE", STATUS_PENDING, str(dataset.root), f"检查执行失败，须按未完成项保留: {exc}", observation={"error": repr(exc)}))
            log_event(dataset, run_id, check_id, started, "失败", repr(exc), started_cpu=started_cpu)


def events_by_key(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _, annotation = event_annotation(value)
    _, events = event_records(annotation)
    output = {}
    for index, event in enumerate(events or []):
        if isinstance(event, dict):
            output[event_key(event, index)] = event
    return output


def compare_event(dataset: Dataset, label_path: Path, json_path: str, reference: dict[str, Any], actual: dict[str, Any], image_meta: dict[str, int] | None) -> list[dict[str, Any]]:
    findings = []
    changed_classes = {
        key: {"reference": reference.get(key), "actual": actual.get(key)}
        for key in ("event_type", "eventtype", "concept", "label", "class", "kind")
        if reference.get(key) != actual.get(key)
    }
    if changed_classes:
        findings.append(make_finding(dataset, "C10", "L2-P", STATUS_FAIL, f"{label_path}#{json_path}", "可信同源 clean 参照中的类别字段与当前记录不同", 1, locator_type="annotationRecord", observation={"changedFields": changed_classes, "reference": event_category(reference), "actual": event_category(actual)}))

    time_fields = dict.fromkeys([key for pair in TIME_FIELD_PAIRS for key in pair] + ["cme_time"])
    for key in time_fields:
        ref_time = parse_time(reference.get(key))
        actual_time = parse_time(actual.get(key))
        if ref_time and actual_time and ref_time != actual_time:
            delta = abs((actual_time - ref_time).total_seconds())
            findings.append(make_finding(dataset, "C10", "L2-T", STATUS_FAIL, f"{label_path}#{json_path}.{key}", "可信同源 clean 参照中的事件时间与当前时间不同", delta, "absoluteTimeOffset", "second", 0, locator_type="timeInterval", observation={"reference": reference.get(key), "actual": actual.get(key)}))

    ref_polygon = numeric_polygon(reference.get("bbox_polygon_pixel"))
    actual_polygon = numeric_polygon(actual.get("bbox_polygon_pixel"))
    if ref_polygon and actual_polygon and len(ref_polygon) == len(actual_polygon):
        displacements = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(actual_polygon, ref_polygon)]
        mean_displacement = sum(displacements) / len(displacements)
        diagonal = math.hypot(image_meta["width"], image_meta["height"]) if image_meta else 1.0
        ref_area = polygon_area(ref_polygon)
        actual_area = polygon_area(actual_polygon)
        area_ratio_error = abs(actual_area / ref_area - 1.0) if ref_area else math.inf
        if mean_displacement > 0.5:
            findings.append(make_finding(dataset, "C10", "L2-G", STATUS_FAIL, f"{label_path}#{json_path}.bbox_polygon_pixel", "标注位置相对同源 clean 参照发生偏移", mean_displacement, "meanVertexDisplacement", "pixel", 0.5, locator_type="imageRegion", observation={"meanPixelDisplacement": mean_displacement, "normalizedDisplacement": mean_displacement / diagonal, "areaRatioError": area_ratio_error}))
        if area_ratio_error > 0.02:
            findings.append(make_finding(dataset, "C10", "L3-MIS", STATUS_FAIL, f"{label_path}#{json_path}.bbox_polygon_pixel", "标注覆盖面积相对同源 clean 参照发生变化", area_ratio_error, "relativeAreaError", "ratio", 0.02, locator_type="imageRegion", observation={"referenceArea": ref_area, "actualArea": actual_area, "relativeAreaError": area_ratio_error, "meanPixelDisplacement": mean_displacement}))
    ref_bbox = reference.get("bbox")
    actual_bbox = actual.get("bbox")
    if (
        isinstance(ref_bbox, list) and isinstance(actual_bbox, list)
        and len(ref_bbox) == len(actual_bbox) == 4
        and all(isinstance(item, (int, float)) and math.isfinite(item) for item in ref_bbox + actual_bbox)
    ):
        displacement = math.hypot(float(actual_bbox[0]) - float(ref_bbox[0]), float(actual_bbox[1]) - float(ref_bbox[1]))
        ref_area = float(ref_bbox[2]) * float(ref_bbox[3])
        actual_area = float(actual_bbox[2]) * float(actual_bbox[3])
        area_ratio_error = abs(actual_area / ref_area - 1.0) if ref_area > 0 else math.inf
        if displacement > 0.5:
            findings.append(make_finding(dataset, "C10", "L2-G", STATUS_FAIL, f"{label_path}#{json_path}.bbox", "bbox 左上角相对同源 clean 参照发生偏移", displacement, "bboxOriginDisplacement", "pixel", 0.5, locator_type="imageRegion", observation={"reference": ref_bbox, "actual": actual_bbox, "areaRatioError": area_ratio_error}))
        if area_ratio_error > 0.02:
            findings.append(make_finding(dataset, "C10", "L3-MIS", STATUS_FAIL, f"{label_path}#{json_path}.bbox", "bbox 覆盖面积相对同源 clean 参照发生变化", area_ratio_error, "relativeAreaError", "ratio", 0.02, locator_type="imageRegion", observation={"reference": ref_bbox, "actual": actual_bbox, "relativeAreaError": area_ratio_error}))
    return findings


def compare_shape(
    dataset: Dataset,
    label_path: Path,
    json_path: str,
    reference: dict[str, Any],
    actual: dict[str, Any],
    image_meta: dict[str, int] | None,
) -> list[dict[str, Any]]:
    findings = []
    if reference.get("label") != actual.get("label"):
        findings.append(make_finding(
            dataset,
            "C10",
            "L2-P",
            STATUS_FAIL,
            f"{label_path}#{json_path}.label",
            "同源 clean 参照中的形状类别与当前类别不同",
            1,
            locator_type="annotationRecord",
            observation={"reference": reference.get("label"), "actual": actual.get("label")},
        ))
    ref_points = shape_polygon(reference)
    actual_points = shape_polygon(actual)
    if not ref_points or not actual_points:
        return findings
    if len(ref_points) != len(actual_points):
        findings.append(make_finding(
            dataset,
            "C10",
            "L2-G",
            STATUS_FAIL,
            f"{label_path}#{json_path}.points",
            "同源 clean 参照与当前形状的顶点数量不同",
            abs(len(actual_points) - len(ref_points)),
            "vertexCountDifference",
            "point",
            0,
            locator_type="imageRegion",
            observation={"referenceCount": len(ref_points), "actualCount": len(actual_points)},
        ))
        return findings
    displacements = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(actual_points, ref_points)]
    mean_displacement = sum(displacements) / len(displacements)
    if mean_displacement <= 0.5:
        return findings
    diagonal = math.hypot(image_meta["width"], image_meta["height"]) if image_meta else 1.0
    ref_area = polygon_area(ref_points)
    actual_area = polygon_area(actual_points)
    area_ratio_error = abs(actual_area / ref_area - 1.0) if ref_area else None
    findings.append(make_finding(
        dataset,
        "C10",
        "L2-G",
        STATUS_FAIL,
        f"{label_path}#{json_path}.points",
        "形状边界相对同源 clean 参照发生偏移",
        mean_displacement,
        "meanVertexDisplacement",
        "pixel",
        0.5,
        locator_type="imageRegion",
        observation={
            "meanPixelDisplacement": mean_displacement,
            "normalizedDisplacement": mean_displacement / diagonal,
            "referenceArea": ref_area,
            "actualArea": actual_area,
            "relativeAreaError": area_ratio_error,
        },
    ))
    return findings


def guard_reference_geometry(dataset: Dataset, reference: Dataset, actual_value: dict[str, Any], ref_value: dict[str, Any], label_path: Path, findings: list[dict[str, Any]], hashes: dict[Path, str]) -> list[dict[str, Any]]:
    """Image transforms require content alignment before semantic confirmation."""
    raw = dataset.data_files.get(label_reference(actual_value) or "")
    clean_raw = reference.data_files.get(label_reference(ref_value) or "")
    if not str(label_reference(actual_value) or "").lower().endswith(".png"):
        return findings
    semantic_types = {"L2-G", "L3-MIS", "L3-06", "L3-07"}
    candidates = [f for f in findings if f["defectType"] in semantic_types]
    size_changed = raw is not None and clean_raw is not None and raw.stat().st_size != clean_raw.stat().st_size
    if not candidates and not size_changed:
        return findings
    same_bytes = False
    if raw is not None and clean_raw is not None and not size_changed:
        for path in (raw, clean_raw):
            if path not in hashes:
                hashes[path] = file_hash(path)
        same_bytes = hashes[raw] == hashes[clean_raw]
    if same_bytes:
        for f in candidates:
            f["_observation"]["imageIdentity"] = {"method":"sha256_exact", "referenceImage":str(clean_raw), "sha256":hashes[raw]}
        return findings
    retained = [f for f in findings if f["defectType"] not in semantic_types]
    retained.append(make_finding(
        dataset, "C10", "REFERENCE_IMAGE_ALIGNMENT_REQUIRED", STATUS_PENDING, str(label_path),
        "当前图像与 clean 内容不同或参照数据缺失；几何、覆盖和数量差异需先完成图像对齐与目标可见性核验",
        locator_type="imageRegion", observation={
            "actualImage":str(raw) if raw else None, "referenceImage":str(clean_raw) if clean_raw else None,
            "imageSizeBytesDifferent":size_changed,
            "unconfirmedDifferences":[{"defectType":f["defectType"],"locator":f["_locator"],"severity":f["severity"],"observation":f["_observation"]} for f in candidates],
            "referenceLabel":str(reference.label_files.get(label_path.name,reference.root)),
        },
    ))
    return retained


def compare_to_reference(dataset: Dataset, reference: Dataset) -> tuple[list[dict[str, Any]], int]:
    failures = []
    hashes: dict[Path, str] = {}
    ref_samples = {sample_key(reference.label_files[name], value): (name, value) for name, value in reference.labels.items()}
    actual_samples = {sample_key(dataset.label_files[name], value): (name, value) for name, value in dataset.labels.items()}
    shared = sorted(set(ref_samples) & set(actual_samples))
    if not shared:
        return [], 0
    for key in sorted(set(ref_samples) - set(actual_samples)):
        ref_name, _ = ref_samples[key]
        failures.append(make_finding(dataset, "C10", "L3-06", STATUS_FAIL, str(dataset.root / "label"), "同源 clean 参照中的样本标注在当前版本缺失", 1, affected_count=1, affected_ratio=1 / max(1, len(ref_samples)), observation={"sampleKey": key, "referenceLabel": ref_name}))
    for key in sorted(set(actual_samples) - set(ref_samples)):
        actual_name, _ = actual_samples[key]
        failures.append(make_finding(dataset, "C10", "L3-07", STATUS_FAIL, str(dataset.label_files[actual_name]), "当前版本出现同源 clean 参照中不存在的样本标注", 1, affected_count=1, affected_ratio=1 / max(1, len(actual_samples)), observation={"sampleKey": key, "actualLabel": actual_name}))
    for key in shared:
        sample_start = len(failures)
        ref_name, ref_value = ref_samples[key]
        actual_name, actual_value = actual_samples[key]
        if family_name(dataset.dataset_id) == "ULF_sample":
            # File identity is stable; the observation-window times themselves
            # may be polluted. Never identify samples by the reported window.
            ref_basic = ref_value.get("basic_annotation") or {}
            actual_basic = actual_value.get("basic_annotation") or {}
            for time_key in ("startTime", "endTime"):
                a, b = parse_time(ref_basic.get(time_key)), parse_time(actual_basic.get(time_key))
                if a and b and a != b:
                    failures.append(make_finding(
                        dataset, "C10", "L2-T", STATUS_FAIL,
                        f"{dataset.label_files[actual_name]}#$.basic_annotation.{time_key}",
                        "观测窗口时刻与可信同源 clean 参照不同", abs((b-a).total_seconds()),
                        "absoluteTimeOffset", "second", 0, locator_type="timeInterval",
                        observation={"reference": ref_basic[time_key], "actual": actual_basic[time_key], "referenceLabel": str(reference.label_files[ref_name])},
                    ))
            ref_science = ref_value.get("science_annotation") or {}
            actual_science = actual_value.get("science_annotation") or {}
            # Unit semantics require a trustworthy reference; columns alone
            # cannot establish physical units.
            if ref_science.get("units") != actual_science.get("units"):
                failures.append(make_finding(
                    dataset, "C10", "L2-M", STATUS_FAIL,
                    f"{dataset.label_files[actual_name]}#$.science_annotation.units",
                    "单位声明与可信同源 clean 参照不同", 1, locator_type="annotationRecord",
                    observation={"reference": ref_science.get("units"), "actual": actual_science.get("units"), "referenceLabel": str(reference.label_files[ref_name])},
                ))
        ref_annotation_key, ref_annotation = event_annotation(ref_value)
        actual_annotation_key, actual_annotation = event_annotation(actual_value)
        ref_records_key, _ = event_records(ref_annotation)
        actual_records_key, _ = event_records(actual_annotation)
        ref_events = events_by_key(ref_value)
        actual_events = events_by_key(actual_value)
        records_path = f"$.{actual_annotation_key or ref_annotation_key or 'event_annotation'}.{actual_records_key or ref_records_key or 'events'}"
        for event_id in sorted(set(ref_events) - set(actual_events)):
            failures.append(make_finding(dataset, "C10", "L3-06", STATUS_FAIL, f"{dataset.label_files[actual_name]}#{records_path}", "同源 clean 参照中的事件标注在当前版本缺失", 1, affected_count=1, affected_ratio=1 / max(1, len(ref_events)), locator_type="annotationRecord", observation={"sampleKey": key, "eventKey": event_id, "referenceLabel": ref_name}))
        for event_id in sorted(set(actual_events) - set(ref_events)):
            failures.append(make_finding(dataset, "C10", "L3-07", STATUS_FAIL, f"{dataset.label_files[actual_name]}#{records_path}", "当前版本出现同源 clean 参照中不存在的事件标注", 1, affected_count=1, affected_ratio=1 / max(1, len(actual_events)), locator_type="annotationRecord", observation={"sampleKey": key, "eventKey": event_id}))
        ref = label_reference(actual_value)
        image_meta = dataset.png_meta.get(ref or "")
        for event_id in sorted(set(ref_events) & set(actual_events)):
            event_index = list(actual_events).index(event_id)
            failures.extend(compare_event(dataset, dataset.label_files[actual_name], f"{records_path}[{event_index}]", ref_events[event_id], actual_events[event_id], image_meta))
        ref_shapes = science_shapes(ref_value)
        actual_shapes = science_shapes(actual_value)
        shape_list_path = "$.science_annotation"
        shape_matches, missing_shape_indexes, extra_shape_indexes = match_science_shapes(ref_shapes, actual_shapes)
        for ref_index, actual_index, match_method, match_cost in shape_matches:
            ref_path, ref_shape = ref_shapes[ref_index]
            actual_path, actual_shape = actual_shapes[actual_index]
            compared = compare_shape(dataset, dataset.label_files[actual_name], actual_path, ref_shape, actual_shape, image_meta)
            for finding in compared:
                finding["_observation"]["shapeMatching"] = {
                    "method": match_method,
                    "cost": match_cost,
                    "referencePath": ref_path,
                    "actualPath": actual_path,
                }
            failures.extend(compared)
        for index in missing_shape_indexes:
            failures.append(make_finding(
                dataset,
                "C10",
                "L3-06",
                STATUS_FAIL,
                f"{dataset.label_files[actual_name]}#{shape_list_path}",
                "同源 clean 参照中的形状标注在当前版本缺失",
                1,
                affected_count=1,
                affected_ratio=1 / max(1, len(ref_shapes)),
                locator_type="annotationRecord",
                observation={"sampleKey": key, "referenceShapePath": ref_shapes[index][0]},
            ))
        for index in extra_shape_indexes:
            failures.append(make_finding(
                dataset,
                "C10",
                "L3-07",
                STATUS_FAIL,
                f"{dataset.label_files[actual_name]}#{actual_shapes[index][0]}",
                "当前版本出现同源 clean 参照中不存在的形状标注",
                1,
                affected_count=1,
                affected_ratio=1 / max(1, len(actual_shapes)),
                locator_type="annotationRecord",
                observation={"sampleKey": key, "actualShapePath": actual_shapes[index][0]},
            ))
        failures[sample_start:] = guard_reference_geometry(
            dataset, reference, actual_value, ref_value, dataset.label_files[actual_name],
            failures[sample_start:], hashes,
        )
    return failures, len(shared)


def choose_reference(group: list[Dataset]) -> Dataset | None:
    declared_clean = [dataset for dataset in group if dataset.tier and dataset.tier.lower() in {"clean", "p0"}]
    if len(declared_clean) == 1:
        return declared_clean[0]
    # Few structural violations do not prove correct semantic annotations.
    # An ambiguous or absent clean reference must not generate confirmed truth.
    return None


def reconcile_ulf_metadata_attribution(dataset: Dataset, reference: Dataset) -> None:
    """Keep a raw-data contradiction without inventing label contamination."""
    if family_name(dataset.dataset_id) != "ULF_sample":
        return
    by_key = {sample_key(reference.label_files[n], v): v for n,v in reference.labels.items()}
    reason_fields = {
        "n_points 与 CSV 数据行数不一致": "n_points",
        "components 与 CSV 数值列不一致": "components",
        "sampling_rate_hz 与 CSV 时间轴典型采样间隔不一致": "sampling_rate_hz",
        "duration_s 与 CSV 时间轴包含末采样间隔的覆盖时长不一致": "duration_s",
    }
    for finding in dataset.findings:
        if finding["checkId"] != "C06" or finding["defectType"] != "L2-M" or finding["status"] != STATUS_FAIL:
            continue
        label_name = Path(finding["_locator"].split("#",1)[0]).name
        actual = dataset.labels.get(label_name)
        ref = by_key.get(sample_key(dataset.label_files[label_name], actual)) if actual else None
        if ref is None:
            continue
        a,b = ref.get("science_annotation") or {}, actual.get("science_annotation") or {}
        obs = finding["_observation"]
        reasons = obs.get("violations", [])
        errors = obs.get("relativeErrors", [])
        confirmed = [(r,e) for r,e in zip(reasons,errors) if r in reason_fields and a.get(reason_fields[r]) != b.get(reason_fields[r])]
        unresolved = [r for r in reasons if r not in {r for r,e in confirmed}]
        if unresolved:
            obs["unattributedDataContradictions"] = unresolved
            obs["referenceMetadata"] = {k:a.get(k) for k in reason_fields.values()}
        if confirmed:
            finding["basis"] = "；".join(r for r,e in confirmed)
            finding["severity"]["value"] = max(e for r,e in confirmed)
            obs["violations"] = [r for r,e in confirmed]
            obs["relativeErrors"] = [e for r,e in confirmed]
        else:
            finding["status"] = STATUS_PENDING
            finding["defectType"] = "DATA_METADATA_ATTRIBUTION_UNRESOLVED"
            finding["confirmedOrPending"] = "待复核事项"
            finding["basis"] = "CSV 与参数存在矛盾，但参数与可信 clean 完全一致；保留数据体异常，不能归因为标注参数污染"
            finding["_observation"]["conflictingFieldsUnchangedFromTrustedClean"] = True
            finding["_observation"]["candidateDefectTypes"] = ["L1-TS", "L2-M"]


def apply_reference_checks(datasets: list[Dataset], run_id: str, enabled_checks: set[str]) -> None:
    if "C10" not in enabled_checks:
        return
    groups: dict[str, list[Dataset]] = collections.defaultdict(list)
    for dataset in datasets:
        groups[family_name(dataset.dataset_id)].append(dataset)
    for group in groups.values():
        reference = choose_reference(group)
        for dataset in group:
            started, started_cpu = start_check(dataset, run_id, "C10")
            try:
                if reference is None or len(group) < 2:
                    for defect_type in ("L2-G", "L2-P", "L3-06", "L3-07", "L3-MIS"):
                        dataset.findings.append(make_finding(dataset, "C10", defect_type, STATUS_PENDING, str(dataset.root), "缺少唯一可信同源 clean 参照或可比较的质量版本；无法对需要语义真值的缺陷作确定性重推", observation={"availableVersions": len(group), "trustedReferenceAvailable": reference is not None}))
                elif dataset is reference:
                    dataset.findings.append(make_finding(dataset, "C10", "REFERENCE_BASELINE", STATUS_PASS, str(dataset.root), "该版本由明确的 clean/P0 声明指定为同源参照", observation={"referenceDataset": reference.dataset_id}))
                else:
                    reconcile_ulf_metadata_attribution(dataset, reference)
                    failures, shared_sample_count = compare_to_reference(dataset, reference)
                    if shared_sample_count == 0:
                        dataset.findings.append(make_finding(
                            dataset, "C10", "REFERENCE_ALIGNMENT_UNAVAILABLE", STATUS_PENDING,
                            str(dataset.root), "当前版本与同源参照没有可稳定对齐的样本，不能将零差异解释为通过",
                            observation={"referenceDataset": reference.dataset_id, "sharedSampleCount": 0},
                        ))
                    elif failures:
                        dataset.findings.extend(failures)
                    else:
                        dataset.findings.append(make_finding(dataset, "C10", "REFERENCE_DIFFERENCE", STATUS_PASS, str(dataset.root), "与同源参照的可比样本及事件字段未发现差异", observation={"referenceDataset": reference.dataset_id}))
                log_event(dataset, run_id, "C10", started, "实际完成", started_cpu=started_cpu)
            except Exception as exc:
                dataset.findings.append(make_finding(dataset, "C10", "CHECK_EXECUTION_FAILURE", STATUS_PENDING,
                    str(dataset.root), f"参照检查失败，须按未完成项保留: {exc}", observation={"error": repr(exc)}))
                log_event(dataset, run_id, "C10", started, "失败", repr(exc), started_cpu=started_cpu)


def apply_backend_checks(datasets: list[Dataset], run_id: str, enabled_checks: set[str], backend_profile: str) -> None:
    """Run family adapters and merge gated signals into formal findings.

    The adapter is placed after reference checks and before IDs and scorecards,
    so a high-evidence signal follows the same evidence, audit, scoring and
    ranking path as every deterministic rule.  ``legacy`` is a true rollback.
    """
    if backend_profile not in {"legacy", "enhanced"}:
        raise ValueError("Unknown backend profile")
    for dataset in datasets:
        dataset.backend_profile = backend_profile
        if backend_profile == "legacy":
            continue
        family = family_name(dataset.dataset_id)
        if family not in backend_strategy.FAMILIES:
            continue
        existing = {(f["checkId"], f["defectType"], f["_locator"]) for f in dataset.findings}
        for label_name, label in sorted(dataset.labels.items()):
            data_name = label_reference(label)
            data_path = dataset.data_files.get(Path(data_name).name) if data_name else None
            csv_stats = None
            if family == "ULF_sample" and data_path is not None:
                try:
                    # Reuse the verified full-axis scanner; never infer a time
                    # axis from a preview or from row count alone.
                    from . import priors
                    csv_stats = priors.csv_statistics(data_path)
                except (OSError, UnicodeError, ValueError, csv.Error, OverflowError):
                    csv_stats = None
            try:
                observation = backend_strategy.build_observation(
                    family, data_path, label, csv_stats=csv_stats, profile=backend_profile
                )
            except (OSError, TypeError, ValueError, UnicodeError) as error:
                observation = {
                    "version": backend_strategy.VERSION, "profile": backend_profile,
                    "family": family, "strategy": "fallback", "scoreEffect": "none",
                    "fallback": "strategy_observation_failed:" + type(error).__name__,
                }
            dataset.backend_observations.append({"label": label_name, "data": data_name, "observation": observation})
            signals = backend_strategy.formal_signals(
                family, observation, label_path=str(dataset.label_files.get(label_name, dataset.root)),
                data_path=str(data_path) if data_path else None,
            )
            for signal in signals:
                if signal.get("checkId") not in enabled_checks:
                    continue
                key = (signal["checkId"], signal["defectType"], signal["locator"])
                if key in existing:
                    continue
                ratio = (signal.get("affectedCount") or 1) / max(1, len(dataset.labels))
                dataset.findings.append(make_finding(
                    dataset, signal["checkId"], signal["defectType"], signal["status"], signal["locator"],
                    signal["basis"], signal.get("severityValue", 0.0), signal.get("severityMetric", "count"),
                    signal.get("severityUnit", "item"), signal.get("tolerance", 0.0),
                    signal.get("affectedCount"), ratio, signal.get("criticality", "一般"),
                    observation=signal.get("observation"), locator_type=signal.get("locatorType", "file"),
                ))
                existing.add(key)
                dataset.backend_formal_signal_count += 1


def finalize_ids(dataset: Dataset) -> None:
    order = sorted(range(len(dataset.findings)), key=lambda index: (dataset.findings[index]["checkId"], dataset.findings[index]["status"], dataset.findings[index]["defectType"], dataset.findings[index]["_locator"], index))
    counters = collections.Counter()
    for index in order:
        finding = dataset.findings[index]
        counters[finding["checkId"]] += 1
        identity = f"{finding['checkId']}-{stable_hash(dataset.dataset_id, 6).upper()}-{counters[finding['checkId']]:06d}"
        finding["findingId"] = identity
        finding["evidenceRef"] = f"EV-{identity}"


def score_dataset(dataset: Dataset, enabled_checks: set[str], run_id: str) -> dict[str, Any]:
    dimensions = []
    uncomputable = []
    for dim_id, label in DIMENSION_LABELS.items():
        relevant = {check for check, dims in CHECK_DIMENSIONS.items() if dim_id in dims and check in enabled_checks}
        findings = [finding for finding in dataset.findings if finding["checkId"] in relevant]
        failures = [finding for finding in findings if finding["status"] == STATUS_FAIL]
        completed_checks = {finding["checkId"] for finding in findings if finding["status"] in {STATUS_PASS, STATUS_FAIL, STATUS_PENDING}}
        determined_checks = {finding["checkId"] for finding in findings if finding["status"] in {STATUS_PASS, STATUS_FAIL}}
        if not completed_checks or not determined_checks:
            dimensions.append({"dimId": dim_id, "name": label, "score": None, "range": None, "grade": None, "meaning": "检查已执行，但证据不足，不能给出该维度分数"})
            uncomputable.append(dim_id)
            continue
        penalty = 0.0
        for finding in failures:
            ratio = finding.get("affectedScope", {}).get("ratio")
            ratio = ratio if isinstance(ratio, (int, float)) else 1 / max(1, len(dataset.data_files) + len(dataset.labels))
            multiplier = SCORE_CRITICAL_MULTIPLIER if finding.get("criticality") == "关键" else 1.0
            penalty += min(SCORE_FINDING_CAP, 100.0 * ratio * multiplier)
        score = round(max(0.0, 100.0 - penalty), 4)
        pending_count = sum(1 for finding in findings if finding["status"] == STATUS_PENDING)
        lower = round(score * len(completed_checks) / max(1, len(relevant)), 4)
        grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
        dimensions.append({"dimId": dim_id, "name": label, "score": score, "range": [lower, score] if pending_count else [score, score], "grade": grade, "meaning": f"{len(failures)} 条确认缺陷，{pending_count} 条待判定"})

    computable_scores = [item["score"] for item in dimensions if isinstance(item["score"], (int, float))]
    total = sum(computable_scores) / len(computable_scores) if computable_scores else None
    critical = [finding for finding in dataset.findings if finding["status"] == STATUS_FAIL and finding.get("criticality") == "关键"]
    if total is not None and critical:
        default_denominator = max(1, len(dataset.data_files) + len(dataset.labels))
        critical_ratio = sum(
            float(finding.get("affectedScope", {}).get("ratio"))
            if isinstance(finding.get("affectedScope", {}).get("ratio"), (int, float))
            else 1 / default_denominator
            for finding in critical
        )
        critical_ceiling = max(0.0, SCORE_CRITICAL_CAP - SCORE_CRITICAL_RATIO_SLOPE * min(1.0, critical_ratio))
        total = min(total, critical_ceiling)
    statuses = collections.Counter(finding["status"] for finding in dataset.findings)
    applicable = len({finding["checkId"] for finding in dataset.findings if finding["status"] != STATUS_NA})
    completed = len({finding["checkId"] for finding in dataset.findings if finding["status"] in {STATUS_PASS, STATUS_FAIL, STATUS_PENDING}})
    top_defects = [finding["findingId"] for finding in dataset.findings if finding["status"] == STATUS_FAIL][:20]
    return {
        "datasetId": dataset.dataset_id,
        "datasetVersion": dataset.dataset_version,
        "declaredUsage": "验收/入库/选数",
        "ruleVersion": RULE_VERSION,
        "specVersion": SPEC_VERSION,
        "runId": run_id,
        "scoringInputs": {
            "dataFileCount": len(dataset.data_files),
            "labelFileCount": len(dataset.labels),
            "defaultAffectedDenominator": max(1, len(dataset.data_files) + len(dataset.labels)),
            "criticalAffectedRatio": round(critical_ratio, 8) if critical else 0.0,
        },
        "totalScore": round(total, 4) if total is not None else None,
        "dimensions": dimensions,
        "coverage": {"completed": completed, "applicable": applicable, "ratio": round(completed / applicable, 6) if applicable else None},
        "checkStatusSummary": dict(statuses),
        "topDefects": top_defects,
        "criticalDefects": [finding["findingId"] for finding in critical],
        "uncomputableDimensions": uncomputable,
        "evidenceRefs": [finding["evidenceRef"] for finding in dataset.findings],
        **({"backend": {
            "profile": dataset.backend_profile,
            "strategyVersion": backend_strategy.VERSION,
            "observationCount": len(dataset.backend_observations),
            "formalSignalCount": dataset.backend_formal_signal_count,
            "meaning": "增强模式的高证据后端信号已进入正式证据、评分和排序链路；legacy 为迁移前规则链路",
        }} if dataset.backend_profile == "enhanced" else {}),
    }


def write_outputs(output: Path, datasets: list[Dataset], scorecards: list[dict[str, Any]], run_id: str, seed: int,
                  enabled_checks: set[str], input_root: Path | None = None, created_at: str | None = None) -> None:
    findings_dir = output / "06_问题清单"
    evidence_dir = output / "07_证据索引" / "evidence"
    score_dir = output / "04_评分卡"
    spec_dir = output / "03_评价规范"
    ranking_dir = output / "05_排序表"
    logs_dir = output / "08_运行日志"
    for path in (findings_dir, evidence_dir, score_dir, spec_dir, ranking_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    source_spec = asset("spec", "quality_spec.yaml")
    (spec_dir / "quality_spec.yaml").write_bytes(source_spec.read_bytes())

    public_findings = []
    evidence_rows = []
    for dataset in datasets:
        for finding in dataset.findings:
            evidence_path = evidence_dir / f"{finding['evidenceRef']}.json"
            payload = {
                "evidenceId": finding["evidenceRef"],
                "findingId": finding["findingId"],
                "datasetId": finding["datasetId"],
                "checkId": finding["checkId"],
                "ruleVersion": finding["ruleVersion"],
                "locatorType": finding["_locatorType"],
                "locator": finding["_locator"],
                "observation": finding["_observation"],
            }
            json_dump(evidence_path, payload)
            evidence_rows.append({
                "evidenceId": finding["evidenceRef"],
                "findingId": finding["findingId"],
                "locatorType": finding["_locatorType"],
                "locator": finding["_locator"],
                "contentHash": file_hash(evidence_path),
                "capturedBy": run_id,
            })
            public_findings.append({key: value for key, value in finding.items() if not key.startswith("_")})

    jsonl_dump(findings_dir / "findings.jsonl", public_findings)
    with (output / "07_证据索引" / "evidence_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("evidenceId", "findingId", "locatorType", "locator", "contentHash", "capturedBy"))
        writer.writeheader()
        writer.writerows(evidence_rows)

    for card in scorecards:
        json_dump(score_dir / f"{card['datasetId']}_{card['datasetVersion']}.json", card)

    ranked = sorted(scorecards, key=lambda card: (-(card["totalScore"] if card["totalScore"] is not None else -1), len(card["criticalDefects"]), card["datasetId"]))
    with (ranking_dir / "ranking_global.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ("rank", "datasetId", "datasetVersion", "totalScore", "dimensionScores", "groupId", "comparisonBasis", "retainSubRanking")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, card in enumerate(ranked, 1):
            writer.writerow({
                "rank": index,
                "datasetId": card["datasetId"],
                "datasetVersion": card["datasetVersion"],
                "totalScore": card["totalScore"],
                "dimensionScores": json.dumps({item["dimId"]: item["score"] for item in card["dimensions"]}, ensure_ascii=False, separators=(",", ":")),
                "groupId": family_name(card["datasetId"]),
                "comparisonBasis": "关键缺陷封顶后按可计算维度均值降序；同分按关键缺陷数、datasetId 稳定打破并列",
                "retainSubRanking": "证据不足维度保留分项区间，不以 0 或满分填补",
            })

    all_events = [event for dataset in datasets for event in dataset.run_events]
    for event in all_events:
        if event.get("seed") is None:
            event["seed"] = seed
        event.setdefault("ruleVersion", RULE_VERSION)
        event.setdefault("systemVersion", VERSION)
    all_events.sort(key=lambda event: event["ts"])
    jsonl_dump(logs_dir / "run.log.jsonl", all_events)
    backend_rows = []
    for dataset in datasets:
        for item in dataset.backend_observations:
            backend_rows.append({
                "datasetId": dataset.dataset_id,
                "datasetVersion": dataset.dataset_version,
                "profile": dataset.backend_profile,
                "label": item.get("label"),
                "data": item.get("data"),
                "observation": item.get("observation"),
            })
    jsonl_dump(logs_dir / "backend_observations.jsonl", backend_rows)
    (logs_dir / "system_env.txt").write_text(
        f"opc_agent_version={VERSION}\npython={platform.python_version()}\nplatform={platform.platform()}\nrule_version={RULE_VERSION}\nnetwork_used=false\nmodel_token_usage=0\n",
        encoding="utf-8",
    )
    json_dump(logs_dir / "seeds.yaml", {"globalSeed": seed, "samplingSeed": seed})

    pending = [finding for finding in public_findings if finding["status"] == STATUS_PENDING]
    (output / "09_未完成事项说明.md").write_text(
        "# 未完成事项说明\n\n"
        f"- 待判定记录：{len(pending)} 条。\n"
        "- 单包模式无法确定性重推类别错标、真实边界偏移、漏标、虚标与覆盖偏差；需同源 clean 版本或人工锚点。\n"
        "- 文档与来源追溯、使用条件须由输入包补充机器可读说明后再计入综合分。\n"
        "- 当前实现不调用模型服务，token 消耗为 0；模型插件只有通过 golden fixture 后才可启用。\n",
        encoding="utf-8",
    )
    manifest = {
        "agentVersion": VERSION,
        "ruleVersion": RULE_VERSION,
        "specVersion": SPEC_VERSION,
        "runId": run_id,
        "createdAt": created_at or utc_now(),
        "generatedAt": utc_now(),
        "datasets": [dataset.dataset_id for dataset in datasets],
        "enabledChecks": sorted(enabled_checks),
        "backendProfile": datasets[0].backend_profile if datasets and len({d.backend_profile for d in datasets}) == 1 else "mixed",
        "seed": seed,
        "loggingVersion": 2 if all(dataset.logging_version == 2 for dataset in datasets) else 1,
        "executionMode": "rules-only" if all(dataset.logging_version == 2 for dataset in datasets) else "external-integration",
        "datasetRoots": [{"datasetId": dataset.dataset_id, "root": str(dataset.root.resolve())} for dataset in datasets],
        "inputRoot": str(input_root.resolve()) if input_root is not None else None,
        "artifacts": [
            "03_评价规范/quality_spec.yaml", "04_评分卡/", "05_排序表/ranking_global.csv", "06_问题清单/findings.jsonl",
            "07_证据索引/evidence_index.csv", "08_运行日志/run.log.jsonl", "08_运行日志/backend_observations.jsonl", "09_未完成事项说明.md",
        ],
    }
    if manifest["loggingVersion"] == 2:
        manifest["artifacts"].append("08_运行日志/runtime_metrics.json")
    json_dump(output / "00_MANIFEST.yaml", manifest)


def run(input_path: Path, output: Path, seed: int, enabled_checks: set[str], backend_profile: str = "legacy") -> dict[str, Any]:
    input_path = input_path.resolve()
    output = output.resolve()
    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_path}")
    if output.exists() or output.is_relative_to(input_path) or input_path.is_relative_to(output):
        raise ValueError("Use a new output directory outside the input and its ancestors")
    if not enabled_checks or enabled_checks - set(ALL_CHECKS):
        raise ValueError("Choose one or more known check IDs")
    if backend_profile not in {"legacy", "enhanced"}:
        raise ValueError("Unknown backend profile")
    if type(seed) is not int:
        raise ValueError("The random seed must be an integer")
    started_all = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = utc_now()
    roots = discover_datasets(input_path)
    if not roots:
        raise SystemExit(f"No dataset directory containing data/ and label/ found under {input_path}")
    tier_lookup = load_tiers(input_path if input_path.is_dir() else input_path.parent)
    # Keep run IDs stable across the reversible backend profiles.  The profile
    # is recorded in the manifest, scorecard and runtime log; output directories
    # are independently allocated, so sharing the historical identity is safe.
    run_signature = "|".join((str(input_path.resolve()), str(seed), RULE_VERSION, ",".join(sorted(enabled_checks))))
    run_id = f"RUN-{stable_hash(run_signature).upper()}"
    logs_dir = output / "08_运行日志"
    logs_dir.mkdir(parents=True, exist_ok=False)
    live_log = logs_dir / "run.log.jsonl"
    live_log.touch(exist_ok=False)
    datasets = [inventory_dataset(root, tier_lookup.get(root.name)) for root in roots]
    for dataset in datasets:
        dataset.seed = seed
        dataset.event_log_path = live_log
        dataset.logging_version = 2
        dataset.backend_profile = backend_profile
    for dataset in datasets:
        base_analysis(dataset, run_id, enabled_checks)
    apply_reference_checks(datasets, run_id, enabled_checks)
    apply_backend_checks(datasets, run_id, enabled_checks, backend_profile)
    for dataset in datasets:
        finalize_ids(dataset)
    scorecards = [score_dataset(dataset, enabled_checks, run_id) for dataset in datasets]
    write_outputs(output.resolve(), datasets, scorecards, run_id, seed, enabled_checks,
                  input_root=input_path, created_at=started_utc)
    usage = process_usage()
    usage["cpuDeltaSeconds"] = round(max(0.0, usage["cpuSeconds"] - started_cpu), 6)
    json_dump(logs_dir / "runtime_metrics.json", {
        "schemaVersion": 1, "runId": run_id, "executionMode": "rules-only",
        "backendProfile": backend_profile,
        "systemVersion": VERSION, "ruleVersion": RULE_VERSION, "seed": seed,
        "elapsedSeconds": round(time.perf_counter() - started_all, 6),
        "processUsage": usage, "memoryScope": "process_lifetime_high_water",
        "modelForwardPasses": 0, "externalApiCalls": 0, "externalModelTokens": 0,
        "acceleratorPeakBytes": None,
        "note": "CPU process measurements; no model inference or accelerator sampler. Audit is a separate stage.",
    })
    return {
        "runId": run_id,
        "datasetCount": len(datasets),
        "findingCount": sum(len(dataset.findings) for dataset in datasets),
        "confirmedDefectCount": sum(1 for dataset in datasets for finding in dataset.findings if finding["status"] == STATUS_FAIL),
        "pendingCount": sum(1 for dataset in datasets for finding in dataset.findings if finding["status"] == STATUS_PENDING),
        "output": str(output.resolve()),
        "backendProfile": backend_profile,
    }


def main() -> None:
    # Retained for callers of the original module; command parsing is separate.
    from .cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
