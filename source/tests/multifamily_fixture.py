"""Controlled 30-package fixture covering every public OPC dataset family.

This fixture mirrors the public tier structure, but it is not a reconstruction
of the hidden competition data.  Every injected change is deterministic so the
benchmark can distinguish implementation regressions from dataset variation.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

from fixture_factory import write_json, write_png


FAMILIES = (
    "AIA_AR_sample",
    "HMI_AR_sample",
    "CME_C2_sample",
    "EPB_seg_sample",
    "DMSP_SSUSI",
    "ULF_sample",
)
TIERS = ("clean", "P1", "P2", "P3", "P4")
TIER_SUFFIX = {
    "clean": "0000000000",
    "P1": "1111111111",
    "P2": "2222222222",
    "P3": "3333333333",
    "P4": "4444444444",
}
COVERAGE_COUNT = {"P1": 1, "P2": 4, "P3": 8, "P4": 12}
COVERAGE_PERCENT = {"clean": 0, "P1": 5, "P2": 20, "P3": 40, "P4": 60}
INTENSITY = {"P1": 1.0, "P2": 2.0, "P3": 3.0, "P4": 4.5}
CANONICAL_MISSING = "L0-MISSING"


EXPECTED_BY_FAMILY = {
    "AIA_AR_sample": {CANONICAL_MISSING, "L2-G", "L2-P"},
    "HMI_AR_sample": {CANONICAL_MISSING, "L2-G", "L2-P"},
    "CME_C2_sample": {CANONICAL_MISSING, "L2-P", "L2-T"},
    "EPB_seg_sample": {CANONICAL_MISSING, "L2-G", "L2-F", "L2-P", "L3-06", "L3-07"},
    "DMSP_SSUSI": {CANONICAL_MISSING, "L2-G", "L2-F", "L2-P", "L3-06", "L3-07"},
    "ULF_sample": {CANONICAL_MISSING, "L2-M", "L2-P", "L2-T"},
}


def _timestamp(index: int, minute_offset: float = 0.0) -> str:
    base = dt.datetime(2026, 1, 1, 0, 0, 0) + dt.timedelta(hours=index, minutes=minute_offset)
    return base.isoformat(timespec="seconds")


def _polygon(shift: float = 0.0, scale: float = 1.0) -> list[list[float]]:
    left = 4.0 + shift
    top = 4.0 + shift
    side = 8.0 * scale
    return [
        [left, top],
        [left + side, top],
        [left + side, top + side],
        [left, top + side],
        [left, top],
    ]


def _event(event_id: str, category: str, index: int, polygon: list[list[float]] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "SOL_standard": event_id,
        "event_type": category,
        "event_starttime": _timestamp(index, 10),
        "event_endtime": _timestamp(index, 20),
    }
    if polygon is not None:
        value["bbox_polygon_pixel"] = polygon
    return value


def _refresh_event_summary(value: dict[str, Any]) -> None:
    for key in ("hek_event_annotation", "ulf_event_annotation"):
        annotation = value.get(key)
        if not isinstance(annotation, dict):
            continue
        events = annotation.get("matched_events", [])
        annotation["has_events"] = bool(events)
        annotation["event_count"] = len(events)
        annotation["event_types"] = sorted(
            {
                str(event.get("kind") or event.get("event_type"))
                for event in events
                if event.get("kind") or event.get("event_type")
            }
        )


def _image_label(family: str, data_name: str, index: int) -> dict[str, Any]:
    stem = Path(data_name).stem
    label_header_name = data_name
    if family in {"EPB_seg_sample", "DMSP_SSUSI"}:
        # The organizer's visualization examples use the JSON filename here,
        # while basic_annotation.fileName points at the PNG.
        label_header_name = f"{stem}.json"
    value: dict[str, Any] = {
        "label_header": {"fileName": label_header_name, "annotator": "controlled-fixture"},
        "basic_annotation": {
            "fileName": data_name,
            "relativePath": "data/",
            "startTime": _timestamp(index),
            "endTime": _timestamp(index, 59),
            "width": 32,
            "height": 32,
        },
    }
    if family in {"AIA_AR_sample", "HMI_AR_sample"}:
        events = [
            _event(f"{family}-{index}-0", "AR", index, _polygon()),
            _event(f"{family}-{index}-1", "AR", index, _polygon(2.0, 0.75)),
        ]
        value["science_annotation"] = {"dateObs": _timestamp(index)}
        value["hek_event_annotation"] = {
            "has_events": True,
            "event_count": len(events),
            "event_types": ["AR"],
            "matched_events": events,
        }
    elif family == "CME_C2_sample":
        events = [_event(f"{family}-{index}-0", "CME", index)]
        value["science_annotation"] = {"dateObs": _timestamp(index)}
        value["hek_event_annotation"] = {
            "has_events": True,
            "event_count": 1,
            "event_types": ["CME"],
            "matched_events": events,
        }
    elif family == "EPB_seg_sample":
        value["science_annotation"] = {
            "mask": {
                "imageWidth": 32,
                "imageHeight": 32,
                "shapes": [
                    {"label": "EPB", "shape_type": "polygon", "points": _polygon()[:-1]},
                    {"label": "EPB", "shape_type": "polygon", "points": _polygon(2.0, 0.75)[:-1]},
                ],
            }
        }
    elif family == "DMSP_SSUSI":
        value["science_annotation"] = {
            "specialShapes": [
                {"label": "PCA", "shape_type": "rectangle", "points": [[4.0, 4.0], [12.0, 12.0]]},
                {"label": "TPA", "shape_type": "rectangle", "points": [[14.0, 5.0], [22.0, 13.0]]},
            ],
            "imageWidth": 32,
            "imageHeight": 32,
        }
    else:
        raise ValueError(f"unsupported image family: {family}")
    return value


def _ulf_label(data_name: str, index: int) -> dict[str, Any]:
    event = {
        "start_dt": _timestamp(index, 10),
        "end_dt": _timestamp(index, 20),
        "kind": "Pi2",
        "source": f"events.jsonl:{index + 1}",
    }
    return {
        "label_header": {"fileName": data_name, "annotator": "controlled-fixture"},
        "basic_annotation": {
            "fileName": data_name,
            "relativePath": "data/",
            "startTime": _timestamp(index),
            "endTime": _timestamp(index, 59),
        },
        "science_annotation": {"sampling_rate_hz": 1, "duration_s": 4, "n_points": 4},
        "ulf_event_annotation": {
            "has_events": True,
            "event_count": 1,
            "event_types": ["Pi2"],
            "matched_events": [event],
        },
    }


def _write_clean_family(publish: Path, family: str, sample_count: int) -> Path:
    root = publish / f"{family}_{TIER_SUFFIX['clean']}"
    for index in range(sample_count):
        if family == "ULF_sample":
            data_name = f"sample_{index:03d}.csv"
            data_path = root / "data" / data_name
            data_path.parent.mkdir(parents=True, exist_ok=True)
            offset = index * 10
            rows = ["t,H,D,Z,T"]
            rows.extend(
                f"{step},{offset + step + 1},{offset + step + 2},{offset + step + 3},{offset + step + 4}"
                for step in range(4)
            )
            data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            write_json(root / "label" / f"{data_name}.json", _ulf_label(data_name, index))
        else:
            data_name = f"sample_{index:03d}.png"
            write_png(root / "data" / data_name, width=32, height=32, value=20 + index)
            label_name = f"{data_name}.json" if family in {"AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample"} else f"sample_{index:03d}.json"
            write_json(root / "label" / label_name, _image_label(family, data_name, index))
    return root


def _shift_time(value: str, seconds: float) -> str:
    parsed = dt.datetime.fromisoformat(value)
    return (parsed + dt.timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _pollute_event_family(root: Path, family: str, tier: str, selected: range, extras: range, intensity: float) -> None:
    for index in selected:
        path = root / "label" / f"sample_{index:03d}.png.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        events = value["hek_event_annotation"]["matched_events"]
        events[0]["event_type"] = "WRONG"
        if family in {"AIA_AR_sample", "HMI_AR_sample"}:
            scale = 1.0 if tier == "P1" else 1.0 + 0.04 * intensity
            events[0]["bbox_polygon_pixel"] = _polygon(intensity, scale)
            if tier != "P1":
                events.pop(1)
        else:
            for key in ("event_starttime", "event_endtime"):
                events[0][key] = _shift_time(events[0][key], 60 * intensity)
        _refresh_event_summary(value)
        write_json(path, value)
    if family in {"AIA_AR_sample", "HMI_AR_sample"} and tier != "P1":
        for index in extras:
            path = root / "label" / f"sample_{index:03d}.png.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            value["hek_event_annotation"]["matched_events"].append(
                _event(f"{family}-{index}-EXTRA", "AR", index, _polygon(1.0, 0.5))
            )
            _refresh_event_summary(value)
            write_json(path, value)


def _pollute_shape_family(root: Path, family: str, selected: range, extras: range, intensity: float) -> None:
    for index in selected:
        path = root / "label" / f"sample_{index:03d}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        basic = value["basic_annotation"]
        basic["width"] = 32 + int(intensity)
        if family == "EPB_seg_sample":
            mask = value["science_annotation"]["mask"]
            mask["imageWidth"] = 32 + int(intensity)
            shapes = mask["shapes"]
            shapes[0]["label"] = "WRONG"
            shapes[0]["points"] = _polygon(intensity)[:-1]
        else:
            science = value["science_annotation"]
            science["imageWidth"] = 32 + int(intensity)
            shapes = science["specialShapes"]
            shapes[0]["label"] = "WRONG"
            shapes[0]["points"] = [[4.0 + intensity, 4.0 + intensity], [12.0 + intensity, 12.0 + intensity]]
        shapes.pop(1)
        write_json(path, value)
    for index in extras:
        path = root / "label" / f"sample_{index:03d}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if family == "EPB_seg_sample":
            shapes = value["science_annotation"]["mask"]["shapes"]
            shapes.append({"label": "EPB", "shape_type": "polygon", "points": _polygon(1.0, 0.5)[:-1]})
        else:
            shapes = value["science_annotation"]["specialShapes"]
            shapes.append({"label": "PCA", "shape_type": "rectangle", "points": [[2.0, 16.0], [8.0, 22.0]]})
        write_json(path, value)


def _pollute_ulf(root: Path, selected: range, intensity: float) -> None:
    for index in selected:
        path = root / "label" / f"sample_{index:03d}.csv.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["science_annotation"]["n_points"] = 4 + int(intensity)
        event = value["ulf_event_annotation"]["matched_events"][0]
        event["kind"] = "WRONG"
        for key in ("start_dt", "end_dt"):
            event[key] = _shift_time(event[key], 60 * intensity)
        _refresh_event_summary(value)
        write_json(path, value)


def _pollute_family(root: Path, family: str, tier: str, sample_count: int) -> None:
    count = COVERAGE_COUNT[tier]
    selected = range(0, count)
    extras = range(sample_count - count, sample_count)
    intensity = INTENSITY[tier]
    if family in {"AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample"}:
        _pollute_event_family(root, family, tier, selected, extras, intensity)
    elif family in {"EPB_seg_sample", "DMSP_SSUSI"}:
        _pollute_shape_family(root, family, selected, extras, intensity)
    elif family == "ULF_sample":
        _pollute_ulf(root, selected, intensity)
    else:
        raise ValueError(f"unsupported family: {family}")

    extension = ".csv" if family == "ULF_sample" else ".png"
    for index in extras:
        (root / "data" / f"sample_{index:03d}{extension}").unlink()


def expected_classes(family: str, tier: str) -> set[str]:
    if tier == "clean":
        return set()
    result = set(EXPECTED_BY_FAMILY[family])
    if family in {"AIA_AR_sample", "HMI_AR_sample"}:
        if tier == "P1":
            result -= {"L3-MIS", "L3-06", "L3-07"}
        else:
            result |= {"L3-MIS", "L3-06", "L3-07"}
    return result


def build_multifamily_fixture(root: Path, sample_count: int = 20) -> tuple[Path, dict[str, dict[str, Any]]]:
    if sample_count != 20:
        raise ValueError("sample_count must be 20 so 1/4/8/12 samples exactly represent 5/20/40/60%")
    publish = root / "publish"
    truth: dict[str, dict[str, Any]] = {}
    rows = []
    for family in FAMILIES:
        clean = _write_clean_family(publish, family, sample_count)
        for tier in TIERS:
            dataset = clean if tier == "clean" else publish / f"{family}_{TIER_SUFFIX[tier]}"
            if tier != "clean":
                shutil.copytree(clean, dataset)
                _pollute_family(dataset, family, tier, sample_count)
            expected = expected_classes(family, tier)
            truth[dataset.name] = {
                "family": family,
                "tier": tier,
                "coveragePercent": COVERAGE_PERCENT[tier],
                "expectedClasses": sorted(expected),
            }
            rows.append(
                {
                    "dataset_id": family,
                    "tier": tier,
                    "randomized_dir": dataset.name,
                    "defect_class_ids": " ".join(sorted(expected)),
                    "coverage_percent": COVERAGE_PERCENT[tier],
                }
            )
    publish.mkdir(parents=True, exist_ok=True)
    with (publish / "publish_tier_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset_id", "tier", "randomized_dir", "defect_class_ids", "coverage_percent"),
        )
        writer.writeheader()
        writer.writerows(rows)
    return publish, truth
