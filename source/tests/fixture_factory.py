import binascii
import csv
import json
import shutil
import struct
import zlib
from pathlib import Path


def write_png(path: Path, width: int = 8, height: int = 8, value: int = 127) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + bytes([value, value, value]) * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)

    content = b"\x89PNG\r\n\x1a\n"
    content += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    content += chunk(b"IDAT", zlib.compress(raw))
    content += chunk(b"IEND", b"")
    path.write_bytes(content)


def event(event_id: str, category: str = "AR", polygon=None, start: str = "2026-01-01 00:00:00"):
    return {
        "SOL_standard": event_id,
        "event_type": category,
        "event_starttime": start,
        "event_endtime": "2026-01-01 01:00:00",
        "bbox_polygon_pixel": polygon or [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]],
    }


def label(file_name: str, start: str, events, width: int = 8, height: int = 8):
    return {
        "label_header": {"fileName": file_name, "annotator": "fixture"},
        "basic_annotation": {
            "fileName": file_name,
            "relativePath": "data/",
            "startTime": start,
            "endTime": start,
            "width": width,
            "height": height,
        },
        "science_annotation": {"dateObs": start},
        "hek_event_annotation": {
            "has_events": bool(events),
            "event_types": sorted({item["event_type"] for item in events}),
            "matched_events": events,
            "event_count": len(events),
        },
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_fixture(root: Path):
    publish = root / "publish"
    clean = publish / "HMI_AR_sample_aaaaaaaaaa"
    corrupt = publish / "HMI_AR_sample_bbbbbbbbbb"
    for dataset in (clean, corrupt):
        (dataset / "data").mkdir(parents=True)
        (dataset / "label").mkdir(parents=True)

    write_png(clean / "data" / "a.png", value=80)
    write_png(clean / "data" / "b.png", value=140)
    write_json(clean / "label" / "a.png.json", label("a.png", "2026-01-01T00:00:00", [event("E1")]))
    write_json(clean / "label" / "b.png.json", label("b.png", "2026-01-02T00:00:00", [event("E2")]))

    shutil.copy2(clean / "data" / "a.png", corrupt / "data" / "a.png")
    shutil.copy2(clean / "data" / "a.png", corrupt / "data" / "duplicate.png")
    changed = label(
        "a.png",
        "2026-01-01T00:00:00",
        [
            event("E1", category="WRONG", polygon=[[2, 2], [7, 2], [7, 7], [2, 7], [2, 2]], start="2026-01-01 00:10:00"),
            event("EXTRA", polygon=[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]),
        ],
        width=9,
        height=8,
    )
    write_json(corrupt / "label" / "a.png.json", changed)
    write_json(corrupt / "label" / "b.png.json", label("b.png", "2026-01-02T00:00:00", []))
    write_json(corrupt / "label" / "orphan.png.json", label("orphan.png", "2026-01-03T00:00:00", []))

    with (publish / "publish_tier_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset_id", "tier", "randomized_dir", "defect_class_ids", "n_defect_classes"))
        writer.writeheader()
        writer.writerow({"dataset_id": "HMI_AR_sample", "tier": "clean", "randomized_dir": clean.name, "defect_class_ids": "", "n_defect_classes": 0})
        writer.writerow({"dataset_id": "HMI_AR_sample", "tier": "P3", "randomized_dir": corrupt.name, "defect_class_ids": "L0-03 L2-G L2-F L2-P L2-T L3-06 L3-07 L3-MIS", "n_defect_classes": 8})
    return publish, clean, corrupt
