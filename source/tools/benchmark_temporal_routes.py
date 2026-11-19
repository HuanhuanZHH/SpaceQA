#!/usr/bin/env python3
"""Read-only P0/P1 timing pilot. Reference agreement is not official accuracy."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import time


CME_FIELDS = ("speed_2nd_initial", "speed_2nd_20R", "accel_flag", "remarks")


def timestamp(value):
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed.astimezone(dt.timezone.utc)


def label_map(root):
    directory = root / "label" if (root / "label").is_dir() else root
    result = {}
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        name = value["basic_annotation"]["fileName"]
        if name in result:
            raise ValueError(f"Ambiguous file identity: {name}")
        result[name] = (path, value)
    return result


def csv_endpoints(path):
    if not path.is_file():
        return None, "missing_csv"
    values = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or reader.fieldnames[0] != "t":
            return None, "unknown_time_column"
        for row in reader:
            try:
                value = float(row["t"])
            except (ValueError, TypeError):
                return None, "invalid_time_axis"
            if not math.isfinite(value):
                return None, "invalid_time_axis"
            values.append(value)
    if len(values) < 2:
        return None, "insufficient_points"
    deltas = [b - a for a, b in zip(values, values[1:])]
    cadence = statistics.median(deltas)
    if cadence <= 0 or any(abs(delta - cadence) > 1e-6 for delta in deltas):
        return None, "irregular_time_axis_not_attributed_to_label"
    return (values[0], values[-1]), "regular_csv"


def ulf_one(task):
    root, name, value = task
    raw = root / "data" / name if (root / "data").is_dir() else root / name
    endpoints, state = csv_endpoints(raw)
    basic, science = value["basic_annotation"], value["science_annotation"]
    start, end = timestamp(basic["startTime"]), timestamp(basic["endTime"])
    row = {"filename": name, "state": state, "reversed": start > end, "anchor_flag": False}
    if endpoints is not None:
        anchor = timestamp(science["dateObs"])
        residuals = [(actual - (anchor + dt.timedelta(seconds=relative))).total_seconds()
                     for actual, relative in zip((start, end), endpoints)]
        row.update(anchor_flag=any(abs(x) > 1e-6 for x in residuals), residual_seconds=residuals)
    return row


def ulf_routes(p0_root, p1_root, workers):
    maps = {"P0": label_map(p0_root), "P1": label_map(p1_root)}
    if maps["P0"].keys() != maps["P1"].keys():
        raise ValueError("ULF P0/P1 label sets differ")
    reference = set()
    for name in maps["P0"]:
        a, b = (maps[tier][name][1]["basic_annotation"] for tier in ("P0", "P1"))
        if any(a[key] != b[key] for key in ("startTime", "endTime")):
            reference.add(name)
    output = {}
    rows_by_tier = {}
    for tier, root in (("P0", p0_root), ("P1", p1_root)):
        tasks = [(root, name, pair[1]) for name, pair in maps[tier].items()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(ulf_one, tasks))
        rows_by_tier[tier] = rows
        flags = {row["filename"] for row in rows if row["anchor_flag"]}
        output[tier] = {
            "samples": len(rows),
            "start_after_end_flags": sum(row["reversed"] for row in rows),
            "dateObs_csv_anchor_flags": len(flags),
            "csv_gate_states": dict(collections.Counter(row["state"] for row in rows)),
        }
        if tier == "P1":
            output[tier]["agreement_with_reference_window_changes"] = {
                "matched": len(flags & reference), "extra": len(flags - reference),
                "missed": len(reference - flags), "reference_count": len(reference),
            }
    return output, rows_by_tier


def cme_routes(p0_root, p1_root):
    maps = {"P0": label_map(p0_root), "P1": label_map(p1_root)}
    if maps["P0"].keys() != maps["P1"].keys():
        raise ValueError("CME P0/P1 label sets differ")
    references = set()
    comparisons = []
    for name in maps["P0"]:
        annotations = [maps[tier][name][1]["hek_event_annotation"] for tier in ("P0", "P1")]
        by_key = []
        for annotation in annotations:
            keyed = collections.defaultdict(list)
            for index, event in enumerate(annotation.get("matched_events", [])):
                key = tuple(event.get(field) for field in CME_FIELDS)
                keyed[key].append((index, event))
            by_key.append(keyed)
        if by_key[0].keys() != by_key[1].keys() or any(len(v) != 1 for keyed in by_key for v in keyed.values()):
            raise ValueError(f"CME event identity cannot be paired uniquely: {name}")
        for key, values in by_key[0].items():
            _, a = values[0]
            index, b = by_key[1][key][0]
            delta = (timestamp(b["cme_time"]) - timestamp(a["cme_time"])).total_seconds()
            if delta:
                references.add((name, index))
                comparisons.append({"filename": name, "event_index": index, "delta_seconds": delta})
    output = {"reference_event_changes": len(references), "reference_affected_samples": len({x[0] for x in references})}
    for tier in ("P0", "P1"):
        groups = collections.defaultdict(list)
        naive = set()
        event_total = 0
        for name, (_, value) in maps[tier].items():
            frame = timestamp(value["basic_annotation"]["startTime"])
            annotation = value["hek_event_annotation"]
            for index, event in enumerate(annotation.get("matched_events", [])):
                event_total += 1
                event_time = timestamp(event["cme_time"])
                if event_time != frame:
                    naive.add((name, index))
                # A group time is used for identity, never assumed equal to the event time.
                key = (annotation.get("image_time"),) + tuple(event.get(field) for field in CME_FIELDS)
                groups[key].append(((name, index), event_time))
        consensus_flags = set()
        eligible = 0
        for rows in groups.values():
            if len(rows) < 3:
                continue
            counts = collections.Counter(value for _, value in rows)
            mode, count = counts.most_common(1)[0]
            if count / len(rows) < 2 / 3:
                continue
            eligible += len(rows)
            consensus_flags.update(identity for identity, value in rows if value != mode)
        row = {"samples": len(maps[tier]), "event_records": event_total,
               "event_not_equal_frame_naive_flags": len(naive),
               "within_P1_or_P0_consensus_eligible_records": eligible,
               "within_tier_consensus_flags": len(consensus_flags)}
        if tier == "P1":
            row["consensus_agreement_with_reference_event_changes"] = {
                "matched": len(consensus_flags & references), "extra": len(consensus_flags - references),
                "missed": len(references - consensus_flags), "reference_count": len(references)}
        output[tier] = row
    return output, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cme, cme_rows = cme_routes(args.input / "CME_C2_sample_g8x0hw7reo", args.input / "CME_C2_sample_1qiaedt0hl")
    print("CME complete:", json.dumps(cme), flush=True)
    ulf, ulf_rows = ulf_routes(args.input / "ULF_sample_lj7rg82ela", args.input / "ULF_sample_n4wb7i9z75", args.workers)
    result = {"scope": "CME and ULF full P0/P1; no fitting or model training", "CME": cme, "ULF": ulf,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "limitations": ["P0-assisted agreement is not hidden official accuracy.",
                              "ULF anchor assumes dateObs is a trusted origin and CSV t is in seconds.",
                              "CME consensus assumes group identity fields remain valid and at least two thirds agree.",
                              "No-reference routes do not consume P0; P0 is used only for this comparison."]}
    for name, payload in (("summary.json", result), ("cme_time_changes.json", cme_rows), ("ulf_samples.json", ulf_rows)):
        (args.output / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
