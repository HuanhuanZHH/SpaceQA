"""Read-only current-input rules, with scientific priors kept out of hard verdicts.

No clean reference, tier, target, split, or package suffix is inspected. This is
an experimental candidate, not a replacement for the frozen competition engine.
Only standard-library dependencies are required. Group bits use G/C/D order.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import re
import statistics
import struct
from dataclasses import dataclass
from pathlib import Path

from . import backend_strategy

VERSION = "prior-rules-2026.11.02-1"
FAMILIES = ("AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample", "EPB_seg_sample", "DMSP_SSUSI", "ULF_sample")
ALLOWED = dict(zip(FAMILIES, ({"AR"}, {"AR"}, {"CME"}, {"EPB"}, {"PCA", "TPA", "unclear"}, {"Pi2"})))
ALIASES = {"Active Region": "AR", "Coronal Hole": "CH"}
TIME_PAIRS = (("startTime", "endTime"), ("event_starttime", "event_endtime"), ("start_dt", "end_dt"), ("dateObs", "dateEnd"))


@dataclass(frozen=True)
class Policy:
    # Caller must establish this convention from release documentation, never
    # from a package's clean/P0 status. Roadshow's basic.startTime is dependent.
    trust_ulf_dateobs: bool = False
    ulf_duration_convention: str = "either"  # inclusive, elapsed, or either
    ulf_category_contract: bool = False
    tolerance_seconds: float = 1e-5
    nominal_cme_cadence_seconds: float = 720.0
    # Enhanced adds AIA94-derived family observations only.  Legacy is the
    # rollback profile for the previous dialog-mode submission.
    backend_profile: str = "enhanced"


def timestamp(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed.astimezone(dt.timezone.utc)
    except (ValueError, OverflowError):
        return None


def number(value):
    if isinstance(value, bool):
        return None
    try:
        answer = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return answer if math.isfinite(answer) else None


def close(a, b, tolerance=1e-5):
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-6, abs_tol=tolerance)


def walk(value, pointer=""):
    if isinstance(value, dict):
        yield pointer, value
        for key, nested in value.items():
            key = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk(nested, pointer + "/" + key)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from walk(nested, pointer + "/" + str(index))


def png_size(path):
    with Path(path).open("rb") as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("invalid PNG IHDR")
    width, height = struct.unpack(">II", header[16:24])
    if not width or not height:
        raise ValueError("zero PNG dimension")
    return width, height


def csv_statistics(path):
    """Read the ENTIRE CSV axis; never infer regularity from a preview or sample."""
    values = []
    invalid = 0
    rows = 0
    time_kind = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = [x.strip() for x in next(reader, [])]
        # Uppercase T is the magnetic total-field component, not the lowercase
        # relative-seconds column t. Case-folding these creates false mismatch.
        time_indexes = [i for i, x in enumerate(header) if x == "t" or x.lower() in {"time", "timestamp", "datetime", "date"}]
        time_index = time_indexes[0] if len(time_indexes) == 1 else None
        for row in reader:
            if not row or not any(x.strip() for x in row):
                continue
            rows += 1
            if time_index is None or len(row) <= time_index:
                invalid += 1
                continue
            raw = row[time_index].strip()
            numeric = number(raw)
            absolute = timestamp(raw) if numeric is None else None
            kind = "relative_seconds" if numeric is not None else "absolute_utc" if absolute else None
            if kind is None or (time_kind is not None and time_kind != kind):
                invalid += 1
                continue
            time_kind = kind
            values.append(numeric if numeric is not None else absolute.timestamp())
    deltas = [b - a for a, b in zip(values, values[1:])]
    cadence = statistics.median(deltas) if deltas else None
    regular = bool(rows >= 2 and len(values) == rows and not invalid and cadence is not None and cadence > 0
                   and all(close(x, cadence) for x in deltas))
    return dict(rows=rows, columns=header, components=[x for i, x in enumerate(header) if i != time_index],
                time_column=header[time_index] if time_index is not None else None, time_kind=time_kind,
                invalid_time_rows=invalid, first_t=values[0] if values else None,
                last_t=values[-1] if values else None, median_dt=cadence,
                regular=regular, full_scan=True, strictly_increasing=bool(deltas) and all(x > 0 for x in deltas))


def _finding(code, pointer, basis, observation, *, certainty="confirmed", score_eligible=True, group="D"):
    return dict(defectType=code, pointer=pointer or "/", basis=basis, observation=observation,
                certainty=certainty, score_eligible=bool(score_eligible and certainty == "confirmed"), group=group)


def _category_findings(family, label, policy):
    findings = []
    allowed = ALLOWED[family]
    # Do not reinterpret numeric eventtype IDs. concept is a textual alias, not
    # a reason to discard a present event_type field.
    for pointer, node in walk(label):
        is_event = any(token in pointer for token in ("/matched_events/", "/events/", "/shapes/", "/specialShapes/"))
        fields = [key for key in ("event_type", "concept", "kind", "label") if key in node] if is_event else []
        if "event_types" in node and isinstance(node["event_types"], list):
            for index, category in enumerate(node["event_types"]):
                if isinstance(category, str) and ALIASES.get(category, category) not in allowed:
                    uncertain = family == "ULF_sample" and not policy.ulf_category_contract
                    findings.append(_finding("L2-P", pointer + "/event_types/" + str(index),
                        "Category outside the release-scoped family dictionary", {"actual": category, "allowed": sorted(allowed)},
                        certainty="candidate" if uncertain else "confirmed"))
        for key in fields:
            category = node[key]
            if category in (None, "") or not isinstance(category, str):
                continue
            if ALIASES.get(category, category) not in allowed:
                uncertain = family == "ULF_sample" and not policy.ulf_category_contract
                findings.append(_finding("L2-P", pointer + "/" + key,
                    "Category outside the release-scoped family dictionary; this does not prove physical absence",
                    {"actual": category, "allowed": sorted(allowed)}, certainty="candidate" if uncertain else "confirmed"))
    # Summary/list inconsistency is real, but already exists in clean ULF.
    for pointer, node in walk(label):
        events = node.get("matched_events")
        if not isinstance(events, list):
            continue
        reasons = []
        if "event_count" in node and node["event_count"] != len(events):
            reasons.append("event_count differs from record count")
        if isinstance(node.get("has_events"), bool) and node["has_events"] != bool(events):
            reasons.append("has_events differs from record presence")
        declared = node.get("event_types")
        types = {ALIASES.get(e.get("kind") or e.get("event_type") or e.get("concept"), e.get("kind") or e.get("event_type") or e.get("concept"))
                 for e in events if isinstance(e, dict)} - {None, ""}
        classified = [e.get("kind") or e.get("event_type") or e.get("concept") for e in events if isinstance(e, dict)]
        complete_categories = bool(events) and len(classified) == len(events) and all(isinstance(x, str) and x for x in classified)
        if isinstance(declared, list) and complete_categories and types != {ALIASES.get(x, x) for x in declared if isinstance(x, str)}:
            reasons.append("event_types differs from record categories")
        if reasons:
            findings.append(_finding("ANNOTATION_STRUCTURE_CONFLICT", pointer,
                "Internal contradiction; no claim that this is newly injected contamination",
                {"violations": reasons, "observed_types": sorted(types)}, certainty="candidate"))
    return findings


def _dimension_findings(label, actual):
    findings = []
    for pointer, node in walk(label):
        # CME event.width is an angle, never the PNG width. Only explicit image
        # metadata containers are compared; scientific object parameters skip.
        if any(token in pointer for token in ("/matched_events", "/events", "/shapes", "/specialShapes", "/bbox")):
            continue
        if not any(token in pointer.lower() for token in ("basic_annotation", "science_annotation", "metadata", "image", "header", "size")):
            continue
        for width_key, height_key in (("width", "height"), ("imageWidth", "imageHeight"), ("imgWidth", "imgHeight"), ("image_width", "image_height")):
            for key, target in zip((width_key, height_key), actual):
                if key in node and (number(node[key]) is None or not close(number(node[key]), target, 0.0)):
                    findings.append(_finding("L2-F", pointer + "/" + key,
                        "Declared image dimension contradicts the current PNG IHDR", {"actual": node[key], "png_dimension": target}))
        size = node.get("imgSize")
        match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*(?:pixels?|px|像素)?\s*", size, re.I) if isinstance(size, str) else None
        if match and tuple(map(int, match.groups())) != actual:
            findings.append(_finding("L2-F", pointer + "/imgSize", "imgSize contradicts the current PNG IHDR",
                                     {"declared": list(map(int, match.groups())), "png_dimensions": list(actual)}))
    return findings


def _ulf_findings(label, stats, policy, time_origin):
    science = label.get("science_annotation") or {}
    basic = label.get("basic_annotation") or {}
    findings = []
    regular = bool(stats.get("full_scan") and stats.get("regular") and not stats.get("invalid_time_rows"))
    cadence = number(stats.get("median_dt"))
    first, last = number(stats.get("first_t")), number(stats.get("last_t"))
    span = last - first if first is not None and last is not None else None
    inclusive = span + cadence if span is not None and cadence is not None else None
    accepted_duration = [inclusive] if policy.ulf_duration_convention == "inclusive" else [span] if policy.ulf_duration_convention == "elapsed" else [span, inclusive]
    n_points = number(science.get("n_points"))
    duration = number(science.get("duration_s"))
    row_agrees = n_points is not None and close(n_points, stats.get("rows"), 0.0)
    duration_agrees = duration is not None and any(close(duration, value) for value in accepted_duration)
    # Independent metadata agreement can support a current complete data body.
    # When both counts and span disagree, a truncated/altered CSV remains a
    # competing explanation and is not converted into annotation contamination.
    checks = []
    if n_points is not None and not row_agrees:
        checks.append(("n_points", science["n_points"], stats.get("rows"), regular and duration_agrees))
    if regular and duration is not None and not duration_agrees:
        checks.append(("duration_s", science["duration_s"], accepted_duration, row_agrees))
    rate = number(science.get("sampling_rate_hz"))
    if regular and cadence and rate is not None and not close(rate, 1 / cadence):
        checks.append(("sampling_rate_hz", science["sampling_rate_hz"], 1 / cadence, row_agrees or duration_agrees))
    components = science.get("components")
    if isinstance(components, list) and set(map(str, components)) != set(stats.get("components", [])):
        checks.append(("components", components, stats.get("components"), True))
    for key, declared, measured, supported in checks:
        findings.append(_finding("L2-M", "/science_annotation/" + key,
            "Current CSV/metadata contradiction" if supported else "CSV/metadata contradiction; data-body corruption or truncation remains plausible",
            {"declared": declared, "measured": measured, "regular_full_csv": regular,
             "row_count_agrees": row_agrees, "duration_agrees": duration_agrees,
             "attribution": "annotation_supported" if supported else "data_or_annotation_unresolved"},
            certainty="confirmed" if supported else "candidate"))
    if not regular:
        findings.append(_finding("CSV_AXIS_DIAGNOSTIC", "/science_annotation",
            "Full CSV time axis is incomplete, irregular, nonfinite, nonmonotonic, or underspecified; do not attribute it to annotation parameters",
            stats, certainty="candidate", group=None))

    if regular:
        origin = time_origin
        if stats.get("time_kind") == "absolute_utc":
            expected = [dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in (first, last)]
            independent = True
            source = "absolute CSV time column"
        else:
            if origin is None and timestamp(science.get("dateObs")):
                origin = dict(value=science["dateObs"], source="science_annotation.dateObs", independent=policy.trust_ulf_dateobs)
            parsed_origin = timestamp(origin.get("value")) if isinstance(origin, dict) else None
            if parsed_origin:
                expected = [parsed_origin + dt.timedelta(seconds=x) for x in (first, last)]
                source = origin.get("source", "unspecified")
                independent = bool(origin.get("independent")) and source != "basic_annotation.startTime"
            else:
                expected, independent, source = None, False, None
        if expected:
            for key, target in zip(("startTime", "endTime"), expected):
                declared = timestamp(basic.get(key))
                if declared and abs((declared - target).total_seconds()) > policy.tolerance_seconds:
                    findings.append(_finding("L2-T", "/basic_annotation/" + key,
                        "Window endpoint contradicts the complete CSV axis and an independent time origin" if independent else "Window endpoint differs from a label-provided time-origin convention; requires provenance",
                        {"declared": basic[key], "expected": target.isoformat(), "offset_seconds": (declared - target).total_seconds(),
                         "origin_source": source, "independent_origin": independent},
                        certainty="confirmed" if independent else "candidate"))
    # No hard hourly alignment or physical-unit inference is made. A cropped
    # window can legitimately start at any minute; naked CSV has no unit truth.
    if "units" in science and science["units"] not in (None, "", "nT (H/Z/T), deg (D)"):
        findings.append(_finding("ULF_UNIT_PROVENANCE_REQUIRED", "/science_annotation/units",
            "Naked CSV cannot establish physical units; obtain documentation or trusted reference",
            {"declared": science["units"]}, certainty="candidate"))
    return findings


def _finish(result):
    for finding in result["findings"]:
        finding["locator"] = result["label_path"] + "#" + finding["pointer"]
        finding["rule_version"] = VERSION
    result["confirmed_bits"] = [int(any(f["group"] == group and f["score_eligible"] for f in result["findings"])) for group in ("G", "C", "D")]
    result["candidate_bits"] = [int(any(f["group"] == group and f["certainty"] == "candidate" for f in result["findings"])) for group in ("G", "C", "D")]
    return result


def evaluate_sample(family, label, data_path, *, label_path="<label>", sample_id=None, csv_stats=None, time_origin=None, policy=None):
    """Evaluate only one current sample. Extra manifest/target fields are ignored.

    csv_stats must come from csv_statistics or an equivalent verified full scan;
    sampled stats lacking full_scan=True never authorize hard temporal claims.
    time_origin may be {value, source, independent}; basic.startTime is always
    dependent even when the caller accidentally marks it independent.
    """
    if family not in FAMILIES:
        raise ValueError("Unknown family; pass the known release family explicitly")
    policy = policy or Policy()
    if policy.ulf_duration_convention not in {"inclusive", "elapsed", "either"}:
        raise ValueError("Unknown duration convention")
    if policy.backend_profile not in {"legacy", "enhanced"}:
        raise ValueError("Unknown backend profile")
    if not isinstance(label, dict):
        raise ValueError("Label must be an object")
    path = Path(data_path) if data_path is not None else None
    findings = _category_findings(family, label, policy)
    for pointer, node in walk(label):
        for start_key, end_key in TIME_PAIRS:
            if not node.get(start_key) or not node.get(end_key):
                continue
            start, end = timestamp(node[start_key]), timestamp(node[end_key])
            if start is None or end is None:
                findings.append(_finding("TIME_INVALID", pointer, "Timestamp cannot be parsed", {start_key: node[start_key], end_key: node[end_key]}))
            elif start > end:
                findings.append(_finding("L2-T", pointer, "Start occurs after end", {"reversal_seconds": (start - end).total_seconds()}))
    result = dict(sample_id=sample_id or str(label_path), family=family, label_path=str(label_path), data_path=str(path) if path else None,
                  findings=findings, observations={}, raw_input_read_only=True)
    # The migration is observational: this never changes findings, confirmed
    # bits, score eligibility, or the existing dialog response schema.
    try:
        result["observations"]["backend_strategy"] = backend_strategy.build_observation(
            family, path, label, profile=policy.backend_profile)
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        result["observations"]["backend_strategy"] = {
            "version": backend_strategy.VERSION, "profile": policy.backend_profile,
            "family": family, "strategy": "fallback", "scoreEffect": "none",
            "fallback": "strategy_observation_failed:" + type(error).__name__,
        }
    if path is None or not path.is_file():
        findings.append(_finding("L0-01", "/basic_annotation/fileName", "Label exists but its explicit current data path is absent",
                                 {"expected_path": str(path) if path else None}, group=None))
        return _finish(result)
    try:
        if family == "ULF_sample":
            stats = csv_stats if csv_stats is not None else csv_statistics(path)
            result["observations"]["csv_statistics"] = stats
            findings.extend(_ulf_findings(label, stats, policy, time_origin))
        elif path.suffix.lower() == ".png":
            actual = png_size(path)
            result["observations"]["png_dimensions"] = list(actual)
            findings.extend(_dimension_findings(label, actual))
    except (OSError, ValueError, UnicodeError, csv.Error, OverflowError) as error:
        findings.append(_finding("RAW_DATA_READ_FAILED", "/basic_annotation/fileName", "Raw-data read failed; do not infer annotation contamination",
                                 {"error": str(error)}, certainty="candidate", group=None))
    if family == "CME_C2_sample":
        # Three different clocks: catalog event time, group anchor, image frame.
        # Comparing cme_time to either image clock is deliberately absent.
        frame = timestamp((label.get("basic_annotation") or {}).get("startTime"))
        anchor = timestamp((label.get("hek_event_annotation") or {}).get("image_time"))
        result["observations"]["cme_frame_offset_seconds"] = (frame - anchor).total_seconds() if frame and anchor else None
        result["observations"]["cme_clock_semantics"] = "cme_time is catalog event time; image_time is group anchor; startTime is frame time"
    return _finish(result)


def evaluate_package(records, *, policy=None):
    """Evaluate one current package, optionally adding within-package consensus.

    Each record supplies family, label (object), data_path, label_path, optional
    sample_id/csv_stats/time_origin/package. Cross-frame grouping is scoped to
    package or, when absent, the label directory; it never pools tiers/packages.
    Consensus discrepancies stay candidates, even with a majority.
    """
    policy = policy or Policy()
    records = list(records)
    results = [evaluate_sample(r["family"], r["label"], r.get("data_path"), label_path=r.get("label_path", "<label>"),
        sample_id=r.get("sample_id", r.get("id")), csv_stats=r.get("csv_stats"), time_origin=r.get("time_origin"), policy=policy) for r in records]
    groups = collections.defaultdict(list)
    identity_fields = ("speed_2nd_initial", "speed_2nd_20R", "accel_flag", "remarks")
    for index, record in enumerate(records):
        if record["family"] != "CME_C2_sample":
            continue
        label = record["label"]
        annotation = label.get("hek_event_annotation") or {}
        anchor = timestamp(annotation.get("image_time"))
        if not anchor:
            continue
        events = annotation.get("matched_events") or []
        keys = []
        for event in events:
            if not isinstance(event, dict) or not all(key in event for key in identity_fields):
                keys.append(None)
            else:
                location = Path(record.get("label_path", "<label>"))
                package = str(record.get("package") or record.get("package_id") or location.parent)
                keys.append(json.dumps([package, anchor.isoformat()] + [event[k] for k in identity_fields], sort_keys=True, ensure_ascii=True))
        counts = collections.Counter(keys)
        for event_index, (event, key) in enumerate(zip(events, keys)):
            if key is None or counts[key] > 1:
                continue  # cannot bind ambiguous objects by array position
            event_time = timestamp(event.get("cme_time"))
            frame = timestamp((label.get("basic_annotation") or {}).get("startTime"))
            if event_time and frame:
                groups[key].append((index, event_index, event_time, frame))
    for rows in groups.values():
        frames = {r[3] for r in rows}
        if len(rows) < 3 or len(frames) < 3:
            continue
        counts = collections.Counter(r[2] for r in rows)
        mode, count = counts.most_common(1)[0]
        if count < 2 or count / len(rows) < 2 / 3:
            continue
        for index, event_index, event_time, frame in rows:
            if event_time != mode:
                results[index]["findings"].append(_finding("L2-T", f"/hek_event_annotation/matched_events/{event_index}/cme_time",
                    "Catalog time differs from the same event's within-package cross-frame majority; event identity/majority need verification",
                    {"actual": event_time.isoformat(), "consensus": mode.isoformat(), "support": count, "records": len(rows),
                     "distinct_frame_times": len(frames), "matching_fields": list(identity_fields),
                     "offset_seconds": (event_time - mode).total_seconds()}, certainty="candidate"))
    return [_finish(result) for result in results]


def load_package(root, family):
    """Local published label/data layout plus colocated roadshow JSON/PNG/CSV.

    Uses explicit fileName before release filename conventions. Metadata's old
    relativePath is not trusted as a reason to declare a present data/ file lost.
    """
    root = Path(root)
    labels = sorted((root / "label").glob("*.json")) if (root / "label").is_dir() else sorted(root.rglob("*.json"))
    records = []
    for label_path in labels:
        label = json.loads(label_path.read_text(encoding="utf-8-sig"))
        basic = label.get("basic_annotation") or {}
        name = basic.get("fileName")
        if not isinstance(name, str) or not name:
            name = label_path.name.removesuffix(".json")
            if not name.lower().endswith((".png", ".csv")):
                name += ".csv" if family == "ULF_sample" else ".png"
        # Release references may contain an obsolete EPB path. Search only this
        # package and require a unique existing basename before resolving it.
        leaf = re.split(r"[/\\]", name)[-1]
        candidates = list(dict.fromkeys([root / "data" / leaf, label_path.parent / leaf, root / leaf]))
        existing = [p for p in candidates if p.is_file()]
        if len(existing) > 1:
            raise ValueError("Ambiguous data basename within package: " + leaf)
        records.append(dict(family=family, label=label, data_path=str(existing[0] if existing else candidates[0]),
                            label_path=str(label_path), sample_id=str(label_path.relative_to(root))))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--output", required=True, help="New JSONL result file; input data is never modified")
    parser.add_argument("--trust-ulf-dateobs", action="store_true", help="Only after independently confirming this release's origin contract")
    parser.add_argument("--ulf-duration-convention", choices=("inclusive", "elapsed", "either"), default="either")
    parser.add_argument("--backend-profile", choices=("enhanced", "legacy"), default="enhanced")
    args = parser.parse_args()
    records = load_package(args.package, args.family)
    results = evaluate_package(records, policy=Policy(trust_ulf_dateobs=args.trust_ulf_dateobs,
                                                     ulf_duration_convention=args.ulf_duration_convention,
                                                     backend_profile=args.backend_profile))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(version=VERSION, samples=len(results), confirmed_samples=sum(any(r["confirmed_bits"]) for r in results),
                         candidate_samples=sum(any(r["candidate_bits"]) for r in results),
                         backendProfile=args.backend_profile,
                         backendStrategies=backend_strategy.summarize([r.get("observations", {}).get("backend_strategy", {}) for r in results]),
                         output=str(output))))


if __name__ == "__main__":
    main()
