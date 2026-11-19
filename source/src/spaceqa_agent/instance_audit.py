"""Independent instance comparison and defect-localization evaluation.

The candidate generator has no access to evaluation truth.  The evaluator never
uses model outputs as truth, and requires an explicit reviewed scope and explicit
localization tolerances.  All public inputs/outputs are JSON-compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "instance-audit-1"
_EPS = 1e-10
CANDIDATE_CLASS_CODES = {
    "missing_annotation": "L3-06", "spurious_annotation": "L3-07",
    "class_error": "L2-P", "boundary_position_deviation": "L2-G",
    "time_boundary_jitter": "L2-T", "undercoverage": "L3-MIS", "overcoverage": "L3-MIS",
}
CANDIDATE_DEFAULTS = {
    "confidence_min": 0.8,
    "association_iou_min": 0.1,
    "association_distance_max": 0.15,
    "boundary_iou_min": 0.75,
    "boundary_distance_max": 0.05,
    "coverage_min": 0.9,
    "area_ratio_tolerance": 0.1,
    "time_endpoint_tolerance_seconds": 0.0,
}


class LocatorError(ValueError):
    """The location cannot be compared without guessing its meaning."""


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocatorError("coordinates must be finite numbers")
    value = float(value)
    if not math.isfinite(value):
        raise LocatorError("coordinates must be finite numbers")
    return value


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise LocatorError("expected a coordinate pair")
    return _number(value[0]), _number(value[1])


def _cross(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area(points: Sequence[Sequence[float]]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1]
               for a, b in zip(points, points[1:] + points[:1])) / 2


def _on_segment(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> bool:
    return (abs(_cross(a, b, p)) <= _EPS and
            min(a[0], b[0]) - _EPS <= p[0] <= max(a[0], b[0]) + _EPS and
            min(a[1], b[1]) - _EPS <= p[1] <= max(a[1], b[1]) + _EPS)


def _segments_intersect(a: Sequence[float], b: Sequence[float],
                        c: Sequence[float], d: Sequence[float]) -> bool:
    x, y, z, w = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if x * y < -_EPS and z * w < -_EPS:
        return True
    return any((abs(v) <= _EPS and _on_segment(s, t, p))
               for v, s, t, p in [(x, a, b, c), (y, a, b, d), (z, c, d, a), (w, c, d, b)])


def _polygon(points: Any) -> list[tuple[float, float]]:
    if not isinstance(points, (list, tuple)):
        raise LocatorError("polygon must be one simple ring, without holes")
    pts = [_pair(p) for p in points]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    # Remove consecutive collinear vertices before ear clipping.
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        for i in range(len(pts)):
            if _on_segment(pts[i - 1], pts[(i + 1) % len(pts)], pts[i]):
                pts.pop(i)
                changed = True
                break
    if len(pts) < 3 or len(set(pts)) != len(pts) or abs(_signed_area(pts)) <= _EPS:
        raise LocatorError("polygon is degenerate")
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if j in {i + 1, (i - 1) % len(pts)} or (i == 0 and j == len(pts) - 1):
                continue
            if _segments_intersect(pts[i], pts[(i + 1) % len(pts)],
                                   pts[j], pts[(j + 1) % len(pts)]):
                raise LocatorError("self-intersecting polygon")
    return pts if _signed_area(pts) > 0 else list(reversed(pts))


def _inside_triangle(p: Sequence[float], a: Sequence[float],
                     b: Sequence[float], c: Sequence[float]) -> bool:
    return min(_cross(a, b, p), _cross(b, c, p), _cross(c, a, p)) >= -_EPS


def _triangles(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    pts = list(points)
    out = []
    while len(pts) > 3:
        for i, p in enumerate(pts):
            a, c = pts[i - 1], pts[(i + 1) % len(pts)]
            if _cross(a, p, c) <= _EPS:
                continue
            if any(_inside_triangle(q, a, p, c) for j, q in enumerate(pts)
                   if j not in {i, (i - 1) % len(pts), (i + 1) % len(pts)}):
                continue
            out.append([a, p, c])
            pts.pop(i)
            break
        else:
            raise LocatorError("polygon could not be triangulated")
    return out + [pts]


def _clip(subject: list[tuple[float, float]],
          clipper: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = list(subject)
    for a, b in zip(clipper, clipper[1:] + clipper[:1]):
        inp, out = out, []
        if not inp:
            break
        for p, q in zip(inp, inp[1:] + inp[:1]):
            cp, cq = _cross(a, b, p), _cross(a, b, q)
            pin, qin = cp >= -_EPS, cq >= -_EPS
            if pin:
                out.append(p)
            if pin != qin:
                t = cp / (cp - cq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    denom = 6 * _signed_area(points)
    x = y = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        w = a[0] * b[1] - b[0] * a[1]
        x += (a[0] + b[0]) * w
        y += (a[1] + b[1]) * w
    return x / denom, y / denom


def _normalize_locator(locator: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(locator, Mapping):
        raise LocatorError("locator must be an object")
    kind = locator.get("type")
    if kind in {"file", "record"}:
        path = locator.get("path")
        if not isinstance(path, str) or not path:
            raise LocatorError("file/record locator requires path")
        out = {"kind": kind, "path": path.replace("\\", "/")}
        if kind == "record":
            if "record_id" not in locator or locator["record_id"] is None:
                raise LocatorError("record locator requires record_id")
            out["record_id"] = str(locator["record_id"])
        return out
    if kind == "timeInterval":
        if locator.get("coordinate_system") != "time_seconds":
            raise LocatorError("timeInterval requires coordinate_system=time_seconds")
        axis = locator.get("time_axis_id")
        if not isinstance(axis, str) or not axis:
            raise LocatorError("timeInterval requires an explicit time_axis_id/origin")
        start, end = _pair(locator.get("timeInterval"))
        if end <= start:
            raise LocatorError("timeInterval must have positive duration")
        axis_range = locator.get("axis_range")
        if axis_range is not None:
            lo, hi = _pair(axis_range)
            if hi <= lo or start < lo - _EPS or end > hi + _EPS:
                raise LocatorError("timeInterval is outside a valid axis_range")
            axis_range = [lo, hi]
        return {"kind": "time", "interval": [start, end], "axis": axis,
                "axis_range": axis_range}
    if kind not in {"bbox", "polygon"}:
        raise LocatorError("unsupported locator type")
    coord = locator.get("coordinate_system")
    if coord not in {"image_pixels", "image_normalized"}:
        raise LocatorError("image locator requires an explicit coordinate_system")
    frame = locator.get("frame_id")
    if not isinstance(frame, str) or not frame:
        raise LocatorError("image locator requires frame_id of the current image")
    origin = locator.get("origin", "top_left")
    if origin not in {"top_left", "bottom_left"} or locator.get("axis_order", "xy") != "xy":
        raise LocatorError("unsupported origin or axis order")
    w = h = 1.0
    if coord == "image_pixels":
        w, h = _pair(locator.get("image_size"))
        if w <= 0 or h <= 0:
            raise LocatorError("image_size must be positive")
    if kind == "bbox":
        box = locator.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise LocatorError("bbox uses [x_min,y_min,x_max,y_max]")
        x0, y0, x1, y1 = map(_number, box)
        if x1 <= x0 or y1 <= y0:
            raise LocatorError("bbox must have positive width and height")
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        points = [_pair(p) for p in locator.get("polygon", [])]
    normalized = [(x / w, (y / h if origin == "top_left" else 1 - y / h))
                  for x, y in points]
    if any(x < -_EPS or y < -_EPS or x > 1 + _EPS or y > 1 + _EPS
           for x, y in normalized):
        raise LocatorError("image coordinates are outside the declared image")
    pts = _polygon(normalized)
    return {"kind": "image", "points": pts, "frame": frame,
            "area": _signed_area(pts), "centroid": _centroid(pts)}


def _location_metrics(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    if a["kind"] != b["kind"]:
        return None
    if a["kind"] in {"file", "record"}:
        equal = a == b
        return {"iou": float(equal), "normalized_distance": 0.0 if equal else None,
                "coverage_reference": float(equal), "coverage_annotation": float(equal),
                "area_ratio": 1.0, "exact": equal}
    if a["kind"] == "time":
        if a["axis"] != b["axis"]:
            return None
        a0, a1 = a["interval"]
        b0, b1 = b["interval"]
        if a["axis_range"] != b["axis_range"]:
            return None
        area_a, area_b = a1 - a0, b1 - b0
        intersection = max(0.0, min(a1, b1) - max(a0, b0))
        axis_range = a["axis_range"]
        distance = (abs((a0 + a1 - b0 - b1) / 2) / (axis_range[1] - axis_range[0])
                    if axis_range else None)
        return {"iou": intersection / (area_a + area_b - intersection),
                "normalized_distance": distance,
                "coverage_reference": intersection / area_a,
                "coverage_annotation": intersection / area_b,
                "area_ratio": area_b / area_a,
                "start_error_seconds": abs(a0 - b0), "end_error_seconds": abs(a1 - b1)}
    if a["frame"] != b["frame"]:
        return None
    intersection = 0.0
    for ta in _triangles(a["points"]):
        for tb in _triangles(b["points"]):
            pts = _clip(ta, tb)
            if len(pts) >= 3:
                intersection += abs(_signed_area(pts))
    intersection = min(a["area"], b["area"], max(0.0, intersection))
    return {"iou": intersection / (a["area"] + b["area"] - intersection),
            "normalized_distance": math.dist(a["centroid"], b["centroid"]) / math.sqrt(2),
            "coverage_reference": intersection / a["area"],
            "coverage_annotation": intersection / b["area"],
            "area_ratio": b["area"] / a["area"]}


def locator_metrics(reference: Mapping[str, Any], annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Return exact polygon/bbox/interval overlap; incompatible frames raise."""
    result = _location_metrics(_normalize_locator(reference), _normalize_locator(annotation))
    if result is None:
        raise LocatorError("incompatible locator coordinate frames/types/time axes")
    return result


def maximum_weight_matching(scores: Sequence[Sequence[float | None]]) -> list[tuple[int, int]]:
    """Maximum-cardinality, then maximum-weight bipartite assignment (not greedy).

    None forbids an edge; scores must be finite in [0,1]. Dummy columns permit
    unmatched rows. The Hungarian algorithm uses only the standard library.
    """
    n = len(scores)
    if not n:
        return []
    m = len(scores[0])
    if any(len(row) != m for row in scores):
        raise ValueError("score matrix must be rectangular")
    if not m:
        return []
    for row in scores:
        for score in row:
            if score is not None and not 0 <= _number(score) <= 1:
                raise ValueError("matching scores must lie in [0,1]")
    bonus = min(n, m) + 1
    costs = [[-(bonus + float(s)) if s is not None else 0.0 for s in row] + [0.0] * n
             for row in scores]
    width = m + n
    u, v = [0.0] * (n + 1), [0.0] * (width + 1)
    p, way = [0] * (width + 1), [0] * (width + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [math.inf] * (width + 1), [False] * (width + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, width + 1):
                if not used[j]:
                    cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(width + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1)
                  if p[j] and scores[p[j] - 1][j - 1] is not None)


def _settings(raw: Mapping[str, Any] | None, defaults: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    if raw:
        unknown = set(raw) - set(defaults)
        if unknown:
            raise ValueError(f"unrecognized threshold(s): {sorted(unknown)}")
        out.update(raw)
    for key, value in out.items():
        if value is None:
            if defaults[key] is not None:
                raise ValueError(f"threshold {key} cannot be null")
            continue
        number = _number(value)
        if number < 0 or (not key.endswith("seconds") and number > 1):
            raise ValueError(f"invalid threshold {key}")
    return out


def _object(raw: Mapping[str, Any], sample_id: str, *, confidence: bool = False,
            allow_unknown_label: bool = False) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise LocatorError("object must be a JSON object")
    if not isinstance(raw.get("id"), str) or not raw["id"]:
        raise LocatorError("object requires a nonempty id")
    sample = raw.get("sample_id", sample_id)
    if not isinstance(sample, str) or not sample:
        raise LocatorError("object requires sample_id")
    label = raw.get("label")
    class_known = raw.get("class_known", True) is True
    if allow_unknown_label and (not class_known or label is None or label == ""):
        label, class_known = None, False
    elif not isinstance(label, str) or not label:
        raise LocatorError("object requires label")
    value = _number(raw.get("confidence", 0.0 if confidence else 1.0))
    if not 0 <= value <= 1:
        raise LocatorError("confidence must lie in [0,1]")
    return {"raw": dict(raw), "id": raw["id"], "sample_id": sample, "label": label,
            "class_known": class_known, "confidence": value, "loc": _normalize_locator(raw.get("locator"))}


def _score(metrics: Mapping[str, Any]) -> float:
    distance = metrics.get("normalized_distance")
    return 0.8 * metrics["iou"] + 0.2 * max(0.0, 1 - (distance if distance is not None else 1))


def _passes(metrics: Mapping[str, Any], iou: float | None,
            distance: float | None, time_tolerance: float | None = None) -> bool:
    if "exact" in metrics:
        return bool(metrics["exact"])
    return ((iou is not None and metrics["iou"] + _EPS >= iou) or
            (distance is not None and metrics.get("normalized_distance") is not None and
             metrics["normalized_distance"] <= distance + _EPS) or
            (time_tolerance is not None and "start_error_seconds" in metrics and
             max(metrics["start_error_seconds"], metrics["end_error_seconds"]) <= time_tolerance + _EPS))


def _base_result(payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    return {"schema_version": VERSION, "mode": mode,
            "rule_version": payload.get("rule_version", VERSION),
            "official_metrics": None, "official_qualification_established": False}


def audit_annotations(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Generate review candidates from independent objects vs current labels.

    Model absence alone can never produce a spurious-annotation candidate.
    A complete, verified-negative reference is required for that hypothesis.
    """
    result = _base_result(payload, "candidates")
    thresholds = _settings(payload.get("thresholds"), CANDIDATE_DEFAULTS)
    result.update(status="complete", thresholds=thresholds, threshold_authority="prototype_not_official",
                  candidates=[], abstentions=[], matches=[])
    sample = payload.get("sample_id", "")
    reference = payload.get("reference", {})
    if not isinstance(reference, Mapping):
        reference = {}
    result["reference_metadata"] = {key: value for key, value in reference.items() if key != "objects"}
    reason = None
    if not isinstance(sample, str) or not sample:
        reason = "sample_id_missing"
    elif payload.get("data_available") is not True:
        reason = "input_missing_or_availability_not_verified"
    elif reference.get("status") != "ok":
        reason = "reference_failed_or_not_available"
    elif reference.get("independent_of_annotations") is not True:
        reason = "reference_independence_not_established"
    elif reference.get("source") not in {"model", "independent_reference"}:
        reason = "reference_source_not_declared"
    elif "objects" not in reference or "annotations" not in payload:
        reason = "object_collection_missing_not_a_verified_empty_list"
    elif not isinstance(reference.get("objects", []), list) or not isinstance(payload.get("annotations", []), list):
        reason = "object_collections_must_be_lists"
    elif payload.get("annotation_parse_status", "ok") != "ok":
        reason = "annotations_not_successfully_parsed"
    if reason:
        result["status"] = "abstained"
        result["abstentions"].append({"sample_id": sample, "reason": reason})
        return result
    refs, anns = [], []
    uncertain_ref = invalid_ann = False
    for key, rows, dest in [("reference", reference.get("objects", []), refs),
                            ("annotation", payload.get("annotations", []), anns)]:
        ids = set()
        for raw in rows:
            try:
                obj = _object(raw, sample, confidence=(key == "reference" and reference.get("source") == "model"),
                              allow_unknown_label=True)
                if obj["sample_id"] != sample:
                    raise LocatorError("candidate request contains another sample")
                if obj["id"] in ids:
                    raise LocatorError("duplicate object id")
                ids.add(obj["id"])
                if obj["loc"]["kind"] not in {"image", "time"}:
                    raise LocatorError("object comparison requires an image or time interval")
                if key == "reference" and obj["confidence"] < thresholds["confidence_min"]:
                    raise LocatorError("reference_confidence_below_threshold")
                dest.append(obj)
            except (ValueError, TypeError) as exc:
                uncertain_ref |= key == "reference"
                invalid_ann |= key == "annotation"
                result["abstentions"].append({"sample_id": sample, "object_type": key,
                                               "object_id": raw.get("id") if isinstance(raw, Mapping) else None,
                                               "reason": str(exc)})
    refs, result["reference_duplicates"] = _deduplicate(refs)
    metrics = {}
    scores = []
    for i, ref in enumerate(refs):
        row = []
        for j, ann in enumerate(anns):
            metric = _location_metrics(ref["loc"], ann["loc"])
            metrics[i, j] = metric
            row.append(_score(metric) if metric and _passes(metric, thresholds["association_iou_min"],
                                                           thresholds["association_distance_max"]) else None)
        scores.append(row)
    pairs = maximum_weight_matching(scores)
    used_r, used_a = set(), set()

    def candidate(kind: str, ref: dict[str, Any] | None, ann: dict[str, Any] | None,
                  metric: dict[str, Any] | None = None) -> None:
        identity = [sample, kind, ref["id"] if ref else None, ann["id"] if ann else None,
                    result["rule_version"]]
        result["candidates"].append({
            "id": "CAND-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16],
            "status": "candidate", "defect_class": CANDIDATE_CLASS_CODES[kind],
            "candidate_kind": kind, "subtype": kind if kind in {"undercoverage", "overcoverage"} else None,
            "sample_id": sample,
            "reference_id": ref["id"] if ref else None, "annotation_id": ann["id"] if ann else None,
            "locator": (ann or ref)["raw"]["locator"],
            "reference_locator": ref["raw"]["locator"] if ref else None,
            "confidence": ref["confidence"] if ref else None, "metrics": metric,
            "expected_label": ref["label"] if ref else None,
            "observed_label": ann["label"] if ann else None,
            "basis": "independent_instance_comparison_not_verified_scientific_truth",
            "rule_version": result["rule_version"]})

    for i, j in pairs:
        used_r.add(i)
        used_a.add(j)
        ref, ann, metric = refs[i], anns[j], metrics[i, j]
        result["matches"].append({"reference_id": ref["id"], "annotation_id": ann["id"], "metrics": metric})
        if not ref["class_known"] or not ann["class_known"]:
            result["abstentions"].append({"object_type": "matched_pair", "reference_id": ref["id"],
                                           "annotation_id": ann["id"], "dimension": "class",
                                           "reason": "object_class_not_known"})
        elif ref["label"] != ann["label"]:
            candidate("class_error", ref, ann, metric)
        if ref["loc"]["kind"] == "time":
            if max(metric["start_error_seconds"], metric["end_error_seconds"]) > thresholds["time_endpoint_tolerance_seconds"] + _EPS:
                candidate("time_boundary_jitter", ref, ann, metric)
            continue
        cov, ratio = thresholds["coverage_min"], thresholds["area_ratio_tolerance"]
        if (metric["coverage_reference"] < cov and metric["coverage_annotation"] >= cov and
                metric["area_ratio"] < 1 - ratio):
            candidate("undercoverage", ref, ann, metric)
        elif (metric["coverage_annotation"] < cov and metric["coverage_reference"] >= cov and
              metric["area_ratio"] > 1 + ratio):
            candidate("overcoverage", ref, ann, metric)
        elif (metric["iou"] < thresholds["boundary_iou_min"] or
              metric["normalized_distance"] > thresholds["boundary_distance_max"]):
            candidate("boundary_position_deviation", ref, ann, metric)
    # A frame/axis mismatch cannot establish an absent object in either direction.
    incompatible = any(metric is None for metric in metrics.values())
    for i, ref in enumerate(refs):
        if i not in used_r:
            if invalid_ann or incompatible:
                result["abstentions"].append({"object_type": "reference", "object_id": ref["id"],
                                               "reason": "invalid_or_incompatible_annotations_prevent_absence_decision"})
            else:
                candidate("missing_annotation", ref, None)
    for j, ann in enumerate(anns):
        if j in used_a:
            continue
        if (reference.get("complete") is True and reference.get("negative_evidence_verified") is True
                and not uncertain_ref and not incompatible):
            candidate("spurious_annotation", None, ann)
        else:
            result["abstentions"].append({"object_type": "annotation", "object_id": ann["id"],
                                           "reason": "absence_of_detection_is_not_verified_negative_evidence"})
    if result["abstentions"]:
        result["status"] = "partial" if result["candidates"] or result["matches"] else "abstained"
    return result


def _defect(raw: Mapping[str, Any]) -> dict[str, Any]:
    obj = dict(raw)
    obj["label"] = raw.get("defect_class")
    if raw.get("critical") is not None and not isinstance(raw["critical"], bool):
        raise LocatorError("critical flag must be boolean or unknown/null")
    # Confidence does not weight metrics. A candidate based on verified negative
    # evidence has no detector probability, so JSON null is a valid omission here.
    if obj.get("confidence") is None:
        obj.pop("confidence", None)
    return _object(obj, "")


def _counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}


def _deduplicate(objects: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    kept, duplicates = [], []
    for obj in sorted(objects, key=lambda x: (-x["confidence"], x["id"])):
        duplicate = None
        for prior in kept:
            if (obj["sample_id"], obj["label"]) != (prior["sample_id"], prior["label"]):
                continue
            metric = _location_metrics(prior["loc"], obj["loc"])
            if metric and metric["iou"] >= 1 - _EPS:
                duplicate = prior
                break
        if duplicate:
            if obj["raw"].get("critical") is True:
                duplicate["raw"]["critical"] = True
            duplicates.append({"id": obj["id"], "duplicate_of": duplicate["id"]})
        else:
            kept.append(obj)
    return kept, duplicates


def evaluate_defects(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate defect class AND location, only in an independent reviewed scope.

    Returns official-*style* metrics, never claims an official competition score.
    No tolerance is silently borrowed from a detector or from the research memo.
    """
    result = _base_result(payload, "evaluate")
    result.update(status="unavailable", metrics=None, matches=[], ignored_predictions=[],
                  duplicates=[], errors=[], evaluation_scope="independent_reviewed_samples_only")
    truth = payload.get("truth", {})
    if not isinstance(truth, Mapping):
        truth = {}
    if truth.get("independent") is not True or truth.get("exhaustive") is not True:
        result["errors"].append("independent_exhaustive_truth_not_established")
        return result
    reviewed = truth.get("reviewed_samples")
    if not isinstance(reviewed, list) or not reviewed or any(not isinstance(s, str) or not s for s in reviewed):
        result["errors"].append("explicit_reviewed_samples_required")
        return result
    reviewed = set(reviewed)
    reviewed_classes = truth.get("reviewed_sample_classes")
    reviewed_pairs = None
    if reviewed_classes is not None:
        reviewed_pairs = set()
        if not isinstance(reviewed_classes, list):
            result["errors"].append("reviewed_sample_classes_must_be_a_list")
            return result
        for entry in reviewed_classes:
            if (not isinstance(entry, Mapping) or entry.get("sample_id") not in reviewed or
                    not isinstance(entry.get("defect_classes"), list) or
                    any(not isinstance(label, str) or not label or label == "*"
                        for label in entry["defect_classes"])):
                result["errors"].append("invalid_reviewed_sample_classes")
                return result
            reviewed_pairs.update((entry["sample_id"], label) for label in entry["defect_classes"])
        if not reviewed_pairs:
            result["errors"].append("at_least_one_reviewed_sample_class_required")
            return result
    raw_thresholds = payload.get("evaluation_thresholds")
    if not isinstance(raw_thresholds, Mapping) or not payload.get("tolerance_source"):
        result["errors"].append("explicit_localization_tolerances_and_source_required")
        return result
    defaults = {"iou_min": None, "distance_max": None, "time_tolerance_seconds": None}
    try:
        tolerances = _settings({k: v for k, v in raw_thresholds.items() if k != "per_class"}, defaults)
        per_class = {name: _settings(values, tolerances)
                     for name, values in raw_thresholds.get("per_class", {}).items()}
        if all(value is None for value in tolerances.values()) and not per_class:
            raise ValueError("at least one localization tolerance must be provided")
        # A zero IoU gate would declare disjoint locations correct.
        if any(t["iou_min"] is not None and t["iou_min"] <= 0 for t in [tolerances, *per_class.values()]):
            raise ValueError("iou_min must be greater than zero")
    except (ValueError, TypeError, AttributeError) as exc:
        result["errors"].append(str(exc))
        return result
    result.update(evaluation_thresholds={**tolerances, "per_class": per_class},
                  tolerance_source=payload["tolerance_source"],
                  truth_source=truth.get("source"), reviewed_samples=sorted(reviewed),
                  truth_independence="user_declared_not_authenticated")
    result["reviewed_sample_classes"] = reviewed_classes
    result["exhaustive_scope_meaning"] = ("only_explicit_sample_class_pairs" if reviewed_pairs is not None
                                           else "all_defect_classes_in_reviewed_samples_declared_exhaustive")
    known, masks, predictions = [], [], []
    if "defects" not in truth or "predictions" not in payload:
        result["errors"].append("explicit_truth_and_prediction_collections_required")
        return result
    if (not isinstance(truth.get("defects", []), list) or
            not isinstance(payload.get("predictions", []), list) or
            any(not isinstance(raw, Mapping) for raw in truth.get("defects", []) + payload.get("predictions", []))):
        result["errors"].append("defect_collections_must_be_lists_of_objects")
        return result
    for raw in truth.get("defects", []):
        state = raw.get("status", "known")
        if raw.get("sample_id") not in reviewed:
            result["errors"].append("truth_defect_outside_reviewed_scope")
            continue
        if state in {"unknown", "excluded", "ignore", "uncertain"}:
            try:
                masks.append(_defect(raw))
            except (ValueError, TypeError):
                # Unlocalized unknown truth masks unmatched predictions in that sample;
                # uncertainty never becomes a negative label.
                unknown_label = raw.get("defect_class")
                if not isinstance(unknown_label, str) or not unknown_label:
                    unknown_label = "*"
                masks.append({"sample_id": raw["sample_id"], "label": unknown_label,
                              "id": raw.get("id"), "loc": None})
        elif state == "known":
            if reviewed_pairs is not None and (raw["sample_id"], raw.get("defect_class")) not in reviewed_pairs:
                result["errors"].append("known_truth_outside_reviewed_sample_class_scope")
                continue
            try:
                known.append(_defect(raw))
            except (ValueError, TypeError) as exc:
                result["errors"].append(f"invalid_known_truth:{raw.get('id')}:{exc}")
        else:
            result["errors"].append(f"unrecognized_truth_status:{state}")
    if result["errors"]:
        return result
    if len({obj["id"] for obj in known}) != len(known):
        result["errors"].append("truth_ids_must_be_unique")
        return result
    unknown_critical_truth_ids = [obj["id"] for obj in known
                                  if not isinstance(obj["raw"].get("critical"), bool)]
    known, truth_duplicates = _deduplicate(known)
    result["truth_duplicates"] = truth_duplicates
    for raw in payload.get("predictions", []):
        if raw.get("sample_id") not in reviewed:
            result["ignored_predictions"].append({"id": raw.get("id"), "reason": "outside_reviewed_scope"})
            continue
        if reviewed_pairs is not None and (raw["sample_id"], raw.get("defect_class")) not in reviewed_pairs:
            result["ignored_predictions"].append({"id": raw.get("id"), "reason": "unreviewed_sample_class"})
            continue
        if raw.get("status") in {"abstain", "abstained", "unknown", "excluded"}:
            result["ignored_predictions"].append({"id": raw.get("id"), "reason": "abstained_prediction"})
            continue
        try:
            predictions.append(_defect(raw))
        except (ValueError, TypeError) as exc:
            # Invalid locations cannot be treated as detections or silently dropped.
            result["errors"].append(f"invalid_prediction:{raw.get('id')}:{exc}")
    if result["errors"]:
        return result
    if len({obj["id"] for obj in predictions}) != len(predictions):
        result["errors"].append("prediction_ids_must_be_unique")
        return result
    predictions, result["duplicates"] = _deduplicate(predictions)
    for obj in known + predictions:
        threshold = per_class.get(obj["label"], tolerances)
        kind = obj["loc"]["kind"]
        applicable = (kind in {"file", "record"} or threshold["iou_min"] is not None or
                      (threshold["distance_max"] is not None and
                       (kind == "image" or obj["loc"].get("axis_range") is not None)) or
                      (kind == "time" and threshold["time_tolerance_seconds"] is not None))
        if not applicable:
            result["errors"].append(f"no_applicable_localization_tolerance:{obj['label']}:{kind}")
    if result["errors"]:
        return result
    scores, metrics = [], {}
    for i, ref in enumerate(known):
        row = []
        threshold = per_class.get(ref["label"], tolerances)
        for j, pred in enumerate(predictions):
            metric = None
            if (ref["sample_id"], ref["label"]) == (pred["sample_id"], pred["label"]):
                metric = _location_metrics(ref["loc"], pred["loc"])
            metrics[i, j] = metric
            row.append(_score(metric) if metric and _passes(metric, threshold["iou_min"],
                                                           threshold["distance_max"], threshold["time_tolerance_seconds"]) else None)
        scores.append(row)
    pairs = maximum_weight_matching(scores)
    used_ref, used_pred = {i for i, _ in pairs}, {j for _, j in pairs}
    for i, j in pairs:
        result["matches"].append({"truth_id": known[i]["id"], "prediction_id": predictions[j]["id"],
                                   "defect_class": known[i]["label"], "metrics": metrics[i, j]})
    ignored = set()
    for j, pred in enumerate(predictions):
        if j in used_pred:
            continue
        for mask in masks:
            if pred["sample_id"] != mask["sample_id"] or mask["label"] not in {"*", "unknown", pred["label"]}:
                continue
            metric = _location_metrics(mask["loc"], pred["loc"]) if mask["loc"] else None
            # Any nonzero overlap with unknown/ignored territory is unscorable;
            # exact file/record equality is required for those locator types.
            if mask["loc"] is None or metric is None or metric["iou"] > 0:
                ignored.add(j)
                result["ignored_predictions"].append({"id": pred["id"], "reason": "unknown_or_excluded_truth", "truth_id": mask["id"]})
                break
    result["unmatched_truth_ids"] = [obj["id"] for i, obj in enumerate(known) if i not in used_ref]
    result["unmatched_prediction_ids"] = [obj["id"] for j, obj in enumerate(predictions) if j not in used_pred | ignored]
    by_class = {}
    for label in sorted({obj["label"] for obj in known + predictions}):
        tp = sum(known[i]["label"] == label for i, _ in pairs)
        fp = sum(obj["label"] == label and j not in used_pred | ignored for j, obj in enumerate(predictions))
        fn = sum(obj["label"] == label and i not in used_ref for i, obj in enumerate(known))
        by_class[label] = _counts(tp, fp, fn)
    critical = [i for i, obj in enumerate(known) if obj["raw"].get("critical") is True]
    critical_tp = sum(i in used_ref for i in critical)
    subset_critical_recall = critical_tp / len(critical) if critical else None
    summary = _counts(len(pairs), len(predictions) - len(used_pred | ignored), len(known) - len(used_ref))
    defined_f1 = [x["F1"] for x in by_class.values() if x["F1"] is not None]
    summary.update(by_class=by_class, macro_F1=sum(defined_f1) / len(defined_f1) if defined_f1 else None,
                   critical_defect_recall=(subset_critical_recall if not unknown_critical_truth_ids else None),
                   declared_known_critical_subset_recall=subset_critical_recall,
                   critical_annotations_complete=not unknown_critical_truth_ids,
                   unknown_critical_truth_ids=unknown_critical_truth_ids,
                   critical_defects=len(critical), critical_defects_matched=critical_tp,
                   unknown_or_excluded_truth=len(masks))
    result.update(status="complete", metrics=summary,
                  metric_authority="official_style_only_not_an_official_score")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["candidates", "evaluate"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result = (audit_annotations if args.mode == "candidates" else evaluate_defects)(payload)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if result["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
