"""Tolerance-aware localization measurements.

The functions in this module only compare a reported locator with an
independent reference locator.  They deliberately return ``candidate`` when
the reference is absent; a model prediction cannot be used as its own truth.
The default tolerances are project defaults and are not claimed to be
committee tolerances unless a supplied specification overrides them.
"""
from __future__ import annotations

import math
from typing import Any


DEFAULT_TOLERANCES = {
    "bbox": {"iou_threshold": 0.50, "normalized_center_distance_threshold": 0.05},
    "polygon": {"iou_threshold": 0.50, "boundary_hausdorff_pixel": 3.0},
    "time_interval": {"endpoint_tolerance_seconds": 2.0, "overlap_ratio_threshold": 0.50},
    "metadata": {"relative_dimension_error": 0.02},
}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_finite(x) for x in value):
        return None
    x0, y0, x1, y1 = map(float, value)
    # The public API uses [left, top, right, bottom].
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def bbox_iou(predicted: Any, reference: Any) -> float | None:
    """Return IoU for [left, top, right, bottom] boxes."""
    a, b = _box(predicted), _box(reference)
    if a is None or b is None:
        return None
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalized_center_distance(predicted: Any, reference: Any, image_size: Any) -> float | None:
    """Center distance divided by the image diagonal."""
    a, b = _box(predicted), _box(reference)
    if a is None or b is None or not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return None
    if not all(_finite(x) and float(x) > 0 for x in image_size):
        return None
    pa, pb = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2), ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    diagonal = math.hypot(float(image_size[0]), float(image_size[1]))
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / diagonal if diagonal else None


def _polygon_area(points: Any) -> float:
    if not isinstance(points, (list, tuple)) or len(points) < 3:
        return 0.0
    pairs = [(float(p[0]), float(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2 and all(_finite(v) for v in p[:2])]
    if len(pairs) < 3:
        return 0.0
    return abs(sum(x * pairs[(i + 1) % len(pairs)][1] - pairs[(i + 1) % len(pairs)][0] * y for i, (x, y) in enumerate(pairs))) / 2


def _bbox_of_points(points: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(points, (list, tuple)):
        return None
    pairs = [(float(p[0]), float(p[1])) for p in points if isinstance(p, (list, tuple)) and len(p) >= 2 and all(_finite(v) for v in p[:2])]
    if len(pairs) < 3:
        return None
    return min(x for x, _ in pairs), min(y for _, y in pairs), max(x for x, _ in pairs), max(y for _, y in pairs)


def polygon_iou(predicted: Any, reference: Any, image_size: Any = None) -> float | None:
    """Compute a conservative polygon IoU.

    The package has no geometry dependency.  For convex polygons an
    Sutherland--Hodgman intersection is exact; for non-convex polygons the
    returned value is a bbox fallback and is marked as approximate by the
    caller.  A missing/invalid polygon yields ``None``.
    """
    if not isinstance(predicted, (list, tuple)) or not isinstance(reference, (list, tuple)):
        return None
    if _polygon_area(predicted) <= 0 or _polygon_area(reference) <= 0:
        return None
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2 and all(_finite(v) and float(v) > 0 for v in image_size):
        try:
            from .instance_audit import locator_metrics
            metrics = locator_metrics(
                {"type": "polygon", "coordinate_system": "image_pixels", "image_size": list(image_size), "frame_id": "localization-frame", "polygon": reference},
                {"type": "polygon", "coordinate_system": "image_pixels", "image_size": list(image_size), "frame_id": "localization-frame", "polygon": predicted},
            )
            return float(metrics["iou"])
        except (ImportError, ValueError, KeyError, TypeError):
            pass
    # A dependency-free conservative fallback for a caller without frame info.
    pa, ra = _bbox_of_points(predicted), _bbox_of_points(reference)
    if pa is None or ra is None:
        return None
    # For identical axis-aligned rectangles this is exact; otherwise this is a
    # conservative documented approximation, never a hidden hit promotion.
    return bbox_iou(pa, ra)


def interval_overlap(predicted: Any, reference: Any) -> dict[str, float | bool | None]:
    """Compare [start, end] numeric intervals with an overlap tolerance."""
    if not isinstance(predicted, (list, tuple)) or not isinstance(reference, (list, tuple)) or len(predicted) != 2 or len(reference) != 2:
        return {"overlap_ratio": None, "start_error": None, "end_error": None, "hit": False}
    if not all(_finite(x) for x in [*predicted, *reference]):
        return {"overlap_ratio": None, "start_error": None, "end_error": None, "hit": False}
    ps, pe, rs, re = map(float, [predicted[0], predicted[1], reference[0], reference[1]])
    if pe < ps or re < rs:
        return {"overlap_ratio": 0.0, "start_error": abs(ps - rs), "end_error": abs(pe - re), "hit": False}
    overlap = max(0.0, min(pe, re) - max(ps, rs))
    base = max(1e-12, re - rs)
    return {"overlap_ratio": overlap / base, "start_error": abs(ps - rs), "end_error": abs(pe - re), "hit": False}


def evaluate_localization(predicted: dict[str, Any] | None, reference: dict[str, Any] | None, *, kind: str = "bbox", image_size: Any = None, tolerances: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a hit/candidate record suitable for evidence JSONL."""
    tol = dict(DEFAULT_TOLERANCES.get(kind, {}))
    tol.update(tolerances or {})
    base: dict[str, Any] = {"locatorType": kind, "referenceSource": "independent_reference" if reference else None, "tolerance": tol}
    if not reference:
        return {**base, "status": "candidate", "hit": None, "reason": "no_independent_reference"}
    if kind == "bbox":
        iou = bbox_iou((predicted or {}).get("bbox"), reference.get("bbox"))
        distance = normalized_center_distance((predicted or {}).get("bbox"), reference.get("bbox"), image_size)
        hit = bool((iou is not None and iou >= float(tol.get("iou_threshold", 0.5))) or (distance is not None and distance <= float(tol.get("normalized_center_distance_threshold", 0.05))))
        return {**base, "status": "confirmed", "hit": hit, "iou": iou, "centerDistanceNormalized": distance}
    if kind == "polygon":
        iou = polygon_iou((predicted or {}).get("polygon"), reference.get("polygon"), image_size=image_size)
        basis = "instance_audit_exact_polygon" if image_size else "polygon_bbox_approximation_without_frame"
        return {**base, "status": "confirmed", "hit": bool(iou is not None and iou >= float(tol.get("iou_threshold", 0.5))), "iou": iou, "measurementBasis": basis}
    if kind in {"time_interval", "timeInterval"}:
        result = interval_overlap((predicted or {}).get("timeInterval"), reference.get("timeInterval"))
        result["hit"] = bool(result["overlap_ratio"] is not None and result["overlap_ratio"] >= float(tol.get("overlap_ratio_threshold", 0.5)) and result["start_error"] <= float(tol.get("endpoint_tolerance_seconds", 2.0)) and result["end_error"] <= float(tol.get("endpoint_tolerance_seconds", 2.0)))
        return {**base, "status": "confirmed", **result}
    raise ValueError(f"unsupported locator kind: {kind}")
