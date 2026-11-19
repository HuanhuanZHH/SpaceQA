"""Family-aware backend strategies inspired by the AIA94 processing package.

The AIA94 package is a deterministic scientific image-processing pipeline, not
a trainable checkpoint.  This module exposes its reusable backend contract
without importing the package's server-specific dependencies or changing the
existing rule verdicts:

* AIA/HMI: metadata-aware exposure/bright-region context;
* EPB/DMSP: image geometry and label-region consistency;
* CME: event-record and temporal context;
* ULF: full-axis time-series context.

The returned object is an observation only.  Callers may store it in a report,
but must not treat it as an independent defect truth or silently use it to
change scores.  If a scientific array/metadata sidecar is unavailable, the
strategy reports a bounded fallback instead of guessing from a PNG/JPEG.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VERSION = "backend-strategy-2026.11.08-1"

IMAGE_FAMILIES = ("AIA_AR_sample", "HMI_AR_sample")
GEOMETRY_FAMILIES = ("EPB_seg_sample", "DMSP_SSUSI")
FAMILIES = IMAGE_FAMILIES + ("CME_C2_sample",) + GEOMETRY_FAMILIES + ("ULF_sample",)


@dataclass(frozen=True)
class StrategySpec:
    family: str
    name: str
    input_kind: str
    reusable_steps: tuple[str, ...]
    forbidden_steps: tuple[str, ...]
    evidence_units: tuple[str, ...]


STRATEGIES = {
    "AIA_AR_sample": StrategySpec(
        "AIA_AR_sample", "aia94_bright_region", "scientific_image_or_png",
        ("exposure_context", "robust_denoise_context", "dual_threshold_region_candidate", "temporal_context"),
        ("treat_region_id_as_physical_event", "use_png_brightness_as_DN_per_s"),
        ("image_pixels", "bbox", "polygon", "frame_id")),
    "HMI_AR_sample": StrategySpec(
        "HMI_AR_sample", "solar_region_adapted", "scientific_image_or_png",
        ("metadata_aware_preprocess", "robust_region_candidate", "temporal_context"),
        ("reuse_AIA94_threshold_without_calibration", "solar_rotation_without_geometry"),
        ("image_pixels", "bbox", "polygon", "frame_id")),
    "CME_C2_sample": StrategySpec(
        "CME_C2_sample", "event_parameter_context", "image_plus_event_json",
        ("event_record_consistency", "time_context", "parameter_provenance"),
        ("solar_bright_region_threshold", "solar_rotation_region_tracking"),
        ("json_pointer", "event_time", "record")),
    "EPB_seg_sample": StrategySpec(
        "EPB_seg_sample", "segmentation_geometry", "image_plus_geometry_json",
        ("geometry_normalization", "region_label_matching", "coverage_measurement"),
        ("solar_rotation_projection", "AIA94_intensity_threshold"),
        ("image_pixels", "bbox", "polygon", "mask")),
    "DMSP_SSUSI": StrategySpec(
        "DMSP_SSUSI", "segmentation_geometry", "image_plus_geometry_json",
        ("geometry_normalization", "region_label_matching", "coverage_measurement"),
        ("solar_rotation_projection", "AIA94_intensity_threshold"),
        ("image_pixels", "bbox", "polygon", "mask")),
    "ULF_sample": StrategySpec(
        "ULF_sample", "timeseries_context", "csv_plus_time_metadata",
        ("full_axis_scan", "robust_temporal_context", "interval_matching", "unit_provenance"),
        ("pixel_segmentation", "solar_rotation_projection", "image_brightness_threshold"),
        ("csv_row", "time_interval", "field_pointer")),
}


def strategy_for(family: str) -> StrategySpec:
    try:
        return STRATEGIES[family]
    except KeyError as exc:
        raise ValueError("Unknown scientific family: " + str(family)) from exc


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _walk(value: Any, pointer: str = ""):
    if isinstance(value, dict):
        yield pointer, value
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _walk(child, pointer + "/" + escaped)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, pointer + "/" + str(index))


def _timestamp(value: Any):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed.replace(tzinfo=_dt.timezone.utc) if parsed.tzinfo is None else parsed.astimezone(_dt.timezone.utc)


def _geometry_summary(label: dict[str, Any]) -> dict[str, Any]:
    counts = {"bbox": 0, "polygon": 0, "mask": 0, "other": 0}
    total = 0
    for pointer, node in _walk(label):
        if not isinstance(node, dict):
            continue
        kind = node.get("type") or node.get("shape_type") or node.get("geometry_type")
        if isinstance(kind, str):
            normalized = kind.lower()
            if "box" in normalized or normalized in {"rectangle", "rect"}:
                counts["bbox"] += 1
            elif "polygon" in normalized or "contour" in normalized:
                counts["polygon"] += 1
            elif "mask" in normalized or "seg" in normalized:
                counts["mask"] += 1
            elif normalized in {"bbox", "polygon", "mask"}:
                counts[normalized] += 1
            else:
                counts["other"] += 1
            total += 1
        for key in ("bbox", "bounding_box", "box", "polygon", "points", "contour", "mask"):
            if key in node and node[key] not in (None, [], {}):
                normalized = "bbox" if key in {"bbox", "bounding_box", "box"} else "polygon" if key in {"polygon", "points", "contour"} else "mask"
                counts[normalized] += 1
                total += 1
    return {"geometryRecords": total, "geometryTypes": counts}


def _find_sidecar(path: Path) -> Path | None:
    candidates = []
    if path.is_dir():
        candidates.append(path / ".zattrs")
    candidates.extend([path.with_suffix(path.suffix + ".zattrs"), path.with_suffix(".zattrs"),
                       path.with_suffix(".metadata.json"), path.with_suffix(".meta.json")])
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _metadata_context(path: Path, label: dict[str, Any]) -> dict[str, Any]:
    metadata = {}
    sidecar = _find_sidecar(path)
    if sidecar:
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, UnicodeError, ValueError):
            metadata = {}
    # A label can carry release metadata, but it is reported as dependent until
    # a separate sidecar/source contract is present.
    for pointer, node in _walk(label):
        if any(token in pointer.lower() for token in ("metadata", "header", "science_annotation")) and isinstance(node, dict):
            for key in ("EXPTIME", "exptime", "exposure", "R_SUN", "r_sun", "CDELT1", "cdelt1", "CDELT2", "cdelt2"):
                if key in node and key not in metadata:
                    metadata[key] = node[key]
    exposure = next((metadata.get(key) for key in ("EXPTIME", "exptime", "exposure") if _finite(metadata.get(key))), None)
    anomaly = bool(exposure is not None and exposure <= 0.02)
    return {"sidecar": str(sidecar) if sidecar else None, "metadataKeys": sorted(str(k) for k in metadata),
            "exposureSeconds": float(exposure) if exposure is not None else None,
            "exposureAnomalyAtPoint02s": anomaly,
            "metadataIndependent": bool(sidecar)}


def _event_context(label: dict[str, Any]) -> dict[str, Any]:
    events = []
    for pointer, node in _walk(label):
        if isinstance(node, dict) and isinstance(node.get("matched_events"), list):
            events.extend((pointer + "/matched_events/" + str(i), event) for i, event in enumerate(node["matched_events"]) if isinstance(event, dict))
    times = []
    for _, event in events:
        for key in ("cme_time", "event_time", "startTime", "start_time"):
            parsed = _timestamp(event.get(key))
            if parsed:
                times.append(parsed.isoformat())
                break
    return {"eventRecords": len(events), "eventTimeRecords": len(times), "distinctEventTimes": len(set(times)), "timeFieldsAreContextOnly": True}


def build_observation(family: str, data_path: str | Path | None, label: dict[str, Any], *, csv_stats: dict[str, Any] | None = None,
                      profile: str = "enhanced") -> dict[str, Any]:
    """Build a non-scoring backend observation for one sample.

    ``profile='legacy'`` returns a minimal marker so callers can switch back
    without changing the response schema.  ``enhanced`` uses the family-aware
    AIA94-derived contract and always reports why a step was skipped.
    """
    spec = strategy_for(family)
    path = Path(data_path) if data_path is not None else None
    exists = bool(path and path.is_file())
    base = {"version": VERSION, "profile": profile, "family": family, "strategy": spec.name,
            "inputKind": spec.input_kind, "inputExists": exists,
            "reusableSteps": list(spec.reusable_steps), "forbiddenSteps": list(spec.forbidden_steps),
            "evidenceUnits": list(spec.evidence_units), "scoreEffect": "none",
            "fallback": None, "candidateCount": 0}
    if profile == "legacy":
        base["fallback"] = "legacy_profile_requested"
        return base
    if not exists:
        base["fallback"] = "missing_input_preserved_by_rules"
        return base
    if family in IMAGE_FAMILIES:
        context = _metadata_context(path, label)
        native = context["sidecar"] is not None
        base["metadataContext"] = context
        base["candidateContract"] = {"type": "region", "threshold": "AIA94-derived parameters require family calibration", "tracking": "temporal_context_if_frame_sequence"}
        if not native:
            base["fallback"] = "display_image_without_scientific_array_or_metadata"
        elif family == "HMI_AR_sample":
            base["fallback"] = "family_specific_intensity_calibration_required"
    elif family in GEOMETRY_FAMILIES:
        base["geometry"] = _geometry_summary(label)
        base["candidateContract"] = {"type": "region", "coordinates": "native_image_pixels", "tracking": "no_solar_rotation"}
        if not base["geometry"]["geometryRecords"]:
            base["fallback"] = "no_explicit_geometry_record_in_label"
    elif family == "CME_C2_sample":
        base["eventContext"] = _event_context(label)
        base["candidateContract"] = {"type": "event_record", "coordinates": "json_pointer", "tracking": "event_time_context"}
    else:  # ULF
        stats = csv_stats or {}
        base["timeSeriesContext"] = {"fullAxisScan": bool(stats.get("full_scan")), "regular": bool(stats.get("regular")),
                                      "timeKind": stats.get("time_kind"), "medianDt": stats.get("median_dt"),
                                      "rows": stats.get("rows"), "invalidTimeRows": stats.get("invalid_time_rows")}
        base["candidateContract"] = {"type": "time_interval", "coordinates": "csv_time_axis", "tracking": "temporal_context"}
        if not stats.get("full_scan"):
            base["fallback"] = "full_csv_axis_required"
    return base


def summarize(observations: list[dict[str, Any]], profile: str | None = None) -> dict[str, Any]:
    """Aggregate strategy observations for a job/API response."""
    counts: dict[str, int] = {}
    fallbacks: dict[str, int] = {}
    profiles: set[str] = set()
    for item in observations:
        name = str(item.get("strategy", "unknown"))
        counts[name] = counts.get(name, 0) + 1
        value = item.get("profile")
        if value:
            profiles.add(str(value))
        if item.get("fallback"):
            reason = str(item["fallback"])
            fallbacks[reason] = fallbacks.get(reason, 0) + 1
    resolved_profile = profile or (next(iter(profiles)) if len(profiles) == 1 else "mixed" if profiles else "unknown")
    return {"version": VERSION, "profile": resolved_profile, "samples": len(observations), "strategies": counts,
            "fallbacks": fallbacks, "scoreEffect": "none", "meaning": "AIA94-inspired backend observations; no score or verdict changes"}


def formal_signals(family: str, observation: dict[str, Any], *, label_path: str,
                   data_path: str | None) -> list[dict[str, Any]]:
    """Return only high-evidence signals eligible for the formal rule engine.

    AIA94's candidate regions are scientific observations, not defects.  They
    therefore stay out of this list unless an independent, explicit metadata
    condition is present.  This gate is intentionally conservative: a missing
    sidecar or an absent geometry record is reported as pending context rather
    than converted into a failure.
    """
    if observation.get("profile") != "enhanced":
        return []
    signals: list[dict[str, Any]] = []
    if family in IMAGE_FAMILIES:
        context = observation.get("metadataContext") or {}
        # The AIA94 pipeline explicitly repairs frames at this documented
        # exposure boundary.  A sidecar makes the value independently
        # measurable; a display-only PNG never reaches this branch.
        if context.get("metadataIndependent") and context.get("exposureAnomalyAtPoint02s"):
            sidecar = context.get("sidecar") or data_path or label_path
            signals.append({
                "checkId": "C06", "defectType": "L2-M", "status": "不通过",
                "locator": str(sidecar), "locatorType": "file",
                "basis": "独立元数据中的 EXPTIME≤0.02 s 触发 AIA94 曝光异常边界；该帧需要曝光修复后才能进入科学强度分析",
                "severityValue": 1, "severityMetric": "exposureAnomaly",
                "severityUnit": "frame", "tolerance": 0.0,
                "criticality": "一般", "affectedCount": 1,
                "observation": {"strategy": observation.get("strategy"), "strategyVersion": observation.get("version"),
                                 "metadataContext": context, "formalGate": "independent_sidecar_metadata"},
            })
    if family == "ULF_sample":
        context = observation.get("timeSeriesContext") or {}
        invalid = context.get("invalidTimeRows")
        if context.get("fullAxisScan") and isinstance(invalid, int) and invalid > 0:
            signals.append({
                "checkId": "C07", "defectType": "L2-T", "status": "不通过",
                "locator": str(data_path or label_path), "locatorType": "timeInterval",
                "basis": "完整 CSV 时间轴包含无法解析或不一致的时间行；该异常由原始时间轴直接核验得到",
                "severityValue": invalid, "severityMetric": "invalidTimeRows",
                "severityUnit": "row", "tolerance": 0.0,
                "criticality": "一般", "affectedCount": invalid,
                "observation": {"strategy": observation.get("strategy"), "strategyVersion": observation.get("version"),
                                 "timeSeriesContext": context, "formalGate": "full_csv_axis"},
            })
    return signals


__all__ = ["VERSION", "STRATEGIES", "StrategySpec", "strategy_for", "build_observation", "formal_signals", "summarize"]
