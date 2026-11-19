"""Robust fusion and auditable expert outlier handling."""
from __future__ import annotations

import math
from statistics import mean, median
from typing import Any


DEFAULT_POLICY = {"method": "median_mad", "z_threshold": 3.5, "minimum_experts": 3, "aggregate": "trimmed_mean", "record_excluded_experts": True}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def robust_fuse(expert_probabilities: dict[str, float], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Filter extreme expert probabilities and aggregate the remainder.

    MAD is used by default because a single overconfident expert should not
    move the reference centre used to judge itself.  If too few values are
    available, all finite experts are retained and the fallback is recorded.
    """
    cfg = dict(DEFAULT_POLICY); cfg.update(policy or {})
    values = {str(k): float(v) for k, v in expert_probabilities.items() if _finite(v)}
    if not values:
        return {"probability": None, "keptExperts": [], "excludedExperts": [], "statistics": {"reason": "no_finite_expert_output"}, "policy": cfg}
    names, nums = list(values), list(values.values())
    centre = median(nums)
    deviations = [abs(v - centre) for v in nums]
    mad = median(deviations)
    scale = max(1e-12, 1.4826 * mad)
    threshold = float(cfg.get("z_threshold", 3.5))
    excluded = []
    kept = []
    # Do not remove the only available evidence, and do not apply a robust
    # rule to tiny ensembles where it would be unstable.
    if len(nums) >= int(cfg.get("minimum_experts", 3)) and mad > 0:
        for name, value in values.items():
            robust_z = abs(value - centre) / scale
            (excluded if robust_z > threshold else kept).append({"name": name, "value": value, "robustZ": robust_z, "reason": "median_mad_threshold"} if robust_z > threshold else name)
    else:
        kept = names
    kept_names = [x["name"] if isinstance(x, dict) else x for x in kept]
    kept_values = [values[name] for name in kept_names]
    if not kept_values:
        kept_names, kept_values, excluded = names, nums, []
    return {"probability": sum(kept_values) / len(kept_values), "keptExperts": kept_names, "excludedExperts": excluded,
            "statistics": {"centreMedian": centre, "mad": mad, "scale": scale, "threshold": threshold, "expertCount": len(nums), "keptCount": len(kept_values)}, "policy": cfg}
