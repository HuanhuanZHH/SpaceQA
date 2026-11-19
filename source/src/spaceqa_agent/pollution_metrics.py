"""Eight-class pollution indices and reproducible ranking."""
from __future__ import annotations

import math
from typing import Any


DEFECT_CODES = ("L0-01", "L2-G", "L2-P", "L2-T", "L2-F", "L3-06", "L3-07", "L3-MIS")
DEFAULT_THRESHOLDS = {code: 0.50 for code in DEFECT_CODES}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _probability(row: dict[str, Any], code: str) -> float | None:
    probs = row.get("defectProbabilities") or row.get("probabilities") or {}
    if isinstance(probs, dict) and _finite(probs.get(code)):
        return max(0.0, min(1.0, float(probs[code])))
    if code in set(row.get("confirmedCodes") or ()):
        # A deterministic finding is certainty for this observed score.  This
        # is explicitly different from an official recall/precision label.
        return 1.0
    return None


def compute_pollution_index(records: list[dict[str, Any]], *, thresholds: dict[str, float] | None = None, sample_count: int | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_THRESHOLDS); cfg.update({k: float(v) for k, v in (thresholds or {}).items() if _finite(v)})
    n = int(sample_count if sample_count is not None else len(records))
    n = max(0, n)
    defects = {}
    for code in DEFECT_CODES:
        probabilities = []
        missing_probability = 0
        for row in records:
            value = _probability(row, code)
            if value is None:
                missing_probability += 1
                continue
            if value >= cfg[code]:
                probabilities.append((str(row.get("sampleId", "")), value))
        probability_sum = sum(v for _, v in probabilities)
        defects[code] = {"threshold": cfg[code], "overThresholdSamples": len(probabilities), "probabilitySum": probability_sum,
                         "pollutionIndexContribution": 100.0 * probability_sum / n if n else None,
                         "missingProbabilitySamples": missing_probability,
                         "sampleDetails": [{"sampleId": sid, "probability": value} for sid, value in probabilities]}
    contributions = [x["pollutionIndexContribution"] for x in defects.values() if _finite(x["pollutionIndexContribution"])]
    return {"formula": "100 * sum(probability_i,c for probability_i,c >= threshold_c) / N", "sampleCount": n, "thresholds": cfg,
            "defects": defects, "pollutionIndex": sum(contributions) if len(contributions) == len(DEFECT_CODES) else None,
            "rankingDirection": "ascending", "meaning": "模型/规则观测污染指数；越低越好。八类缺陷允许同一样本重复计入；没有独立真值时不是官方污染率。"}


def rank_pollution(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort packages by pollution index, retaining genuine ties."""
    usable = sorted(cards, key=lambda row: (float(row["pollutionIndex"]) if _finite(row.get("pollutionIndex")) else float("inf"), str(row.get("datasetId", row.get("dirId", "")))))
    ranks = []
    for index, row in enumerate(usable, 1):
        value = row.get("pollutionIndex")
        rank = 1 + sum(_finite(other.get("pollutionIndex")) and _finite(value) and float(other["pollutionIndex"]) < float(value) for other in usable)
        ranks.append({**row, "rank": rank, "tiedDatasetIds": [other.get("datasetId", other.get("dirId")) for other in usable if other is not row and other.get("pollutionIndex") == value]})
    return ranks
