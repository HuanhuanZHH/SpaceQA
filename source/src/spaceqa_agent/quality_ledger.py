"""Eight-class, located quality ledger and reproducible participant scorecards.

The scientific state and the state of a narrower executable check are separate.
An in-vocabulary class, plausible polygon, or low model probability cannot prove
an annotation correct. No provider confidence is used as physical severity.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
from pathlib import Path

VERSION = "eight-class-ledger-1.0"
SEED = 20261104
DEFAULT_TOLERANCES = {
    "bbox": {"iouThreshold": 0.50, "normalizedCenterDistanceThreshold": 0.05},
    "polygon": {"iouThreshold": 0.50, "boundaryHausdorffPixel": 3.0},
    "timeInterval": {"endpointToleranceSeconds": 2.0, "overlapRatioThreshold": 0.50},
    "metadata": {"relativeDimensionError": 0.02},
}
CLASSES = {
    "L0-01": "数据文件缺失", "L2-G": "边界与位置偏差",
    "L2-P": "类别标注错误", "L2-T": "时段起止时刻抖动",
    "L2-F": "元数据尺寸矛盾", "L3-06": "漏标", "L3-07": "虚标",
    "L3-MIS": "覆盖不足或覆盖过度",
}
CHECKS = {c: f"C{i:02d}" for i, c in enumerate(CLASSES, 31)}
GROUPS = {c: "G" if c in {"L2-G", "L3-MIS"} else "C" if c in {"L3-06", "L3-07"} else "D" for c in CLASSES}
DIMENSIONS = {
    "completeness": "完整性", "annotationQuality": "标注质量",
    "sampleCorrespondence": "样本对应", "coverageDistribution": "覆盖与分布",
    "traceability": "文档与来源追溯", "usageConditions": "使用条件",
}
DIMENSION_CODES = {
    "completeness": ["L0-01"],
    "annotationQuality": ["L2-G", "L2-P", "L2-T", "L2-F", "L3-06", "L3-07", "L3-MIS"],
    "sampleCorrespondence": ["L0-01", "L2-F"],
    "coverageDistribution": [], "traceability": [], "usageConditions": [],
}


def canonical(code):
    return "L0-01" if code == "L0-03" else code


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def walk(value, pointer=""):
    if isinstance(value, dict):
        yield pointer, value
        for k, v in value.items():
            yield from walk(v, pointer + "/" + k.replace("~", "~0").replace("/", "~1"))
    elif isinstance(value, list):
        for i, v in enumerate(value): yield from walk(v, pointer + "/" + str(i))


def objects(annotation):
    """Only current instance arrays; event summaries are not extra instances."""
    for pointer, value in walk(annotation):
        for key in ("matched_events", "shapes"):
            if key in value and isinstance(value[key], list):
                for i, obj in enumerate(value[key]):
                    if isinstance(obj, dict): yield pointer + "/" + key + "/" + str(i), obj


def category(obj):
    for k in ("label", "event_type", "kind"):
        if isinstance(obj.get(k), str): return k, obj[k]
    return None, None


def geometry(obj):
    pts = obj.get("bbox_polygon_pixel", obj.get("points"))
    if not isinstance(pts, list) or len(pts) < 2: return None
    if not all(isinstance(p, list) and len(p) == 2 and all(finite(x) for x in p) for p in pts): return None
    return [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]


def locate(sample, pointer=None, obj=None):
    loc = {"type": "annotationRecord" if pointer is not None else "file",
           "path": sample["label_path"], "annotation_pointer": pointer}
    box = geometry(obj or {})
    if box is not None:
        loc.update(type="imageRegion", bbox=box, coordinate_system="image_pixels",
                   origin="top_left", axis_order="xy", image_size=sample.get("image_size"),
                   data_path=sample["data_path"])
    return loc


def finding(sample, code, locator, detail, severity, *, source="rule", stage="confirmed", measurements=None, aliases=None, probability=None, impact=None):
    code = canonical(code)
    identity = digest([sample["id"], code, locator, detail])[:20]
    check = CHECKS.get(code, "C39")
    return dict(findingId=check + "-" + identity, checkId=check, ruleVersion=VERSION,
                dirId=sample["package"], sampleId=sample["id"], defectType=code,
                aliasCodes=aliases or [], verdict="fail" if stage == "confirmed" else "undetermined",
                findingStage=stage, sourceKind=source, scoreEligible=stage == "confirmed" and code in CLASSES,
                locator=locator, detail=detail, severity=severity, measurements=measurements or {},
                probability=(1.0 if probability is None and stage == "confirmed" else probability),
                impact=impact or {"label": "待根据任务语义复核影响"},
                evidence=dict(label_path=sample["label_path"], data_path=sample["data_path"],
                              label_sha256=sample.get("label_sha256"),
                              data_sha256_from_inventory=sample.get("data_sha256"),
                              data_hash_verification=sample.get("data_hash_verification", "not_rehashed")))


def analyze_sample(sample, annotation, *, allowed_labels=None, contract_source=None):
    """Execute mechanical scopes; preserve scientific unknowns for all eight classes."""
    nobj = list(objects(annotation))
    image = sample.get("image_size")
    data_present = bool(sample["data_exists"])
    is_csv = sample["data_path"].lower().endswith(".csv")
    checks = {c: dict(status="undetermined", scopeStatus="undetermined",
                      reason="需要独立目标/事件参考或经验证的定位判断") for c in CLASSES}
    rows = []
    checks["L0-01"] = dict(status="pass" if data_present else "fail", scopeStatus="pass" if data_present else "fail",
                           reason="标注引用文件的实盘存在性；L0-03合并为同一项")
    if not data_present:
        loc = locate(sample); loc["expected_data_path"] = sample["data_path"]
        rows.append(finding(sample, "L0-01", loc, "标注存在，对应数据文件缺失",
                            dict(metric="missing_file", value=1., unit="file", normalized=1.), aliases=["L0-03"]))
    # No image dimensions on a CSV, or on an AIA label without dimension fields.
    dimension_checks = []
    if image and data_present:
        for pointer, value in walk(annotation):
            # Object-local width/height may describe a box, never the whole image.
            if any(x in pointer.split("/") for x in ("matched_events", "shapes", "objects", "events")): continue
            declared = []
            if isinstance(value.get("imgSize"), str):
                import re
                m = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*(?:pixels?|px|像素)?\s*", value["imgSize"], re.I)
                if m: declared.append((pointer + "/imgSize", [int(x) for x in m.groups()]))
            for w, h in (("imageWidth", "imageHeight"), ("imgWidth", "imgHeight"), ("image_width", "image_height")):
                if w in value and h in value and all(finite(value[k]) and value[k] > 0 for k in (w, h)):
                    declared.append((pointer, [value[w], value[h]]))
            for ptr, wh in declared:
                bad = list(wh) != list(image); dimension_checks.append(bad)
                if bad:
                    error = max(abs(wh[i] - image[i]) / image[i] for i in range(2))
                    rows.append(finding(sample, "L2-F", locate(sample, ptr),
                        f"声明尺寸{wh}，PNG实测尺寸{image}",
                        dict(metric="max_relative_dimension_error", value=error, unit="ratio", normalized=min(1., error)),
                        measurements=dict(declared=wh, actual=image, max_pixel_error=max(abs(wh[i]-image[i]) for i in range(2)))))
    if dimension_checks:
        checks["L2-F"] = dict(status="fail" if any(dimension_checks) else "pass", scopeStatus="fail" if any(dimension_checks) else "pass",
                              reason="声明的整图尺寸与当前PNG头逐字段比较")
    elif is_csv:
        checks["L2-F"] = dict(status="not_applicable", scopeStatus="not_applicable", reason="CSV没有图像尺寸")
    elif data_present and image:
        checks["L2-F"] = dict(status="not_applicable", scopeStatus="not_applicable", reason="当前标注未声明整图尺寸；不因字段缺省报尺寸矛盾")
    if is_csv:
        for c in ("L2-G", "L3-MIS"):
            checks[c] = dict(status="not_applicable", scopeStatus="not_applicable", reason="此版定义为图像空间边界/面积；时序端点另列L2-T")
    # An explicit, recorded participant vocabulary contract is narrower than
    # scientific class correctness. Passing this scope never passes L2-P.
    cats = []; bad_cats = 0
    if allowed_labels is not None:
        if not contract_source: raise ValueError("A category contract requires its source/basis")
        for ptr, obj in nobj:
            key, label = category(obj)
            if label is None: continue
            cats.append(label)
            if label not in allowed_labels:
                bad_cats += 1
                rows.append(finding(sample, "L2-P", locate(sample, ptr + "/" + key, obj),
                    f"类别{label!r}不属于声明的任务类别集合{sorted(allowed_labels)}；依据：{contract_source}",
                    dict(metric="invalid_category_instance", value=1., unit="object", normalized=None),
                    measurements=dict(actual=label, allowed=sorted(allowed_labels), contract_source=contract_source,
                                      affected_object_fraction=None)))
        checks["L2-P"] = dict(status="fail" if bad_cats else "undetermined", scopeStatus="fail" if bad_cats else "pass",
                              reason="已核验类别契约；集合内类别的语义正确性仍待判")
        for row in rows:
            if row["defectType"] == "L2-P":
                row["severity"].update(metric="affected_object_fraction", value=bad_cats / max(1, len(cats)), unit="ratio", normalized=bad_cats / max(1, len(cats)))
                row["measurements"].update(invalid_object_count=bad_cats, classified_object_count=len(cats), affected_object_fraction=bad_cats / max(1, len(cats)))
    # Structural contradictions are diagnostics, not proof of scientific jitter,
    # target displacement, missing or spurious targets.
    diagnostics = []
    from .engine import parse_time
    for ptr, obj in nobj:
        box = geometry(obj)
        if image and box and (box[0] < 0 or box[1] < 0 or box[2] > image[0] or box[3] > image[1]):
            diagnostics.append(dict(type="geometry_out_of_frame", pointer=ptr, bbox=box))
        for a, b in (("start_dt", "end_dt"), ("event_starttime", "event_endtime")):
            t0, t1 = parse_time(obj.get(a)), parse_time(obj.get(b))
            if t0 and t1 and t1 < t0:
                diagnostics.append(dict(type="reversed_event_interval", pointer=ptr, start=obj[a], end=obj[b], seconds=(t0-t1).total_seconds()))
    family = sample["family"]
    basic=annotation.get("basic_annotation") or {};science=annotation.get("science_annotation") or {}
    observation_start=parse_time(basic.get("startTime")) or parse_time(science.get("dateObs")) or parse_time(basic.get("timestamp"))
    observation_end=parse_time(basic.get("endTime")) or observation_start
    return dict(sampleId=sample["id"], dirId=sample["package"], family=family, checks=checks,
                findings=rows, diagnostics=diagnostics, dataPresent=data_present, objectCount=len(nobj),
                categoryHistogram=dict(collections.Counter(category(o)[1] or "missing" for _, o in nobj)),
                emptyAnnotation=not nobj, labelSha256=sample.get("label_sha256"),
                imageSize=image, labelPath=sample["label_path"], dataPath=sample["data_path"],
                sourceVersion=(annotation.get("label_header") or {}).get("labelVersion"),
                provenanceAuthor=(annotation.get("label_header") or {}).get("annotator"),
                observationInterval=[observation_start.isoformat(),observation_end.isoformat()] if observation_start and observation_end else None)


def attach_candidates(result, candidates):
    """Normalize existing model/LLM instance findings without score promotion."""
    seen = set()
    for c in candidates:
        code = canonical(c.get("defectType", c.get("defect_class")))
        if code not in CLASSES: raise ValueError("Unknown fine defect category")
        loc = c.get("locator")
        if not isinstance(loc, dict) or not loc: raise ValueError("A fine candidate needs a locator")
        sid = c.get("sampleId", c.get("sample_id", result["sampleId"]))
        if sid != result["sampleId"]: raise ValueError("Candidate escapes the current sample")
        if code in {"L2-G", "L3-06", "L3-07", "L3-MIS"} and not (loc.get("bbox") or loc.get("polygon") or loc.get("timeInterval")):
            raise ValueError("A visual/event candidate needs spatial or temporal coordinates")
        key = digest([code, loc])
        if key in seen: continue
        seen.add(key)
        probability = c.get("probability", c.get("confidence", c.get("fusionClassProbability", c.get("fusion_class_probability"))))
        if not finite(probability):
            probability = None
        result["findings"].append(dict(c, defectType=code, findingStage="candidate", verdict="undetermined", scoreEligible=False,
                                       probability=probability, impact=c.get("impact") or {"label": "模型候选需复核"}))
    return result


def compare_instances(result, payload):
    """Connect existing spatial/temporal experts to fine classes and severity.

    The independent predictor sees the raw observation, not the current label.
    A model reference stays candidate-only even when it reports high confidence.
    Actual adjudication belongs to the hash-bound human/reference review tool.
    """
    from .instance_audit import audit_annotations
    if payload.get("sample_id") != result["sampleId"]:
        raise ValueError("Instance comparison must use the current sample ID")
    comparison = audit_annotations(payload)
    candidates = []
    for c in comparison["candidates"]:
        metrics = c.get("metrics") or {}; kind = c["candidate_kind"]
        severity = None
        if kind == "boundary_position_deviation" and finite(metrics.get("normalized_distance")):
            severity = dict(metric="centroid_distance_over_image_diagonal", value=metrics["normalized_distance"], unit="ratio")
        elif kind == "undercoverage" and finite(metrics.get("coverage_reference")):
            severity = dict(metric="uncovered_reference_fraction", value=1-metrics["coverage_reference"], unit="ratio")
        elif kind == "overcoverage" and finite(metrics.get("coverage_annotation")):
            severity = dict(metric="excess_annotation_fraction", value=1-metrics["coverage_annotation"], unit="ratio")
        elif kind == "time_boundary_jitter" and all(finite(metrics.get(k)) for k in ("start_error_seconds", "end_error_seconds")):
            severity = dict(metric="max_endpoint_error", value=max(metrics["start_error_seconds"], metrics["end_error_seconds"]), unit="second")
        elif kind in {"missing_annotation", "spurious_annotation"}:
            severity = dict(metric="unmatched_target_count", value=1., unit="object")
        candidates.append(dict(c, findingId=c["id"], checkId=CHECKS[c["defect_class"]],
            ruleVersion=VERSION, dirId=result["dirId"], sampleId=result["sampleId"],
            sourceKind=payload.get("reference",{}).get("source","model"), severity=severity,
            severityBasis="相对独立预测对象的程序测量；参考本身未获科学确认，不能视作真实污染程度"))
    attach_candidates(result, candidates)
    result["instanceComparison"] = comparison
    return result


def aggregate(results):
    if not results: raise ValueError("Empty packages are retained as unscorable, not perfect")
    ids = [r["sampleId"] for r in results]
    if len(ids) != len(set(ids)): raise ValueError("Duplicate samples would corrupt denominators")
    n = len(results); counts = {}; severity_rows = []; confirmed_ids = set()
    for code in CLASSES:
        by_sample = {}; state = collections.Counter(r["checks"][code]["status"] for r in results)
        scope = collections.Counter(r["checks"][code]["scopeStatus"] for r in results)
        candidates = set(); instances = []
        for r in results:
            findings = {f.get("findingId", digest(f)): f for f in r["findings"] if canonical(f.get("defectType")) == code}
            for f in findings.values():
                if f.get("findingStage") == "confirmed" and f.get("scoreEligible") is True:
                    by_sample.setdefault(r["sampleId"], []).append(f); instances.append(f); confirmed_ids.add(r["sampleId"])
                else: candidates.add(r["sampleId"])
        applicable = n - state["not_applicable"]; unknown = state["undetermined"]
        confirmed = len(by_sample)
        counts[code] = dict(name=CLASSES[code], denominator=n, applicableSamples=applicable,
            confirmedSamples=confirmed, locatedInstances=len(instances), candidateSamples=len(candidates),
            statusCounts={s: state[s] for s in ("pass", "fail", "undetermined", "not_applicable")},
            scopeStatusCounts={s: scope[s] for s in ("pass", "fail", "undetermined", "not_applicable")},
            observedConfirmedProportion=confirmed / n,
            proportionEstimate=confirmed / n if unknown == 0 else None,
            proportionBounds=[confirmed / n, (confirmed + unknown) / n],
            relativeError=None, relativeErrorReason="缺独立逐样本真值，不能用模型输出自证误差",
            scientificCheckCompletion=(applicable - unknown) / applicable if applicable else None)
        for sid, fs in by_sample.items():
            valid = [f for f in fs if finite((f.get("severity") or {}).get("value"))]
            if not valid: continue
            metrics = {(f["severity"]["metric"], f["severity"]["unit"]) for f in valid}
            for metric, unit in metrics:
                selected = [f for f in valid if (f["severity"]["metric"], f["severity"]["unit"]) == (metric, unit)]
                severity_rows.append(dict(dirId=results[0]["dirId"], sampleId=sid, defectType=code,
                    metric=metric, unit=unit, severity=max(f["severity"]["value"] for f in selected),
                    findingIds=[f["findingId"] for f in selected],
                    orderingBasis="同一缺陷、同一计量定义与单位；值越大越严重，置信度不参与"))
    # Same sample counts once in the density score. Severity is a separate axis
    # so increasing defect density can never perversely improve this score.
    score = 100 * (1 - len(confirmed_ids) / n)
    dimensions = []
    for dim, label in DIMENSIONS.items():
        codes = DIMENSION_CODES[dim]
        relevant = [r for r in results if any(r["checks"][c]["status"] != "not_applicable" for c in codes)] if codes else []
        fail_ids = {r["sampleId"] for r in results if any(f.get("scoreEligible") and f.get("findingStage") == "confirmed" and f.get("defectType") in codes for f in r["findings"])}
        requested = sum(counts[c]["applicableSamples"] for c in codes)
        completed = sum(counts[c]["applicableSamples"] - counts[c]["statusCounts"]["undetermined"] for c in codes)
        observed = 100 * (1 - len(fail_ids) / len(relevant)) if relevant else None
        dimensions.append(dict(id=dim, name=label, defectCodes=codes, confirmedSamples=len(fail_ids),
            denominator=len(relevant), observedTechnicalScore=observed,
            score=observed if requested and requested == completed else None,
            state="fail" if fail_ids else "pass" if requested and requested == completed else "undetermined",
            checkCoverage=completed / requested if requested else None,
            applicableCheckDenominator=requested, completedChecks=completed,
            reason="仅当前可验证检查范围的技术分；完整维度分须覆盖全部适用检查" if codes else
                   "有统计与用途结论；缺预期覆盖基准/可核验来源/用途验收阈值，不虚构数值"))
    req = sum(x["applicableSamples"] for x in counts.values())
    done = sum(x["applicableSamples"] - x["statusCounts"]["undetermined"] for x in counts.values())
    histogram = collections.Counter()
    for r in results: histogram.update(r["categoryHistogram"])
    from .engine import parse_time
    windows=sorted((parse_time(r["observationInterval"][0]),parse_time(r["observationInterval"][1])) for r in results if r.get("observationInterval"))
    gaps=[];overlaps=0
    if windows:
        covered_until=windows[0][1]
        for a,b in windows[1:]:
            gap=(a-covered_until).total_seconds()
            if gap>0:gaps.append(gap)
            elif gap<0:overlaps+=1
            covered_until=max(covered_until,b)
    temporal=dict(observedWindows=len(windows),first=windows[0][0].isoformat() if windows else None,
        last=max(b for _,b in windows).isoformat() if windows else None,
        positiveGapCount=len(gaps),maxPositiveGapSeconds=max(gaps) if gaps else None,
        overlappingWindows=overlaps,
        interpretation="标注声明的观测时间分布；缺连续覆盖目标，不把抽样日期间隔直接当缺陷")
    for code in CLASSES:
        rows = [r for r in severity_rows if r["defectType"] == code]
        for metric in {r["metric"] for r in rows}:
            ordered = sorted([r for r in rows if r["metric"] == metric], key=lambda r: (-r["severity"], r["sampleId"]))
            for row in ordered: row["severityRank"] = 1 + sum(x["severity"] > row["severity"] for x in ordered)
    # Probability based pollution indices are intentionally separate from the
    # confirmed technical score.  A candidate may contribute to this
    # diagnostic index without becoming a confirmed defect.
    try:
        from .pollution_metrics import compute_pollution_index
        pollution_records = []
        for row in results:
            probs = {}
            confirmed_codes = []
            for f in row.get("findings", []):
                code = canonical(f.get("defectType"))
                value = f.get("probability", f.get("confidence", f.get("fusionClassProbability", f.get("fusion_class_probability"))))
                if code in CLASSES and finite(value): probs[code] = max(probs.get(code, 0.0), float(value))
                if code in CLASSES and f.get("findingStage") == "confirmed" and f.get("scoreEligible") is True: confirmed_codes.append(code)
            pollution_records.append({"sampleId": row["sampleId"], "defectProbabilities": probs, "confirmedCodes": confirmed_codes})
        pollution = compute_pollution_index(pollution_records, sample_count=n)
    except Exception as exc:
        pollution = {"pollutionIndex": None, "error": str(exc)}
    return dict(version=VERSION, dirId=results[0]["dirId"], family=results[0]["family"], sampleCount=n,
        confirmedProblemSamples=len(confirmed_ids), observedTechnicalScore=score,
        fullQualityScore=None, officialScore=None, officialPrecision=None, officialRecall=None, officialF1=None,
        scoringFormula="100*(1-distinct confirmed eligible problem samples / all registered label samples)",
        scoreMeaning="已实现范围内的观测技术分；不是软件得分、完整质量分或官方q",
        defects=counts, dimensions=dimensions, severity=severity_rows,
        scientificCheckCoverage=done / req if req else None, applicableChecks=req, completedChecks=done,
        distribution=dict(emptyAnnotations=sum(r["emptyAnnotation"] for r in results),
            emptyAnnotationRatio=sum(r["emptyAnnotation"] for r in results)/n,
            objectCount=sum(r["objectCount"] for r in results),
            meanObjectDensity=math.fsum(r["objectCount"] for r in results)/n,
            categoryHistogram=dict(histogram),
            temporal=temporal,
            interpretation="空标注与分布差异是描述统计；没有目标/事件参考时不等于漏标或虚标"),
        provenance=dict(labelVersions=dict(collections.Counter(str(r.get("sourceVersion")) for r in results)),
                        anonymousAuthors=sum(r.get("provenanceAuthor") in {None, "", "anonymous"} for r in results),
                        interpretation="匿名化按发放约定处理；不能当作来源错误扣分"),
        usage=dict(acceptance="undetermined", ingestion="conditional", selection="conditional",
                   advice="隔离缺文件/契约冲突记录；定位疑点进入人工复核后再用于科学选数"),
        pollution=pollution,
        localizationPolicy={"tolerances": DEFAULT_TOLERANCES, "referenceRequired": True,
                            "withoutReference": "candidate; 不将模型输出自证为命中"},
        expertOutlierPolicy={"method": "median_mad", "zThreshold": 3.5, "minimumExperts": 3,
                             "aggregate": "trimmed_mean", "recordExcludedExperts": True})


def rank(cards):
    """Reproducible technical ordering, with genuine ties explicitly retained."""
    # Prefer the formal pollution index whenever a card has one.  Older cards
    # retain the technical-score fallback and disclose that basis.
    if all(c.get("pollution", {}).get("pollutionIndex") is not None for c in cards):
        ordered = sorted(cards, key=lambda c: (c["pollution"]["pollutionIndex"], c["dirId"]))
        basis = "pollutionIndex ascending"
    else:
        ordered = sorted(cards, key=lambda c: (-c["observedTechnicalScore"], c["dirId"]))
        basis = "observedTechnicalScore descending; pollution unavailable"
    families = collections.Counter()
    for index, c in enumerate(ordered, 1):
        families[c["family"]] += 1
        c.update(rankGlobalDisplay=index, rankInDatasetDisplay=families[c["family"]],
                 technicalCompetitionRank=1+sum(o["observedTechnicalScore"] > c["observedTechnicalScore"] for o in ordered),
                 tiedDirectoryIds=[o["dirId"] for o in ordered if o["dirId"] != c["dirId"] and o["observedTechnicalScore"] == c["observedTechnicalScore"]],
                 rankAuthority=basis + "; 同分目录名顺序仅供显示，不能声称真实质量不并列")
    return ordered


def csv_write(path, rows):
    if not rows: return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists(): parser.error("Choose a new output file")
    rows = [json.loads(line) for line in args.sample_results.read_text(encoding="utf-8").splitlines() if line.strip()]
    dump(args.output, aggregate(rows))


if __name__ == "__main__": main()
