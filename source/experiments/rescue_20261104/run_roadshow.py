"""Fresh eight-class mechanical checks on every registered roadshow sample.

Frozen HGB outputs are displayed as coarse review signals, never fine truth.
This runner is additive: no data, previous outputs, weights or service is changed.
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

from spaceqa_agent import quality_ledger as q
from spaceqa_agent.engine import png_header

CONTRACTS = {
    "AIA_AR_sample": dict(labels=["AR"], source="参赛方任务类别契约：AIA_AR活动区；路演要求6.2以Coronal Hole示范类别不符。此契约不能验证集合内AR是否真实"),
    "HMI_AR_sample": dict(labels=["AR"], source="参赛方活动区任务类别契约；不等于独立科学类别参考"),
    "EPB_seg_sample": dict(labels=["EPB"], source="参赛方EPB目标掩膜类别契约；__OOV__是集合外标签；集合内EPB仍需图像复核"),
    "CME_C2_sample": dict(labels=["CME"], source="参赛方CME任务类别契约；不验证物理真实性"),
    "DMSP_SSUSI": dict(labels=["PCA", "TPA", "unclear"], source="现有公开任务标签集合；不验证集合内语义正确性"),
    # Other physically valid ULF classes are not automatically invalid Pi2 labels.
    "ULF_sample": None,
}


def fresh(row):
    data, label = Path(row["data_path"]), Path(row["label_path"])
    before = label.stat()
    raw = label.read_bytes(); ann = json.loads(raw.decode("utf-8-sig"))
    after = label.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Label changed during the fresh check")
    sample = {k: row[k] for k in ("id", "package", "family", "data_path", "label_path")}
    sample.update(label_sha256=hashlib.sha256(raw).hexdigest(), data_exists=data.is_file(),
                  data_sha256=row.get("data_sha256"), data_hash_verification="inventory SHA retained; fresh current PNG header/existence, not full-image rehash")
    if data.is_file() and data.suffix.lower() == ".png":
        m = png_header(data); sample["image_size"] = [m["width"], m["height"]]
    return sample, ann


def process(row):
    sample, annotation = fresh(row)
    contract = CONTRACTS.get(sample["family"])
    result = q.analyze_sample(sample, annotation,
        allowed_labels=contract["labels"] if contract else None,
        contract_source=contract["source"] if contract else None)
    return sample, result


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument("--inventory-root", type=Path)
    entry.add_argument("--input", type=Path, help="New unpacked package or directory of packages")
    parser.add_argument("--hgb-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package", action="append")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists(): raise ValueError("Choose an unused output directory")
    source_root = (args.inventory_root or args.input).resolve()
    for root in (source_root, args.hgb_root.resolve() if args.hgb_root else None):
        if root and (out == root or root in out.parents or out in root.parents): raise ValueError("Keep output outside input roots")
    out.mkdir(parents=True)
    started = time.monotonic(); cards = []; all_evidence = []; status_rows = []; severity = []; raw_results = []
    state = dict(version=q.VERSION, seed=q.SEED, startUTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        environment=dict(python=sys.version, platform=platform.platform()),
        scoringAuthority="participant observed technical density; official four-unit formula unavailable",
        modelInference="verified existing HGB prediction artifact reused; zero new expert/API calls",
        categoryContracts=CONTRACTS, packages=[])
    q.dump(out/"policy.json", dict(version=q.VERSION, classes=q.CLASSES, aliases={"L0-03": "L0-01"},
        dimensionCodes=q.DIMENSION_CODES, categoryContracts=CONTRACTS,
        scoreFormula="100*(1-union(confirmed eligible sample IDs)/fixed registered label count)",
        severity="missing file: 1; category: invalid target fraction; dimensions: max relative axis discrepancy; same type and metric only",
        unknown="retained in applicability denominators; missing baselines are never perfect or zero scores",
        ranking="observed technical density across packages; directory ID is an administrative tie break only",
        exclusions="HGB candidates, API opinions, image perturbations, metadata statistics and unsupported physical defects never score",
        officialFormula=None))
    if args.inventory_root:
        packages = [(p, None) for p in sorted(args.inventory_root.iterdir()) if (p/"inventory.jsonl").is_file()]
    else:
        roots = [args.input] if any(args.input.glob("*.json")) or (args.input/"label").is_dir() else [p for p in sorted(args.input.iterdir()) if p.is_dir()]
        packages = [(p, inventory_new_package(p)) for p in roots]
    for p, direct_rows in packages:
        if args.package and p.name not in args.package: continue
        tick = time.monotonic(); print("START", p.name, flush=True)
        original = direct_rows if direct_rows is not None else [json.loads(line) for line in (p/"inventory.jsonl").open() if line.strip()]
        if len({r["id"] for r in original}) != len(original): raise ValueError("Duplicate inventory IDs")
        rows = []; fresh_labels = []; failed = []
        for row in original:
            try:
                sample, result = process(row)
                rows.append(result); fresh_labels.append(dict(sampleId=sample["id"], label_path=sample["label_path"], label_sha256=sample["label_sha256"]))
            except (OSError, ValueError, TypeError, UnicodeError) as exc:
                # Missing/malformed inputs stay in the fixed denominator. Fail
                # the package rather than silently omit or invent a clean row.
                failed.append(dict(sampleId=row["id"], reason=str(exc)))
        if failed:
            q.dump(out/(p.name+"_failed.json"), failed)
            raise RuntimeError(f"{len(failed)} failed inputs retained; package is incomplete")
        target = out/"packages"/p.name; target.mkdir(parents=True)
        write_jsonl(target/"samples.jsonl", rows)
        write_jsonl(target/"input_hashes.jsonl", fresh_labels)
        findings = [f for r in rows for f in r["findings"]]
        write_jsonl(target/"findings.jsonl", findings)
        card = q.aggregate(rows)
        if args.hgb_root:
            pred = args.hgb_root/p.name/"predictions.jsonl"
            if pred.is_file():
                signals = [json.loads(line) for line in pred.open()]
                if {x["id"] for x in signals} != {x["sampleId"] for x in rows}: raise ValueError("HGB output/input IDs differ")
                card["hgbCoarseCandidateSamples"] = sum(any(x["model_candidate_bits"]) for x in signals)
                card["hgbArtifactSha256"] = q.file_sha(pred)
                card["hgbFineLocationsAvailable"] = False
                byid = {r["sampleId"]: r for r in rows}
                review_queue = []
                for x in signals:
                    if any(x["model_candidate_bits"]):
                        r = byid[x["id"]]
                        review_queue.append(dict(sampleId=x["id"], dirId=p.name, label_path=r["labelPath"],
                            data_path=r["dataPath"], family=r["family"], fusion_probabilities=x["probabilities"],
                            groups=x["model_candidate_bits"], confirmedClasses=sorted({f["defectType"] for f in r["findings"]}),
                            action="图像+label+专家信号复核；必须给细分类别和原始坐标，不能只复述G/C/D"))
                write_jsonl(target/"review_queue.jsonl", review_queue)
        card["freshMechanicalSeconds"] = time.monotonic()-tick
        q.dump(target/"scorecard.json", card)
        cards.append(card); severity.extend(card["severity"])
        all_evidence.extend(findings)
        for r in rows:
            for code, ch in r["checks"].items():
                status_rows.append(dict(dirId=p.name, sampleId=r["sampleId"], defectType=code, checkId=q.CHECKS[code],
                    ruleVersion=q.VERSION, verdict=ch["status"], scopeVerdict=ch["scopeStatus"],
                    fieldPath=r["labelPath"], detail=ch["reason"]))
        # Deterministic 10% source recomputation: same checks, fresh reads, exact
        # match. Not an independent scientific annotation-accuracy audit.
        selected = random.Random(q.SEED).sample(range(len(original)), min(len(original), max(3, math_ceil(len(original)*.1))))
        mismatches = []
        for i in selected:
            _, repeat = process(original[i])
            if repeat != rows[i]: mismatches.append(original[i]["id"])
        if mismatches: raise ValueError("Fresh deterministic source replay differs")
        state["packages"].append(dict(package=p.name, sampleCount=len(rows), conclusions=len(findings),
            replaySampleCount=len(selected), replayMismatches=len(mismatches),
            sampleLedgerSha256=q.file_sha(target/"samples.jsonl"), inventorySha256=q.file_sha(p/"inventory.jsonl") if direct_rows is None else q.digest(original),
            freshMechanicalSeconds=card["freshMechanicalSeconds"], secondsIncludingReplay=time.monotonic()-tick))
        print("DONE", p.name, len(rows), card["confirmedProblemSamples"], round(card["observedTechnicalScore"],3), flush=True)
    ranked = q.rank(cards)
    for card in ranked: q.dump(out/"packages"/card["dirId"]/"scorecard.json", card)
    q.dump(out/"scorecards.json", ranked)
    write_jsonl(out/"findings.jsonl", all_evidence)
    q.csv_write(out/"八类缺陷比例.csv", [dict(dirId=c["dirId"], defectType=code, **d) for c in ranked for code,d in c["defects"].items()])
    q.csv_write(out/"同类缺陷严重程度排序.csv", severity)
    q.csv_write(out/"四状态检查记录.csv", status_rows)
    q.csv_write(out/"六维评分与技术排序.csv", [dict(dirId=c["dirId"], family=c["family"],
        rankGlobalDisplay=c["rankGlobalDisplay"], rankInDatasetDisplay=c["rankInDatasetDisplay"],
        observedTechnicalScore=c["observedTechnicalScore"], fullQualityScore=None,
        confirmedProblemSamples=c["confirmedProblemSamples"], sampleCount=c["sampleCount"],
        scientificCheckCoverage=c["scientificCheckCoverage"], **{d["name"]+"完整得分":d["score"] for d in c["dimensions"]},
        **{d["name"]+"观测技术分":d["observedTechnicalScore"] for d in c["dimensions"]},
        tiedDirectoryIds=c["tiedDirectoryIds"]) for c in ranked])
    q.csv_write(out/"证据表_SpaceQA_八类整改版.csv", [dict(findingId=f["findingId"], checkId=f["checkId"],
        ruleVersion=f["ruleVersion"], dirId=f["dirId"], objectId=f["sampleId"],
        fieldPath=f["locator"].get("annotation_pointer"), locatorType=f["locator"]["type"], verdict=f["verdict"],
        detail=f["detail"], defectType=f["defectType"], locator=f["locator"], severity=f["severity"], evidence=f["evidence"]) for f in all_evidence])
    state.update(totalSeconds=time.monotonic()-started, finishUTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        status="completed", packageCount=len(cards), sampleCount=sum(c["sampleCount"] for c in cards),
        confirmedProblemSamples=sum(c["confirmedProblemSamples"] for c in cards),
        sourceHash=q.file_sha(Path(__file__)), ledgerModuleHash=q.file_sha(Path(q.__file__)))
    q.dump(out/"run.json", state)
    print(json.dumps({k:state[k] for k in ("status","packageCount","sampleCount","confirmedProblemSamples","totalSeconds")}), flush=True)


def math_ceil(x):
    import math
    return math.ceil(x)


def inventory_new_package(root):
    from spaceqa_agent.engine import family_name, label_reference
    root = root.resolve(); family = family_name(root.name)
    if family not in CONTRACTS: raise ValueError("Unregistered family: specify a supported six-family package name")
    data = collections.defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".csv"}: data[path.name].append(path)
    labels = sorted((root/"label").glob("*.json")) if (root/"label").is_dir() else sorted(root.glob("*.json"))
    rows = []
    for path in labels:
        ann = json.loads(path.read_text(encoding="utf-8-sig")); ref = label_reference(ann)
        if not ref: raise ValueError("Unresolved data reference: " + str(path))
        if Path(ref).name != ref or "\\" in ref: raise ValueError("Reference must be a basename under the registered package")
        matches = data.get(ref, [])
        if len(matches)>1: raise ValueError("Ambiguous data reference: " + ref)
        expected = matches[0] if matches else (root/"data"/ref if (root/"data").is_dir() else root/ref)
        rows.append(dict(id=root.name+":"+path.name, package=root.name, family=family,
                         data_path=str(expected), label_path=str(path)))
    if not rows: raise ValueError("No resolvable annotated samples: " + str(root))
    return rows


if __name__ == "__main__": main()
