"""Recalculate the historical 12-package receipt when SSH is unavailable.

This is NOT a fresh source-data scan or a new inference experiment. Only facts
present in the frozen receipt are used; absent instance counts stay unknown.
"""
import argparse
import collections
import json
from pathlib import Path, PurePosixPath
from spaceqa_agent import quality_ledger as q


def replay(snapshot, old_report, output):
    if output.exists(): raise ValueError("Choose an unused output directory")
    output.mkdir(parents=True)
    source=json.loads(snapshot.read_text(encoding="utf-8-sig"))
    snapshot_sha=q.file_sha(snapshot)
    old={x["package"]:x for x in json.loads(old_report.read_text(encoding="utf-8-sig"))["rows"]}
    cards=[]; allfindings=[]; allstates=[]; allseverity=[]; samples_by_package={}
    rejected=[]
    for package in source["packages"]:
        tools={t["tool"]:t for t in package["toolResults"]}
        inv=tools["inventory_package"]; rules=tools["check_rules"]
        name=rules["scorecards"][0]["datasetId"]
        baseline=old[name];family=baseline["family"]
        samples={}
        for s in inv["samples"]:
            state={c:dict(status="undetermined",scopeStatus="undetermined",reason="旧快照没有该类独立定位判断") for c in q.CLASSES}
            state["L0-01"]=dict(status="pass",scopeStatus="pass",reason="历史配对检查，无新实盘复查")
            state["L2-F"]=dict(status="pass",scopeStatus="pass",reason="历史EPB整图尺寸检查，无新实盘复查") if family.startswith("EPB") else dict(status="not_applicable",scopeStatus="not_applicable",reason="CSV无尺寸/AIA未声明整图尺寸")
            if family.startswith("ULF"):
                for c in ("L2-G","L3-MIS"):state[c]=dict(status="not_applicable",scopeStatus="not_applicable",reason="本版图像空间定义不适用于CSV")
            samples[s["sampleId"]]=dict(sampleId=s["sampleId"],dirId=name,family=family,checks=state,findings=[],
                emptyAnnotation=False,objectCount=0,categoryHistogram={},labelPath=s["labelPath"],dataPath=None,
                sourceVersion=None,provenanceAuthor=None)
        for f in rules["findings"]:
            code=q.canonical(f.get("defectType") or (f.get("legacyFinding") or {}).get("defectType"));sid=f.get("sampleId")
            if code not in q.CLASSES or not f.get("scoreEligible") or f.get("verdict")!="fail":continue
            if sid not in samples:
                rejected.append(dict(dirId=name,findingId=f["findingId"],reason="样本映射不唯一，保留而不换算分母"));continue
            loc=f["locator"];metric=(f.get("measurements") or {}).get("severity",{})
            severity=dict(metric=metric.get("metric"),value=metric.get("value"),unit=metric.get("unit"),normalized=None)
            # Old class severity=1 was merely a finding count. It does not
            # establish the affected-object fraction. Never fabricate a value.
            if code=="L2-P":severity=None
            item=dict(findingId=f["findingId"],checkId=f["checkId"],ruleVersion=f["ruleVersion"],
                dirId=name,sampleId=sid,defectType=code,aliasCodes=["L0-03"] if code=="L0-01" else [],
                findingStage="confirmed",scoreEligible=True,sourceKind="historical_rule_receipt",verdict="fail",
                locator=dict(type=loc.get("locatorType"),path=loc.get("value")),detail=f.get("basis"),severity=severity,
                measurements=f.get("measurements"),evidence=dict(snapshot=str(snapshot),snapshotSha256=snapshot_sha,
                    evidenceRefs=f.get("evidenceRefs"),freshSourceRecheck=False))
            samples[sid]["findings"].append(item)
            samples[sid]["checks"][code]=dict(status="fail",scopeStatus="fail",reason="继承历史规则证据；新鲜原始数据与实际定位仍需复核")
        results=list(samples.values());card=q.aggregate(results)
        # The frozen receipt supplies real distribution aggregates. It does not
        # supply the original annotation JSON for reconstructing every target.
        dist=tools.get("check_distribution",{}).get("distribution")
        card["distribution"]=dict(source="historical check_distribution",raw=dist)
        card["provenance"]=dict(source="historical check_provenance",raw=tools.get("check_provenance",{}).get("dimensions"))
        card.update(replayOnly=True,hgbCoarseCandidateSamples=baseline["modelCandidateSamples"],
                    hgbFineLocationsAvailable=False,freshMechanicalSeconds=None,
                    previousConfirmedSamples=baseline["confirmedSamples"],
                    previousScore=baseline["technicalScore"])
        if card["confirmedProblemSamples"]!=baseline["confirmedSamples"]:
            card["priorCountDifference"]=card["confirmedProblemSamples"]-baseline["confirmedSamples"]
            card["priorCountDifferenceReason"]="仅映射当前八类与逐样本证据；历史聚合与本账本差异保留待查"
            print("COUNT DIFFERENCE",name,card["confirmedProblemSamples"],baseline["confirmedSamples"])
        cards.append(card);allfindings.extend(f for r in results for f in r["findings"]);allseverity.extend(card["severity"])
        samples_by_package[name]=results
        for r in results:
            for code,ch in r["checks"].items():allstates.append(dict(dirId=name,sampleId=r["sampleId"],defectType=code,checkId=q.CHECKS[code],ruleVersion=q.VERSION,verdict=ch["status"],scopeVerdict=ch["scopeStatus"],fieldPath=r["labelPath"],detail=ch["reason"]))
    ranked=q.rank(cards)
    q.dump(output/"scorecards.json",ranked)
    for name,rows in samples_by_package.items():
        p=output/"packages"/name;p.mkdir(parents=True)
        with (p/"samples.jsonl").open("w",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    q.csv_write(output/"八类缺陷比例_快照重算.csv",[dict(dirId=c["dirId"],defectType=code,**d) for c in ranked for code,d in c["defects"].items()])
    q.csv_write(output/"严重程度_已有数值证据.csv",allseverity)
    q.csv_write(output/"四状态检查记录.csv",allstates)
    q.csv_write(output/"六维评分与技术排序.csv",[dict(dirId=c["dirId"],family=c["family"],rankGlobalDisplay=c["rankGlobalDisplay"],rankInDatasetDisplay=c["rankInDatasetDisplay"],observedTechnicalScore=c["observedTechnicalScore"],fullQualityScore=None,confirmedProblemSamples=c["confirmedProblemSamples"],sampleCount=c["sampleCount"],scientificCheckCoverage=c["scientificCheckCoverage"],**{d["name"]+"完整得分":d["score"] for d in c["dimensions"]},**{d["name"]+"观测技术分":d["observedTechnicalScore"] for d in c["dimensions"]},tiedDirectoryIds=c["tiedDirectoryIds"]) for c in ranked])
    with (output/"findings.jsonl").open("w",encoding="utf-8") as f:
        for row in allfindings:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    q.dump(output/"run.json",dict(status="frozen_snapshot_recomputed",freshSourceScan=False,newModelInference=False,
        version=q.VERSION,seed=q.SEED,snapshot=str(snapshot),snapshotSha256=snapshot_sha,
        oldReportSha256=q.file_sha(old_report),packages=len(cards),samples=sum(c["sampleCount"] for c in cards),
        confirmedProblemSamples=sum(c["confirmedProblemSamples"] for c in cards),rejected=rejected,
        source=q.file_sha(Path(__file__)),ledgerModule=q.file_sha(Path(q.__file__)),
        limitation="SSH中断后基于本地完整历史回执复算；不是新鲜源数据扫描或新的准确率实验"))
    print(json.dumps({"packages":len(cards),"samples":sum(c["sampleCount"] for c in cards),"confirmed":sum(c["confirmedProblemSamples"] for c in cards)},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",required=True,type=Path)
    parser.add_argument("--old-report",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args();replay(a.snapshot,a.old_report,a.output)
