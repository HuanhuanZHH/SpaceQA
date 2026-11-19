"""Bounded six-family public replay; no fitting and no B selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from portable_experts import FAMILIES, RESEARCH_ROOT, read, rows, sha, write, write_rows, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--source", choices=("public", "roadshow"), default="public")
    parser.add_argument("--roadshow-root", type=Path, default=Path("/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101"))
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(RESEARCH_ROOT) and not out.exists(), "New research-only output required")
    out.mkdir(parents=True)
    lock = read(args.lock)
    training = Path(lock["training_root"])
    chosen = {family: [] for family in FAMILIES}
    # Selection uses only type, package, availability and ordering. Truth,
    # predictions and label contents never determine selected samples.
    inventories = [training / "inventory.jsonl"] if args.source == "public" else sorted((args.roadshow_root / "packages").glob("*/inventory.jsonl"))
    for inventory, row in ((p,r) for p in inventories for r in rows(p)):
        family = row["family"]
        if (args.source == "public" and row["tier"] != "P1") or row["input_missing"] or len(chosen[family]) >= 2:
            continue
        row["package_id"] = row.get("package_id", row.get("package"))
        if chosen[family] and row["package_id"] != chosen[family][0]["package_id"]:
            continue
        keys = ("id", "family", "package_id", "label_path", "image", "input_sha256", "features", "prompt", "input_missing")
        chosen[family].append({k: row[k] for k in keys})
    unavailable = [family for family, values in chosen.items() if len(values) == 0]
    if args.source == "public":
        require(not unavailable, "Six-family sample selection incomplete")
    chosen = {family:values for family,values in chosen.items() if values}
    require(chosen and all(len(v) == 2 for v in chosen.values()), "Available-family sample selection incomplete")
    ids = {r["id"] for group in chosen.values() for r in group}
    needed = {r["route"] for family in chosen for r in lock["family_routes"][family]["experts"]}
    references = {}
    source_hashes = {}
    for route in sorted(needed):
        if args.source == "public":
            paths = [(Path(lock["policy_root"]) / "gbdt/predictions" if route == "gbdt_prior_v1" else training / "predictions") / (route + ".jsonl")]
        else:
            packages = {sample[0]["package_id"] for family,sample in chosen.items() if any(e["route"] == route for e in lock["family_routes"][family]["experts"])}
            paths = [(Path(lock["policy_root"]) / "gbdt/roadshow_predictions" / (package + ".jsonl")
                      if route == "gbdt_prior_v1" else args.roadshow_root / "packages" / package / "models" / (route + ".jsonl")) for package in sorted(packages)]
        reference = out / ("reference_" + route + ".jsonl")
        kept = [r for path in paths for r in rows(path) if r["id"] in ids]
        write_rows(reference, kept)
        references[route] = reference
        source_hashes.update({str(path):sha(path) for path in paths})
    write(out / "fixture_provenance.json", dict(selection="first two available P1 rows per family; same package" if args.source == "public" else "first two available rows per family; first sorted package; no truth selection", records=len(ids), unavailable_families=unavailable,
           source_inventory_sha256={str(p):sha(p) for p in inventories}, prediction_sha256=source_hashes,
           source=args.source, B_inputs_opened=args.source == "roadshow", B_fitted=False, targets_used=False, ground_truth_available=False))
    summary = []
    for family, samples in chosen.items():
        reference = out / (family + ".reference.jsonl")
        write_rows(reference, samples)
        selection = out / (family + ".selection.json")
        write(selection, [r["label_path"] for r in samples])
        label = Path(samples[0]["label_path"])
        package = label.parent.parent if label.parent.name == "label" else label.parent
        command = [sys.executable, str(Path(__file__).with_name("portable_experts.py")), "run",
            "--input", str(package), "--family", family, "--output", str(out / family), "--lock", str(args.lock),
            "--device", args.device, "--batch-size", str(args.batch_size), "--release-profile", "public-v1" if args.source == "public" else "roadshow-b-v1",
            "--selection-json", str(selection), "--reference-inventory", str(reference), "--probability-atol", ".02"]
        for item in lock["family_routes"][family]["experts"]:
            command += ["--reference-prediction", item["route"] + "=" + str(references[item["route"]])]
        begin = time.monotonic()
        process = subprocess.run(command, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}, capture_output=True, text=True)
        (out / (family + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        completion = out / family / "complete.json"
        item = read(completion) if completion.exists() else {"status": "failed"}
        item.update(family=family, returncode=process.returncode, process_seconds=time.monotonic()-begin)
        summary.append(item)
        write(out / "progress.json", dict(status="running", finished=len(summary), families=summary))
        print(json.dumps(item), flush=True)
    passed = all(x["returncode"] == 0 and x["status"] == "passed_research_smoke" for x in summary)
    write(out / "complete.json", dict(status=("passed_six_family_smoke" if args.source == "public" else "passed_available_family_smoke") if passed else "failed_smoke",
          samples=len(ids), unavailable_families=unavailable, families=summary, B_fitted=False, thresholds_changed=False, official_metrics_computed=False,
          full_dataset_evaluation=False, batch_size=args.batch_size, device=args.device))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
