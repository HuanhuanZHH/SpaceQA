"""Export small, independently audited runtime metadata without input data/weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(root):
    state_path = root / "external_supervisor_state_v2.json"
    state = read(state_path)
    final_path = root / "external_supervisor_complete_v2.json"
    # The final worker can exit less than two seconds after the last periodic
    # snapshot. The supervisor writes a final completion receipt separately.
    if final_path.is_file():
        final = read(final_path)
        state = {"status": final["status"], "active": [], "pending": [],
                 "completed": final["results"], "updated_unix": final_path.stat().st_mtime}
        state_path = final_path
    results = []
    for folder in sorted((root / "runs").iterdir()):
        paths = {key: folder / name for key, name in {
            "execution": "complete.json",
            "external_receipt": "external_runtime_receipt_v2.json",
            "independent_audit": "independent_audit_external_v2.json",
        }.items()}
        if not all(path.is_file() for path in paths.values()):
            continue
        bundle = {key: read(path) for key, path in paths.items()}
        execution = bundle["execution"]
        receipt = bundle["external_receipt"]
        is_gpu = receipt["cpu_slot"] is None
        physical_mapping_ok = (set(receipt["actual_gpu_uuids_seen"]) == {receipt["expected_physical_gpu0_uuid"]}) if is_gpu else not receipt["actual_gpu_uuids_seen"]
        passed = (
            receipt["under_3h_by_external_wall"] is True
            and receipt["timeout_or_guard_failure"] is None
            and bundle["independent_audit"]["status"] == "pass"
            and not bundle["independent_audit"]["errors"]
            and physical_mapping_ok
            and (receipt["exit_code"] == 0 or receipt["adopted_existing_worker"])
        )
        results.append({
            "package": folder.name,
            "family": execution["package"]["family"],
            "tier": execution["package"]["tier"],
            "release_profile": execution["package"]["release_profile"],
            "source_bytes": execution["package"]["counts"]["bytes"],
            "coverage": execution["coverage"],
            "stage_times": execution["stage_times"],
            "resource": execution["resource"],
            "expert_routes": bundle["independent_audit"]["model_route_names"],
            "fresh_forward_batches": execution["expert_completion"]["fresh_forward_batches"],
            "external_receipt": bundle["external_receipt"],
            "independent_audit": bundle["independent_audit"],
            "measured_host_pass": passed,
            "actual_physical_mapping_verified": physical_mapping_ok,
            "adopted_exit_code_limitation": receipt["adopted_existing_worker"] and receipt["exit_code"] is None,
            "source_artifact_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths.values()
            },
            "rules_and_fresh_candidates_integrated": execution["rules_and_fresh_candidates_integrated"],
            "A10_acceptance_established": False,
            "official_accuracy_metrics": None,
        })
    planned_count = len(read(root / "plan_v1.json")["packages"])
    passed_count = sum(row["measured_host_pass"] for row in results)
    return {
        "status": state["status"],
        "authoritative_live_state": str(state_path),
        "updated_unix": state["updated_unix"],
        "active": state["active"],
        "pending": state["pending"],
        "completed_count": len(results),
        "planned_count": planned_count,
        "passed_measured_host_count": passed_count,
        "all_planned_packages_passed_on_measured_host": passed_count == planned_count and not state["active"] and not state["pending"],
        "results": results,
        "limit_scope": "Each individual complete worker process, not sum of package times",
        "limitations": [
            "Measured A100 host; no A10 hardware acceptance.",
            "Fresh expert candidates remain separate from rules scorecards.",
            "No new official localized F1, recall or precision is established.",
            "OS page cache is not flushed; shared-host resource contention is recorded.",
            "Adopted workers have no wait exit code; successful products, independent audit and conservative exit interval are retained instead.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(collect(args.root), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
