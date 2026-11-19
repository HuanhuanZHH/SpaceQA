"""Summarize verified small forwards without inventing accuracy or time claims."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from portable_experts import RESEARCH_ROOT, read, require, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(RESEARCH_ROOT) and not args.output.exists(), "New research summary required")
    summaries = []
    accepted_samples = gpu_samples = cpu_samples = 0
    labels = set()
    current_entry_sha = sha(Path(__file__).with_name("portable_experts.py"))
    for folder in args.runs:
        report = read(folder / "complete.json")
        group = dict(run=str(folder), status=report["status"], report_sha256=sha(folder / "complete.json"),
                     samples=report.get("samples"), batch_size=report.get("batch_size"),
                     unavailable_families=report.get("unavailable_families", []), families=[])
        individual = "family" in report and "fresh_forward_samples" in report
        items = [report] if individual else report.get("families", [])
        for item in items:
            item_dir = folder if individual else folder / item["family"]
            check = read(item_dir / "reference_comparison.json")
            request = read(item_dir / "request.json")
            deltas = [x["max_abs_probability_delta"] for x in check["probabilities"] if x["max_abs_probability_delta"] is not None]
            counts = item["fresh_forward_samples"]
            accepted_samples += sum(counts.values())
            cpu_samples += sum(n for route,n in counts.items() if route == "gbdt_prior_v1" or item["device"] == "cpu")
            gpu_samples += sum(n for route,n in counts.items() if route != "gbdt_prior_v1" and item["device"].startswith("cuda:"))
            for record in (json.loads(line) for line in (item_dir / "inventory.jsonl").read_text().splitlines() if line.strip()):
                if not record["input_missing"]:
                    labels.add(record["label_path"])
            group["families"].append(dict(family=item["family"], status=item["status"],
                samples=item["records"], fresh_forward_batches=item["fresh_forward_batches"],
                fresh_forward_samples=counts,
                gpu_peak_allocated_bytes=item["gpu_peak_allocated_bytes"], seconds=item["seconds"],
                max_abs_probability_delta=max(deltas, default=None),
                all_frozen_candidate_bits_equal=all(x["frozen_candidate_bits_equal"] for x in check["probabilities"]),
                exact_preprocessing_equal=all(all(v for k,v in row.items() if k.endswith("equal")) for row in check["preprocessing"]),
                input_hashes_unchanged=item["input_hashes_unchanged"],
                executed_entry_sha256=request["source_sha256"],
                executed_entry_matches_current=request["source_sha256"] == current_entry_sha))
        if "checks" in report:
            group["checks"] = [{"case": x["case"], "passed": x["passed"]} for x in report["checks"]]
        summaries.append(group)
    tests = subprocess.run([sys.executable, str(Path(__file__).with_name("test_portable_experts.py"))], capture_output=True, text=True)
    attempts = []
    for completion in sorted(RESEARCH_ROOT.glob("portable_*/**/complete.json")):
        record = read(completion)
        if "fresh_forward_samples" in record:
            attempts.append(dict(run=str(completion.parent), status=record["status"],
                                 fresh_samples=sum(record["fresh_forward_samples"].values())))
    result = dict(version="portable-validation-summary-2026.11.03-2", runs=summaries,
                  counts=dict(accepted_comparison_sample_forwards=accepted_samples,
                              accepted_gpu_sample_forwards=gpu_samples, accepted_cpu_tree_sample_evaluations=cpu_samples,
                              distinct_original_label_records=len(labels),
                              all_preserved_attempt_sample_forwards=sum(a["fresh_samples"] for a in attempts)),
                  preserved_attempts=attempts,
                  latest_source_tests=dict(returncode=tests.returncode, passed=tests.returncode == 0,
                                           stdout=tests.stdout, stderr=tests.stderr, expected_test_count=7),
                  provenance_note="Historical requests retain their executed source SHA. Current source adds guards and is independently tested; differing source SHAs are not relabeled as current-code model runs.",
                  official_metrics_computed=False, full_dataset_evaluation=False, retraining=False,
                  thresholds_changed=False, cached_probability_reuse=0,
                  hardware="icarus0 A100 40GB; existing Qwen service preserved; GPU1 not used",
                  code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    result["documentation_sha256"] = sha(Path(__file__).with_name("PORTABLE_EXPERTS.md"))
    write(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
