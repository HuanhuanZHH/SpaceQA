"""Verify this research archive's byte hashes and aggregate confusion formulas.

This check reads only tracked-sized archive artifacts. It does not read private
datasets, run training, regenerate predictions, or validate scientific truth.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


ARCHIVE = Path(__file__).resolve().parent
REPO = ARCHIVE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(value):
    if value in (None, "", "null", "None"):
        return None
    return float(value)


def verify_counts(value, where, errors):
    checks = 0
    for prefix in ("", "all_tiers_", "G_", "C_", "D_"):
        keys = [prefix + name for name in ("TP", "FP", "FN", "TN")]
        if not all(key in value and numeric(value[key]) is not None for key in keys):
            continue
        tp, fp, fn, tn = (numeric(value[key]) for key in keys)
        if min(tp, fp, fn, tn) < 0 or any(x != int(x) for x in (tp, fp, fn, tn)):
            errors.append(where + ": invalid confusion count")
        wanted = dict(precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                      F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None, FPR=fp/(fp+tn) if fp+tn else None)
        for key, expected in wanted.items():
            if prefix + key not in value:
                continue
            observed = numeric(value[prefix+key])
            if (key == "FPR" and not prefix and "/gbdt/" in where
                    and any(name in where for name in ("/metrics.json/", "/complete.json/"))):
                # Standalone GBDT reports SAMPLE FPR over fully-known negative
                # samples; its TP/FP/FN/TN are BIT counts. Do not conflate them.
                if observed is not None and not 0 <= observed <= 1:
                    errors.append(where + ": sample FPR outside [0,1]")
                continue
            # Some historical tables leave F1 undefined for no positive truth,
            # whereas others report zero when false positives remain.
            valid = (observed is None and (expected is None or key == "F1" and tp+fn == 0)) or (
                observed is not None and expected is not None and math.isclose(observed, expected, abs_tol=1e-10, rel_tol=1e-9))
            if not valid:
                errors.append(where + ": " + prefix + key + " formula mismatch")
            checks += 1
    return checks


def verify_json(value, where, errors):
    if isinstance(value, dict):
        return verify_counts(value, where, errors) + sum(verify_json(v, where+"/"+str(k), errors) for k,v in value.items())
    if isinstance(value, list):
        return sum(verify_json(v, where+"/"+str(i), errors) for i,v in enumerate(value))
    return 0


def main():
    manifest = json.loads((ARCHIVE / "archive_manifest.json").read_text(encoding="utf-8"))
    errors, checks = [], 0
    for record in manifest["files"]:
        path = REPO / record["path"]
        if not path.is_file() or sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            errors.append("Missing or changed archive file: " + record["path"])
            continue
        if path.suffix == ".json":
            checks += verify_json(json.loads(path.read_text(encoding="utf-8-sig")), record["path"], errors)
        elif path.suffix == ".csv":
            for i, row in enumerate(csv.DictReader(path.open(encoding="utf-8-sig", newline=""))):
                checks += verify_counts(row, record["path"]+":"+str(i+2), errors)
    root = REPO / "results/reference/prior_optimization_20261102"
    base = root / "comparison_v1/frozen_validation_policy.json"
    extension = root / "gbdt_extension_v1/frozen_validation_extension.json"
    headwise = root / "headwise_v1/frozen_validation_policy.json"
    rules = json.loads(extension.read_text(encoding="utf-8-sig"))
    links = [(rules["base_policy_sha256"], base), (rules["gbdt_training_sha256"], root/"gbdt/training.json"),
             (rules["gbdt_training_barrier_sha256"], root/"gbdt/training_barrier.json")]
    for folder, policy in (("integrated_default", extension), ("integrated_review", headwise)):
        summary = json.loads((root/folder/"integration_summary.json").read_text(encoding="utf-8-sig"))
        links.append((summary["frozenPolicySha256"], policy))
        if summary["groundTruthAvailable"] or summary["F1"] is not None or summary["officialScore"] is not None:
            errors.append(folder + ": roadshow truth/score must remain unavailable")
    for expected, path in links:
        if sha(path) != expected:
            errors.append("Frozen policy/source link mismatch: " + str(path))
    report = dict(status="pass" if not errors else "fail", files=len(manifest["files"]),
                  confusion_formula_checks=checks, frozen_link_checks=len(links), errors=errors,
                  standalone_gbdt_sample_FPR="range checked only; private fully-known negative sample counts required for recomputation",
                  private_inputs_read=False, scientific_truth_verified=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
