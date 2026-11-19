"""Synthetic missing-input checks without running a neural model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from portable_experts import RESEARCH_ROOT, read, require, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(RESEARCH_ROOT) and not out.exists(), "New research output required")
    out.mkdir(parents=True)
    from PIL import Image
    missing_label = out / "fixtures/orphan"
    missing_label.mkdir(parents=True)
    Image.new("RGB", (8,8), "white").save(missing_label / "orphan.png")
    missing_data = out / "fixtures/missing_data"
    (missing_data / "label").mkdir(parents=True)
    (missing_data / "data").mkdir()
    write(missing_data / "label/absent.png.json", {})
    checks = []
    for name, source, key in [("label_absent", missing_label, "source_label_absent"), ("data_absent", missing_data, "source_data_absent")]:
        target = out / name
        command = [sys.executable, str(Path(__file__).with_name("portable_experts.py")), "run", "--input", str(source),
                   "--family", "AIA_AR_sample", "--output", str(target), "--lock", str(args.lock),
                   "--device", "cpu", "--batch-size", "1", "--release-profile", "public-v1"]
        process = subprocess.run(command, capture_output=True, text=True)
        (out / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        complete = read(target / "complete.json") if (target / "complete.json").exists() else {}
        checks.append(dict(case=name, returncode=process.returncode, completion=complete,
            passed=process.returncode == 0 and complete.get(key) == 1 and complete.get("inferred") == 0 and complete.get("abstentions") == 1 and not any(complete.get("fresh_forward_batches", {}).values())))
    passed = all(x["passed"] for x in checks)
    write(out / "complete.json", dict(status="passed" if passed else "failed", synthetic_inputs_only=True, checks=checks))
    print(json.dumps(dict(status="passed" if passed else "failed", checks=checks)))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
