#!/usr/bin/env python3
"""Explicit local replay of verified roadshow archives; no remote scheduler."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    code = workspace / "opc-agent/experiments/roadshow_evaluation_20261101"
    sys.path.insert(0, str(code))
    spec = importlib.util.spec_from_file_location("spaceqa_local_roadshow", code / "roadshow.py")
    roadshow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roadshow)
    roadshow.OUT.mkdir(parents=True, exist_ok=True)
    integrity = roadshow.check_archives()
    if integrity["verified"] != integrity["expected_packages"]:
        raise ValueError("All roadshow archives must match the provided SHA256SUMS before local replay")
    frozen = roadshow.freeze()
    policy = roadshow.OUT / "dispatch_current.json"
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for item in integrity["packages"]:
        roadshow.prepare_package(item)
        # A new process per package releases model memory before the next package.
        command = [sys.executable, "-u", str(code / "roadshow.py"), "worker",
                   "--package", item["package"], "--gpu", args.gpu, "--dispatch", str(policy)]
        subprocess.run(command, check=True, env=environment)
        folder = roadshow.OUT / "packages" / item["package"] / "models"
        if not all((folder / (name + ".complete.json")).exists() for name in frozen["experts"]):
            raise RuntimeError("Worker could not complete; the selected GPU may be occupied")
    roadshow.summarize(frozen)
    roadshow.status("replay_complete_without_ground_truth", evaluated_packages=len(integrity["packages"]),
                    compared_models=len(frozen["experts"]), F1=None,
                    reason="Independent scientific defect truth is unavailable; aggregate candidate replay only")


if __name__ == "__main__":
    main()
