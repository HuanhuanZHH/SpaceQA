#!/usr/bin/env python3
"""Verify reference artifacts and stage isolated, path-relocated research runs.

This launcher never edits the historical experiment files or original inputs.
The prepared-source cache is an explicit, private dependency. Staging uses links
for large files and copies/remaps metadata only; it does not download datasets.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY_ROOT = "/staff/tangjiarui/OPC"
RUN_NAME = "all-family-training-20261031"
MODEL_FILES = (
    "dinov2-small/config.json", "ijepa_vith14_1k/config.json",
    "Qwen3-VL-8B-Instruct/config.json",
    "image-model-pilot/yolo11n.pt", "image-model-pilot/yolo11n-seg.pt",
    "image-model-pilot/yolov8n-seg.pt",
    "image-model-pilot/resnet18-f37072fd.pth",
    "image-model-pilot/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
)
SOURCE_FILES = ("dataset.jsonl", "prepare_state.json", "registration_state.json")
SOURCE_DIRS = ("raw_views", "views", "registration", "blind-rules")


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def config_read(path: Path) -> dict:
    config = load(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction configuration schema")
    root = os.path.expandvars(config["external_root"])
    if "$" in root or not Path(root).is_absolute():
        raise ValueError("Set SPACEQA_EXTERNAL_ROOT to an absolute path, or set external_root in a *.local.json")
    result = dict(config, external_root=str(Path(root).resolve()))
    for key in ("manifest", "prepared_source", "models", "image_python", "qwen_python", "roadshow_source"):
        value = os.path.expandvars(str(config[key])).format(external_root=result["external_root"])
        if not Path(value).is_absolute() or "$" in value:
            raise ValueError(f"{key} must resolve to an absolute path")
        # Resolving the final python symlink would discard its virtualenv prefix.
        result[key] = str(Path(value).absolute() if key.endswith("_python") else Path(value).resolve())
    # Dataset paths inside private manifests may originate on a different host.
    result["path_mappings"] = {LEGACY_ROOT: result["external_root"], **config.get("path_mappings", {})}
    if result.get("gpu", "1") != "1":
        raise ValueError("The byte-frozen queue schedules GPU1. Use gpu=1; other GPU layouts need a separately registered runner.")
    return result


def preflight(config: dict, *, check_environment: bool = True) -> dict:
    required = [("manifest", Path(config["manifest"])), ("prepared_source", Path(config["prepared_source"]))]
    source = Path(config["prepared_source"])
    required += [("prepared_source/" + name, source / name) for name in SOURCE_FILES + SOURCE_DIRS]
    for name in ("06_问题清单/findings.jsonl", "07_证据索引/evidence_index.csv"):
        required.append(("blind_rules/" + name, source / "blind-rules" / name))
    required += [("models/" + name, Path(config["models"]) / name) for name in MODEL_FILES]
    required += [(key, Path(config[key])) for key in ("image_python", "qwen_python")]
    absent = [name for name, path in required if not path.exists()]
    errors = []
    for field, filename in (("expected_manifest_sha256", Path(config["manifest"])),
                            ("expected_source_dataset_sha256", source / "dataset.jsonl")):
        if config.get(field) and filename.is_file() and sha(filename) != config[field]:
            errors.append(field + " does not match")
    for name in ("prepare_state.json", "registration_state.json"):
        if (source / name).exists() and load(source / name).get("status") != "complete":
            errors.append(name + " is not complete")
    for folder in ("dinov2-small", "ijepa_vith14_1k", "Qwen3-VL-8B-Instruct"):
        model = Path(config["models"]) / folder
        if model.is_dir() and not any(model.glob("*.safetensors")) and not any(model.glob("pytorch_model*.bin")):
            errors.append(folder + " has config but no local model weights")
    environments = {}
    differences = []
    if check_environment:
        if os.name != "posix":
            errors.append("Historical GPU training requires Linux (fcntl/process-group semantics)")
        reference_environment = load(REPO / "results/reference/environments.json")
        expected_environments = reference_environment["environments"]
        package_names = sorted(set().union(*(set(values) for values in expected_environments.values())))
        probe = (
            "import json,sys,importlib.metadata as m; import torch; "
            "names=" + repr(package_names) + "; "
            "v={};\nfor n in names:\n try:v[n]=m.version(n)\n except m.PackageNotFoundError:v[n]=None\n"
            "print(json.dumps(dict(python=sys.version,python_version=sys.version.split()[0],torch_cuda=torch.version.cuda,packages=v,cuda_available=torch.cuda.is_available(),devices=torch.cuda.device_count())))"
        )
        for key in ("image_python", "qwen_python"):
            if not Path(config[key]).is_file():
                continue
            try:
                process = subprocess.run([config[key], "-c", probe], capture_output=True, text=True, timeout=45)
                if process.returncode:
                    errors.append(key + " import probe failed: " + process.stderr[-500:])
                else:
                    environments[key] = json.loads(process.stdout.strip().splitlines()[-1])
                    expected = expected_environments["image-models" if key == "image_python" else "qwen3vl"]
                    for package, version in expected.items():
                        actual = environments[key]["packages"].get(package)
                        if actual != version:
                            differences.append({"environment": key, "package": package,
                                                "expected": version, "actual": actual})
                    for field in ("python", "torch_cuda"):
                        actual_field = "python_version" if field == "python" else field
                        actual = environments[key][actual_field]
                        if actual != reference_environment[field]:
                            differences.append({"environment": key, "package": field,
                                                "expected": reference_environment[field], "actual": actual})
                    if not environments[key]["cuda_available"] or environments[key]["devices"] < 2:
                        errors.append(key + " must see CUDA GPU1")
            except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
                errors.append(key + " probe failed: " + str(exc))
    if differences:
        errors.append("Environment versions differ from reference; see version_differences")
    return {"status": "pass" if not absent and not errors else "fail", "missing": absent,
            "errors": errors, "environments": environments,
            "version_differences": differences,
            "prepared_source_regeneration": "External cache and registration inputs required; raw datasets alone are insufficient.",
            "retraining_bitwise_identical": False,
            "reason": "Historical GPU kernels and time/readiness-based encoder warmup are not bitwise deterministic."}


def remap(value, mapping: dict):
    if isinstance(value, dict):
        return {key: remap(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item, mapping) for item in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return new.rstrip("/") + value[len(old):]
    return value


def relocate_python(source: str, workspace: Path) -> str:
    """Only root-path string constants may differ in the generated source."""
    target = source.replace(LEGACY_ROOT, workspace.as_posix())

    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                node.value = node.value.replace(workspace.as_posix(), LEGACY_ROOT)
            return node

    if ast.dump(ast.parse(source)) != ast.dump(Normalize().visit(ast.parse(target))):
        raise ValueError("Relocation changed executable semantics beyond root-path strings")
    return target


def copy_jsonl(source: Path, target: Path, mapping: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8-sig") as incoming, target.open("x", encoding="utf-8", newline="\n") as outgoing:
        for line in incoming:
            if line.strip():
                outgoing.write(json.dumps(remap(json.loads(line), mapping), ensure_ascii=True) + "\n")


def stage_rule_run(source: Path, target: Path, mapping: dict) -> None:
    """Relocate evidence locators as well as dataset paths, without rewriting findings."""
    target.mkdir()
    for item in source.iterdir():
        if item.name != "07_证据索引":
            (target / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
            continue
        index_dir = target / item.name
        index_dir.mkdir()
        for evidence in item.iterdir():
            if evidence.name != "evidence_index.csv":
                (index_dir / evidence.name).symlink_to(evidence.resolve(), target_is_directory=evidence.is_dir())
                continue
            with evidence.open(encoding="utf-8-sig", newline="") as incoming:
                reader = csv.DictReader(incoming)
                with (index_dir / evidence.name).open("x", encoding="utf-8-sig", newline="") as outgoing:
                    writer = csv.DictWriter(outgoing, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    for record in reader:
                        record["locator"] = remap(record["locator"], mapping)
                        writer.writerow(record)


def stage(config: dict, workspace: Path) -> dict:
    workspace = workspace.resolve()
    if workspace.exists():
        raise FileExistsError("Workspace must be new; existing experiment outputs are never overwritten")
    report = preflight(config, check_environment=False)
    if report["status"] != "pass":
        raise ValueError("Missing prerequisites: " + json.dumps(report))
    if workspace == REPO or REPO in workspace.parents or workspace in REPO.parents:
        raise ValueError("Use a separate workspace outside the Git checkout")
    workspace.mkdir(parents=True)
    checkout = workspace / "opc-agent"
    checkout.mkdir()
    changes = {}
    for dirname in ("experiments", "tools", "spec", "agent"):
        for path in sorted((REPO / dirname).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix not in {".py", ".md", ".json", ".yaml", ".yml"}:
                continue
            dest = checkout / path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            # Package runtime and wrappers are intentionally not used by the frozen experiment.
            if path.relative_to(REPO).as_posix() == "tools/audit_run.py":
                continue
            if path.suffix == ".py":
                dest.write_text(relocate_python(data.decode("utf-8"), workspace), encoding="utf-8", newline="")
            else:
                dest.write_bytes(data)
            changes[str(path.relative_to(REPO))] = {"source_sha256": sha(path), "staged_sha256": sha(dest)}
    legacy = REPO / "reproducibility/legacy_runtime/opc_agent.py"
    shutil.copyfile(legacy, checkout / "opc_agent.py")
    annotation = workspace / "phase2_visual_audit_xzj/src/annotation_diff.py"
    annotation.parent.mkdir(parents=True)
    shutil.copyfile(REPO / "reproducibility/external_preprocessing/annotation_diff.py", annotation)
    manifest = workspace / "phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl"
    source_manifest = Path(config["manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    nontrivial_mapping = any(old != new for old, new in config["path_mappings"].items())
    if nontrivial_mapping:
        copy_jsonl(source_manifest, manifest, config["path_mappings"])
    else:
        manifest.symlink_to(source_manifest)
    source = Path(config["prepared_source"])
    staged_source = checkout / "runs/full-matrix-20261031"
    staged_source.mkdir(parents=True)
    dataset_mapping = {**config["path_mappings"],
                       LEGACY_ROOT + "/opc-agent/runs/full-matrix-20261031": source.as_posix()}
    for path in source.iterdir():
        if path.name == "dataset.jsonl":
            if any(old != new for old, new in dataset_mapping.items()):
                copy_jsonl(path, staged_source / path.name, dataset_mapping)
            else:
                (staged_source / path.name).symlink_to(path.resolve())
        elif path.name == "blind-rules" and nontrivial_mapping:
            stage_rule_run(path, staged_source / path.name, config["path_mappings"])
        elif path.name in SOURCE_FILES + SOURCE_DIRS:
            (staged_source / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (workspace / "models").symlink_to(Path(config["models"]), target_is_directory=True)
    roadshow = workspace / "OPC_路演测试数据/OPC_Roadshow_B_20261105"
    if Path(config["roadshow_source"]).is_dir():
        roadshow.parent.mkdir(parents=True)
        roadshow.symlink_to(config["roadshow_source"], target_is_directory=True)
    for name, key in ((".venv-image-models", "image_python"), (".venv-qwen3vl", "qwen_python")):
        interpreter = Path(config[key])
        if interpreter.parent.name != "bin":
            raise ValueError("Historical Linux runner expects virtualenv/bin/python: " + key)
        (workspace / name).symlink_to(interpreter.parent.parent, target_is_directory=True)
    extra_sources = {str(checkout / "opc_agent.py"): sha(checkout / "opc_agent.py"),
                     str(annotation): sha(annotation),
                     str(REPO / "scripts/roadshow_local.py"): sha(REPO / "scripts/roadshow_local.py"),
                     str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    receipt = {"schema_version": 1, "workspace": str(workspace), "source_revision": "2026-11-02",
               "legacy_source_is_unchanged": True, "relocations": changes,
               "extra_source_hashes": extra_sources,
               "private_source_dataset_sha256": sha(source / "dataset.jsonl"),
               "private_manifest_sha256": sha(Path(config["manifest"])),
               "config": config, "note": "New scientific run; no silent resume of historical checkpoints under relocated code."}
    write_new(workspace / "reproduction_receipt.json", receipt)
    return {"status": "staged", "workspace": str(workspace), "relocated_source_files": len(changes),
            "large_files": "linked, not copied", "receipt": str(workspace / "reproduction_receipt.json")}


def command_plan(config: dict, workspace: Path, phase: str) -> list[list[str]]:
    code = workspace / "opc-agent/experiments/all_family_training_20261031"
    image = config["image_python"]
    if phase == "prepare":
        return [[image, "-u", str(code / "prepare.py")]]
    if phase == "train":
        # Exact original order: every route is trained on all six families before its own test.
        return [[image, "-u", str(code / "runner.py")]]
    if phase == "evaluate":
        return [[image, "-u", str(code / "barrier.py")],
                [image, "-u", str(code / "evaluate.py"), "score"],
                [image, "-u", str(code / "validate.py"), "--results"]]
    if phase == "roadshow":
        # This explicit local driver never invokes the archived remote-GPU watcher.
        return [[image, "-u", str(REPO / "scripts/roadshow_local.py"),
                 "--workspace", str(workspace), "--gpu", "1"]]
    raise ValueError(phase)


def run(config: dict, workspace: Path, phase: str, dry_run: bool) -> dict:
    workspace = workspace.resolve()
    receipt = load(workspace / "reproduction_receipt.json")
    if receipt["config"] != config:
        raise ValueError("Config differs from staged receipt; create a new workspace")
    commands = command_plan(config, workspace, phase)
    if dry_run:
        return {"status": "dry_run", "phase": phase, "commands": commands}
    if os.name != "posix":
        raise ValueError("GPU research replay is supported on Linux only")
    environment_check = preflight(config)
    if environment_check["status"] != "pass":
        raise ValueError("Environment preflight failed: " + json.dumps(environment_check))
    for relative, hashes in receipt["relocations"].items():
        if sha(workspace / "opc-agent" / relative) != hashes["staged_sha256"]:
            raise ValueError("Staged code changed after receipt: " + relative)
    for filename, expected in receipt["extra_source_hashes"].items():
        if sha(Path(filename)) != expected:
            raise ValueError("Imported source changed after receipt: " + filename)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                       OPC_ALL_FAMILY_OUT=str(workspace / "opc-agent/runs" / RUN_NAME))
    for command in commands:
        subprocess.run(command, cwd=workspace / "opc-agent", env=environment, check=True)
    return {"status": "complete", "phase": phase}


def snapshot(manifest: Path) -> dict:
    if manifest.exists():
        raise FileExistsError("Snapshot exists. Explicitly select a new manifest for a new revision.")
    records = {}
    extensions = {".py", ".json", ".csv", ".md", ".sh", ".ps1", ".toml", ".yaml", ".yml", ".txt", ".html", ".css", ".js"}
    for path in sorted(REPO.rglob("*")):
        rel = path.relative_to(REPO)
        if any(part in {".git", "__pycache__", ".venv", "build", "dist", ".work"} or part.endswith(".egg-info") for part in rel.parts):
            continue
        if not path.is_file() or path == manifest or path.name.endswith(".local.json"):
            continue
        if path.suffix in extensions or path.name in {"Dockerfile", ".gitignore", ".gitattributes", ".dockerignore"}:
            records[rel.as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_new(manifest, {"schema_version": 1, "hash_algorithm": "sha256", "files": records})
    return {"status": "snapshotted", "files": len(records), "manifest": str(manifest)}


def verify(manifest: Path) -> dict:
    files = load(manifest)["files"]
    errors = []
    for relative, expected in files.items():
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO) or not path.is_file() or sha(path) != expected["sha256"]:
            errors.append(relative)
    frozen_checks = 0
    frozen_manifest = REPO / "results/reference/code_manifest.json"
    if frozen_manifest.exists():
        for relative, expected in load(frozen_manifest).items():
            path = REPO / "experiments/all_family_training_20261031" / relative
            if not path.is_file() or sha(path) != expected:
                errors.append("Frozen scientific source changed: " + relative)
            frozen_checks += 1
    # Recalculate each recorded confusion matrix without GPU or private data.
    matrix_checks = 0
    with (REPO / "results/reference/metrics_long.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            tp, fp, fn = (int(row[key]) for key in ("TP", "FP", "FN"))
            for field, expected in (("recall", tp / (tp + fn) if tp + fn else None),
                                    ("precision", tp / (tp + fp) if tp + fp else None),
                                    ("F1", 2 * tp / (2 * tp + fp + fn) if tp + fn else None)):
                if expected is None:
                    if row[field] != "":
                        errors.append(f"Invalid undefined {field}: {row['route']}/{row['family']}/{row['tier']}")
                elif not math.isclose(float(row[field]), expected, rel_tol=1e-12, abs_tol=1e-12):
                    errors.append(f"Metric mismatch: {row['route']}/{row['family']}/{row['tier']}/{field}")
                matrix_checks += 1
    result = {"status": "pass" if not errors else "fail", "files": len(files),
              "frozen_scientific_source_checks": frozen_checks,
              "confusion_metric_checks": matrix_checks, "errors": errors}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "snapshot"):
        sub = commands.add_parser(name)
        sub.add_argument("--manifest", type=Path, default=REPO / "configs/source_manifest.json")
    for name in ("preflight", "stage", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in ("stage", "run"):
            sub.add_argument("--workspace", type=Path, required=True)
        if name == "preflight":
            sub.add_argument("--files-only", action="store_true")
        if name == "run":
            sub.add_argument("--stage", choices=["prepare", "train", "evaluate", "roadshow"], required=True)
            sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "verify":
            result = verify(args.manifest)
        elif args.command == "snapshot":
            result = snapshot(args.manifest)
        else:
            config = config_read(args.config)
            if args.command == "preflight":
                result = preflight(config, check_environment=not args.files_only)
            elif args.command == "stage":
                result = stage(config, args.workspace)
            else:
                result = run(config, args.workspace, args.stage, args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("status") == "fail" else 0
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
