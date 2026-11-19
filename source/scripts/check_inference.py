#!/usr/bin/env python3
"""Read-only GPU replay of frozen experts against historical probabilities.

Only aggregate counts and numerical differences are written. This is a small
checkpoint/inference check, not a new accuracy evaluation or a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def subset_summary(indices, actual, expected, thresholds, np, tolerance):
    if not indices:
        return {"samples": 0, "max_probability_delta": None,
                "within_tolerance": False, "threshold_decisions_match": False}
    current = actual[indices]
    historical = expected[indices]
    cutoffs = thresholds[indices]
    delta = np.abs(current - historical)
    changed = (current >= cutoffs) != (historical >= cutoffs)
    return {"samples": len(indices), "max_probability_delta": float(delta.max()),
            "mean_probability_delta": float(delta.mean()),
            "within_tolerance": bool((delta <= tolerance).all()),
            "threshold_decisions_match": not bool(changed.any()),
            "changed_decision_bits": int(changed.sum())}


def check(args):
    workspace = args.workspace.resolve()
    run = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError("Choose a new output file outside the historical run")
    code = workspace / "opc-agent/experiments/all_family_training_20261031"
    if not (workspace / "reproduction_receipt.json").is_file() or not (code / "models.py").is_file():
        raise ValueError("--workspace must be an already staged reproduction workspace")
    policy_path = REPO / "configs/dispatch_20_models.json"
    selected = read(policy_path)["selected"]
    routes = sorted({value["route"] for value in selected.values()} - {"rules"})
    if len(routes) != 4 or "qlora" in routes:
        raise ValueError("Expected the frozen four-expert static policy")

    # Set all offline and cache controls before importing optional ML libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="spaceqa-inference-", dir=output.parent) as cache:
        os.environ["YOLO_CONFIG_DIR"] = cache
        os.environ["HF_HOME"] = cache
        os.environ["TORCH_HOME"] = cache
        import numpy as np
        import torch
        from PIL import Image

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU result is substituted")
        sys.path.insert(0, str(code))
        common = load_module("common", code / "common.py")
        if Path(common.ROOT).resolve() != workspace or Path(common.REPO).resolve() != workspace / "opc-agent":
            raise ValueError("Staged common.py does not point at the isolated workspace")
        models = load_module("models", code / "models.py")
        batch_size = 16
        image_size = (384, 768)
        summaries = []

        for route in routes:
            print(json.dumps({"stage": "checking_frozen_expert", "route": route}), flush=True)
            checkpoint = run / "checkpoints" / (route + ".pt")
            report = read(run / "training" / (route + ".json"))
            expected_sha = report["checkpoint_hashes"]["checkpoints/" + route + ".pt"]
            if sha(checkpoint) != expected_sha:
                raise ValueError("Historical checkpoint hash mismatch for " + route)
            # Saved prediction membership is used only to choose available
            # replay examples, never to choose an expert or tune a threshold.
            success = {item["id"] for item in rows(run / "predictions" / (route + ".jsonl"))
                       if item.get("status") == "success"}
            selected_families = [family for family, value in selected.items() if value["route"] == route]
            chosen = {family: [] for family in selected_families}
            baseline = []
            positions = set()
            present_index = 0
            for item in rows(run / "inventory.jsonl"):
                if item["input_missing"]:
                    continue
                if present_index < batch_size:
                    baseline.append(item["id"])
                    positions.add(0)
                family = item["family"]
                if (family in chosen and len(chosen[family]) < 2 and item.get("split") == "test"
                        and item["id"] in success):
                    chosen[family].append(item["id"])
                    positions.add((present_index // batch_size) * batch_size)
                present_index += 1
            if any(not values for values in chosen.values()):
                raise ValueError("A selected family has no available held-out replay examples")
            # Recreate the whole historical 16-sample batch around each chosen
            # test example, rather than changing its BF16 batch context.
            batches = {start: [] for start in positions}
            present_index = 0
            for item in rows(run / "inventory.jsonl"):
                if item["input_missing"]:
                    continue
                start = (present_index // batch_size) * batch_size
                if start in batches:
                    batches[start].append(item)
                present_index += 1
            samples = [item for start in sorted(batches) for item in batches[start]]
            wanted = {item["id"] for item in samples}
            saved_predictions = {item["id"]: item for item in rows(run / "predictions" / (route + ".jsonl"))
                                 if item["id"] in wanted}
            if set(saved_predictions) != wanted:
                raise ValueError("Historical probabilities are incomplete for " + route)
            for item in samples:
                old = saved_predictions[item["id"]]
                if old["status"] != "success" or old["input_sha256"] != item["input_sha256"]:
                    raise ValueError("Historical prediction/input identity mismatch for " + route)

            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = models.build(route, pretrained=False, widths=saved.get("widths"),
                                 hidden=saved.get("hidden")).cuda().eval()
            model.load_state_dict(saved["state"], strict=True)
            mean = saved["mean"].cuda()
            std = saved["std"].cuda()
            values = []
            with torch.inference_mode():
                for start in sorted(batches):
                    image_arrays = []
                    for item in batches[start]:
                        with Image.open(item["image"]) as image:
                            image_arrays.append(np.array(image.resize((image_size[1], image_size[0]),
                                                Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255)
                    images = torch.from_numpy(np.stack(image_arrays)).cuda()
                    features = torch.tensor([item["features"] for item in batches[start]], dtype=torch.float32).cuda()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = model(images, (features - mean) / std)
                    values.append(logits.float().sigmoid().cpu().numpy())
            actual = np.concatenate(values)
            expected = np.array([saved_predictions[item["id"]]["probabilities"] for item in samples])
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("Invalid inference probabilities for " + route)
            thresholds = np.array([report["thresholds"][common.FAMILIES.index(item["family"])] for item in samples])
            indices = {item["id"]: i for i, item in enumerate(samples)}
            result = {"route": route, "checkpoint_sha256": expected_sha,
                      "all_context_samples": subset_summary(list(range(len(samples))), actual, expected, thresholds, np, args.tolerance),
                      "first_historical_batch": subset_summary([indices[value] for value in baseline], actual, expected, thresholds, np, args.tolerance),
                      "selected_heldout_families": {family: subset_summary([indices[value] for value in ids], actual, expected, thresholds, np, args.tolerance)
                                                    for family, ids in chosen.items()}}
            summaries.append(result)
            del model, saved, mean, std, images, features
            torch.cuda.empty_cache()

        passed = all(item["all_context_samples"]["within_tolerance"]
                     and item["all_context_samples"]["threshold_decisions_match"] for item in summaries)
        payload = {"schema_version": 1, "status": "pass" if passed else "fail",
                   "check": "frozen_checkpoint_inference_replay", "models": summaries,
                   "policy_sha256": sha(policy_path), "model_count": len(summaries),
                   "batch_size": batch_size, "image_size_height_width": list(image_size),
                   "autocast_dtype": "bfloat16", "absolute_probability_tolerance": args.tolerance,
                   "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
                   "families_using_rules_only": [family for family, value in selected.items() if value["route"] == "rules"],
                   "training_performed": False, "original_run_modified": False,
                   "sample_identifiers_in_output": False,
                   "limitation": "Small replay; no new accuracy estimate. Batch contexts are reconstructed from nonmissing inventory order; historical resumed batches may differ."}
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--tolerance", type=float, default=5e-3)
    args = parser.parse_args()
    if not 0 < args.tolerance <= 5e-3:
        parser.error("Tolerance must be positive and no larger than 5e-3")
    payload = check(args)
    print(json.dumps({"status": payload["status"], "model_count": payload["model_count"]}))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
