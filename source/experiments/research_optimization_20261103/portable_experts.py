"""Research-only fresh inference with immutable default experts and transforms.

This entry neither trains nor recalibrates. Model heads are G/C/D technical
candidates, never official nine-class scientific truth. No web-service changes.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

# Dynamic imports must not emit caches into the frozen source trees.
sys.dont_write_bytecode = True

FAMILIES = ("AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample", "EPB_seg_sample", "DMSP_SSUSI", "ULF_sample")
GROUPS = ("G", "C", "D")
RESEARCH_ROOT = Path("/staff/tangjiarui/OPC/research_optimization_20261103")
VERSION = "portable-experts-research-2026.11.03-1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_rows(path, values):
    with Path(path).open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def output_directory(path, input_path=None):
    out = Path(path).resolve()
    require(out.is_relative_to(RESEARCH_ROOT.resolve()) and out != RESEARCH_ROOT.resolve(), "Output must be a new child of the research root")
    require(input_path is None or not (out == input_path or out.is_relative_to(input_path)), "Output overlaps source input")
    require(not out.exists(), "Refusing to overwrite any existing run")
    out.mkdir(parents=True)
    return out


def selected_routes(base, extension):
    require(base["status"] == extension["status"] == "frozen", "Policy is not frozen")
    require(base["test_split_parsed"] is False and extension["test_split_parsed"] is False, "Selection was not frozen before test")
    name = base["recommended_variant"]
    matches = [value for key, value in base["recalibrated"].items() if name.startswith(key + "_")]
    require(len(matches) == 1, "Unsupported or ambiguous frozen default")
    old = matches[0]
    result = {}
    for family in FAMILIES:
        mode = extension["selected"][family]["variant"]
        routes = []
        if mode in {"base_validation_recommendation", "base_gc_gbdt_d"}:
            choice = old["choices"][family]
            thresholds = list(choice["thresholds"])
            if old["gc_only"] or mode == "base_gc_gbdt_d":
                thresholds[2] = 1.01
            if choice["route"] != "rules":
                routes.append(dict(route=choice["route"], thresholds=thresholds))
        if mode in {"gbdt_all_groups", "gbdt_gc_only", "base_gc_gbdt_d"}:
            thresholds = list(extension["thresholds"][family][mode])
            if mode == "gbdt_gc_only":
                thresholds[2] = 1.01
            if mode == "base_gc_gbdt_d":
                thresholds[:2] = [1.01, 1.01]
            routes.append(dict(route="gbdt_prior_v1", thresholds=thresholds))
        require(routes, "Unsupported rules-only or empty route")
        result[family] = dict(variant=mode, experts=routes)
    return result


def freeze(args):
    """Build a provenance lock from already-frozen A policies, not new labels."""
    out = output_directory(args.output)
    code, training, prior = args.code_root.resolve(), args.training_root.resolve(), args.policy_root.resolve()
    basepath = prior / "comparison_v1/frozen_validation_policy.json"
    extensionpath = prior / "gbdt_extension_v1/frozen_validation_extension.json"
    base, extension = read(basepath), read(extensionpath)
    require(sha(basepath) == extension["base_policy_sha256"], "Base/extension chain changed")
    routes = selected_routes(base, extension)
    files = {str(p): sha(p) for p in (basepath, extensionpath)}
    manifest = read(training / "code_manifest.json")
    for name, expected in manifest.items():
        path = code / "experiments/all_family_training_20261031" / name
        require(sha(path) == expected, "Frozen training source changed: " + name)
        files[str(path)] = expected
    # Indirect imports are locked too; they are not bundled or silently copied.
    extra = [code / "experiments/full_matrix_20261031/common.py", code / "experiments/full_matrix_20261031/prepare.py",
             code / "tools/prepare_model_comparison.py", code / "opc_agent.py",
             code / "experiments/image_models_20261031/models.py",
             code.parent / "phase2_visual_audit_xzj/src/annotation_diff.py",
             args.prior_code_root / "prior_rules.py", args.prior_code_root / "gbdt/train_gbdt.py"]
    for path in extra:
        files[str(path.resolve())] = sha(path)
    require(files[str((args.prior_code_root / "prior_rules.py").resolve())] == base["prior_module_sha256"], "Prior module differs from frozen policy")
    for route in sorted({r["route"] for x in routes.values() for r in x["experts"]} - {"gbdt_prior_v1"}):
        reportpath = training / "training" / (route + ".json")
        report = read(reportpath)
        require(report["status"] == "complete" and report["test_read"] is False, "Training report not frozen")
        require(len(report["actual_train_coverage"]) == 30, "Expert was not trained on all 30 packages")
        files[str(reportpath)] = sha(reportpath)
        for relative, expected in report["checkpoint_hashes"].items():
            path = training / relative
            require(sha(path) == expected, "Checkpoint differs from frozen training")
            files[str(path)] = expected
    bootstrap = code.parent / "models/image-model-pilot/yolov8n-seg.pt"
    files[str(bootstrap)] = sha(bootstrap)
    gbdt = prior / "gbdt"
    train = read(gbdt / "training.json")
    barrier = read(gbdt / "training_barrier.json")
    require(sha(gbdt / "training.json") == extension["gbdt_training_sha256"] == barrier["training_sha256"], "GBDT training chain mismatch")
    require(sha(gbdt / "training_barrier.json") == extension["gbdt_training_barrier_sha256"], "GBDT barrier changed")
    require(sha(gbdt / "vectorizer.pkl") == barrier["vectorizer_sha256"], "Vectorizer changed")
    for p in [gbdt / "training.json", gbdt / "training_barrier.json", gbdt / "vectorizer.pkl"]:
        files[str(p)] = sha(p)
    for i, group in enumerate(GROUPS):
        path = gbdt / ("model_" + group + ".pkl")
        require(sha(path) == train["heads"][i]["model_sha256"], "GBDT head changed")
        files[str(path)] = sha(path)
    lock = dict(version=VERSION, status="frozen_research_entry", family_routes=routes, files=files,
                code_root=str(code), training_root=str(training), policy_root=str(prior),
                prior_code_root=str(args.prior_code_root.resolve()), prior_policy=base["prior_policy"],
                threshold_tuning=False, training=False, uses_B_for_selection=False)
    write(out / "lock.json", lock)
    print(json.dumps(dict(status="lock_created", output=str(out / "lock.json"), files=len(files))))


def validate_lock(path):
    lock = read(path)
    require(lock["version"] == VERSION and lock["status"] == "frozen_research_entry", "Unknown lock")
    for name, expected in lock["files"].items():
        require(sha(name) == expected, "Locked source/model/policy changed: " + name)
    return lock


def transforms(lock, output):
    code = Path(lock["code_root"])
    sys.path.insert(0, str(code / "experiments/all_family_training_20261031"))
    common = load_module("common", code / "experiments/all_family_training_20261031/common.py")
    prep = load_module("portable_frozen_allprep", code / "experiments/all_family_training_20261031/prepare.py")
    sys.modules["common"] = prep.reference
    try:
        full = load_module("portable_frozen_fullprep", code / "experiments/full_matrix_20261031/prepare.py")
    finally:
        sys.modules["common"] = common
    full.OUT = output / "current_views"
    for name in ["raw_views", "views"]:
        (full.OUT / name).mkdir(parents=True)
    return common, prep, full


def pair_labels(root, family, selection=None, limit=None):
    root = root.resolve()
    labels = sorted(root.rglob("*.json"))
    if selection is not None:
        labels = [Path(p).resolve() for p in selection]
        require(len(labels) == len(set(labels)), "Duplicate selected labels")
        require(all(p.is_relative_to(root) and p.is_file() and p.suffix == ".json" for p in labels), "Selected label escapes package or is absent")
    if limit:
        labels = labels[:limit]
    result = []
    for label in labels:
        if (root / "data").is_dir() and label.parent == root / "label":
            data = root / "data" / label.name.removesuffix(".json")
        elif family == "AIA_AR_sample":
            data = root / "data" / label.name.removesuffix(".json")
        elif family == "ULF_sample":
            data = label.with_name(label.name.removesuffix(".json"))
        else:
            data = label.with_suffix(".png")
        require(label.resolve().is_relative_to(root) and data.resolve().is_relative_to(root), "Input symlink escapes package")
        result.append((label, data))
    return result


def validate_pair_names(pairs):
    # Frozen rule Dataset dictionaries use basenames. Refuse any nested input
    # that would overwrite a record, rather than silently losing checks.
    for index, kind in ((0, "label"), (1, "data")):
        names = [pair[index].name for pair in pairs]
        require(len(names) == len(set(names)), "Ambiguous duplicate " + kind + " basename; frozen legacy adapter cannot safely represent this input")


def legacy_bits(full, root, pairs):
    validate_pair_names(pairs)
    rules = full.rules
    # Same adapter as historical B preprocessing; it only adds concept support.
    original = rules.event_category
    rules.event_category = lambda event: event.get("concept") or original(event)
    dataset = rules.Dataset(root=root, dataset_id=root.name)
    dataset.label_files = {p.name: p for p, _ in pairs}
    dataset.data_files = {p.name: p for _, p in pairs if p.is_file()}
    rules.base_analysis(dataset, "PORTABLE-RESEARCH", set(rules.ALL_CHECKS) - {"C10"})
    answer = {str(p): [0, 0, 0] for p, _ in pairs}
    for finding in dataset.findings:
        if finding["status"] != rules.STATUS_FAIL:
            continue
        code = finding["defectType"]
        bit = 0 if code.startswith("GEOMETRY_") or code in {"L2-G", "L3-MIS"} else 1 if code in {"L3-06", "L3-07"} else None
        locator = finding.get("_locator") or finding.get("locator") or finding.get("evidence", {}).get("locator")
        key = str(locator).split("#")[0]
        if bit is not None and key in answer:
            answer[key][bit] = 1
    return answer, dataset.run_events


def prepare(args, lock, out):
    import numpy as np
    from PIL import Image
    common, allprep, full = transforms(lock, out)
    pairs = pair_labels(args.input, args.family, read(args.selection_json) if args.selection_json else None, args.limit)
    old_bits, events = legacy_bits(full, args.input.resolve(), pairs)
    prior = load_module("portable_frozen_priors", Path(lock["prior_code_root"]) / "prior_rules.py")
    policy = prior.Policy(**lock["prior_policy"])
    records = []
    for label_path, data_path in pairs:
        sid = hashlib.sha256(str(label_path).encode()).hexdigest()
        record = dict(id=sid, family=args.family, label_path=str(label_path), data_path=str(data_path),
                      label_sha256=sha(label_path), data_sha256=None, input_missing=not data_path.is_file(),
                      probabilities=None, image=None, training_image=None, input_sha256=None, features=None,
                      rule_bits=old_bits[str(label_path)], status="input_missing" if not data_path.is_file() else "preparing")
        try:
            label = read(label_path)
            require(isinstance(label, dict), "Annotation must be an object")
            observation = prior.evaluate_sample(args.family, label, data_path, label_path=str(label_path), sample_id=sid, policy=policy)
            record["prior_observation"] = observation
            if record["input_missing"]:
                records.append(record)
                continue
            record["data_sha256"] = sha(data_path)
            header = full.rules.png_header(data_path) if data_path.suffix.lower() == ".png" else {}
            row = dict(id=sid, family=args.family, dataset_family=args.family, data_exists=True,
                       data_path=str(data_path), data_sha256=record["data_sha256"], base_sample_id=sid,
                       tier="unseen", image_width=header.get("width"), image_height=header.get("height"))
            display_label = copy.deepcopy(label)
            if args.release_profile == "roadshow-b-v1" and args.family == "ULF_sample":
                origin = label.get("basic_annotation", {}).get("startTime")
                if origin:
                    display_label.setdefault("science_annotation", {})["dateObs"] = origin
            view = full.feature_view(display_label, row)
            stats = read(full.OUT / "raw_views" / (record["data_sha256"] + ".json"))
            row["features"] = view["features"]
            features, prompt = allprep.current_input(row, label, stats)
            small = out / "student_views" / (sid + ".png")
            small.parent.mkdir(exist_ok=True)
            with Image.open(view["image"]) as image:
                image.resize((256, 128), Image.Resampling.BILINEAR).save(small)
            require(np.isfinite(features).all(), "Non-finite prepared features")
            record.update(image=view["image"], training_image=str(small), features=features, prompt=prompt,
                          input_sha256=common.digest([sha(view["image"]), prompt, features]), status="prepared")
            require(record["label_sha256"] == sha(label_path) and record["data_sha256"] == sha(data_path), "Raw input changed during preprocessing")
        except Exception as error:
            record.update(input_missing=True, status="preparation_failed", error=repr(error))
        records.append(record)
    if args.selection_json is None and args.limit is None:
        paired_data = {p.resolve() for _, p in pairs}
        for data_path in sorted(args.input.rglob("*")):
            if not data_path.is_file() or data_path.suffix.lower() not in {".png", ".csv"} or data_path.resolve() in paired_data:
                continue
            require(data_path.resolve().is_relative_to(args.input), "Orphan data symlink escapes package")
            records.append(dict(id=hashlib.sha256(str(data_path).encode()).hexdigest(), family=args.family,
                label_path=None, label_sha256=None, data_path=str(data_path), data_sha256=sha(data_path),
                input_missing=True, probabilities=None, image=None, training_image=None, input_sha256=None,
                features=None, rule_bits=[0,0,0], status="label_missing"))
    require(records, "No current observation or annotation found")
    write(out / "legacy_check_events.json", events)
    write_rows(out / "inventory.jsonl", records)
    return records, common


def fresh_probabilities(args, lock, records, common):
    import numpy as np
    present = [r for r in records if not r["input_missing"]]
    predictions, calls, samples, bootstrap_calls, gpu_peak = {}, {}, {}, {}, 0
    routes = lock["family_routes"][args.family]["experts"]
    for route_entry in routes:
        route = route_entry["route"]
        predictions[route] = {}
        calls[route] = 0
        samples[route] = 0
        bootstrap_calls[route] = 0
        if not present:
            continue
        if route == "gbdt_prior_v1":
            code = Path(lock["prior_code_root"]) / "gbdt/train_gbdt.py"
            gbdt = load_module("portable_frozen_gbdt", code)
            folder = Path(lock["policy_root"]) / "gbdt"
            # Pickle is loaded only after every byte was verified against the lock.
            models = [pickle.loads((folder / ("model_" + g + ".pkl")).read_bytes()) for g in GROUPS]
            vectorizer = pickle.loads((folder / "vectorizer.pkl").read_bytes())
            for offset in range(0, len(present), args.batch_size):
                batch = present[offset:offset + args.batch_size]
                # Public training used the explicitly whitelisted current anchor;
                # historical B replay omitted it. Keep these release profiles
                # distinct so a replay does not silently change its inputs.
                use_anchor = args.release_profile == "public-v1" and read(folder / "training.json").get("additional_CME_input") != "none"
                ff = []
                for record in batch:
                    anchor = (read(record["label_path"]).get("hek_event_annotation") or {}).get("image_time") if use_anchor and record["family"] == "CME_C2_sample" else None
                    ff.append(gbdt.features(record, cme_image_time=anchor))
                matrix = vectorizer.transform(ff)
                pp = gbdt.probabilities(models, matrix)
                calls[route] += 1
                samples[route] += len(batch)
                predictions[route].update({r["id"]: p.tolist() for r, p in zip(batch, pp)})
        else:
            import torch
            from PIL import Image
            require(args.device == "cpu" or args.device.startswith("cuda:"), "Explicit CPU or CUDA device required")
            if args.device.startswith("cuda:"):
                require(args.device == "cuda:0", "This research run may use GPU0 only")
                require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
                torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gb / (torch.cuda.get_device_properties(0).total_memory / 1024**3), 0)
                torch.cuda.reset_peak_memory_stats(0)
            source = Path(lock["code_root"]) / "experiments/all_family_training_20261031/models.py"
            model_code = load_module("portable_frozen_models", source)
            saved = torch.load(Path(lock["training_root"]) / "checkpoints" / (route + ".pt"), map_location="cpu", weights_only=False)
            model = model_code.build(route, pretrained=False, widths=saved.get("widths"), hidden=saved.get("hidden"))
            # Frozen YOLO constructor probes backbone shape once with two zero
            # images. It is separate from real-sample classifier forwards.
            bootstrap_calls[route] = int(route.startswith("yolo"))
            model.load_state_dict(saved["state"], strict=True)
            model = model.to(args.device).eval()
            mean, std = saved["mean"].to(args.device), saved["std"].to(args.device)
            size = common.SPATIAL_SIZE if route in common.ARCHS else common.SIZE
            with torch.inference_mode():
                for offset in range(0, len(present), args.batch_size):
                    batch = present[offset:offset + args.batch_size]
                    images = []
                    for record in batch:
                        path = record["image"] if size == common.SPATIAL_SIZE else record["training_image"]
                        with Image.open(path) as image:
                            images.append(np.array(image.resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32).transpose(2, 0, 1) / 255)
                    x = torch.from_numpy(np.stack(images)).to(args.device)
                    f = torch.tensor([r["features"] for r in batch], dtype=torch.float32, device=args.device)
                    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda:") else contextlib.nullcontext()
                    with amp:
                        logits, _ = model(x, (f - mean) / std)
                    pp = logits.float().sigmoid().cpu().numpy()
                    require(np.isfinite(pp).all(), "Non-finite model probabilities")
                    calls[route] += 1
                    samples[route] += len(batch)
                    predictions[route].update({r["id"]: p.tolist() for r, p in zip(batch, pp)})
            if args.device.startswith("cuda:"):
                torch.cuda.synchronize(0)
                gpu_peak = max(gpu_peak, torch.cuda.max_memory_allocated(0))
            del model, saved
            if args.device.startswith("cuda:"):
                torch.cuda.empty_cache()
    return predictions, dict(fresh_forward_batches=calls, fresh_forward_samples=samples,
        constructor_backbone_probe_batches=bootstrap_calls,
        tree_head_predict_proba_calls={r: n*3 for r,n in calls.items() if r == "gbdt_prior_v1"},
        cached_probability_reuse=0, gpu_peak_allocated_bytes=gpu_peak)


def compare_reference(args, records, predictions):
    if not args.reference_inventory:
        return dict(status="not_requested", passed=None)
    # Reference truth/targets are deliberately discarded; no fitting is done.
    wanted = {r["label_path"] for r in records}
    reference = {r["label_path"]: {k: r.get(k) for k in ("id", "label_path", "image", "input_sha256", "features", "prompt", "input_missing")}
                 for r in rows(args.reference_inventory) if r.get("label_path") in wanted}
    checks = []
    for record in records:
        old = reference.get(record["label_path"])
        check = dict(id=record["id"], reference_id=old.get("id") if old else None, input_missing=record["input_missing"])
        check["reference_present"] = old is not None
        if old and not record["input_missing"]:
            check.update(image_sha_equal=sha(record["image"]) == sha(old["image"]),
                         features_equal=record["features"] == old["features"], prompt_equal=record["prompt"] == old["prompt"],
                         input_sha_equal=record["input_sha256"] == old["input_sha256"])
        elif old:
            check["missing_state_equal"] = bool(old["input_missing"]) == bool(record["input_missing"])
        checks.append(check)
    probability_checks = []
    for argument in args.reference_prediction:
        route, name = argument.split("=", 1)
        require(route in predictions, "Reference route not inferred by this family")
        ids = {r["id"] for r in reference.values()}
        oldp = {r["id"]: r for r in rows(name) if r["id"] in ids}
        for record in records:
            old = reference.get(record["label_path"])
            before = oldp.get(old["id"]) if old else None
            now = predictions[route].get(record["id"])
            previous = before.get("probabilities") if before else None
            delta = max(abs(a-b) for a,b in zip(now, previous)) if now is not None and previous is not None else None
            thresholds = args._lock["family_routes"][args.family]["experts"]
            threshold = next(x["thresholds"] for x in thresholds if x["route"] == route)
            bits_equal = [a >= t for a,t in zip(now,threshold)] == [a >= t for a,t in zip(previous,threshold)] if delta is not None else now is None and previous is None and before is not None
            probability_checks.append(dict(id=record["id"], route=route, reference_found=before is not None,
                max_abs_probability_delta=delta, probability_within_tolerance=delta <= args.probability_atol if delta is not None else now is None and previous is None and before is not None,
                frozen_candidate_bits_equal=bits_equal))
    passed = all(all(v for k,v in c.items() if k.endswith("equal") or k == "reference_present") for c in checks)
    passed = passed and all(c["reference_found"] and c["probability_within_tolerance"] and c["frozen_candidate_bits_equal"] for c in probability_checks)
    return dict(status="passed" if passed else "failed", passed=passed, preprocessing=checks, probabilities=probability_checks,
                probability_atol=args.probability_atol, reference_inventory_sha256=sha(args.reference_inventory),
                reference_prediction_sha256={a.split("=",1)[0]: sha(a.split("=",1)[1]) for a in args.reference_prediction})


def sources_unchanged(records):
    return all((r["label_path"] is None or r["label_sha256"] == sha(r["label_path"]))
               and (r["data_sha256"] is None or r["data_sha256"] == sha(r["data_path"])) for r in records)


def run(args):
    started = time.monotonic()
    require(1 <= args.batch_size <= 4, "Research batch must be 1..4")
    require(0 < args.gpu_memory_gb <= 4, "Research GPU allocator budget must be <=4GB")
    require(args.input.is_dir(), "Input package directory absent")
    args.input = args.input.resolve()
    lock = validate_lock(args.lock)
    args._lock = lock
    out = output_directory(args.output, args.input)
    os.environ["YOLO_CONFIG_DIR"] = str(out / "ultralytics_config")
    os.environ["CUDA_CACHE_PATH"] = str(out / "cuda_cache")
    write(out / "request.json", dict(version=VERSION, input=str(args.input), family=args.family, device=args.device,
          batch_size=args.batch_size, release_profile=args.release_profile, lock_sha256=sha(args.lock),
          source_sha256=sha(__file__), training=False, threshold_tuning=False, model_candidate_score_eligible=False))
    try:
        records, common = prepare(args, lock, out)
        predictions, fresh = fresh_probabilities(args, lock, records, common)
        exported = []
        for record in records:
            consulted, bits = [], [0, 0, 0]
            for route in lock["family_routes"][args.family]["experts"]:
                p = predictions[route["route"]].get(record["id"])
                candidate = [int(a >= t) for a,t in zip(p, route["thresholds"])] if p is not None else [0,0,0]
                bits = [int(a or b) for a,b in zip(bits,candidate)]
                consulted.append(dict(**route, probabilities=p, candidate_bits=candidate))
            current = list(record.get("prior_observation", {}).get("confirmed_bits", [0,0,0]))
            current[:2] = [int(current[g] or record["rule_bits"][g]) for g in range(2)]
            exported.append(dict(sample_id=record["id"], family=args.family, label_path=record["label_path"],
                data_path=record["data_path"], input_sha256=record["input_sha256"], models=consulted,
                input_missing=record["input_missing"], status="abstain" if record["input_missing"] else "fresh_candidate",
                reason=record.get("error", record["status"]), model_candidate_bits=bits,
                current_rule_bits=current, combined_flag_bits=[int(a or b) for a,b in zip(current,bits)],
                scientific_verdict="undetermined", model_findings_score_eligible=False, ground_truth_available=False,
                F1=None, recall=None, precision=None))
        write_rows(out / "selected_candidates.jsonl", exported)
        reference = compare_reference(args, records, predictions)
        write(out / "reference_comparison.json", reference)
        unchanged = sources_unchanged(records)
        failures = sum(r["status"] == "preparation_failed" for r in records)
        passed = unchanged and failures == 0 and reference["passed"] is not False
        result = dict(status="passed_research_smoke" if passed else "failed_research_smoke", records=len(records),
            inferred=sum(not r["input_missing"] for r in records), abstentions=sum(r["input_missing"] for r in records),
            source_data_absent=sum(not Path(r["data_path"]).is_file() for r in records),
            source_label_absent=sum(r["label_path"] is None for r in records),
            preparation_failures=failures, input_hashes_unchanged=unchanged, seconds=time.monotonic()-started,
            family=args.family, device=args.device, batch_size=args.batch_size, release_profile=args.release_profile,
            reference_comparison_passed=reference["passed"], **fresh,
            full_dataset_evaluation=False, official_metric_computed=False, deployment_acceptance=False,
            limitations=["Sample-level G/C/D proxy flags are not nine-class localized truth", "Small replay is not a full-dataset accuracy or A10 timing acceptance"])
        write(out / "complete.json", result)
        print(json.dumps(result))
        return 0 if passed else 2
    except BaseException as error:
        write(out / "failure.json", dict(status="failed", error=repr(error), seconds=time.monotonic()-started))
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--code-root", type=Path, required=True)
    f.add_argument("--training-root", type=Path, required=True)
    f.add_argument("--policy-root", type=Path, required=True)
    f.add_argument("--prior-code-root", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--family", choices=FAMILIES, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--lock", type=Path, required=True)
    r.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    r.add_argument("--batch-size", type=int, required=True)
    r.add_argument("--gpu-memory-gb", type=float, default=4)
    r.add_argument("--release-profile", choices=("public-v1", "roadshow-b-v1"), required=True)
    r.add_argument("--selection-json", type=Path)
    r.add_argument("--limit", type=int)
    r.add_argument("--reference-inventory", type=Path)
    r.add_argument("--reference-prediction", action="append", default=[])
    r.add_argument("--probability-atol", type=float, default=.02)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "freeze":
        freeze(arguments)
    else:
        sys.exit(run(arguments))
