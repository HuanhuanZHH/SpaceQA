"""Exploratory all-family CPU tree baseline using current-only whitelisted inputs.

Three heads train on all families before the held-out test file is opened.
No pixels, clean comparisons, tier, sample IDs, source paths, or targets enter X.
Target/truth_valid are used solely as supervision or masked scoring variables.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

FAMILIES = ["AIA_AR_sample", "HMI_AR_sample", "CME_C2_sample", "EPB_seg_sample", "DMSP_SSUSI", "ULF_sample"]
TIERS = ["P0", "P1", "P2", "P3", "P4"]
GROUPS = ["G", "C", "D"]
ROUTE = "gbdt_prior_v1"
NUMERIC_PARAMETERS = ["central_pa", "width", "mpa", "linear_speed", "speed_2nd_initial", "speed_2nd_final", "speed_2nd_20R", "acceleration", "mass", "kinetic_energy"]
CATEGORIES = ["AR", "Active Region", "CH", "Coronal Hole", "CME", "EPB", "PCA", "TPA", "unclear", "Pi2", "Pc3", "Pc5", "other"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def emit(stage, **value):
    print(json.dumps(dict(stage=stage, **value)), flush=True)


def numeric(value):
    try:
        answer = float(value)
        return answer if math.isfinite(answer) else None
    except (ValueError, TypeError, OverflowError):
        return None


def stamp(value):
    if not isinstance(value, str):
        return None
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=dt.timezone.utc)
        return result.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def aggregate(out, key, values):
    values = [numeric(x) for x in values]
    values = [x for x in values if x is not None]
    out[key + ".n"] = float(len(values))
    for name in ("min", "max", "mean", "std"):
        out[key + "." + name] = float(getattr(np, name)(values)) if values else np.nan


def subtract(a, b):
    return a - b if a is not None and b is not None else np.nan


def features(row, *, cme_image_time=None):
    """Explicit feature whitelist; sample identities/targets never determine X."""
    source = row.get("features")
    if not isinstance(source, list) or len(source) != 256:
        raise ValueError("Expected the frozen current-only 256 feature values")
    summary = json.loads(row["prompt"].split("Current fields: ", 1)[1])
    family = row["family"]
    if family not in FAMILIES or summary.get("family") != family:
        raise ValueError("Unknown or inconsistent current observation family")
    out = {f"current.{i:03d}": numeric(value) if numeric(value) is not None else np.nan for i, value in enumerate(source)}
    for item in FAMILIES:
        out["family." + item] = float(item == family)
    decoded = summary.get("decoded") or {}
    declared = summary.get("declared") or {}
    window = summary.get("window") or {}
    events = [x for x in summary.get("events", []) if isinstance(x, dict)]
    geometry = [x for x in summary.get("geometry", []) if isinstance(x, dict)]
    for key in ("width", "height", "rows", "first_t", "last_t", "median_dt", "regular"):
        out["decoded." + key] = numeric(decoded.get(key)) if numeric(decoded.get(key)) is not None else np.nan
    for key in ("sampling_rate_hz", "duration_s", "n_points"):
        out["declared." + key] = numeric(declared.get(key)) if numeric(declared.get(key)) is not None else np.nan
    first, last, cadence = (numeric(decoded.get(k)) for k in ("first_t", "last_t", "median_dt"))
    span = subtract(last, first)
    out["ulf.elapsed_span"] = span
    out["ulf.point_residual"] = subtract(numeric(declared.get("n_points")), numeric(decoded.get("rows")))
    out["ulf.duration_residual_elapsed"] = subtract(numeric(declared.get("duration_s")), numeric(span))
    coverage = span + cadence if numeric(span) is not None and cadence is not None else None
    out["ulf.duration_residual_inclusive"] = subtract(numeric(declared.get("duration_s")), coverage)
    out["ulf.rate_residual"] = subtract(numeric(declared.get("sampling_rate_hz")), 1 / cadence if cadence and cadence > 0 else None)
    actual_components = decoded.get("components") if isinstance(decoded.get("components"), list) else []
    reported_components = declared.get("components") if isinstance(declared.get("components"), list) else []
    for name in ("H", "D", "Z", "T", "F"):
        out["ulf.component.actual." + name] = float(name in actual_components)
        out["ulf.component.declared." + name] = float(name in reported_components)
    out["ulf.component_set_residual"] = len(set(actual_components) ^ set(reported_components))
    unit = declared.get("units")
    for name in ("nT", "pT", "deg"):
        out["ulf.unit_text." + name] = float(isinstance(unit, str) and name in unit)
    start, end, origin = stamp(window.get("startTime")), stamp(window.get("endTime")), stamp(declared.get("dateObs"))
    out["time.window_span"] = subtract(end, start)
    out["time.start_minus_dateObs"] = subtract(start, origin)
    out["time.end_minus_dateObs"] = subtract(end, origin)
    out["time.start_axis_residual"] = subtract(start, origin + first if origin is not None and first is not None else None)
    out["time.end_axis_residual"] = subtract(end, origin + last if origin is not None and last is not None else None)
    out["time.window_reversed"] = float(start is not None and end is not None and start > end)

    for axis in ("width", "height"):
        actual = numeric(decoded.get(axis))
        dimensions = [d.get("value") for d in summary.get("dimensions", []) if isinstance(d, dict) and d.get("axis") == axis]
        aggregate(out, "dimension." + axis + ".declared", dimensions)
        aggregate(out, "dimension." + axis + ".residual", [subtract(numeric(x), actual) for x in dimensions])

    out["objects.events"] = len(events)
    out["objects.geometry"] = len(geometry)
    out["objects.valid_geometry"] = sum(bool(g.get("valid")) for g in geometry)
    aggregate(out, "geometry.point_count", [g.get("points_count") for g in geometry])
    widths, heights, areas, centroid_x, centroid_y, bad_bounds = [], [], [], [], [], 0
    width, height = numeric(decoded.get("width")), numeric(decoded.get("height"))
    for g in geometry:
        bounds = g.get("bounds")
        if isinstance(bounds, list) and len(bounds) == 4 and all(numeric(x) is not None for x in bounds):
            x0, y0, x1, y1 = map(float, bounds)
            bw, bh = max(0, x1 - x0), max(0, y1 - y0)
            widths.append(bw / width if width else bw)
            heights.append(bh / height if height else bh)
            areas.append(bw * bh / (width * height) if width and height else bw * bh)
            bad_bounds += int(x0 < 0 or y0 < 0 or (width and x1 > width) or (height and y1 > height))
        center = g.get("centroid")
        if isinstance(center, list) and len(center) == 2:
            if numeric(center[0]) is not None:
                centroid_x.append(float(center[0]) / width if width else float(center[0]))
            if numeric(center[1]) is not None:
                centroid_y.append(float(center[1]) / height if height else float(center[1]))
    for key, value in (("width", widths), ("height", heights), ("bbox_area", areas), ("centroid_x", centroid_x), ("centroid_y", centroid_y)):
        aggregate(out, "geometry." + key, value)
    out["geometry.out_of_bounds_objects"] = bad_bounds
    for key in ("polygon", "rectangle", "angular_sector", "none"):
        out["geometry.kind." + key] = sum(g.get("kind") == key for g in geometry)

    for key in ("event_type", "concept", "kind", "label"):
        counts = collections.Counter(event.get(key) if event.get(key) in CATEGORIES else "other" for event in events if key in event)
        for category in CATEGORIES:
            out["category." + key + "." + category] = counts[category]
    for key in NUMERIC_PARAMETERS:
        aggregate(out, "parameter." + key, [event.get(key) for event in events])
    for key in ("cme_time", "start_dt", "end_dt"):
        times = [stamp(event.get(key)) for event in events]
        for anchor_name, anchor in (("window_start", start), ("dateObs", origin), ("image_time", stamp(cme_image_time))):
            offsets = [value - anchor for value in times if value is not None and anchor is not None]
            aggregate(out, "event_time." + key + ".minus_" + anchor_name, offsets)
            if key == "cme_time":
                for interval in (60, 720):
                    aggregate(out, f"event_time.cme.quantization_residual_{interval}." + anchor_name,
                              [abs(value - round(value / interval) * interval) for value in offsets])
    # Repeating-event majority is not injected here: its peers may lie in
    # another split, and pooling train/validation/test would cause leakage.
    for annotation_key in ("hek_event_annotation", "ulf_event_annotation"):
        event_summary = (summary.get("event_summary") or {}).get(annotation_key) or {}
        out["summary." + annotation_key + ".present"] = float(bool(event_summary))
        out["summary." + annotation_key + ".event_count"] = numeric(event_summary.get("event_count")) if numeric(event_summary.get("event_count")) is not None else np.nan
        out["summary." + annotation_key + ".has_events"] = float(bool(event_summary.get("has_events")))
        out["summary." + annotation_key + ".count_residual"] = subtract(numeric(event_summary.get("event_count")), len(events))
        types = event_summary.get("event_types") if isinstance(event_summary.get("event_types"), list) else []
        for category in CATEGORIES:
            out["summary." + annotation_key + ".type." + category] = float(category in types)
    # Bound extreme current parameters for stable histograms. Missing remains
    # NaN, handled natively by HGB; no imputations are fitted on held-out data.
    return {key: float(np.clip(value, -1e30, 1e30)) if numeric(value) is not None else np.nan for key, value in out.items()}


def load_split(path, split, *, read_cme_anchor, output):
    start = time.monotonic()
    rows, extra = [], []
    anchor_cache = {}
    anchor_failures = 0
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            assert record["split"] == split
            row = {key: record.get(key) for key in ("id", "family", "tier", "target", "truth_valid", "input_sha256", "input_missing", "training_group_id")}
            rows.append(row)
            if row["input_missing"]:
                extra.append(None)
                continue
            anchor = None
            if read_cme_anchor and row["family"] == "CME_C2_sample":
                # Paths only locate the CURRENT label; never become model X.
                # Hash its bytes before accessing the single whitelisted field.
                path_now = record["label_path"]
                if path_now not in anchor_cache:
                    content = Path(path_now).read_bytes()
                    if record.get("label_sha256") and hashlib.sha256(content).hexdigest() != record["label_sha256"]:
                        raise ValueError("Current CME label changed from frozen inventory")
                    anchor_cache[path_now] = (json.loads(content).get("hek_event_annotation") or {}).get("image_time")
                anchor = anchor_cache[path_now]
                anchor_failures += int(stamp(anchor) is None)
            extra.append(features(record, cme_image_time=anchor))
            if index and index % 10000 == 0:
                emit("extract_current_features", split=split, rows=index, seconds=time.monotonic() - start)
    write(output / (split + "_features.json"), dict(split=split, rows=len(rows), present=sum(x is not None for x in extra),
        cme_anchor_unavailable=anchor_failures, seconds=time.monotonic() - start, source_sha256=sha(path),
        feature_inputs="existing current-only 256 floats, explicit Current fields summary whitelist, optional current CME image_time"))
    return rows, extra


def coverage(rows, eligible=None):
    counts = collections.Counter((r["family"], r["tier"]) for i, r in enumerate(rows) if eligible is None or eligible[i])
    assert all(counts[(f, t)] > 0 for f in FAMILIES for t in TIERS), dict(counts)
    return {f + "|" + t: counts[(f, t)] for f in FAMILIES for t in TIERS}


def probabilities(models, matrix):
    return np.column_stack([model.predict_proba(matrix)[:, 1] for model in models])


def metrics(rows, p):
    if not rows:
        return dict(n=0, TP=0, FP=0, FN=0, TN=0, F1=None, precision=None, recall=None, FPR=None)
    y = np.array([r["target"] for r in rows], bool)
    valid = np.array([r["truth_valid"] for r in rows], bool)
    tp, fp, fn, tn = (int(x.sum()) for x in (p & y & valid, p & ~y & valid, ~p & y & valid, ~p & ~y & valid))
    negative = valid.all(1) & ~y.any(1)
    return dict(n=len(rows), TP=tp, FP=fp, FN=fn, TN=tn, F1=2 * tp / (2 * tp + fp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
                FPR=float(p[negative].any(1).mean()) if negative.any() else None, unknown_bits=int((~valid).sum()),
                missing_inputs=sum(bool(r["input_missing"]) for r in rows))


def calibrate(rows, pp):
    thresholds = np.full((6, 3), 1.01)
    traces = []
    for fi, family in enumerate(FAMILIES):
        for bit, name in enumerate(GROUPS):
            subset = np.array([r["family"] == family and bool(r["truth_valid"][bit]) and not r["input_missing"] for r in rows])
            y = np.array([r["target"][bit] for r in rows], bool)[subset]
            values = pp[subset, bit]
            options = []
            for threshold in [1.01] + np.linspace(.01, .99, 99).tolist():
                pred = values >= threshold
                tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
                score = 2 * tp / max(2 * tp + fp + fn, 1)
                options.append((score, -fp, threshold, tp, fn))
            best = max(options)
            thresholds[fi, bit] = best[2]
            traces.append(dict(family=family, group=name, threshold=best[2], val_F1=best[0], TP=best[3], FP=-best[1], FN=best[4], known_samples=len(y)))
    return thresholds, traces


def run(args):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction import DictVectorizer
    import sklearn
    output = Path(args.output).resolve()
    allowed = Path("/staff/tangjiarui/OPC/inference_prior_20261102/gbdt").resolve()
    if output != allowed:
        raise ValueError("New artifacts may only be written under the authorized gbdt directory")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError("Completed run already exists; do not silently overwrite")
    start = time.monotonic()
    write(output / "state.json", dict(status="preparing_train", pid=os.getpid(), route=ROUTE))
    tr, tf = load_split(args.source / "train.jsonl", "train", read_cme_anchor=args.read_cme_anchor, output=output)
    va, vf = load_split(args.source / "val.jsonl", "val", read_cme_anchor=args.read_cme_anchor, output=output)
    assert not ({r["training_group_id"] for r in tr} & {r["training_group_id"] for r in va})
    train_present = np.array([x is not None for x in tf])
    vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
    xtr = vectorizer.fit_transform([x for x in tf if x is not None])
    trfit = [r for r, present in zip(tr, train_present) if present]
    coverage_all = coverage(trfit)
    del tf
    val_present = np.array([x is not None for x in vf])
    xval = vectorizer.transform([x for x in vf if x is not None])
    del vf
    models, bit_evidence = [], []
    y = np.asarray([r["target"] for r in trfit], np.int8)
    valid = np.asarray([r["truth_valid"] for r in trfit], bool)
    hyperparameters = dict(max_iter=150, max_leaf_nodes=15, learning_rate=.08, min_samples_leaf=30,
                           l2_regularization=1.0, early_stopping=False, random_state=20261102, class_weight="balanced")
    for bit, name in enumerate(GROUPS):
        eligible = valid[:, bit]
        counts = collections.Counter(r["family"] for r, ok in zip(trfit, eligible) if ok)
        assert set(counts) == set(FAMILIES), counts
        known = y[eligible, bit]
        assert set(np.unique(known)) == {0, 1}
        weights = np.array([len(known) / (len(FAMILIES) * counts[r["family"]]) for r, ok in zip(trfit, eligible) if ok])
        write(output / "state.json", dict(status="training", group=name, pid=os.getpid(), all_family_coverage=coverage_all))
        head_start = time.monotonic()
        model = HistGradientBoostingClassifier(**hyperparameters)
        model.fit(xtr[eligible], known, sample_weight=weights)
        model_path = output / ("model_" + name + ".pkl")
        model_path.write_bytes(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
        evidence = dict(group=name, samples=int(eligible.sum()), positives=int(known.sum()), families=dict(counts),
                        packages=coverage(trfit, eligible), model_sha256=sha(model_path), seconds=time.monotonic() - head_start)
        bit_evidence.append(evidence)
        write(output / ("training_" + name + ".json"), evidence)
        models.append(model)
        emit("head_training_complete", **evidence)
    (output / "vectorizer.pkl").write_bytes(pickle.dumps(vectorizer, protocol=pickle.HIGHEST_PROTOCOL))
    ppval = np.full((len(va), 3), np.nan)
    ppval[val_present] = probabilities(models, xval)
    thresholds, trace = calibrate(va, ppval)
    feature_names = vectorizer.get_feature_names_out().tolist()
    write(output / "feature_names.json", feature_names)
    training = dict(route=ROUTE, thresholds=thresholds.tolist(), threshold_order=FAMILIES, groups=GROUPS,
        threshold_source="val only; threshold ties prefer fewer false positives then higher threshold", threshold_trace=trace,
        hyperparameters=hyperparameters, sklearn_version=sklearn.__version__, numpy_version=np.__version__,
        train_sha256=sha(args.source / "train.jsonl"), val_sha256=sha(args.source / "val.jsonl"),
        train_coverage=coverage_all, heads=bit_evidence, feature_count=len(feature_names), feature_source_sha256=sha(__file__),
        forbidden_inputs=["target", "truth_valid", "tier", "split", "IDs", "data/label/source paths", "clean comparisons", "P0 counterpart"],
        additional_CME_input="current label image_time only, label SHA verified" if args.read_cme_anchor else "none",
        thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    write(output / "training.json", training)
    barrier = dict(status="all_family_training_and_val_thresholds_frozen", time_unix=time.time(), training_sha256=sha(output / "training.json"),
                   vectorizer_sha256=sha(output / "vectorizer.pkl"), source_sha256=sha(__file__),
                   train_coverage=coverage_all, heads_complete=3, test_opened_before_this_barrier=False)
    write(output / "training_barrier.json", barrier)
    # First and only test-file access occurs after all three models, feature
    # vocabulary, and validation thresholds are frozen and recorded on disk.
    te, ef = load_split(args.source / "test.jsonl", "test", read_cme_anchor=args.read_cme_anchor, output=output)
    assert not ({r["training_group_id"] for r in te} & {r["training_group_id"] for r in tr + va})
    coverage(te)
    test_present = np.array([x is not None for x in ef])
    xtest = vectorizer.transform([x for x in ef if x is not None])
    del ef
    test_start = time.monotonic()
    ppte = np.full((len(te), 3), np.nan)
    ppte[test_present] = probabilities(models, xtest)
    inference_seconds = time.monotonic() - test_start
    # Full-inventory probability interface for independent fusion evaluation.
    pptr = np.full((len(tr), 3), np.nan)
    pptr[train_present] = probabilities(models, xtr)
    prediction = output / "predictions" / (ROUTE + ".jsonl")
    prediction.parent.mkdir(exist_ok=True)
    with prediction.open("w", encoding="utf-8") as stream:
        for rows, pp in ((tr, pptr), (va, ppval), (te, ppte)):
            for row, p in zip(rows, pp):
                success = bool(np.isfinite(p).all())
                stream.write(json.dumps(dict(id=row["id"], input_sha256=row["input_sha256"], probabilities=p.tolist() if success else None,
                    status="success" if success else "input_missing", seconds=0.0), allow_nan=False) + "\n")
    decisions = np.zeros((len(te), 3), bool)
    for index, row in enumerate(te):
        if np.isfinite(ppte[index]).all():
            decisions[index] = ppte[index] >= thresholds[FAMILIES.index(row["family"])]
    long = []
    for family in FAMILIES:
        for tier in TIERS:
            chosen = [i for i, r in enumerate(te) if r["family"] == family and r["tier"] == tier]
            long.append(dict(family=family, tier=tier, route=ROUTE, **metrics([te[i] for i in chosen], decisions[chosen])))
    family_results = []
    for family in FAMILIES:
        chosen = [i for i, r in enumerate(te) if r["family"] == family]
        family_results.append(dict(family=family, **metrics([te[i] for i in chosen], decisions[chosen])))
    report = dict(status="complete", route=ROUTE, scope="exploratory repeated heldout_test; P0-derived technical proxies, not official scientific truth",
        all30=metrics(te, decisions), p1_p4=metrics([r for r in te if r["tier"] != "P0"], decisions[[i for i,r in enumerate(te) if r["tier"] != "P0"]]),
        per_family=family_results, per_package=long, train_rows=len(tr), val_rows=len(va), test_rows=len(te),
        training_barrier_sha256=sha(output / "training_barrier.json"), prediction_sha256=sha(prediction),
        model_bytes=sum((output / ("model_" + k + ".pkl")).stat().st_size for k in GROUPS),
        test_model_inference_seconds=inference_seconds, elapsed_seconds=time.monotonic() - start,
        test_sha256=sha(args.source / "test.jsonl"), no_pixels_preprocessed=True, GPU_used=False,
        feature_names_sha256=sha(output / "feature_names.json"))
    write(output / "metrics.json", report)
    write(prediction.with_suffix(".complete.json"), dict(status="complete", route=ROUTE, records=len(tr)+len(va)+len(te),
        prediction_hash=sha(prediction), barrier_hash=sha(output / "training_barrier.json")))
    write(output / "complete.json", report)
    write(output / "state.json", dict(status="complete", route=ROUTE, all30=report["all30"], elapsed_seconds=report["elapsed_seconds"]))
    emit("complete", **report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031"))
    parser.add_argument("--output", type=Path, default=Path("/staff/tangjiarui/OPC/inference_prior_20261102/gbdt"))
    parser.add_argument("--read-cme-anchor", action="store_true")
    options = parser.parse_args()
    try:
        run(options)
    except Exception as error:
        write(options.output / "state.json", dict(status="failed", error=repr(error), pid=os.getpid()))
        raise
