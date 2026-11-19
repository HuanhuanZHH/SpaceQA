"""Local-only browser service for read-only datasets and auditable scan jobs.

No arbitrary paths, shell commands, or model services are accepted from clients.
Use an SSH local-forward to access a remote instance from a browser.
"""
from __future__ import annotations

import collections
import contextlib
import csv
import hashlib
import http.server
import ipaddress
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from . import __version__, backend_strategy, engine

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_FINDINGS_BYTES = 256 * 1024 * 1024
FAMILY_NAMES = tuple(engine.EXPECTED_EVENT_TYPES) + ("ULF_sample",)
FINISHED = {"succeeded", "failed", "cancelled", "interrupted"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message
        super().__init__(message)


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _opaque(prefix: str, text: Any) -> str:
    return prefix + hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]


def _read_json(path: Path, default=None, limit=MAX_JSON_BYTES):
    if not path.is_file():
        return default
    if path.stat().st_size > limit:
        raise ApiError(413, "This JSON artifact exceeds the preview size limit")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ApiError(422, "Invalid JSON artifact: " + path.name) from exc


def _atomic_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _iso_now():
    return engine.utc_now()


def _family(name):
    for family in FAMILY_NAMES:
        if name == family or name.startswith(family + "_"):
            return family
    return engine.family_name(name)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _page(items, offset=0, limit=50):
    return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit}


def _params(query):
    def integer(name, default, low, high):
        try:
            value = int(query.get(name, [default])[0])
        except (ValueError, TypeError):
            raise ApiError(400, f"Invalid {name}")
        if value < low or value > high:
            raise ApiError(400, f"{name} must be between {low} and {high}")
        return value
    return integer("offset", 0, 0, 10_000_000), integer("limit", 50, 1, 200)


def _walk_packages(root: Path):
    """Bounded discovery; data/label trees and symlinks are never traversed."""
    if not root.is_dir():
        return
    visited = 0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        path = Path(directory)
        visited += 1
        if visited > 12_000:
            break
        depth = len(path.relative_to(root).parts)
        subdirs[:] = [n for n in subdirs if not n.startswith(".") and not (path / n).is_symlink()]
        suffixes = {Path(n).suffix.lower() for n in files}
        nested = "data" in subdirs and "label" in subdirs
        flat = ".json" in suffixes and (bool(suffixes & {".png", ".csv"}) or _family(path.name) in FAMILY_NAMES)
        if nested or flat:
            yield path
            subdirs[:] = []
        elif depth >= 5:
            subdirs[:] = []


def _walk_runs(root: Path):
    if not root.is_dir():
        return
    visited = 0
    for directory, subdirs, files in os.walk(root, followlinks=False):
        path = Path(directory)
        visited += 1
        if visited > 12_000:
            break
        depth = len(path.relative_to(root).parts)
        subdirs[:] = [n for n in subdirs if not n.startswith(".") and not (path / n).is_symlink()
                      and n not in {"data", "label", "evidence", "previews", "montages", "extracted"}]
        if "00_MANIFEST.yaml" in files:
            yield path
            subdirs[:] = []
        elif depth >= 5:
            subdirs[:] = []


class Application:
    """Server state. Historical runs are read-only; only state_dir is writable."""

    def __init__(self, data_roots, run_roots, state_dir, *, start_worker=True):
        self.data_roots = tuple(dict.fromkeys(Path(p).expanduser().resolve() for p in data_roots))
        self.run_roots = tuple(dict.fromkeys(Path(p).expanduser().resolve() for p in run_roots))
        if not self.data_roots:
            raise ValueError("At least one data root is required")
        if any(not p.is_dir() for p in self.data_roots + self.run_roots):
            raise ValueError("All data and historical run roots must be existing directories")
        self.state_dir = Path(state_dir).expanduser().resolve()
        if any(_inside(self.state_dir, root) or _inside(root, self.state_dir) for root in self.data_roots):
            raise ValueError("State directory and input roots must not overlap")
        if any(_inside(self.state_dir, root) for root in self.run_roots):
            # An enclosing search root is safe; an existing run is never writable.
            if any((root / "00_MANIFEST.yaml").is_file() for root in self.run_roots if _inside(self.state_dir, root)):
                raise ValueError("State directory must not be inside a historical run")
        self.jobs_dir = self.state_dir / "jobs"
        self.outputs_dir = self.state_dir / "runs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache_lock = threading.RLock()
        self.datasets: dict[str, dict] = {}
        self.runs: dict[str, Path] = {}
        self.jobs: dict[str, dict] = {}
        self._sample_cache: dict[str, tuple[float, dict]] = {}
        self._finding_cache: collections.OrderedDict = collections.OrderedDict()
        self._discovered_at = 0.0
        self.stop_event = threading.Event()
        self.wake_event = threading.Event()
        self.process: subprocess.Popen | None = None
        self.active_job_id: str | None = None
        self._worker = None
        self._load_jobs()
        if start_worker:
            self._worker = threading.Thread(target=self._work, name="spaceqa-job-worker", daemon=True)
            self._worker.start()

    def _load_jobs(self):
        for path in sorted(self.jobs_dir.glob("job-*.json")):
            try:
                job = _read_json(path)
                if not isinstance(job, dict) or not re.fullmatch(r"job-[a-f0-9]{24}", job.get("id", "")):
                    continue
                if job.get("status") in {"queued", "running"}:
                    job.update(status="interrupted", stage="interrupted", finishedAt=_iso_now(),
                               error="Service restarted; previous process state cannot be trusted. Submit a new evaluation.")
                    _atomic_json(path, job)
                self.jobs[job["id"]] = job
            except (OSError, ApiError, ValueError):
                continue

    def close(self):
        self.stop_event.set()
        self.wake_event.set()
        with self.lock:
            process = self.process
            if process and process.poll() is None:
                process.terminate()
        if self._worker:
            self._worker.join(timeout=8)

    def health(self):
        return {"status": "ok", "service": "spaceqa-web", "version": __version__, "serverTime": _iso_now(),
                "dataRoots": [str(p) for p in self.data_roots], "runRoots": [str(p) for p in self.run_roots],
                "stateDirectory": str(self.state_dir), "capabilities": {
                    "modes": [{"id": "rules", "label": "Deterministic rules + independent audit", "available": True},
                              {"id": "rules_priors", "label": "Rules + audit + independent current-input priors", "available": True}],
                     "backendStrategy": {"version": backend_strategy.VERSION, "default": "enhanced",
                                          "rollback": "legacy", "formalSignals": True,
                                          "scoreEffect": "gated high-evidence signals only"},
                    "liveExpertInference": False, "historicalExpertResults": True,
                    "preview": ["PNG", "CSV", "JSON annotations"],
                    "warnings": ["New jobs execute rules and optional independent priors. Historical expert candidates may be cached; they are not fresh GPU inference."]},
                "system": self._system_summary()}

    def _system_summary(self):
        result = {"hostname": socket.gethostname(), "cpuCount": os.cpu_count(), "loadAverage": None,
                  "memoryTotalBytes": None, "memoryAvailableBytes": None,
                  "storageTotalBytes": None, "storageFreeBytes": None,
                  "activeJobId": self.active_job_id}
        try:
            result["loadAverage"] = list(os.getloadavg())
        except (AttributeError, OSError):
            pass
        try:
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                memory[key] = int(value.strip().split()[0]) * 1024
            result.update(memoryTotalBytes=memory.get("MemTotal"), memoryAvailableBytes=memory.get("MemAvailable"))
        except (OSError, ValueError, IndexError):
            pass
        try:
            usage = shutil.disk_usage(self.state_dir)
            result.update(storageTotalBytes=usage.total, storageFreeBytes=usage.free)
        except OSError:
            pass
        return result

    def discover(self, force=False):
        with self.cache_lock:
            if not force and time.monotonic() - self._discovered_at < 300:
                return
            datasets = {}
            for root in self.data_roots:
                for path in _walk_packages(root):
                    canonical = path.resolve()
                    if not _inside(canonical, root):
                        continue
                    identity = _opaque("ds-", canonical)
                    nested = (canonical / "data").is_dir() and (canonical / "label").is_dir()
                    data_dir, label_dir = (canonical / "data", canonical / "label") if nested else (canonical, canonical)
                    # Counts intentionally exclude arbitrary non-data resources.
                    with os.scandir(data_dir) as entries:
                        data_count = sum(Path(e.name).suffix.lower() in {".png", ".csv"} and e.is_file(follow_symlinks=False) for e in entries)
                    with os.scandir(label_dir) as entries:
                        label_count = sum(e.name.lower().endswith(".json") and e.is_file(follow_symlinks=False) for e in entries)
                    datasets[identity] = {"id": identity, "name": canonical.name, "family": _family(canonical.name),
                                          "path": str(canonical), "labelCount": label_count, "dataCount": data_count,
                                          "source": root.name, "layout": "data-label" if nested else "flat"}
            runs = {}
            for root in self.run_roots + (self.outputs_dir,):
                for path in _walk_runs(root):
                    canonical = path.resolve()
                    if _inside(canonical, root):
                        runs[_opaque("run-", canonical)] = canonical
            self.datasets, self.runs = datasets, runs
            self._discovered_at = time.monotonic()

    def list_datasets(self):
        self.discover()
        return sorted(self.datasets.values(), key=lambda x: (x["family"], x["name"], x["path"]))

    def dataset(self, identity):
        self.discover()
        value = self.datasets.get(identity)
        if value is None:
            raise ApiError(404, "Unknown dataset")
        path = Path(value["path"])
        if not any(_inside(path, root) for root in self.data_roots):
            raise ApiError(403, "Dataset is outside configured input roots")
        return value

    def run_path(self, identity):
        self.discover()
        path = self.runs.get(identity)
        if path is None:
            raise ApiError(404, "Unknown run")
        if not any(_inside(path, root) for root in self.run_roots + (self.outputs_dir,)):
            raise ApiError(403, "Run is outside configured roots")
        return path

    @staticmethod
    def _safe_artifact(root, relative):
        path = root / relative
        if not _inside(path, root):
            raise ApiError(403, "Artifact escapes its registered directory")
        return path

    def _manifest(self, root):
        value = _read_json(self._safe_artifact(root, "00_MANIFEST.yaml"), {})
        return value if isinstance(value, dict) else {}

    def _cards(self, root):
        folder = self._safe_artifact(root, "04_评分卡")
        if not folder.is_dir():
            return []
        return [_read_json(p) for p in sorted(folder.glob("*.json"))[:1000] if _inside(p, root)]

    def _run_item(self, identity, root, manifest=None):
        manifest = manifest if manifest is not None else self._manifest(root)
        cards = self._cards(root)
        return {"id": identity, "name": root.name, "path": str(root), "runId": manifest.get("runId"),
                "datasetCount": len(manifest.get("datasets", [])),
                "findingCount": sum(sum((c.get("checkStatusSummary") or {}).values()) for c in cards if isinstance(c, dict)),
                "createdAt": manifest.get("createdAt") or manifest.get("createdUtc") or manifest.get("generatedAt"),
                "mode": manifest.get("executionMode", "historical / inspect metadata"),
                "systemVersion": manifest.get("agentVersion"), "ruleVersion": manifest.get("ruleVersion")}

    def list_runs(self):
        self.discover()
        items = []
        for identity, root in self.runs.items():
            try:
                items.append(self._run_item(identity, root))
            except (OSError, ApiError, ValueError):
                continue
        return sorted(items, key=lambda x: (x.get("createdAt") or "", x["name"]), reverse=True)

    def findings(self, identity):
        root = self.run_path(identity)
        path = self._safe_artifact(root, "06_问题清单/findings.jsonl")
        if not path.is_file():
            return []
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if stat.st_size > MAX_FINDINGS_BYTES:
            raise ApiError(413, "Finding artifact exceeds the browser cache limit")
        with self.cache_lock:
            if key in self._finding_cache:
                self._finding_cache.move_to_end(key)
                return self._finding_cache[key]
            rows = []
            with path.open(encoding="utf-8-sig") as handle:
                for line in handle:
                    if line.strip():
                        try:
                            value = json.loads(line)
                        except ValueError:
                            continue  # A worker may currently be writing its final line.
                        if isinstance(value, dict):
                            rows.append(value)
            self._finding_cache[key] = rows
            while len(self._finding_cache) > 3:
                self._finding_cache.popitem(last=False)
            return rows

    def run_detail(self, identity):
        root = self.run_path(identity)
        manifest = self._manifest(root)
        findings = self.findings(identity)
        ranking_path = self._safe_artifact(root, "05_排序表/ranking_global.csv")
        ranking = []
        if ranking_path.is_file() and ranking_path.stat().st_size < MAX_JSON_BYTES:
            with ranking_path.open(encoding="utf-8-sig", newline="") as stream:
                ranking = list(csv.DictReader(stream))[:1000]
        logs = []
        log = self._safe_artifact(root, "08_运行日志/run.log.jsonl")
        if log.is_file():
            with log.open(encoding="utf-8-sig") as handle:
                for line in collections.deque(handle, maxlen=100):
                    try:
                        logs.append(json.loads(line))
                    except ValueError:
                        pass
        audit = None
        for name in ("web_audit.json", "integration_audit.json", "audit.json", "audit_result.json"):
            candidate = self._safe_artifact(root, name)
            if candidate.is_file():
                audit = _read_json(candidate)
                break
        warnings = ["Scores follow the project's reproducible specification, not official hidden-set scores."]
        if any(str(f.get("defectType", "")).startswith("MODEL_") for f in findings):
            warnings.append("Model candidate findings may be cached historical inference. Check executionMode; pending candidates do not directly reduce the score.")
        cards = self._cards(root)
        return {**self._run_item(identity, root, manifest), "findingCount": len(findings), "manifest": manifest,
                "scorecards": cards, "ranking": ranking, "statusCounts": dict(collections.Counter(f.get("status", "unknown") for f in findings)),
                "defectCounts": dict(collections.Counter(f.get("defectType", "unknown") for f in findings)),
                "audit": audit, "logs": logs, "warnings": warnings}

    def filter_findings(self, identity, query):
        rows = self.findings(identity)
        for parameter, key in (("status", "status"), ("dataset", "datasetId"), ("defect", "defectType")):
            value = query.get(parameter, [""])[0]
            if value:
                rows = [r for r in rows if r.get(key) == value]
        phrase = query.get("q", [""])[0].casefold()[:300]
        if phrase:
            rows = [r for r in rows if phrase in json.dumps(r, ensure_ascii=False).casefold()]
        return _page(rows, *_params(query))

    def evidence(self, identity, finding_id):
        root = self.run_path(identity)
        finding = next((r for r in self.findings(identity) if r.get("findingId") == finding_id), None)
        if finding is None:
            raise ApiError(404, "Unknown finding")
        evidence_id = finding.get("evidenceRef", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", evidence_id):
            raise ApiError(422, "Invalid evidence reference")
        path = self._safe_artifact(root, "07_证据索引/evidence/" + evidence_id + ".json")
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise ApiError(404, "Evidence payload not found")
        expected = None
        index = self._safe_artifact(root, "07_证据索引/evidence_index.csv")
        if index.is_file() and index.stat().st_size < MAX_FINDINGS_BYTES:
            with index.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    if row.get("evidenceId") == evidence_id:
                        expected = row.get("contentHash")
                        break
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matched = bool(expected and secrets.compare_digest(expected, digest))
        matched = matched and all(payload.get(k) == finding.get(k) for k in ("findingId", "datasetId", "checkId", "ruleVersion"))
        sample = self._evidence_sample(finding, payload)
        return {"finding": finding, "evidence": payload, "sourceVerified": matched,
                "verificationScope": "Saved evidence payload SHA and identity fields; scientific correctness is not established by this check.",
                "sample": sample}

    def _evidence_sample(self, finding, payload):
        self.discover()
        locator_text = str(payload.get("locator", "")).split("#", 1)[0]
        locator = Path(locator_text)
        matches = [ds for ds in self.datasets.values() if ds["name"] == finding.get("datasetId")]
        exact_roots = [ds for ds in matches if locator.is_absolute() and _inside(locator, Path(ds["path"]))]
        if exact_roots:
            matches = exact_roots
        elif len(matches) != 1:
            # Same-name packages may exist in old/new releases. Never guess
            # which one produced a historical finding from a filename substring.
            return None
        for ds in matches:
            catalog = self.sample_catalog(ds["id"])
            for sample_id, sample in catalog.items():
                label = sample.get("labelName")
                data = sample.get("dataName")
                if locator.name not in {label, data}:
                    continue
                if exact_roots and sample.get("_label") and locator.suffix.lower() == ".json" and locator.resolve() != sample["_label"].resolve():
                    continue
                if locator.name:
                    return {"datasetId": ds["id"], "sampleId": sample_id,
                            "url": f"/api/datasets/{ds['id']}/samples/{sample_id}",
                            "matchMethod": "canonical-source-path" if exact_roots else "unique-package-and-exact-filename"}
        return None

    def sample_catalog(self, identity):
        ds = self.dataset(identity)
        with self.cache_lock:
            cached = self._sample_cache.get(identity)
            if cached and time.monotonic() - cached[0] < 300:
                return cached[1]
            root = Path(ds["path"])
            nested = ds["layout"] == "data-label"
            label_dir = root / "label" if nested else root
            data_dir = root / "data" if nested else root
            if not _inside(data_dir, root) or not _inside(label_dir, root):
                raise ApiError(403, "Data or label directory escapes its registered dataset")
            with os.scandir(data_dir) as entries:
                data = {e.name: Path(e.path) for e in entries if Path(e.name).suffix.lower() in {".png", ".csv"} and e.is_file(follow_symlinks=False)}
            with os.scandir(label_dir) as entries:
                labels = [Path(e.path) for e in entries if e.name.lower().endswith(".json") and e.is_file(follow_symlinks=False)]
            result, inferred = {}, set()
            for p in sorted(labels):
                candidate = p.name[:-5]
                if candidate not in data:
                    candidate = next((n for n in (p.stem + ".png", p.stem + ".csv") if n in data), candidate)
                inferred.add(candidate)
                sample_id = _opaque("sample-", str(p.relative_to(root)))
                result[sample_id] = {"id": sample_id, "name": p.name, "labelName": p.name, "dataName": candidate,
                                     "kind": "timeseries" if candidate.lower().endswith(".csv") else "image" if candidate.lower().endswith(".png") else "unknown",
                                     "missingData": candidate not in data, "hasLabel": True, "_label": p}
            for name, p in sorted(data.items()):
                if name in inferred:
                    continue
                sample_id = _opaque("sample-", "data/" + name)
                result[sample_id] = {"id": sample_id, "name": name, "labelName": None, "dataName": name,
                                     "kind": "image" if p.suffix.lower() == ".png" else "timeseries",
                                     "missingData": False, "hasLabel": False, "_data": p}
            self._sample_cache[identity] = (time.monotonic(), result)
            return result

    def _resolve_sample(self, identity, sample_id):
        ds = self.dataset(identity)
        row = self.sample_catalog(identity).get(sample_id)
        if row is None:
            raise ApiError(404, "Unknown sample")
        row = dict(row)
        root = Path(ds["path"])
        data_dir = root / "data" if ds["layout"] == "data-label" else root
        warnings = []
        label = {}
        if row.get("_label"):
            label_path = row["_label"]
            if not _inside(label_path, root):
                raise ApiError(403, "Sample label escapes its dataset")
            try:
                value = _read_json(label_path, {})
                if isinstance(value, dict):
                    label = value
                else:
                    warnings.append("The label is not a JSON object.")
            except ApiError as exc:
                warnings.append(exc.message)
            ref = engine.label_reference(label)
            if isinstance(ref, str) and ref:
                # Known release metadata contains old absolute paths. Only an
                # in-package basename may resolve; never follow metadata paths.
                row["dataName"] = ref.replace("\\", "/").rsplit("/", 1)[-1]
        data_path = row.get("_data") or data_dir / row["dataName"]
        if not _inside(data_path, root):
            raise ApiError(403, "Sample data escapes its dataset")
        row["missingData"] = not data_path.is_file()
        row["kind"] = "image" if data_path.suffix.lower() == ".png" else "timeseries" if data_path.suffix.lower() == ".csv" else "unknown"
        if row["missingData"]:
            warnings.append("Referenced data file is missing. The label is retained for inspection.")
        return ds, row, data_path, label, warnings

    def list_samples(self, identity, query):
        rows = list(self.sample_catalog(identity).values())
        q = query.get("q", [""])[0].casefold()[:300]
        if q:
            rows = [r for r in rows if q in r["name"].casefold() or q in r["dataName"].casefold()]
        offset, limit = _params(query)
        items = []
        for r in rows[offset:offset + limit]:
            _, resolved, _, _, _ = self._resolve_sample(identity, r["id"])
            items.append({k: v for k, v in resolved.items() if not k.startswith("_")})
        return {"items": items, "total": len(rows), "offset": offset, "limit": limit}

    @staticmethod
    def _annotations(label):
        result = []
        colors = ("#33c5ad", "#f5bd48", "#9b8aff", "#62a9ff", "#ff8d9a")
        for pointer, shape in engine.science_shapes(label):
            points = engine.shape_polygon(shape)
            if points:
                result.append({"type": "polygon", "points": points, "label": str(shape.get("label", "region")),
                               "color": colors[len(result) % len(colors)], "pointer": pointer})
        annotation_name, annotation = engine.event_annotation(label)
        list_name, events = engine.event_records(annotation)
        for index, event in enumerate(events):
            points = engine.numeric_polygon(event.get("bbox_polygon_pixel"))
            if points:
                result.append({"type": "polygon", "points": points, "label": str(engine.event_category(event) or "event"),
                               "color": colors[len(result) % len(colors)],
                               "pointer": f"$.{annotation_name}.{list_name}[{index}].bbox_polygon_pixel"})
        for pointer, record in engine.walk_dicts(label):
            bbox = record.get("bbox")
            if (isinstance(bbox, list) and len(bbox) == 4 and
                    all(type(v) in (int, float) and math.isfinite(v) for v in bbox)):
                x, y, width, height = bbox
                if width > 0 and height > 0:
                    result.append({"type": "box", "points": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                                   "label": str(record.get("label") or record.get("event_type") or "bbox"),
                                   "color": colors[len(result) % len(colors)], "pointer": pointer + ".bbox"})
        return result[:1000]

    @staticmethod
    def _series(path):
        if path.stat().st_size > 64 * 1024 * 1024:
            return [], 0, ["CSV exceeds 64MB browser preview limit; use offline tools for the complete file."]
        series, count, stride = [], 0, 1
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            first = next(reader, None)
            if first is None:
                return [], 0, []
            numeric = []
            for index, cell in enumerate(first):
                try:
                    val = float(cell)
                    if math.isfinite(val):
                        numeric.append(index)
                except (TypeError, ValueError):
                    pass
            # Time-like first column is an axis, not an additional measurement.
            first_header = header[0].strip() if header else ""
            time_like = (first_header == "t" or first_header.lower() in {"t_s", "second", "seconds"} or
                         re.search(r"time|date|index|epoch", first_header, re.I))
            if len(numeric) > 1 and numeric[0] == 0 and time_like:
                numeric = numeric[1:]
            numeric = numeric[:6]
            series = [{"name": header[i] if i < len(header) else f"column_{i + 1}", "points": []} for i in numeric]
            for row in _prepend(first, reader):
                if count >= 1_000_000:
                    break
                if count % stride == 0:
                    for s, index in zip(series, numeric):
                        try:
                            value = float(row[index])
                            if math.isfinite(value):
                                s["points"].append([count, value])
                        except (IndexError, ValueError):
                            pass
                count += 1
                if any(len(s["points"]) > 2400 for s in series):
                    for s in series:
                        s["points"] = s["points"][::2]
                    stride *= 2
        return series, count, (["CSV preview uses deterministic subsampling; x-axis is source row index, not calibrated physical time."] if stride > 1 else ["CSV x-axis is source row index."])

    def sample_detail(self, identity, sample_id):
        ds, row, path, label, warnings = self._resolve_sample(identity, sample_id)
        width = height = None
        image_url = None
        series, rows = [], 0
        if not row["missingData"] and row["kind"] == "image":
            try:
                dims = engine.png_header(path)
                width, height = dims["width"], dims["height"]
                image_url = f"/api/datasets/{identity}/samples/{sample_id}/image"
            except (OSError, ValueError) as exc:
                warnings.append("PNG header could not be decoded: " + str(exc))
        elif not row["missingData"] and row["kind"] == "timeseries":
            try:
                series, rows, chart_warnings = self._series(path)
                warnings.extend(chart_warnings)
            except (OSError, UnicodeError, csv.Error) as exc:
                warnings.append("CSV preview unavailable: " + str(exc))
        return {**{k: v for k, v in row.items() if not k.startswith("_")}, "datasetId": identity,
                "imageUrl": image_url, "label": label, "annotations": self._annotations(label),
                "width": width, "height": height, "series": series, "totalRows": rows,
                "warnings": warnings, "annotationMeaning": "Original annotation overlay, not a newly detected defect mask."}

    def sample_image(self, identity, sample_id):
        _, row, path, _, _ = self._resolve_sample(identity, sample_id)
        if row["missingData"] or row["kind"] != "image":
            raise ApiError(404, "PNG data is unavailable")
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ApiError(413, "PNG exceeds the 64MB preview limit")
        engine.png_header(path)
        return path

    def list_jobs(self):
        with self.lock:
            return [self._public_job(self.jobs[k]) for k in sorted(self.jobs, key=lambda k: self.jobs[k].get("createdAt", ""), reverse=True)]

    def _public_job(self, job):
        result = {k: v for k, v in job.items() if not k.startswith("_")}
        if job.get("status") == "running":
            logfile = self.jobs_dir / (job["id"] + ".log")
            if logfile.is_file():
                with logfile.open("rb") as stream:
                    stream.seek(max(0, logfile.stat().st_size - 16_384))
                    result["logTail"] = stream.read().decode("utf-8", errors="replace").splitlines()[-60:]
            output = Path(job["_output"])
            logs_path = output / "08_运行日志/run.log.jsonl"
            if logs_path.is_file() and _inside(logs_path, self.outputs_dir):
                completed = set()
                with logs_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            row = json.loads(line)
                            if row.get("status") == "实际完成":
                                completed.add((row.get("datasetId"), row.get("checkId")))
                        except ValueError:
                            pass
                result["completedChecks"] = len(completed)
                result["expectedChecks"] = len(engine.ALL_CHECKS)
        return result

    def job(self, identity):
        with self.lock:
            job = self.jobs.get(identity)
            if job is None:
                raise ApiError(404, "Unknown job")
            return self._public_job(job)

    def submit(self, payload):
        if not isinstance(payload, dict):
            raise ApiError(400, "Job body must be an object")
        unknown = set(payload) - {"datasetId", "mode", "seed", "backendProfile"}
        if unknown:
            raise ApiError(400, "Unsupported job fields: " + ", ".join(sorted(unknown)))
        mode = payload.get("mode", "rules")
        if mode not in {"rules", "rules_priors"}:
            raise ApiError(400, "Only rules and rules_priors are currently enabled for new evaluations")
        identity = payload.get("datasetId")
        if not isinstance(identity, str):
            raise ApiError(400, "datasetId must be a registered dataset ID")
        ds = self.dataset(identity)
        if mode == "rules_priors" and ds["family"] not in FAMILY_NAMES:
            raise ApiError(400, "Prior checks require a supported scientific data family")
        backend_profile = payload.get("backendProfile", "enhanced")
        if backend_profile not in {"legacy", "enhanced"}:
            raise ApiError(400, "backendProfile must be enhanced or legacy")
        seed = payload.get("seed", 20261021)
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ApiError(400, "Seed must be an integer between 0 and 4294967295")
        with self.lock:
            if sum(j.get("status") in {"queued", "running"} for j in self.jobs.values()) >= 8:
                raise ApiError(409, "Evaluation queue is full; wait for a running job to finish")
            job_id = "job-" + secrets.token_hex(12)
            output = self.outputs_dir / job_id
            if output.exists() or any(_inside(output, p) for p in self.data_roots):
                raise ApiError(409, "Cannot allocate a new output directory")
            value = {"id": job_id, "status": "queued", "stage": "queued", "createdAt": _iso_now(),
                     "startedAt": None, "finishedAt": None, "datasetId": identity, "datasetName": ds["name"],
                     "mode": mode, "seed": seed, "backendProfile": backend_profile,
                     "runId": None, "error": None, "logTail": [],
                     "timeoutSeconds": 10800,
                     "_input": ds["path"], "_output": str(output)}
            self.jobs[job_id] = value
            self._save_job(value)
            self.wake_event.set()
            return self._public_job(value)

    def _save_job(self, job):
        _atomic_json(self.jobs_dir / (job["id"] + ".json"), job)

    def cancel(self, identity):
        with self.lock:
            value = self.jobs.get(identity)
            if value is None:
                raise ApiError(404, "Unknown job")
            if value.get("status") in FINISHED:
                return self._public_job(value)
            value.update(status="cancelled", stage="cancelled", finishedAt=_iso_now())
            self._save_job(value)
            if self.active_job_id == identity and self.process and self.process.poll() is None:
                self.process.terminate()
            return self._public_job(value)

    def _process_step(self, job, argv, stage, capture_path=None):
        with self.lock:
            if job["status"] == "cancelled" or self.stop_event.is_set():
                return False
            job["stage"] = stage
            self._save_job(job)
        env = dict(os.environ)
        package_parent = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        log_path = self.jobs_dir / (job["id"] + ".log")
        with contextlib.ExitStack() as stack:
            log = stack.enter_context(log_path.open("a", encoding="utf-8"))
            capture = stack.enter_context(capture_path.open("w", encoding="utf-8")) if capture_path else log
            log.write(f"\n[{_iso_now()}] {stage}\n")
            log.flush()
            with self.lock:
                self.process = subprocess.Popen(argv, stdout=capture, stderr=log, env=env,
                                                stdin=subprocess.DEVNULL, cwd=package_parent, shell=False)
                process = self.process
            while process.poll() is None:
                if time.monotonic() > job.get("_deadline", float("inf")):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise RuntimeError("Evaluation exceeded its 3-hour total execution limit")
                if self.stop_event.wait(0.3):
                    process.terminate()
                    break
                with self.lock:
                    if job["status"] == "cancelled" and process.poll() is None:
                        process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5)
            with self.lock:
                self.process = None
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        with self.lock:
            job["logTail"] = tail
            self._save_job(job)
        if returncode:
            if job["status"] != "cancelled" and not self.stop_event.is_set():
                raise RuntimeError(f"{stage} exited with code {returncode}. See the job log.")
            return False
        return True

    def _work(self):
        while not self.stop_event.is_set():
            with self.lock:
                queued = [j for j in self.jobs.values() if j.get("status") == "queued"]
                job = min(queued, key=lambda j: j["createdAt"]) if queued else None
                if job:
                    job.update(status="running", stage="scan", startedAt=_iso_now())
                    job["_deadline"] = time.monotonic() + job.get("timeoutSeconds", 10800)
                    self.active_job_id = job["id"]
                    self._save_job(job)
            if job is None:
                self.wake_event.wait(1)
                self.wake_event.clear()
                continue
            try:
                ds = self.dataset(job["datasetId"])
                if ds["path"] != job["_input"]:
                    raise RuntimeError("Registered input changed before job execution")
                output = Path(job["_output"])
                if not _inside(output, self.outputs_dir) or output.exists():
                    raise RuntimeError("Output directory is not a new service-owned path")
                scan_argv = [sys.executable, "-m", "spaceqa_agent", "scan", "--input", ds["path"],
                             "--output", str(output), "--seed", str(job["seed"]),
                             "--backend-profile", job.get("backendProfile", "enhanced")]
                if not self._process_step(job, scan_argv, "scan"):
                    continue
                audit_capture = self.jobs_dir / (job["id"] + ".audit.json")
                if not self._process_step(job, [sys.executable, "-m", "spaceqa_agent", "audit", str(output)], "audit", audit_capture):
                    continue
                result = _read_json(audit_capture)
                _atomic_json(output / "web_audit.json", result)
                if not isinstance(result, dict) or result.get("status") != "pass":
                    raise RuntimeError("Independent audit failed")
                if job["mode"] == "rules_priors":
                    prior_output = self.jobs_dir / (job["id"] + ".priors.jsonl")
                    prior_capture = self.jobs_dir / (job["id"] + ".priors.summary.json")
                    prior_command = [sys.executable, "-m", "spaceqa_agent", "check-priors", "--input", ds["path"],
                                     "--family", ds["family"], "--backend-profile", job.get("backendProfile", "enhanced"),
                                     "--output", str(prior_output)]
                    if not self._process_step(job, prior_command, "priors", prior_capture):
                        continue
                    job["priorSummary"] = _read_json(prior_capture, {})
                    job["priorSummary"]["scoringEffect"] = "Prior observations are independent and original rule scores remain unchanged; formal scan backend signals are recorded in the scorecard."
                with self.lock:
                    if job["status"] != "cancelled":
                        job.update(status="succeeded", stage="complete", finishedAt=_iso_now(),
                                   runId=_opaque("run-", output.resolve()), audit=result)
                        self._save_job(job)
                with self.cache_lock:
                    self.runs[_opaque("run-", output.resolve())] = output.resolve()
            except Exception as exc:
                with self.lock:
                    if job["status"] != "cancelled":
                        job.update(status="failed", stage="failed", error=str(exc), finishedAt=_iso_now())
                        self._save_job(job)
            finally:
                with self.lock:
                    if self.stop_event.is_set() and job["status"] == "running":
                        job.update(status="interrupted", stage="interrupted", finishedAt=_iso_now(), error="Service stopped during evaluation")
                        self._save_job(job)
                    self.active_job_id = None


def _prepend(first, iterator):
    yield first
    yield from iterator


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "SpaceQA"
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.application

    def log_message(self, fmt, *args):
        # No bodies or query parameters (which may contain private filenames).
        sys.stderr.write(f"[{_iso_now()}] {self.command} {urlsplit(self.path).path} {args[1] if len(args) > 1 else ''}\n")

    def _send(self, status, payload):
        body = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if status >= 400:
            self.close_connection = True
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'")
        self.end_headers()
        with path.open("rb") as stream:
            shutil.copyfileobj(stream, self.wfile)

    def _route(self, post=False):
        self._check_host()
        parsed = urlsplit(self.path)
        path = unquote(parsed.path)
        if "\x00" in path or "\\" in path or any(p == ".." for p in path.split("/")):
            raise ApiError(400, "Invalid URL path")
        parts = [p for p in path.split("/") if p]
        query = parse_qs(parsed.query)
        if post:
            self._check_origin()
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "Invalid request length")
            if length < 0 or length > 16_384:
                raise ApiError(413, "Request body exceeds 16KB")
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeError):
                raise ApiError(400, "Request body must be JSON")
            if parts == ["api", "jobs"]:
                return self._send(202, self.app.submit(payload))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "cancel":
                return self._send(200, self.app.cancel(parts[2]))
            raise ApiError(404, "Unknown operation")
        if parts == ["api", "health"]:
            return self._send(200, self.app.health())
        if parts == ["api", "datasets"]:
            if query.get("refresh", [""])[0] == "1":
                self.app.discover(force=True)
            return self._send(200, _page(self.app.list_datasets(), *_params(query)))
        if parts == ["api", "runs"]:
            return self._send(200, _page(self.app.list_runs(), *_params(query)))
        if parts == ["api", "jobs"]:
            return self._send(200, _page(self.app.list_jobs(), *_params(query)))
        if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
            return self._send(200, self.app.job(parts[2]))
        if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
            identity = parts[2]
            if len(parts) == 3:
                return self._send(200, self.app.run_detail(identity))
            if len(parts) == 4 and parts[3] == "findings":
                return self._send(200, self.app.filter_findings(identity, query))
            if len(parts) == 5 and parts[3] == "evidence":
                return self._send(200, self.app.evidence(identity, parts[4]))
        if len(parts) >= 4 and parts[:2] == ["api", "datasets"] and parts[3] == "samples":
            identity = parts[2]
            if len(parts) == 4:
                return self._send(200, self.app.list_samples(identity, query))
            if len(parts) == 5:
                return self._send(200, self.app.sample_detail(identity, parts[4]))
            if len(parts) == 6 and parts[5] == "image":
                return self._file(self.app.sample_image(identity, parts[4]), "image/png")
        if parts and parts[0] == "api":
            raise ApiError(404, "Unknown API endpoint")
        static = Path(__file__).resolve().parent / "web"
        # Keep the former evaluation URL usable while serving the same
        # evidence ledger as the current console.  The compatibility page is
        # read-only and calls /api/runs; it never maintains a second result
        # store.  /demo.html is a short alias used in recordings.
        static_name = path.lstrip("/") if parts else "index.html"
        if static_name == "demo.html":
            static_name = "evaluation.html"
        resource = static / static_name
        if not _inside(resource, static) or not resource.is_file():
            raise ApiError(404, "Static resource not found")
        return self._file(resource)

    def _check_origin(self):
        # JSON is required, and cross-site requests/rebinding are rejected even
        # though this service deliberately binds loopback rather than public IP.
        if self.headers.get_content_type() != "application/json":
            raise ApiError(415, "Content-Type must be application/json")
        host = self.headers.get("Host", "")
        self._check_host()
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc != host:
                raise ApiError(403, "Cross-origin operations are not allowed")
        if self.headers.get("Sec-Fetch-Site") not in {None, "same-origin", "none"}:
            raise ApiError(403, "Cross-site operations are not allowed")

    def _check_host(self):
        host = self.headers.get("Host", "")
        if not host or any(c in host for c in " /\\\n"):
            raise ApiError(403, "Invalid Host")
        hostname = urlsplit("http://" + host).hostname
        if hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ApiError(403, "Use a localhost browser address or SSH local-forward")

    def _dispatch(self, post=False):
        try:
            self._route(post)
        except ApiError as exc:
            self._send(exc.status, {"error": exc.message})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._send(500, {"error": "Unable to read the requested artifact", "detail": str(exc)[:300]})

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(True)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self):
        self.application.close()
        super().server_close()


def create_server(host="127.0.0.1", port=8765, *, data_roots, run_roots=(), state_dir, start_worker=True):
    try:
        addresses = {ipaddress.ip_address(value[4][0]) for value in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    except (socket.gaierror, ValueError) as exc:
        raise ValueError("Invalid listen address") from exc
    if not addresses or any(not address.is_loopback for address in addresses):
        raise ValueError("This service must bind a loopback address; use an SSH local-forward")
    # stdlib server is IPv4 by default; localhost is deliberately normalized.
    if host == "localhost":
        host = "127.0.0.1"
    application = Application(data_roots, run_roots, state_dir, start_worker=start_worker)
    try:
        if ":" in host:
            class IPv6Server(Server):
                address_family = socket.AF_INET6
            server = IPv6Server((host, port), Handler)
        else:
            server = Server((host, port), Handler)
    except Exception:
        application.close()
        raise
    server.application = application
    return server


def serve(host="127.0.0.1", port=8765, *, data_roots, run_roots=(), state_dir):
    server = create_server(host, port, data_roots=data_roots, run_roots=run_roots, state_dir=state_dir)
    previous_term = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.getsignal(signal.SIGTERM)
        # shutdown() must run outside the serve_forever thread. On normal
        # service-manager termination this also stops the active scan child.
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(f"SpaceQA browser service: http://{server.server_address[0]}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
