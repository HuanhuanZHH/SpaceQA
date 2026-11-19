import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/staff/tangjiarui/OPC')
REPO = ROOT/'opc-agent'
OUT = REPO/'runs/full-matrix-20261031'
OLD = REPO/'runs/model-comparison-20261031-v2'
MANIFEST = ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
AUDIT = ROOT/'phase2_visual_audit_xzj'
FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample', 'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']
TIERS = ['P0', 'P1', 'P2', 'P3', 'P4']
SEEDS = [20261031, 20261101, 20261102]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


oldprep = module('full_matrix_oldprep', REPO/'tools/prepare_model_comparison.py')
ann = module('full_matrix_annotation_diff', AUDIT/'src/annotation_diff.py')
ann.PATHS['ULF_sample'] = ('ulf_event_annotation', 'matched_events')
rules = module('full_matrix_official_rules', REPO/'opc_agent.py')


def lines(path):
    with Path(path).open() as f:
        return [json.loads(s) for s in f if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def dump(name, obj):
    path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)); temp.replace(path)


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)
