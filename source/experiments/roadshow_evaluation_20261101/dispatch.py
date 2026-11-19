"""Deterministic dataset dispatch. Expert selection uses original A-val only."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path('/staff/tangjiarui/OPC/opc-agent')
BASE = REPO / 'runs/all-family-training-20261031'
OUT = REPO / 'runs/roadshow-evaluation-20261101'
FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample',
            'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)


def lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def identify(directory_name, annotation=None):
    matches = [f for f in FAMILIES if directory_name == f or directory_name.startswith(f + '_')]
    inferred = []
    if annotation is not None:
        if 'ulf_event_annotation' in annotation:
            inferred.append('ULF_sample')
        if 'mask' in annotation.get('science_annotation', {}):
            inferred.append('EPB_seg_sample')
        if 'hek_event_annotation' in annotation:
            # AIA/HMI have the same event schema, so the known dataset ID disambiguates them.
            inferred.extend(f for f in ['AIA_AR_sample', 'HMI_AR_sample'] if f in matches)
    if len(matches) != 1:
        if len(set(inferred)) == 1:
            return inferred[0]
        raise ValueError('Unknown or ambiguous dataset type: ' + directory_name)
    if inferred and matches[0] not in inferred:
        raise ValueError('Dataset ID conflicts with annotation schema')
    return matches[0]


def f1(y, valid, pred):
    tp = int((pred & y & valid).sum()); fp = int((pred & ~y & valid).sum())
    fn = int((~pred & y & valid).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fn else None


def freeze():
    progress = json.loads((BASE / 'progress/status.json').read_text())
    names = progress['evaluated_routes']
    immutable = json.loads((BASE / 'code_manifest.json').read_text())
    source = REPO / 'experiments/all_family_training_20261031'
    assert all(sha(source / n) == h for n, h in immutable.items())
    val = [{k: r[k] for k in ['id', 'family', 'tier', 'target', 'truth_valid', 'input_missing', 'input_sha256']}
           for r in lines(BASE / 'val.jsonl')]
    y = np.array([r['target'] for r in val], bool); valid = np.array([r['truth_valid'] for r in val], bool)
    family = np.array([FAMILIES.index(r['family']) for r in val]); tier = np.array([int(r['tier'][1]) for r in val])
    index = {r['id']: i for i, r in enumerate(val)}
    spec = importlib.util.spec_from_file_location('static_blind_rule_adapter', REPO / 'experiments/expert_routing_20261101/routing.py')
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    rule = adapter.blind_rules(val)
    choices = []; details = {}; weight_hashes = {}
    for name in ['rules'] + names:
        if name == 'rules':
            prediction = rule; threshold = [[1.01] * 3] * 6; size = 0
        else:
            report = json.loads((BASE / 'training' / (name + '.json')).read_text())
            assert report['status'] == 'complete' and report['test_read'] is False
            assert len(report['actual_train_coverage']) == 30
            completion = json.loads((BASE / 'evaluations' / name / 'complete.json').read_text())
            assert completion['route'] == name and completion['packages'] == 30
            probabilities = np.zeros((len(val), 3)); seen = set()
            for item in lines(BASE / 'predictions' / (name + '.jsonl')):
                i = index.get(item['id'])
                if i is None:
                    continue
                assert i not in seen and val[i]['input_sha256'] == item['input_sha256']
                seen.add(i)
                if item['status'] == 'success':
                    probabilities[i] = item['probabilities']
            assert len(seen) == len(val)
            threshold = report['thresholds']
            prediction = (probabilities >= np.array(threshold)[family]) | rule
            paths = [BASE / p for p in report['checkpoint_hashes']]
            size = sum(p.stat().st_size for p in paths)
            if name == 'qlora':
                # The adapter alone is not an independently runnable model package.
                size += sum(p.stat().st_size for p in (REPO.parent / 'models/Qwen3-VL-8B-Instruct').rglob('*') if p.is_file())
            for p in paths:
                actual = sha(p); assert actual == report['checkpoint_hashes'][str(p.relative_to(BASE))]
                weight_hashes[str(p.relative_to(BASE))] = actual
        details[name] = dict(thresholds=threshold, checkpoint_bytes=size)
        for fi, fname in enumerate(FAMILIES):
            scores = [f1(y[m], valid[m], prediction[m]) for t in range(1, 5) if (m := ((family == fi) & (tier == t))).any()]
            scores = [v for v in scores if v is not None]
            p0 = (family == fi) & (tier == 0) & valid.all(1) & ~y.any(1)
            choice = dict(family=fname, route=name, val_macro_F1=float(np.mean(scores)) if scores else 0.,
                                val_P0_FPR=float(prediction[p0].any(1).mean()) if p0.any() else None,
                                checkpoint_bytes=size, threshold_source='original A val')
            if name == 'qlora':
                choice.update(deployment_eligible=False,
                              reason='Research replay only until standalone persisted quantization and full image budget are verified')
            choices.append(choice)
    selected = {}
    for fname in FAMILIES:
        best = max([v for v in choices if v['family'] == fname and v.get('deployment_eligible', True)],
                   key=lambda r: (r['val_macro_F1'], -(r['val_P0_FPR'] or 0), -r['checkpoint_bytes'], r['route']))
        selected[fname] = dict(best, thresholds=details[best['route']]['thresholds'][FAMILIES.index(fname)])
    stage = f"{len(names):02d}_models"
    frozen = dict(status='frozen', stage=stage, expert_count=len(names), experts=names, selected=selected,
                  original_val_sha256=sha(BASE / 'val.jsonl'), checkpoint_hashes=weight_hashes,
                  model_details=details, all_choices=choices,
                  selector='known dataset ID plus schema cross-check; no learned router',
                  selection_source='Original A validation only; no B data or B outcomes used',
                  roadshow_ground_truth_available=False,
                  note='Selection F1 is original A-val technical proxy, never roadshow F1')
    target = OUT / 'dispatch' / f'dispatch_{stage}.json'
    if target.exists():
        assert json.loads(target.read_text()) == frozen, 'Frozen stage changed; open a new explicit stage'
    else:
        write(target, frozen)
    write(OUT / 'dispatch_current.json', frozen)
    print(json.dumps(dict(stage=stage, selected={f: dict(route=v['route'], A_val_macro_F1=v['val_macro_F1'])
                                                for f, v in selected.items()})))
    return frozen


if __name__ == '__main__':
    freeze()
