"""Post-training, group-held-out expert routing. Never modifies base experiments."""
import argparse
import collections
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

BASE = Path('/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031')
OUT = Path('/staff/tangjiarui/OPC/opc-agent/runs/expert-routing-20261101')
FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample',
            'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']
ARCHS = ['unet', 'attention_unet', 'unetpp', 'resnet18_unet', 'deeplab_mobilenet',
         'yolo11_detect_backbone', 'yolo11_segment_backbone',
         'yolov8_segment_backbone', 'dino_finetune', 'ijepa_finetune']
ROUTES = ['qlora'] + [f'{s}_seed{seed}' for s in ['supervised', 'distilled', 'pruned']
                       for seed in [20261031, 20261101, 20261102]] + ARCHS
PROBE = 'supervised_seed20261031'
SEEDS = [20261101, 20261102, 20261103]
CONFIG = dict(version=1, pools=[2, 3, 4, 6], seeds=SEEDS, epochs=60, batch_size=512,
              learning_rate=0.001, hidden=64, meta_fractions=[0.6, 0.2, 0.2],
              split_seed=20261101, weight_budget_bytes=10_000_000_000,
              runtime_reserve_bytes=8_000_000_000, safety_reserve_bytes=2_000_000_000,
              image_limit_bytes=20_000_000_000, per_dataset_limit_seconds=10800,
              added_p0_false_positive_cap=0.02,
              objective='macro F1 over 24 family/P1-P4 cells; P1 tie-break; then smaller footprint',
              pool_selection='greedy masked-loss complement on meta_fit, weighted equally over 30 cells',
              features='current-only 256 structural features + six schema flags + probe probabilities + rule bits',
              forbidden_inputs=['tier', 'path', 'id', 'target', 'truth_valid', 'P0 comparison'],
              inference='routing features select experts before their outputs are read')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def state(status, **fields):
    write(OUT / 'state.json', dict(status=status, pid=os.getpid(), updated_unix=time.time(), **fields))
    print(json.dumps(dict(status=status, **fields), ensure_ascii=False), flush=True)


def partition(records):
    """Partition the original validation set by whole training_group_id."""
    membership = collections.defaultdict(set)
    for r in records:
        membership[r['training_group_id']].add(r['family'])
    groups = collections.defaultdict(set)
    for group, families in membership.items():
        groups[min(families)].add(group)
    assigned = {}
    for family, values in groups.items():
        ordered = sorted(values, key=lambda g: hashlib.sha256(f'{CONFIG["split_seed"]}:{g}'.encode()).hexdigest())
        assert len(ordered) >= 5, (family, len(ordered))
        first, second = max(1, int(len(ordered) * .6)), max(2, int(len(ordered) * .8))
        for i, group in enumerate(ordered):
            subset = 'meta_fit' if i < first else 'meta_cal' if i < second else 'meta_select'
            if group in assigned:
                assert assigned[group] == subset, 'Cross-family duplicate group requires unified assignment'
            assigned[group] = subset
    return np.array([assigned[r['training_group_id']] for r in records])


def compact(r):
    return {k: r[k] for k in ['id', 'family', 'tier', 'training_group_id', 'features',
                              'target', 'truth_valid', 'input_missing', 'input_sha256']}


def ready():
    required = {'training_barrier.json': 'pass', 'evaluation_complete.json': 'complete',
                'result_validation.json': 'pass', 'queue_state.json': 'complete'}
    waiting = []
    for name, expected in required.items():
        path = BASE / name
        if not path.exists() or json.loads(path.read_text()).get('status') != expected:
            waiting.append(name)
    return waiting


def footprint():
    result = []
    for name in ROUTES:
        path = BASE / 'checkpoints' / (name if name == 'qlora' else name + '.pt')
        files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else ([path] if path.exists() else [])
        count = sum(p.stat().st_size for p in files)
        # A LoRA adapter is unusable without its original 8B base. Do not count it alone.
        if name == 'qlora':
            result.append(dict(route=name, checkpoint_bytes=count, eligible=False,
                               reason='Adapter needs BF16 base (~17.55 GB); persisted quantized standalone artifact not validated'))
        else:
            result.append(dict(route=name, checkpoint_bytes=count, eligible=count > 0,
                               reason='Final full checkpoint; runtime and Docker tar remain to be measured'))
    return result


def preflight():
    validation = [compact(r) for r in rows(BASE / 'val.jsonl')]
    labels = partition(validation)
    coverage = {}
    for name in ['meta_fit', 'meta_cal', 'meta_select']:
        subset = [r for r, label in zip(validation, labels) if label == name]
        counts = collections.Counter(r['family'] + '|' + r['tier'] for r in subset)
        assert len(counts) == 30, (name, counts)
        coverage[name] = dict(records=len(subset), groups=len({r['training_group_id'] for r in subset}),
                              family_tier_counts=dict(sorted(counts.items())))
    result = dict(status='pass', config=CONFIG, coverage=coverage, checkpoints=footprint(),
                  base_training_unmodified=True, gpu_used=False, test_inputs_read=False,
                  expert_checkpoint_selection_used_original_val=True,
                  limitation='Meta partitions are disjoint, but experts used original val for early stopping; not nested cross-validation',
                  pending=ready())
    write(OUT / 'preflight.json', result)
    return result


def blind_rules(records):
    """Read original current-only rule evidence without rewriting its artifacts."""
    ids = {r['id']: i for i, r in enumerate(records)}
    locators = {}
    for r in rows(BASE / 'inventory.jsonl'):
        if r['id'] in ids:
            locators[r['label_path']] = ids[r['id']]
            locators[r['data_path']] = ids[r['id']]
    root = BASE.parent / 'full-matrix-20261031' / 'blind-rules'
    with (root / '07_证据索引/evidence_index.csv').open(encoding='utf-8-sig') as stream:
        evidence = {r['evidenceId']: r['locator'].split('#')[0] for r in csv.DictReader(stream)}
    result = np.zeros((len(records), 3), dtype=bool)
    for r in rows(root / '06_问题清单/findings.jsonl'):
        assert r['checkId'] != 'C10', 'Reference-assisted results forbidden'
        if r['status'] != '不通过':
            continue
        code = r['defectType']
        if code.startswith('GEOMETRY_') or code in ['L2-G', 'L3-MIS']:
            bit = 0
        elif code in ['L3-06', 'L3-07']:
            bit = 1
        elif code in ['L2-P', 'L2-F', 'L2-M', 'L2-T', 'ANNOTATION_STRUCTURE_CONFLICT']:
            bit = 2
        else:
            continue
        index = locators.get(evidence[r['evidenceRef']])
        if index is not None:
            result[index, bit] = True
    return result


def load_probabilities(records, experts):
    mapping = {r['id']: i for i, r in enumerate(records)}
    array = np.zeros((len(records), len(experts), 3), np.float32)
    costs = np.zeros((len(records), len(experts)), np.float32)
    for j, name in enumerate(experts):
        complete = json.loads((BASE / 'predictions' / (name + '.complete.json')).read_text())
        source = BASE / 'predictions' / (name + '.jsonl')
        assert complete['prediction_hash'] == sha(source), name
        found = set()
        for item in rows(source):
            index = mapping.get(item['id'])
            if index is None:
                continue
            assert index not in found, (name, item['id'])
            found.add(index)
            assert item['input_sha256'] == records[index]['input_sha256']
            if item['status'] == 'success':
                array[index, j] = item['probabilities']
                costs[index, j] = item['seconds']
            else:
                assert records[index]['input_missing'], (name, item['id'], item['status'])
        assert len(found) == len(records), name
    assert np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all()
    return array, costs


def arrays(records):
    return (np.array([r['target'] for r in records], bool),
            np.array([r['truth_valid'] for r in records], bool),
            np.array([FAMILIES.index(r['family']) for r in records]),
            np.array([int(r['tier'][1]) for r in records]))


def metrics(y, valid, decision):
    tp = int((decision & y & valid).sum()); fp = int((decision & ~y & valid).sum())
    fn = int((~decision & y & valid).sum()); tn = int((~decision & ~y & valid).sum())
    negative = valid.all(1) & ~y.any(1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                F1=2 * tp / (2 * tp + fp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                FPR=float(decision[negative].any(1).mean()) if negative.any() else None)


def objective(y, valid, pred, family, tier):
    values = []; p1 = []
    for fi in range(6):
        for ti in range(1, 5):
            mask = (family == fi) & (tier == ti)
            value = metrics(y[mask], valid[mask], pred[mask])['F1']
            if value is not None:
                values.append(value)
                if ti == 1:
                    p1.append(value)
    return (float(np.mean(values)) if values else 0., float(np.mean(p1)) if p1 else 0.)


def calibrate(prob, rule, y, valid, family, tier):
    thresholds = np.full((6, 3), 1.01)
    for fi in range(6):
        for bit in range(3):
            mask = (family == fi) & valid[:, bit]
            clean = (family == fi) & (tier == 0) & valid.all(1) & ~y.any(1) & ~rule.any(1)
            choices = []
            for threshold in [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.01]:
                p = (prob[:, bit] >= threshold) | rule[:, bit]
                if clean.any() and p[clean].mean() > CONFIG['added_p0_false_positive_cap']:
                    continue
                tp = int((p & y[:, bit] & mask).sum()); fp = int((p & ~y[:, bit] & mask).sum())
                fn = int((~p & y[:, bit] & mask).sum())
                value = 2 * tp / (2 * tp + fp + fn) if tp + fn else 0.
                choices.append((value, threshold))
            thresholds[fi, bit] = max(choices)[1] if choices else 1.01
    return thresholds


def decisions(prob, thresholds, family, rule):
    return (prob >= np.asarray(thresholds)[family]) | rule


def model_features(records, probe, rule):
    """Explicit allowlist. No IDs, tier, paths, reference changes or truth states."""
    structural = np.array([r['features'] for r in records], np.float32)
    assert structural.shape[1] == 256
    family = np.eye(6, dtype=np.float32)[[FAMILIES.index(r['family']) for r in records]]
    return np.concatenate([structural, family, probe, rule.astype(np.float32)], axis=1)


def train_gates(x, probs, y, valid, fit, cal, sample_weights, pool, folder, epochs=None):
    import torch
    torch.set_num_threads(2)
    mean = x[fit].mean(0); std = np.maximum(x[fit].std(0), .001)
    z = np.clip((x - mean) / std, -10, 10).astype(np.float32)
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
                                    torch.nn.Linear(64, len(pool)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=.01)
        tx = torch.from_numpy(z); pp = torch.from_numpy(probs[:, pool]); yy = torch.from_numpy(y.astype(np.float32))
        vv = torch.from_numpy(valid.astype(np.float32)); ww = torch.from_numpy(sample_weights.astype(np.float32))
        indices = np.flatnonzero(fit); rng = np.random.default_rng(seed)
        trace = []; best = None
        for epoch in range(epochs or CONFIG['epochs']):
            model.train(); rng.shuffle(indices)
            for off in range(0, len(indices), CONFIG['batch_size']):
                index = indices[off:off + CONFIG['batch_size']]
                mix = model(tx[index]).softmax(1)
                p = (mix[:, :, None] * pp[index]).sum(1).clamp(.00001, .99999)
                loss = torch.nn.functional.binary_cross_entropy(p, yy[index], reduction='none')
                weighted = vv[index] * ww[index, None]
                loss = (loss * weighted).sum() / weighted.sum().clamp_min(1)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad():
                mix = model(tx[cal]).softmax(1)
                p = (mix[:, :, None] * pp[cal]).sum(1).clamp(.00001, .99999)
                loss = torch.nn.functional.binary_cross_entropy(p, yy[cal], reduction='none')
                loss = float((loss * vv[cal]).sum() / vv[cal].sum().clamp_min(1))
            trace.append(dict(epoch=epoch + 1, meta_cal_masked_bce=loss))
            if best is None or loss < best[0]:
                best = (loss, epoch + 1, {k: v.detach().clone() for k, v in model.state_dict().items()})
        models.append(dict(seed=seed, selected_epoch=best[1],
                           state={k: v.tolist() for k, v in best[2].items()}, trace=trace))
    bundle = dict(mean=mean.tolist(), std=std.tolist(), pool=pool, models=models,
                  input_dimensions=x.shape[1], input_allowlist=CONFIG['features'])
    write(folder / 'gates.json', bundle)
    return bundle


def gate_predict(bundle, x):
    z = np.clip((x - np.array(bundle['mean'])) / np.array(bundle['std']), -10, 10)
    outputs = []
    for item in bundle['models']:
        weights = item['state']
        hidden = np.maximum(z @ np.array(weights['0.weight']).T + np.array(weights['0.bias']), 0)
        logits = hidden @ np.array(weights['2.weight']).T + np.array(weights['2.bias'])
        values = np.exp(logits - logits.max(1, keepdims=True))
        outputs.append(values / values.sum(1, keepdims=True))
    return np.mean(outputs, axis=0)


def top_weights(weights, topk):
    index = np.argsort(-weights, axis=1, kind='stable')[:, :topk]
    result = np.zeros_like(weights)
    np.put_along_axis(result, index, np.take_along_axis(weights, index, axis=1), axis=1)
    return result / result.sum(1, keepdims=True)


def selection_scores(probs, rule, y, valid, family):
    scores = np.zeros((6, 3, probs.shape[1]))
    for fi in range(6):
        for bit in range(3):
            mask = (family == fi) & valid[:, bit]
            for j in range(probs.shape[1]):
                p = (probs[:, j, bit] >= .5) | rule[:, bit]
                tp = int((p & y[:, bit] & mask).sum()); fp = int((p & ~y[:, bit] & mask).sum())
                fn = int((~p & y[:, bit] & mask).sum())
                scores[fi, bit, j] = 2 * tp / (2 * tp + fp + fn) if tp + fn else 0
    return scores


def greedy_pool(scores, sizes, probe, capacity, losses=None, sample_weights=None):
    pool = [probe]
    while len(pool) < capacity:
        candidates = []
        for j in range(scores.shape[2]):
            if j in pool or sum(sizes[k] for k in pool) + sizes[j] > CONFIG['weight_budget_bytes']:
                continue
            complement = (-float(np.average(losses[:, pool + [j]].min(1), weights=sample_weights))
                          if losses is not None else 0.)
            candidates.append((complement, float(scores[:, :, pool + [j]].max(2).mean()), -sizes[j], -j, j))
        if not candidates:
            break
        pool.append(max(candidates)[4])
    return pool


def csvwrite(path, records):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def run():
    assert not ready(), ready()
    # Verify immutable base weights, split hashes and source before reading test inputs.
    import sys
    code = BASE.parent.parent / 'experiments/all_family_training_20261031'
    sys.path.insert(0, str(code))
    from barrier import verify_barrier
    barrier = verify_barrier()
    state('preparing_meta_validation')
    validation = [compact(r) for r in rows(BASE / 'val.jsonl')]
    split = partition(validation); y, valid, family, tier = arrays(validation)
    available = ~np.array([r['input_missing'] for r in validation]) & valid.any(1)
    fit = (split == 'meta_fit') & available; cal = (split == 'meta_cal') & available
    select = split == 'meta_select'
    for mask in [fit, cal, select]:
        assert len({(r['family'], r['tier']) for r, use in zip(validation, mask) if use}) == 30
    eligible = [r for r in footprint() if r['eligible']]
    experts = [r['route'] for r in eligible]; sizes = [r['checkpoint_bytes'] for r in eligible]
    probe = experts.index(PROBE)
    probs, _ = load_probabilities(validation, experts); rule = blind_rules(validation)
    x = model_features(validation, probs[:, probe], rule)
    counts = collections.Counter((family[i], tier[i]) for i in np.flatnonzero(fit))
    sample_weights = np.array([1 / counts.get((f, t), 1) for f, t in zip(family, tier)])
    sample_weights /= sample_weights[fit].mean()
    scores = selection_scores(probs[fit], rule[fit], y[fit], valid[fit], family[fit])
    pp = np.clip(probs[fit], .00001, .99999)
    effective = np.maximum(pp, rule[fit, None, :].astype(float))
    effective = np.clip(effective, .00001, .99999)
    losses = -(y[fit, None, :] * np.log(effective) + (~y[fit, None, :]) * np.log(1 - effective))
    losses = (losses * valid[fit, None, :]).sum(2) / valid[fit].sum(1)[:, None].clip(min=1)
    write(OUT / 'meta_split.json', dict(source_sha256=sha(BASE / 'val.jsonl'),
        source_role='original expert validation, never base gradient train/test',
        assignments=[dict(id=r['id'], group=r['training_group_id'], subset=s) for r, s in zip(validation, split)]))
    candidates = []; policies = {}; calibration = []

    def add(name, prob, policy, used):
        threshold = calibrate(prob[cal], rule[cal], y[cal], valid[cal], family[cal], tier[cal])
        decision = decisions(prob[select], threshold, family[select], rule[select])
        overall, p1 = objective(y[select], valid[select], decision, family[select], tier[select])
        need = sorted({int(j) for j in used}); bytes_ = sum(sizes[j] for j in need)
        summary = dict(route=name, selection_macro_F1=overall, selection_P1_F1=p1,
                       resident_checkpoint_bytes=bytes_, expert_count=len(need),
                       experts=[experts[j] for j in need])
        policy.update(thresholds=threshold.tolist(), resident=need)
        candidates.append(summary); policies[name] = policy
        calibration.append(dict(route=name, thresholds=threshold.tolist()))

    add('rules', np.zeros((len(validation), 3)), dict(kind='rules'), [])
    global_choice = int(np.argmax(scores.mean((0, 1))))
    add('best_single_validation', probs[:, global_choice], dict(kind='single', expert=global_choice), [global_choice])
    for capacity in CONFIG['pools']:
        pool = greedy_pool(scores, sizes, probe, capacity, losses, sample_weights[fit])
        state('training_router', capacity=capacity, experts=[experts[j] for j in pool])
        local = scores[:, :, pool]
        by_family = np.array(pool)[local.mean(1).argmax(1)]
        by_class = np.array(pool)[local.argmax(2)]
        selected = probs[np.arange(len(validation)), by_family[family]]
        add(f'router_k{capacity}_family', selected,
            dict(kind='family', assignment=by_family.tolist()), by_family)
        selected = probs[np.arange(len(validation))[:, None], by_class[family], np.arange(3)[None, :]]
        add(f'router_k{capacity}_defect', selected,
            dict(kind='defect', assignment=by_class.tolist()), by_class.reshape(-1))
        add(f'router_k{capacity}_uniform', probs[:, pool].mean(1), dict(kind='uniform', pool=pool), pool)
        bundle = train_gates(x, probs, y, valid, fit, cal, sample_weights, pool, OUT / f'pool_{capacity}')
        weights = gate_predict(bundle, x)
        for topk in [1, 2]:
            sparse = top_weights(weights, min(topk, len(pool)))
            mixed = (sparse[:, :, None] * probs[:, pool]).sum(1)
            add(f'router_k{capacity}_learned_top{topk}', mixed,
                dict(kind='learned', pool=pool, topk=topk, bundle=f'pool_{capacity}/gates.json', probe=probe), pool)
    # This decision is made solely from meta_select, before this script opens test.jsonl.
    best = max(candidates, key=lambda r: (r['selection_macro_F1'], r['selection_P1_F1'],
                                         -r['resident_checkpoint_bytes'], r['route']))
    frozen = dict(status='frozen', selected_route=best['route'], selected_on='meta_select only',
                  experts=experts, expert_bytes=sizes, candidates=candidates, policies=policies,
                  config=CONFIG, code_sha256=sha(__file__), base_barrier_sha256=sha(BASE / 'training_barrier.json'),
                  expert_checkpoint_hashes=barrier['checkpoint_hashes'], test_read_for_fit_or_selection=False,
                  limitation='Original expert val was used for checkpoint selection; no nested cross-validation')
    write(OUT / 'frozen_policy.json', frozen)
    csvwrite(OUT / 'validation_candidates.csv', [{**r, 'experts': '+'.join(r['experts'])} for r in candidates])
    state('evaluating_frozen_policies', selected=best['route'])
    evaluate(frozen)


def apply_policy(policy, records, probs, rule):
    n = len(records); called = np.zeros(probs.shape[:2], bool)
    family = np.array([FAMILIES.index(r['family']) for r in records]); rr = np.arange(n)
    kind = policy['kind']
    if kind == 'rules':
        value = np.zeros((n, 3))
    elif kind == 'single':
        j = policy['expert']; called[:, j] = True; value = probs[:, j]
    elif kind == 'family':
        j = np.array(policy['assignment'])[family]; called[rr, j] = True; value = probs[rr, j]
    elif kind == 'defect':
        j = np.array(policy['assignment'])[family]
        called[rr[:, None], j] = True; value = probs[rr[:, None], j, np.arange(3)[None, :]]
    elif kind == 'uniform':
        called[:, policy['pool']] = True; value = probs[:, policy['pool']].mean(1)
    elif kind == 'learned':
        bundle = json.loads((OUT / policy['bundle']).read_text())
        weights = top_weights(gate_predict(bundle, model_features(records, probs[:, policy['probe']], rule)), policy['topk'])
        called[:, policy['pool']] = weights > 0; called[:, policy['probe']] = True
        value = (weights[:, :, None] * probs[:, policy['pool']]).sum(1)
    else:
        raise ValueError(kind)
    missing = np.array([r['input_missing'] for r in records])
    called[missing] = False; value[missing] = 0
    return decisions(value, policy['thresholds'], family, rule), called


def evaluate(frozen):
    test = [compact(r) for r in rows(BASE / 'test.jsonl')]
    val_groups = {r['group'] for r in json.loads((OUT / 'meta_split.json').read_text())['assignments']}
    assert not val_groups & {r['training_group_id'] for r in test}
    experts = frozen['experts']; probs, costs = load_probabilities(test, experts); rule = blind_rules(test)
    y, valid, family, tier = arrays(test); long = []; perclass = []; trace = []; independently_checked = 0
    for name, policy in frozen['policies'].items():
        decision, called = apply_policy(policy, test, probs, rule)
        for fi, fname in enumerate(FAMILIES):
            for ti in range(5):
                mask = (family == fi) & (tier == ti); subset = np.flatnonzero(mask)
                summary = metrics(y[mask], valid[mask], decision[mask])
                common = dict(scope='heldout_test', route=name, family=fname, tier=f'P{ti}')
                long.append(dict(common, n=int(mask.sum()), **summary,
                    known_positive_bits=int((y[mask] & valid[mask]).sum()),
                    fully_known_samples=int(valid[mask].all(1).sum()),
                    partial_or_unknown_samples=int((~valid[mask].all(1)).sum()),
                    missing_inputs=sum(test[i]['input_missing'] for i in subset),
                    mean_expert_calls=float(called[mask].sum(1).mean()),
                    cached_model_seconds_estimate=float((costs[mask] * called[mask]).sum()),
                    resident_checkpoint_bytes=sum(frozen['expert_bytes'][j] for j in policy['resident'])))
                for bit, label in enumerate(['G', 'C', 'D']):
                    summary = metrics(y[mask, bit:bit+1], valid[mask, bit:bit+1], decision[mask, bit:bit+1])
                    # Independent scalar accumulation tests unknown masking and every confusion cell.
                    count = [0, 0, 0, 0]
                    for i in subset:
                        if valid[i, bit]:
                            yy, pp = bool(y[i, bit]), bool(decision[i, bit])
                            count[0 if yy and pp else 1 if pp else 2 if yy else 3] += 1
                    assert count == [summary[k] for k in ['TP', 'FP', 'FN', 'TN']]
                    independently_checked += 1
                    perclass.append(dict(common, class_name=label, **summary))
        if name == frozen['selected_route']:
            for i, r in enumerate(test):
                trace.append(dict(id=r['id'], decisions=decision[i].astype(int).tolist(),
                    called_experts=[experts[j] for j in np.flatnonzero(called[i])],
                    input_missing=r['input_missing'], rule_decisions=rule[i].astype(int).tolist()))
    csvwrite(OUT / 'metrics_long.csv', long); csvwrite(OUT / 'metrics_per_class.csv', perclass)
    with (OUT / 'selected_routing_trace.jsonl').open('w') as stream:
        for r in trace:
            stream.write(json.dumps(r) + '\n')
    expanded = []
    for name in frozen['policies']:
        row = dict(route=name)
        for r in long:
            if r['route'] == name:
                for metric in ['F1', 'precision', 'recall', 'FPR', 'n']:
                    row[f'{r["family"]}_{r["tier"]}_{metric}'] = r[metric]
        assert len(row) == 151
        expanded.append(row)
    csvwrite(OUT / 'heldout_test_expanded.csv', expanded)
    policy = frozen['policies'][frozen['selected_route']]
    write(OUT / 'deployment_candidate.json', dict(selected_route=frozen['selected_route'],
        selection_split='meta_select', resident_experts=[experts[j] for j in policy['resident']],
        actual_checkpoint_bytes=sum(frozen['expert_bytes'][j] for j in policy['resident']),
        planned_runtime_reserve_bytes=CONFIG['runtime_reserve_bytes'],
        planned_safety_reserve_bytes=CONFIG['safety_reserve_bytes'],
        actual_docker_tar_bytes=None, a10_end_to_end_seconds=None, deployment_accepted=False,
        reason='Candidate only: actual offline Docker tar <=20GB and complete per-package A10 <=3h not yet verified'))
    write(OUT / 'validation.json', dict(status='pass', independent_confusion_checks=independently_checked,
        package_checks=30 * len(frozen['policies']), group_overlap=0,
        fitted_or_selected_using_test=False, frozen_before_test=True))
    report = ['# 专家路由阶段结果', '', f'验证集冻结选中：`{frozen["selected_route"]}`。',
        '每个方案覆盖六类×五档30包；指标为公开P0技术参照G/C/D，未知位掩码，不是官方隐藏定位得分。',
        '选型只用meta_select；测试表用于最终比较。专家曾用原val选检查点，因此本次不是嵌套交叉验证。',
        '路由输入不含档位、路径、P0差异或真值；规则结论保留。动态路由先算轻学生，再按当前特征选专家。',
        '模型调用耗时列来自缓存原始计时估算，不包含冷启动、文件读取、证据和报告，不能验收A10三小时。',
        '镜像仍需实际构建并离线验收；量化8B未验证打包前不进入部署专家池。', '',
        '`heldout_test_expanded.csv`为151列逐包表，`metrics_long.csv`含逐包资源统计。',
        '`frozen_policy.json`与各pool的`gates.json`是可复现路由权重；`selected_routing_trace.jsonl`记录专家选择。']
    (OUT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    write(OUT / 'complete.json', dict(status='complete', packages=30, test_records=len(test),
        policies=len(frozen['policies']), selected_route=frozen['selected_route'],
        selection_split='meta_select', validation_sha256=sha(OUT / 'validation.json'),
        deployment_accepted=False))
    state('complete', selected_route=frozen['selected_route'], policies=len(frozen['policies']))


def main():
    global BASE, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'wait', 'run'])
    parser.add_argument('--base', type=Path, default=BASE); parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    BASE, OUT = args.base, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'preflight':
        result = preflight(); print(json.dumps(dict(status=result['status'], coverage=result['coverage'], pending=result['pending'])))
        return
    import fcntl
    lock = (OUT / 'runner.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    fingerprint = sha(__file__)
    write(OUT / 'registered.json', dict(config=CONFIG, code_sha256=fingerprint,
        registered_unix=time.time(), trigger='base queue, full-training barrier, 30-package evaluation and validation complete',
        gpu_used=False))
    if args.mode == 'wait':
        start = time.monotonic(); preflight()
        while ready():
            waiting = ready()
            base_state = json.loads((BASE / 'queue_state.json').read_text())
            if base_state.get('status') == 'failed':
                raise RuntimeError('Base queue failed; route fitting stopped: ' + str(base_state.get('error')))
            if time.monotonic() - start > 72 * 3600:
                raise TimeoutError('72-hour wait exceeded; no training performed')
            state('waiting_for_base_training_and_validation', waiting=waiting,
                  base_stage=base_state.get('stage'), gpu_used=False)
            time.sleep(60)
    assert sha(__file__) == fingerprint, 'Routing code changed after registration'
    run()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        state('failed', error=repr(error))
        raise
