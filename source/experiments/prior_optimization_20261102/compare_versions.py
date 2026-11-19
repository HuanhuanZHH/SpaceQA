#!/usr/bin/env python3
"""Re-evaluate cached all-family predictions against current-input rule variants.

This is an exploratory historical re-evaluation, not an official score or a new
untouched test. New thresholds and routing are fitted on validation only and
persisted before the test split is parsed. No GPU or training code is imported.
All output is confined to /staff/tangjiarui/OPC/inference_prior_20261102/.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ALLOWED_OUTPUT = Path('/staff/tangjiarui/OPC/inference_prior_20261102')
FAMILIES = ('AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample',
            'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample')
TIERS = ('P0', 'P1', 'P2', 'P3', 'P4')
GROUPS = ('G', 'C', 'D')
THRESHOLDS = tuple([round(i / 20, 2) for i in range(1, 20)] + [.99, 1.01])
VERSION = 'cached-fusion-comparison-2026.11.02-1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def write_csv(path, records):
    records = list(records)
    require(bool(records), 'Refusing an empty metric table')
    fields = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (list, dict)) else v
                             for k, v in record.items()})


def event(status, **fields):
    print(json.dumps(dict(status=status, **fields), ensure_ascii=True), flush=True)


def metrics(counts):
    tp, fp, fn, tn = counts
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                FPR=fp / (fp + tn) if fp + tn else None)


def count_bit(target, known, predicted):
    if not known:
        return None
    return 0 if target and predicted else 1 if predicted else 2 if target else 3


def add_counts(a, b):
    return [x + y for x, y in zip(a, b)]


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def old_rule_group(code):
    if code.startswith('GEOMETRY_') or code in {'L2-G', 'L3-MIS'}:
        return 0
    if code in {'L3-06', 'L3-07'}:
        return 1
    if code in {'L2-P', 'L2-F', 'L2-M', 'L2-T', 'ANNOTATION_STRUCTURE_CONFLICT'}:
        return 2
    return None


def load_old_rule_locators(root):
    evidence_path = next(p for p in root.iterdir() if p.name.startswith('07_')) / 'evidence_index.csv'
    finding_path = next(p for p in root.iterdir() if p.name.startswith('06_')) / 'findings.jsonl'
    with evidence_path.open(encoding='utf-8-sig', newline='') as stream:
        evidence = {r['evidenceId']: r['locator'].split('#')[0] for r in csv.DictReader(stream)}
    result = {}
    for record in rows(finding_path):
        require(record['checkId'] != 'C10', 'Clean-reference C10 is forbidden')
        if record['status'] != '\u4e0d\u901a\u8fc7':
            continue
        group = old_rule_group(record['defectType'])
        if group is not None:
            locator = evidence[record['evidenceRef']]
            result.setdefault(locator, [False] * 3)[group] = True
    return result, dict(findings_sha256=sha(finding_path), evidence_sha256=sha(evidence_path))


def load_split(path, expected, expected_hash):
    require(sha(path) == expected_hash, 'Split hash mismatch: ' + expected)
    answer, seen = [], set()
    for record in rows(path):
        require(record['split'] == expected, 'Unexpected split membership')
        require(record['id'] not in seen, 'Duplicate sample ID')
        require(record['family'] in FAMILIES and record['tier'] in TIERS, 'Unknown family/tier')
        require(len(record['target']) == len(record['truth_valid']) == 3, 'Expected three truth bits')
        seen.add(record['id'])
        answer.append({k: record.get(k) for k in (
            'id', 'family', 'tier', 'target', 'truth_valid', 'input_sha256', 'input_missing',
            'label_path', 'data_path', 'data_sha256', 'label_sha256', 'package_id',
            'base_sample_id', 'training_group_id')})
    coverage = collections.Counter((r['family'], r['tier']) for r in answer)
    require(len(coverage) == 30, 'Split does not cover all 30 family/tier packages')
    return answer


def old_bits(records, locators):
    result = []
    for record in records:
        bits = [False] * 3
        for key in ('label_path', 'data_path'):
            bits = [a or b for a, b in zip(bits, locators.get(record[key], [False] * 3))]
        result.append(bits)
    return result


def verify_route(base, route, split_hashes):
    report_path = base / 'training' / (route + '.json')
    barrier_path = base / 'route_barriers' / (route + '.json')
    complete_path = base / 'predictions' / (route + '.complete.json')
    report, barrier, complete = map(read_json, (report_path, barrier_path, complete_path))
    require(report.get('status') == 'complete' and report.get('test_read') is False,
            'Training not independently complete: ' + route)
    require(set(report['training_families']) == set(FAMILIES)
            and len(report['actual_train_coverage']) == 30
            and min(report['actual_train_coverage'].values()) > 0,
            'Route was not trained on all six families and all five tiers: ' + route)
    require(barrier.get('status') == 'pass' and barrier['split_hashes'] == split_hashes,
            'Route barrier invalid: ' + route)
    require(barrier['routes'][route]['report_sha256'] == sha(report_path), 'Training-report hash mismatch')
    require(complete['status'] == 'complete' and complete['route'] == route, 'Prediction incomplete')
    require(complete['barrier_hash'] == sha(barrier_path), 'Prediction/barrier mismatch')
    return dict(report_sha256=sha(report_path), barrier_sha256=sha(barrier_path),
                complete_sha256=sha(complete_path), prediction_sha256=complete['prediction_hash'],
                records=complete['records'], historical_inference_seconds=complete['seconds'])


def load_probabilities(base, route, records, expected):
    mapping = {r['id']: i for i, r in enumerate(records)}
    result = [None] * len(records)
    seen = set()
    h, count = hashlib.sha256(), 0
    with (base / 'predictions' / (route + '.jsonl')).open('rb') as stream:
        for line in stream:
            h.update(line)
            if not line.strip():
                continue
            item = json.loads(line)
            count += 1
            index = mapping.get(item['id'])
            if index is None:
                continue
            require(index not in seen, 'Duplicate prediction for route ' + route)
            seen.add(index)
            record = records[index]
            require(item['input_sha256'] == record['input_sha256'], 'Prediction input hash mismatch')
            require(item['status'] in {'success', 'input_missing'}, 'Unexpected prediction status')
            require((item['status'] == 'input_missing') == bool(record['input_missing']),
                    'Prediction missing-input state mismatch')
            if item['status'] == 'success':
                p = item['probabilities']
                require(len(p) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in p),
                        'Invalid probabilities')
                result[index] = p
    require(len(seen) == len(records), 'Missing predictions for ' + route)
    require(count == expected['records'] and h.hexdigest() == expected['prediction_sha256'],
            'Prediction file hash/count mismatch: ' + route)
    return result


class PriorRunner:
    def __init__(self, module, policy, out, workers):
        self.module, self.policy, self.out, self.workers = module, policy, out, workers
        self.csv_cache = {}
        self.label_checks = 0
        self.csv_checks = 0

    def _csv(self, record):
        path = Path(record['data_path'])
        require(sha(path) == record['data_sha256'], 'Current CSV differs from recorded raw hash')
        stats = self.module.csv_statistics(path)
        require(stats.get('full_scan') is True, 'CSV axis scan incomplete')
        key = record['data_sha256']
        write_json(self.out / 'csv_statistics' / (key + '.json'),
                   dict(source_sha256=key, source_path=str(path), statistics=stats))
        return key, stats

    def evaluate(self, records, split):
        unique_csv = {}
        for record in records:
            if record['family'] == 'ULF_sample' and not record['input_missing']:
                key = record['data_sha256']
                if key not in self.csv_cache:
                    unique_csv.setdefault(key, record)
        event('scanning_current_csv_axes', split=split, new_unique_csv=len(unique_csv))
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            for key, stats in pool.map(self._csv, unique_csv.values()):
                self.csv_cache[key] = stats
                self.csv_checks += 1

        packages = collections.defaultdict(list)
        for i, record in enumerate(records):
            packages[record['package_id']].append((i, record))

        def package(items):
            inputs = []
            for index, record in items:
                path = Path(record['label_path'])
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == record['label_sha256'],
                        'Current label differs from historical manifest: ' + str(path))
                inputs.append(dict(id=record['id'], family=record['family'],
                                   label=json.loads(raw.decode('utf-8-sig')), data_path=record['data_path'],
                                   label_path=record['label_path'], package=record['package_id'],
                                   csv_stats=self.csv_cache.get(record['data_sha256'])))
            outputs = self.module.evaluate_package(inputs, policy=self.policy)
            require(len(outputs) == len(items), 'Prior output count mismatch')
            return [(index, output) for (index, record), output in zip(items, outputs)]

        outputs = [None] * len(records)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
            for batch in pool.map(package, packages.values()):
                for index, result in batch:
                    require(result['sample_id'] == records[index]['id'], 'Prior sample identity mismatch')
                    outputs[index] = result
        self.label_checks += len(records)
        folder = self.out / 'prior_outputs'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / (split + '.jsonl')).open('w', encoding='utf-8') as stream:
            for result in outputs:
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        event('current_prior_complete', split=split, records=len(outputs),
              confirmed=collections.Counter(f['defectType'] for r in outputs for f in r['findings'] if f['score_eligible']),
              candidates=sum(f['certainty'] == 'candidate' for r in outputs for f in r['findings']))
        return [[bool(x) for x in r['confirmed_bits']] for r in outputs]


def retained_prior(pure, old):
    # Retain current-only structural geometry/count rules. Replace all legacy D
    # decisions with the new scientifically scoped D rules.
    return [[p[0] or r[0], p[1] or r[1], p[2]] for p, r in zip(pure, old)]


def decisions(records, probs, rules, choices, *, gc_only=False, force_rule_families=()):
    answer = []
    for index, record in enumerate(records):
        family = record['family']
        choice = choices[family]
        route = choice['route']
        p = probs[route][index] if route != 'rules' and family not in force_rule_families else None
        answer.append([bool(rules[index][g]) or (p is not None and (not gc_only or g < 2)
                       and p[g] >= choice['thresholds'][g]) for g in range(3)])
    return answer


def summarize(records, predictions, variant):
    counts = collections.defaultdict(lambda: [0] * 4)
    n, missing, unknown, fp_samples, negative_samples = (collections.Counter() for _ in range(5))
    for record, predicted in zip(records, predictions):
        key = (record['family'], record['tier'])
        n[key] += 1
        missing[key] += bool(record['input_missing'])
        unknown[key] += sum(not x for x in record['truth_valid'])
        known_negative = all(record['truth_valid']) and not any(record['target'])
        negative_samples[key] += known_negative
        fp_samples[key] += known_negative and any(predicted)
        for group in range(3):
            index = count_bit(record['target'][group], record['truth_valid'][group], predicted[group])
            if index is not None:
                counts[key + (group,)][index] += 1
    packages, groups = [], []
    total_all, total_positive = [0] * 4, [0] * 4
    group_totals = [[0] * 4 for _ in GROUPS]
    for family in FAMILIES:
        for tier in TIERS:
            key = family, tier
            combined = [0] * 4
            row = dict(variant=variant, family=family, tier=tier, n=n[key],
                       missing_inputs=missing[key], unknown_truth_bits=unknown[key],
                       fully_known_negative_samples=negative_samples[key],
                       false_positive_samples=fp_samples[key],
                       sample_FPR=fp_samples[key] / negative_samples[key] if negative_samples[key] else None)
            for group, name in enumerate(GROUPS):
                c = counts[key + (group,)]
                combined = add_counts(combined, c)
                row.update({name + '_' + k: v for k, v in metrics(c).items()})
                groups.append(dict(variant=variant, family=family, tier=tier, group=name, **metrics(c)))
                if tier != 'P0':
                    group_totals[group] = add_counts(group_totals[group], c)
            row.update(metrics(combined))
            packages.append(row)
            total_all = add_counts(total_all, combined)
            if tier != 'P0':
                total_positive = add_counts(total_positive, combined)
    p0 = [r for r in packages if r['tier'] == 'P0']
    summary = dict(variant=variant, records=len(records), overall=metrics(total_positive),
                   overall_all_tiers=metrics(total_all),
                   macro_F1_24=mean(r['F1'] for r in packages if r['tier'] != 'P0'),
                   P1_macro_F1_6=mean(r['F1'] for r in packages if r['tier'] == 'P1'),
                   P0_sample_FPR=sum(r['false_positive_samples'] for r in p0) / sum(r['n'] for r in p0),
                   P0_family_macro_sample_FPR=mean(r['sample_FPR'] for r in p0),
                   groups={g: metrics(group_totals[i]) for i, g in enumerate(GROUPS)})
    return summary, packages, groups


def family_objective(records, predicted):
    counts = [[0] * 4 for _ in TIERS]
    fp0, n0 = 0, 0
    for record, p in zip(records, predicted):
        tier = int(record['tier'][1])
        if tier == 0:
            n0 += 1
            fp0 += any(p)
        for g in range(3):
            index = count_bit(record['target'][g], record['truth_valid'][g], p[g])
            if index is not None:
                counts[tier][index] += 1
    scores = [metrics(c)['F1'] for c in counts]
    return dict(macro_F1=mean(scores[1:]) or 0., P1_F1=scores[1] or 0., P0_sample_FPR=fp0 / n0)


def objective_key(value, thresholds=()):
    return (value['macro_F1'], value['P1_F1'], -value['P0_sample_FPR'], sum(thresholds))


def calibrate_family(records, probabilities, rules, *, gc_only, cap):
    """Two deterministic coordinate passes, validation data only.

    The objective is mean bit-micro F1 across the four positive tiers, subject
    to P0 sample-FPR <= max(base-rule FPR, cap). 1.01 disables a model bit.
    """
    thresholds = [1.01] * 3
    base = family_objective(records, rules)
    allowed_fpr = max(base['P0_sample_FPR'], cap)
    current = [list(x) for x in rules]
    value = base
    for _ in range(2):
        for group in range(2 if gc_only else 3):
            best = (objective_key(value, thresholds), thresholds[group], value)
            for threshold in THRESHOLDS:
                trial = []
                for p, rule, before in zip(probabilities, rules, current):
                    bits = list(before)
                    bits[group] = rule[group] or (p is not None and p[group] >= threshold)
                    trial.append(bits)
                score = family_objective(records, trial)
                if score['P0_sample_FPR'] > allowed_fpr + 1e-12:
                    continue
                candidate_thresholds = list(thresholds)
                candidate_thresholds[group] = threshold
                key = objective_key(score, candidate_thresholds)
                if key > best[0]:
                    best = key, threshold, score
            thresholds[group], value = best[1], best[2]
            for p, rule, before in zip(probabilities, rules, current):
                before[group] = rule[group] or (p is not None and p[group] >= thresholds[group])
    return dict(route=None, thresholds=thresholds, **value, allowed_P0_sample_FPR=allowed_fpr)


def select_routes(records, probabilities, rules, sizes, cap):
    selected, diagnostics = {}, []
    for mode in ('all_groups', 'gc_only'):
        all_choices = {}
        budget_choices = {}
        for family in FAMILIES:
            ids = [i for i, r in enumerate(records) if r['family'] == family]
            subset, subset_rules = [records[i] for i in ids], [rules[i] for i in ids]
            base = dict(route='rules', thresholds=[1.01] * 3, checkpoint_bytes=0,
                        **family_objective(subset, subset_rules))
            choices = [base]
            for route in sorted(probabilities):
                choice = calibrate_family(subset, [probabilities[route][i] for i in ids], subset_rules,
                                          gc_only=mode == 'gc_only', cap=cap)
                choice.update(route=route, checkpoint_bytes=sizes[route])
                choices.append(choice)
            for choice in choices:
                diagnostics.append(dict(mode=mode, family=family, **choice))

            def key(choice):
                return (choice['macro_F1'], choice['P1_F1'], -choice['P0_sample_FPR'],
                        -choice['checkpoint_bytes'], choice['route'])

            all_choices[family] = max(choices, key=key)
            # The original deployment planning reserves 8 GB for runtime and
            # 2 GB safety from 20 GB, leaving 10 GB for combined weights.
            budget_choices[family] = max((c for c in choices if c['checkpoint_bytes'] <= 10_000_000_000), key=key)
            event('family_validation_selection', mode=mode, family=family,
                  all20=all_choices[family]['route'], packable=budget_choices[family]['route'])
        for name, choices in (('all20', all_choices), ('packable', budget_choices)):
            unique = {c['route']: c['checkpoint_bytes'] for c in choices.values() if c['route'] != 'rules'}
            total = sum(unique.values())
            require(name != 'packable' or total <= 10_000_000_000, 'Combined selected weights exceed planning budget')
            selected[name + '_' + mode] = dict(choices=choices, checkpoint_bytes=total,
                                                checkpoint_only=True, gc_only=mode == 'gc_only',
                                                deployment_weight_budget_eligible=total <= 10_000_000_000)
    return selected, diagnostics


def candidate_predictions(records, probabilities, old, pure, prior, frozen):
    result = {'old_rules': old, 'pure_new_prior': pure, 'new_prior_retained_gc': prior}
    static = frozen['old_static_choices']
    result['old_static_expert_or_rules'] = decisions(records, probabilities, old, static)
    result['old_static_expert_or_new_prior'] = decisions(records, probabilities, prior, static)
    result['old_static_gc_expert_d_prior'] = decisions(records, probabilities, prior, static, gc_only=True)
    result['old_static_gc_cme_ulf_rules_only'] = decisions(
        records, probabilities, prior, static, gc_only=True, force_rule_families=('CME_C2_sample', 'ULF_sample'))
    for name, policy in frozen['recalibrated'].items():
        result[name + '_expert_or_new_prior'] = decisions(
            records, probabilities, prior, policy['choices'], gc_only=policy['gc_only'])
        result[name + '_cme_ulf_rules_only'] = decisions(
            records, probabilities, prior, policy['choices'], gc_only=policy['gc_only'],
            force_rule_families=('CME_C2_sample', 'ULF_sample'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031'))
    parser.add_argument('--dispatch', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/dispatch_current.json'))
    parser.add_argument('--old-rules', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/full-matrix-20261031/blind-rules'))
    parser.add_argument('--prior-module', type=Path, default=Path(__file__).with_name('prior_rules.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--p0-cap', type=float, default=.02)
    parser.add_argument('--trust-ulf-dateobs', action='store_true')
    parser.add_argument('--ulf-duration-convention', choices=('inclusive', 'elapsed', 'either'), default='either')
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()), 'Output must stay inside the new inference-prior workspace')
    require(out != ALLOWED_OUTPUT.resolve(), 'Use a new child output directory')
    require(not out.exists(), 'Refusing to overwrite comparison output')
    require(1 <= args.workers <= 32 and 0 <= args.p0_cap <= 1, 'Invalid resource/calibration options')
    out.mkdir(parents=True)
    started = time.monotonic()
    module_spec = importlib.util.spec_from_file_location('spaceqa_prior_candidate', args.prior_module)
    prior_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = prior_module
    module_spec.loader.exec_module(prior_module)
    policy = prior_module.Policy(trust_ulf_dateobs=args.trust_ulf_dateobs,
                                 ulf_duration_convention=args.ulf_duration_convention)
    runner = PriorRunner(prior_module, policy, out, args.workers)
    dispatch = read_json(args.dispatch)
    require(dispatch['status'] == 'frozen' and set(dispatch['selected']) == set(FAMILIES), 'Invalid historical dispatch')
    routes = dispatch['experts']
    require(len(routes) == 20 and len(set(routes)) == 20, 'Expected 20 completed models')
    barrier = read_json(args.base / 'training_barrier.json')
    require(barrier['status'] == 'pass', 'Global training barrier incomplete')
    split_hashes = barrier['split_hashes']
    require(dispatch['original_val_sha256'] == split_hashes['val'], 'Historical dispatch validation hash mismatch')
    require(sha(args.base / 'train.jsonl') == split_hashes['train'], 'Training split hash mismatch')
    provenance = {route: verify_route(args.base, route, split_hashes) for route in routes}
    sizes = {r: dispatch['model_details'][r]['checkpoint_bytes'] for r in routes}
    locator_bits, rule_provenance = load_old_rule_locators(args.old_rules)
    validation = load_split(args.base / 'val.jsonl', 'val', split_hashes['val'])
    val_old = old_bits(validation, locator_bits)
    val_pure = runner.evaluate(validation, 'val')
    val_prior = retained_prior(val_pure, val_old)
    val_probabilities = {}
    for route in routes:
        val_probabilities[route] = load_probabilities(args.base, route, validation, provenance[route])
        event('validated_cached_predictions', split='val', route=route)
    recalibrated, diagnostics = select_routes(validation, val_probabilities, val_prior, sizes, args.p0_cap)
    frozen = dict(version=VERSION, status='frozen', selection_split='original_validation_only',
                  validation_sha256=split_hashes['val'],
                  script_sha256=sha(__file__), prior_module_sha256=sha(args.prior_module),
                  prior_policy=dataclasses.asdict(policy), calibration_thresholds=THRESHOLDS,
                  calibration_coordinate_passes=2, p0_sample_FPR_cap=args.p0_cap,
                  objective='macro of four positive-tier bit-micro F1 per family; P1 tie-break; lower P0 FPR; smaller weights',
                  old_static_choices={f: dispatch['selected'][f] for f in FAMILIES},
                  recalibrated=recalibrated, provenance=provenance, rule_provenance=rule_provenance,
                  historical_dispatch_sha256=sha(args.dispatch), test_split_parsed=False,
                  proxy_truth='P0-derived G/C/D technical proxy; unknown bits masked; not independent scientific truth',
                  evaluation_status='exploratory historical re-evaluation; historical test outcomes were viewed during development')
    val_results = candidate_predictions(validation, val_probabilities, val_old, val_pure, val_prior, frozen)
    val_summaries, val_packages, val_groups = [], [], []
    for name, predicted in val_results.items():
        summary, packages, groups = summarize(validation, predicted, name)
        val_summaries.append(summary)
        val_packages.extend(packages)
        val_groups.extend(groups)
    # Pick a single recommendation before parsing the test split. Limit the
    # selectable deployment pools to the weight-budget-eligible variants.
    eligible = [x for x in val_summaries if x['variant'] in {'pure_new_prior', 'new_prior_retained_gc'}
                or x['variant'].startswith('packable_')]
    best = max(eligible, key=lambda x: (x['macro_F1_24'] or 0., x['P1_macro_F1_6'] or 0.,
                                        -x['P0_sample_FPR'], x['variant']))
    frozen['recommended_variant'] = best['variant']
    frozen['recommended_validation_summary'] = best
    write_json(out / 'frozen_validation_policy.json', frozen)
    frozen_hash = sha(out / 'frozen_validation_policy.json')
    write_csv(out / 'validation_route_selection.csv', diagnostics)
    write_csv(out / 'all30_validation.csv', val_packages)
    write_csv(out / 'groups_validation.csv', val_groups)
    event('validation_policy_frozen', sha256=frozen_hash, recommended=best['variant'])

    # The test split is first parsed strictly after the frozen policy is written.
    test = load_split(args.base / 'test.jsonl', 'test', split_hashes['test'])
    val_groups_set = {r['training_group_id'] for r in validation}
    require(not val_groups_set.intersection(r['training_group_id'] for r in test), 'Validation/test group leakage')
    old = old_bits(test, locator_bits)
    pure = runner.evaluate(test, 'test')
    prior = retained_prior(pure, old)
    probabilities = {}
    for route in routes:
        probabilities[route] = load_probabilities(args.base, route, test, provenance[route])
        event('validated_cached_predictions', split='test', route=route)
    test_results = candidate_predictions(test, probabilities, old, pure, prior, frozen)
    summaries, package_rows, group_rows = [], [], []
    for name, predicted in test_results.items():
        summary, packages, groups = summarize(test, predicted, name)
        summary['validation_selected_recommendation'] = name == frozen['recommended_variant']
        summaries.append(summary)
        package_rows.extend(packages)
        group_rows.extend(groups)
    require(sha(out / 'frozen_validation_policy.json') == frozen_hash, 'Frozen policy changed after test evaluation')
    write_csv(out / 'all30_test.csv', package_rows)
    write_csv(out / 'groups_test.csv', group_rows)
    flattened = []
    for summary in summaries:
        row = {k: v for k, v in summary.items() if k not in {'overall', 'overall_all_tiers', 'groups'}}
        row.update(summary['overall'])
        row.update({'all_tiers_' + k: v for k, v in summary['overall_all_tiers'].items()})
        for name, value in summary['groups'].items():
            row.update({name + '_' + k: v for k, v in value.items()})
        flattened.append(row)
    write_csv(out / 'summary_test.csv', flattened)
    result = dict(status='complete', version=VERSION, gpu_forward_passes=0, training_updates=0,
                  output=str(out), seconds=time.monotonic() - started,
                  validation_records=len(validation), test_records=len(test),
                  test_package_count=30, variants=len(summaries),
                  frozen_validation_policy_sha256=frozen_hash,
                  selected_before_test=frozen['recommended_variant'],
                  test_used_for_selection=False,
                  interpretation=frozen['proxy_truth'], evaluation_status=frozen['evaluation_status'],
                  current_label_sha256_checks=runner.label_checks,
                  full_csv_axis_and_raw_sha256_checks=runner.csv_checks,
                  cached_model_input_hash_checks=(len(test) + len(validation)) * len(routes),
                  cached_prediction_file_sha256_checks=2 * len(routes),
                  raw_image_bytes_rehashed=False,
                  current_raw_note='Label bytes and each unique ULF CSV were rehashed. Image bytes were not fully rehashed; current PNG IHDR was read, and model caches were checked against historical prepared input hashes.',
                  timing_note='CPU cached-probability comparison; not new model inference or A10/full-pipeline acceptance',
                  deployment_note='Weights-only 10 GB planning constraint; final Docker/OCI image still requires measurement',
                  validation_summaries=val_summaries, test_summaries=summaries)
    write_json(out / 'comparison.json', result)
    event('comparison_complete', output=str(out), seconds=result['seconds'],
          selected_before_test=result['selected_before_test'], summaries=summaries)


if __name__ == '__main__':
    main()
