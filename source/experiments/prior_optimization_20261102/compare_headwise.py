#!/usr/bin/env python3
"""Last cached experiment: family × defect-head compact-expert dispatch.

Two validation-only profiles are frozen before test parsing. F1 selection treats
G/C/D separately; review selection first seeks >=.8 recall under a P0 head-FPR
cap. No Qwen/8B model, new training, or test-based route fitting is allowed.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import time

from compare_versions import (ALLOWED_OUTPUT, FAMILIES, GROUPS, THRESHOLDS,
    event, load_old_rule_locators, load_probabilities, load_split, metrics,
    old_bits, read_json, require, retained_prior, rows, sha, summarize, write_csv,
    write_json)
from compare_gbdt_extension import prior_from_file


def candidates(records, probability_sets, rules, group, sizes, cap):
    """Descending threshold sweep, O(routes × n log n), no repeated row scans."""
    base, p1 = [0] * 4, [0] * 4
    p0_fp, p0_n = 0, 0
    for record, rule in zip(records, rules):
        if not record['truth_valid'][group]:
            continue
        y, predicted = bool(record['target'][group]), bool(rule[group])
        if record['tier'] == 'P0':
            require(not y, 'P0 proxy bit unexpectedly positive')
            p0_n += 1
            p0_fp += predicted
        else:
            slot = 0 if y and predicted else 1 if predicted else 2 if y else 3
            base[slot] += 1
            if record['tier'] == 'P1':
                p1[slot] += 1
    base_fpr = p0_fp / p0_n if p0_n else 0.
    allowed = max(base_fpr, cap)

    def result(route, threshold, counts, p1_counts, fp0):
        return dict(route=route, threshold=threshold, checkpoint_bytes=sizes.get(route, 0),
                    **metrics(counts), P1_F1=metrics(p1_counts)['F1'],
                    P0_false_positive_samples=fp0, P0_known_negative_samples=p0_n,
                    P0_head_FPR=fp0 / p0_n if p0_n else None,
                    allowed_P0_head_FPR=allowed, validation_positive_bits=counts[0] + counts[2])

    answer = [result('rules', 1.01, base, p1, p0_fp)]
    for route, probabilities in sorted(probability_sets.items()):
        events = []
        for i, (record, rule, p) in enumerate(zip(records, rules, probabilities)):
            if p is not None and record['truth_valid'][group] and not rule[group]:
                events.append((p[group], bool(record['target'][group]), record['tier']))
        events.sort(key=lambda x: -x[0])
        counts, p1_counts, fp0 = list(base), list(p1), p0_fp
        cursor = 0
        for threshold in sorted(THRESHOLDS, reverse=True):
            while cursor < len(events) and events[cursor][0] >= threshold:
                probability, y, tier = events[cursor]
                if tier == 'P0':
                    fp0 += 1
                else:
                    before, after = (2, 0) if y else (3, 1)
                    counts[before] -= 1
                    counts[after] += 1
                    if tier == 'P1':
                        p1_counts[before] -= 1
                        p1_counts[after] += 1
                cursor += 1
            answer.append(result(route, threshold, counts, p1_counts, fp0))
    return answer


def select(options, profile):
    feasible = [c for c in options if (c['P0_head_FPR'] or 0.) <= c['allowed_P0_head_FPR'] + 1e-12]
    require(bool(feasible), 'Pure rules must satisfy their own baseline cap')
    positives = feasible[0]['validation_positive_bits']
    if positives == 0:
        choice = next(c for c in feasible if c['route'] == 'rules')
        return dict(choice, recall_target_applicable=False, recall_target_achieved=None,
                    selection_note='No validation positives for this family/head; pure rules, recall target N/A')

    def secondary(c):
        return (c['F1'] or 0., c['recall'] or 0., c['P1_F1'] or 0.,
                -(c['P0_head_FPR'] or 0.), -c['checkpoint_bytes'], c['threshold'], c['route'])

    if profile == 'f1':
        winner = max(feasible, key=secondary)
    else:
        meeting = [c for c in feasible if c['recall'] is not None and c['recall'] >= .8]
        winner = max(meeting, key=secondary) if meeting else max(feasible, key=lambda c: (c['recall'] or 0.,) + secondary(c))
    return dict(winner, recall_target_applicable=True,
                recall_target_achieved=winner['recall'] >= .8,
                selection_note='Maximum F1 within P0 cap' if profile == 'f1' else
                'Recall >= .8 achievable; maximum F1 among those candidates' if winner['recall'] >= .8 else
                'Recall .8 infeasible under P0 cap and cached compact pool; maximum attainable recall then F1')


def decide(records, probabilities, rules, selected):
    predictions = []
    for i, record in enumerate(records):
        bits = list(rules[i])
        for group, name in enumerate(GROUPS):
            choice = selected[record['family']][name]
            p = probabilities[choice['route']][i] if choice['route'] != 'rules' else None
            bits[group] = bool(bits[group] or (p is not None and p[group] >= choice['threshold']))
        predictions.append(bits)
    return predictions


def footprint(selected, base, gbdt):
    routes = {c['route'] for family in selected.values() for c in family.values()} - {'rules'}
    checkpoints, initialization = {}, {}
    root = Path('/staff/tangjiarui/OPC')
    for route in routes:
        if route == 'gbdt_prior_v1':
            files = [gbdt / f'model_{g}.pkl' for g in GROUPS] + [gbdt / 'vectorizer.pkl']
        else:
            files = [base / 'checkpoints' / (route + '.pt')]
        for path in files:
            checkpoints[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
        extra = None
        if route.startswith('yolo'):
            name = {'yolo11_detect_backbone': 'yolo11n.pt', 'yolo11_segment_backbone': 'yolo11n-seg.pt', 'yolov8_segment_backbone': 'yolov8n-seg.pt'}[route]
            extra = root / 'models/image-model-pilot' / name
        elif route in {'dino_finetune', 'ijepa_finetune'}:
            extra = root / 'models' / ('dinov2-small' if route == 'dino_finetune' else 'ijepa_vith14_1k')
        if extra is not None:
            files = [p for p in extra.rglob('*') if p.is_file()] if extra.is_dir() else [extra]
            require(bool(files) and all(p.is_file() for p in files), 'Runtime initialization artifact missing')
            initialization[str(extra)] = dict(bytes=sum(p.stat().st_size for p in files),
                                               verification='current file sizes; full initializer hashes not repeated in this cached comparison')
    checkpoint_bytes = sum(x['bytes'] for x in checkpoints.values())
    initialization_bytes = sum(x['bytes'] for x in initialization.values())
    require(checkpoint_bytes < 10_000_000_000, 'Combined selected checkpoints exceed 10 GB')
    return dict(routes=sorted(routes), checkpoints=checkpoints, checkpoint_bytes=checkpoint_bytes,
                initialization=initialization, initialization_bytes=initialization_bytes,
                checkpoints_plus_initialization_bytes=checkpoint_bytes + initialization_bytes,
                container_measured=False, note='Runtime packages, source and final image still require actual packaging and timing acceptance')


def independent_checks(records, decisions_by_policy, tables):
    audit = {}
    for profile, predictions in decisions_by_policy.items():
        count = collections.defaultdict(lambda: collections.Counter(TP=0, FP=0, FN=0, TN=0))
        for record, prediction in zip(records, predictions):
            for g in range(3):
                if not record['truth_valid'][g]:
                    continue
                y, p = bool(record['target'][g]), bool(prediction[g])
                field = 'TP' if y and p else 'FN' if y else 'FP' if p else 'TN'
                count[(record['family'], record['tier'])][field] += 1
        matched = 0
        for row in tables:
            if row['variant'] == profile:
                require(all(count[(row['family'], row['tier'])][k] == row[k] for k in ('TP', 'FP', 'FN', 'TN')), 'Independent confusion mismatch')
                matched += 4
        require(matched == 120, 'Expected all 30 independently checked packages')
        audit[profile] = dict(status='pass', independent_confusion_checks=matched, samples=len(records))
    return audit


def main():
    root = Path('/staff/tangjiarui/OPC/inference_prior_20261102')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031'))
    parser.add_argument('--comparison', type=Path, default=root / 'comparison_v1')
    parser.add_argument('--gbdt', type=Path, default=root / 'gbdt')
    parser.add_argument('--old-rules', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/full-matrix-20261031/blind-rules'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--p0-head-cap', type=float, default=.05)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()) and out != ALLOWED_OUTPUT.resolve(), 'Unsafe output directory')
    require(not out.exists(), 'Refusing to overwrite headwise output')
    out.mkdir(parents=True)
    started = time.monotonic()
    original = read_json(args.comparison / 'frozen_validation_policy.json')
    split_hashes = read_json(args.base / 'training_barrier.json')['split_hashes']
    dispatch = read_json(Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/dispatch_current.json'))
    sizes = {r: v['checkpoint_bytes'] for r, v in dispatch['model_details'].items() if r != 'rules'}
    compact = sorted(r for r in original['provenance'] if r != 'qlora' and sizes[r] < 10_000_000_000)
    require(len(compact) == 19, 'Expected 19 original compact routes excluding Qwen')
    marker = read_json(args.gbdt / 'predictions/gbdt_prior_v1.complete.json')
    require(marker['status'] == 'complete' and marker['barrier_hash'] == sha(args.gbdt / 'training_barrier.json'), 'GBDT training provenance mismatch')
    sizes['gbdt_prior_v1'] = sum((args.gbdt / f'model_{g}.pkl').stat().st_size for g in GROUPS) + (args.gbdt / 'vectorizer.pkl').stat().st_size
    gbdt_expected = dict(records=marker['records'], prediction_sha256=marker['prediction_hash'])
    locators, old_source = load_old_rule_locators(args.old_rules)
    validation = load_split(args.base / 'val.jsonl', 'val', split_hashes['val'])
    val_rules = retained_prior(prior_from_file(validation, args.comparison / 'prior_outputs/val.jsonl'), old_bits(validation, locators))
    probabilities = {r: load_probabilities(args.base, r, validation, original['provenance'][r]) for r in compact}
    probabilities['gbdt_prior_v1'] = load_probabilities(args.gbdt, 'gbdt_prior_v1', validation, gbdt_expected)
    policies = {p: {f: {} for f in FAMILIES} for p in ('f1', 'high_recall_review')}
    diagnostics = []
    for family in FAMILIES:
        indexes = [i for i, r in enumerate(validation) if r['family'] == family]
        records = [validation[i] for i in indexes]
        rules = [val_rules[i] for i in indexes]
        probs = {route: [p[i] for i in indexes] for route, p in probabilities.items()}
        for group, name in enumerate(GROUPS):
            options = candidates(records, probs, rules, group, sizes, args.p0_head_cap)
            for profile in policies:
                policies[profile][family][name] = select(options, profile)
            diagnostics.extend(dict(family=family, group=name, **c) for c in options)
        event('headwise_family_selected', family=family,
              selections={p: {g: c['route'] for g, c in selected[family].items()} for p, selected in policies.items()})
    frozen = dict(status='frozen', version='family-headwise-2026.11.02-1',
                  selection_source='original validation only', test_split_parsed=False,
                  validation_sha256=split_hashes['val'], thresholds=THRESHOLDS,
                  p0_per_head_cap=args.p0_head_cap, recall_target=.8,
                  profiles=policies, script_sha256=sha(__file__),
                  proxy_truth='P0-derived technical proxy; unknown bits masked; exploratory historical re-evaluation',
                  no_8b=True, no_new_training=True,
                  footprints={p: footprint(selected, args.base, args.gbdt) for p, selected in policies.items()},
                  source_rule_hashes=old_source,
                  original_cache_provenance={r: original['provenance'][r] for r in compact},
                  gbdt_cache=gbdt_expected,
                  source_prior_module_sha256=original['prior_module_sha256'])
    write_json(out / 'frozen_validation_policy.json', frozen)
    frozen_hash = sha(out / 'frozen_validation_policy.json')
    write_csv(out / 'validation_candidates.csv', diagnostics)
    val_summaries, val_tables = [], []
    for profile, selected in policies.items():
        summary, table, _ = summarize(validation, decide(validation, probabilities, val_rules, selected), profile)
        val_summaries.append(summary)
        val_tables.extend(table)
    write_csv(out / 'all30_validation.csv', val_tables)
    event('headwise_profiles_frozen_before_test', sha256=frozen_hash)

    test = load_split(args.base / 'test.jsonl', 'test', split_hashes['test'])
    require(not {r['training_group_id'] for r in validation}.intersection(r['training_group_id'] for r in test), 'Group leakage')
    test_rules = retained_prior(prior_from_file(test, args.comparison / 'prior_outputs/test.jsonl'), old_bits(test, locators))
    test_probabilities = {r: load_probabilities(args.base, r, test, original['provenance'][r]) for r in compact}
    test_probabilities['gbdt_prior_v1'] = load_probabilities(args.gbdt, 'gbdt_prior_v1', test, gbdt_expected)
    summaries, packages, group_rows, predicted = [], [], [], {}
    for profile, selected in policies.items():
        predicted[profile] = decide(test, test_probabilities, test_rules, selected)
        summary, table, group_table = summarize(test, predicted[profile], profile)
        summaries.append(summary)
        packages.extend(table)
        group_rows.extend(group_table)
    audit = independent_checks(test, predicted, packages)
    require(sha(out / 'frozen_validation_policy.json') == frozen_hash, 'Policy changed after test')
    write_csv(out / 'all30_test.csv', packages)
    write_csv(out / 'groups_test.csv', group_rows)
    write_json(out / 'independent_validation.json', audit)
    summary_rows = []
    for s in summaries:
        row = dict(profile=s['variant'], **s['overall'], P0_sample_FPR=s['P0_sample_FPR'], macro_F1_24=s['macro_F1_24'], P1_macro_F1_6=s['P1_macro_F1_6'])
        for group, m in s['groups'].items():
            row.update({group + '_' + k: v for k, v in m.items()})
        summary_rows.append(row)
    write_csv(out / 'summary_test.csv', summary_rows)
    result = dict(status='complete', seconds=time.monotonic() - started,
                  validation_records=len(validation), test_records=len(test), packages=30,
                  profile_count=2, policies_frozen_before_test=True,
                  frozen_validation_policy_sha256=frozen_hash, test_used_for_selection=False,
                  cached_prediction_file_sha256_checks=40, input_hash_checks=40*(len(validation)+len(test))//2,
                  gpu_forward_passes=0, new_training=False,
                  interpretation=frozen['proxy_truth'], footprints=frozen['footprints'],
                  validation_summaries=val_summaries, test_summaries=summaries,
                  infeasible_validation_recall_targets={p: [dict(family=f, group=g, recall=c['recall'], note=c['selection_note'])
                        for f, groups in selected.items() for g, c in groups.items()
                        if c['recall_target_applicable'] and not c['recall_target_achieved']] for p, selected in policies.items()})
    write_json(out / 'comparison.json', result)
    event('headwise_complete', output=str(out), seconds=result['seconds'], summaries=summaries)


if __name__ == '__main__':
    main()
