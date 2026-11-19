#!/usr/bin/env python3
"""Validation-frozen extension adding all-family GBDT to an existing comparison.

The original 20-model experiment and its frozen policy remain unchanged.
Only cached predictions are used; outputs stay in a new inference-prior child.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

from compare_versions import (ALLOWED_OUTPUT, FAMILIES, PriorRunner, add_counts,
    calibrate_family, decisions, event, family_objective, load_old_rule_locators,
    load_probabilities, load_split, old_bits, read_json, require, retained_prior,
    rows, sha, summarize, write_csv, write_json)
from evaluate_roadshow_prior import selected_choices


def prior_from_file(records, path):
    observations = {r['sample_id']: r for r in rows(path)}
    require(set(observations) == {r['id'] for r in records}, 'Prior IDs disagree with split')
    return [[bool(x) for x in observations[r['id']]['confirmed_bits']] for r in records]


def predictions_for(records, prior, base_prob, gbdt_prob, base_choices, base_gc, force_rules, threshold_sets, baseline_prior=None):
    if base_choices:
        base = decisions(records, base_prob, prior, base_choices, gc_only=base_gc,
                         force_rule_families=('CME_C2_sample', 'ULF_sample') if force_rules else ())
        basegc = decisions(records, base_prob, prior, base_choices, gc_only=True,
                           force_rule_families=('CME_C2_sample', 'ULF_sample') if force_rules else ())
    else:
        base, basegc = baseline_prior if baseline_prior is not None else prior, prior
    predictions = {'base_validation_recommendation': base}
    for mode in ('gbdt_all_groups', 'gbdt_gc_only'):
        choices = {f: dict(route='gbdt_prior_v1', thresholds=threshold_sets[f][mode]) for f in FAMILIES}
        predictions[mode] = decisions(records, {'gbdt_prior_v1': gbdt_prob}, prior, choices, gc_only=mode.endswith('gc_only'))
    answer = []
    for i, record in enumerate(records):
        p = gbdt_prob[i]
        bits = list(basegc[i])
        bits[2] = prior[i][2] or (p is not None and p[2] >= threshold_sets[record['family']]['base_gc_gbdt_d'][2])
        answer.append(bits)
    predictions['base_gc_gbdt_d'] = answer
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path('/staff/tangjiarui/OPC/inference_prior_20261102')
    parser.add_argument('--comparison', type=Path, default=root / 'comparison_v1')
    parser.add_argument('--gbdt', type=Path, default=root / 'gbdt')
    parser.add_argument('--base', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031'))
    parser.add_argument('--old-rules', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/full-matrix-20261031/blind-rules'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--p0-cap', type=float, default=.02)
    args = parser.parse_args()
    started = time.monotonic()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()) and out != ALLOWED_OUTPUT.resolve(), 'Unsafe output directory')
    require(not out.exists(), 'Refusing to overwrite extension output')
    out.mkdir(parents=True)
    old_frozen_path = args.comparison / 'frozen_validation_policy.json'
    old_frozen = read_json(old_frozen_path)
    require(old_frozen['status'] == 'frozen', 'Original comparison not frozen')
    split_hashes = read_json(args.base / 'training_barrier.json')['split_hashes']
    training = read_json(args.gbdt / 'training.json')
    barrier_path = args.gbdt / 'training_barrier.json'
    barrier = read_json(barrier_path)
    require(barrier['training_sha256'] == sha(args.gbdt / 'training.json'), 'GBDT training-report hash mismatch')
    require(training['train_sha256'] == split_hashes['train'] and training['val_sha256'] == split_hashes['val'],
            'GBDT source splits differ from original all-family splits')
    require(barrier['heads_complete'] == 3 and barrier['test_opened_before_this_barrier'] is False,
            'GBDT all-family training barrier failed')
    require(len(training['train_coverage']) == 30 and min(training['train_coverage'].values()) > 0,
            'GBDT training does not cover all 30 family/tier packages')
    marker_path = args.gbdt / 'predictions/gbdt_prior_v1.complete.json'
    marker = read_json(marker_path)
    require(marker['status'] == 'complete' and marker['barrier_hash'] == sha(barrier_path), 'GBDT prediction barrier mismatch')
    expected = dict(records=marker['records'], prediction_sha256=marker['prediction_hash'])
    base_choices, base_gc, force_rules = selected_choices(old_frozen)
    base_routes = {c['route'] for c in base_choices.values()} - {'rules'}
    locators, rule_source = load_old_rule_locators(args.old_rules)
    validation = load_split(args.base / 'val.jsonl', 'val', split_hashes['val'])
    val_pure = prior_from_file(validation, args.comparison / 'prior_outputs/val.jsonl')
    val_prior = retained_prior(val_pure, old_bits(validation, locators))
    val_gbdt = load_probabilities(args.gbdt, 'gbdt_prior_v1', validation, expected)
    val_base_probs = {r: load_probabilities(args.base, r, validation, old_frozen['provenance'][r]) for r in base_routes}
    if base_choices:
        val_basegc = decisions(validation, val_base_probs, val_prior, base_choices, gc_only=True,
                               force_rule_families=('CME_C2_sample', 'ULF_sample') if force_rules else ())
    else:
        val_basegc = val_prior
    threshold_sets = {}
    for family in FAMILIES:
        ids = [i for i, r in enumerate(validation) if r['family'] == family]
        subset = [validation[i] for i in ids]
        p = [val_gbdt[i] for i in ids]
        rules = [val_prior[i] for i in ids]
        threshold_sets[family] = {}
        for mode, gc in [('gbdt_all_groups', False), ('gbdt_gc_only', True)]:
            threshold_sets[family][mode] = calibrate_family(subset, p, rules, gc_only=gc, cap=args.p0_cap)['thresholds']
        just_d = [[0., 0., x[2]] if x is not None else None for x in p]
        threshold_sets[family]['base_gc_gbdt_d'] = calibrate_family(
            subset, just_d, [val_basegc[i] for i in ids], gc_only=False, cap=args.p0_cap)['thresholds']
        event('gbdt_family_calibrated', family=family)
    val_predictions = predictions_for(validation, val_prior, val_base_probs, val_gbdt, base_choices,
                                      base_gc, force_rules, threshold_sets,
                                      baseline_prior=val_pure if old_frozen['recommended_variant'] == 'pure_new_prior' else None)
    selected, diagnostics = {}, []
    for family in FAMILIES:
        ids = [i for i, r in enumerate(validation) if r['family'] == family]
        options = []
        for name, predictions in val_predictions.items():
            value = family_objective([validation[i] for i in ids], [predictions[i] for i in ids])
            options.append(dict(variant=name, family=family, **value))
        selected[family] = max(options, key=lambda r: (r['macro_F1'], r['P1_F1'], -r['P0_sample_FPR'], r['variant']))
        diagnostics.extend(options)
    val_predictions['val_selected_gbdt_extension'] = [val_predictions[selected[r['family']]['variant']][i]
                                                     for i, r in enumerate(validation)]
    frozen = dict(status='frozen', source='original validation only', test_split_parsed=False,
                  validation_sha256=split_hashes['val'], base_policy_sha256=sha(old_frozen_path),
                  gbdt_training_sha256=sha(args.gbdt / 'training.json'), gbdt_training_barrier_sha256=sha(barrier_path),
                  gbdt_prediction_sha256=expected['prediction_sha256'], thresholds=threshold_sets,
                  selected=selected, base_variant=old_frozen['recommended_variant'],
                  script_sha256=sha(__file__), p0_cap=args.p0_cap,
                  status_note='Exploratory historical re-evaluation; no official scientific truth')
    write_json(out / 'frozen_validation_extension.json', frozen)
    frozen_hash = sha(out / 'frozen_validation_extension.json')
    write_csv(out / 'validation_selection.csv', diagnostics)
    val_summaries, val_rows = [], []
    for name, predictions in val_predictions.items():
        summary, package_rows, _ = summarize(validation, predictions, name)
        val_summaries.append(summary)
        val_rows.extend(package_rows)
    write_csv(out / 'all30_validation.csv', val_rows)
    event('gbdt_extension_frozen_before_test', sha256=frozen_hash, selected=selected)

    test = load_split(args.base / 'test.jsonl', 'test', split_hashes['test'])
    require(not {r['training_group_id'] for r in validation}.intersection(r['training_group_id'] for r in test),
            'Validation/test group leakage')
    test_pure = prior_from_file(test, args.comparison / 'prior_outputs/test.jsonl')
    test_prior = retained_prior(test_pure, old_bits(test, locators))
    test_gbdt = load_probabilities(args.gbdt, 'gbdt_prior_v1', test, expected)
    test_base_probs = {r: load_probabilities(args.base, r, test, old_frozen['provenance'][r]) for r in base_routes}
    test_predictions = predictions_for(test, test_prior, test_base_probs, test_gbdt, base_choices, base_gc, force_rules, threshold_sets,
                                       baseline_prior=test_pure if old_frozen['recommended_variant'] == 'pure_new_prior' else None)
    test_predictions['val_selected_gbdt_extension'] = [test_predictions[selected[r['family']]['variant']][i]
                                                      for i, r in enumerate(test)]
    summaries, package_rows, group_rows = [], [], []
    for name, predictions in test_predictions.items():
        summary, packages, groups = summarize(test, predictions, name)
        summaries.append(summary)
        package_rows.extend(packages)
        group_rows.extend(groups)
    require(sha(out / 'frozen_validation_extension.json') == frozen_hash, 'Frozen GBDT extension changed after test')
    write_csv(out / 'all30_test.csv', package_rows)
    write_csv(out / 'groups_test.csv', group_rows)
    flat = []
    for value in summaries:
        flat.append(dict(variant=value['variant'], macro_F1_24=value['macro_F1_24'],
                         P1_macro_F1_6=value['P1_macro_F1_6'], P0_sample_FPR=value['P0_sample_FPR'], **value['overall']))
    write_csv(out / 'summary_test.csv', flat)
    write_json(out / 'comparison.json', dict(status='complete', variants=len(summaries),
                seconds=time.monotonic() - started, frozen_policy_sha256=frozen_hash,
                selection_source='original validation only; selection frozen before test parsing',
                test_used_for_selection=False, validation_records=len(validation), test_records=len(test),
                packages=30, proxy_truth='P0-derived technical proxy; unknown bits masked; not an official score',
                selected=selected, test_summaries=summaries, validation_summaries=val_summaries))
    event('gbdt_extension_complete', output=str(out), summaries=summaries)


if __name__ == '__main__':
    main()
