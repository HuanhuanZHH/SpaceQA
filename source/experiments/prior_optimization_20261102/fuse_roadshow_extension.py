#!/usr/bin/env python3
"""Join fresh roadshow prior findings with validation-frozen expert candidates.

All source runs remain immutable. No rule output is regenerated, and no model
is fitted or forwarded. Model flags never become confirmed scientific errors.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import time

from compare_versions import ALLOWED_OUTPUT, event, read_json, require, rows, sha, write_csv, write_json
from evaluate_roadshow_prior import selected_choices


def load_cached(path, marker_path, inventory, *, inventory_hash=None, training_hash=None):
    marker = read_json(marker_path)
    require(marker['status'] == 'complete', 'Cached prediction is incomplete')
    require(sha(path) == marker['prediction_sha256'], 'Cached roadshow prediction hash mismatch')
    if inventory_hash is not None:
        require(marker['inventory_sha256'] == inventory_hash, 'GBDT roadshow inventory hash mismatch')
    if training_hash is not None:
        require(marker['frozen_training_sha256'] == training_hash, 'GBDT roadshow training hash mismatch')
        require(marker['use_B_for_training_or_thresholds'] is False, 'GBDT used B for fitting or calibration')
    result = {}
    for record in rows(path):
        require(record['id'] not in result, 'Duplicate cached prediction ID')
        require(record['status'] in {'success', 'input_missing', 'data_missing'}, 'Unexpected prediction state')
        p = record.get('probabilities')
        if record['status'] == 'success':
            require(p is not None and len(p) == 3 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in p), 'Invalid model probabilities')
        else:
            require(p is None, 'Missing-input prediction contains probabilities')
        result[record['id']] = record
    require(len(result) == marker['records'] and set(result) == {r['id'] for r in inventory}, 'Cached model/inventory IDs differ')
    for record in inventory:
        prediction = result[record['id']]
        require((prediction['status'] != 'success') == bool(record['input_missing']), 'Missing state mismatch')
        if 'input_sha256' in prediction:
            require(prediction['input_sha256'] == record['input_sha256'], 'GBDT roadshow per-record input hash mismatch')
    return result, marker


def main():
    root = Path('/staff/tangjiarui/OPC/inference_prior_20261102')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roadshow', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101'))
    parser.add_argument('--prior-output', type=Path, default=root / 'roadshow_v2')
    parser.add_argument('--base-policy', type=Path, default=root / 'comparison_v1/frozen_validation_policy.json')
    parser.add_argument('--extension-policy', type=Path, default=root / 'gbdt_extension_v1/frozen_validation_extension.json')
    parser.add_argument('--gbdt', type=Path, default=root / 'gbdt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()) and out != ALLOWED_OUTPUT.resolve(), 'Unsafe output directory')
    require(not out.exists(), 'Refusing to overwrite fusion output')
    out.mkdir(parents=True)
    started = time.monotonic()
    prior_complete = read_json(args.prior_output / 'prior_complete.json')
    require(prior_complete['status'] == 'complete' and prior_complete['records'] == 24782 and prior_complete['packages'] == 12, 'Roadshow fresh prior audit incomplete')
    base = read_json(args.base_policy)
    extension = read_json(args.extension_policy)
    extension_hash = sha(args.extension_policy)
    require(base['status'] == extension['status'] == 'frozen', 'Policies are not frozen')
    require(base['test_split_parsed'] is False and extension['test_split_parsed'] is False, 'Policies were not frozen before test parsing')
    require(extension['base_policy_sha256'] == sha(args.base_policy), 'Extension used a different base policy')
    require(extension['gbdt_training_sha256'] == sha(args.gbdt / 'training.json'), 'GBDT training source mismatch')
    base_choices, base_gc, base_force = selected_choices(base)
    package_summaries = []
    prediction_checks = 0
    total_records = 0
    for package in sorted((args.roadshow / 'packages').iterdir()):
        inventory_path = package / 'inventory.jsonl'
        if not inventory_path.is_file():
            continue
        inventory = [{k: r.get(k) for k in ('id', 'family', 'data_path', 'label_path', 'input_missing', 'input_sha256', 'rule_bits')}
                     for r in rows(inventory_path)]
        family = inventory[0]['family']
        mode = extension['selected'][family]['variant']
        old_choice = base_choices.get(family, dict(route='rules', thresholds=[1.01] * 3))
        old_route = 'rules' if base_force and family in {'CME_C2_sample', 'ULF_sample'} else old_choice['route']
        need_base = mode in {'base_validation_recommendation', 'base_gc_gbdt_d'} and old_route != 'rules'
        need_gbdt = mode in {'gbdt_all_groups', 'gbdt_gc_only', 'base_gc_gbdt_d'}
        model, gbdt = {}, {}
        sources = {}
        if need_base:
            model, marker = load_cached(package / 'models' / (old_route + '.jsonl'), package / 'models' / (old_route + '.complete.json'), inventory)
            sources[old_route] = marker['prediction_sha256']
            prediction_checks += 1
        if need_gbdt:
            gbdt, marker = load_cached(args.gbdt / 'roadshow_predictions' / (package.name + '.jsonl'),
                                      args.gbdt / 'roadshow_predictions' / (package.name + '.complete.json'), inventory,
                                      inventory_hash=sha(inventory_path), training_hash=extension['gbdt_training_sha256'])
            sources['gbdt_prior_v1'] = marker['prediction_sha256']
            prediction_checks += 1
        prior_path = args.prior_output / 'packages' / package.name / 'prior_observations.jsonl'
        prior = {r['sample_id']: r for r in rows(prior_path)}
        require(set(prior) == {r['id'] for r in inventory}, 'Fresh roadshow prior IDs differ from inventory')
        folder = out / 'packages' / package.name
        folder.mkdir(parents=True)
        (folder / 'prior_observations.jsonl').symlink_to(prior_path)
        counts = collections.Counter()
        with (folder / 'selected_candidates.jsonl').open('w', encoding='utf-8') as stream:
            for record in inventory:
                sid = record['id']
                observation = prior[sid]
                current = list(observation['confirmed_bits'])
                if base['recommended_variant'] != 'pure_new_prior':
                    current = [int(current[g] or record['rule_bits'][g]) if g < 2 else current[g] for g in range(3)]
                model_bits = [0, 0, 0]
                consulted = []
                if need_base:
                    p = model[sid].get('probabilities')
                    bits = [int(p is not None and p[g] >= old_choice['thresholds'][g]
                                and (g < 2 or (mode == 'base_validation_recommendation' and not base_gc))) for g in range(3)]
                    model_bits = [int(a or b) for a, b in zip(model_bits, bits)]
                    consulted.append(dict(route=old_route, probabilities=p, thresholds=old_choice['thresholds'], candidate_bits=bits))
                if need_gbdt:
                    p = gbdt[sid].get('probabilities')
                    thresholds = extension['thresholds'][family][mode]
                    enabled = (False, False, True) if mode == 'base_gc_gbdt_d' else (True, True, False) if mode == 'gbdt_gc_only' else (True, True, True)
                    bits = [int(enabled[g] and p is not None and p[g] >= thresholds[g]) for g in range(3)]
                    model_bits = [int(a or b) for a, b in zip(model_bits, bits)]
                    consulted.append(dict(route='gbdt_prior_v1', probabilities=p, thresholds=thresholds, candidate_bits=bits))
                combined = [int(a or b) for a, b in zip(current, model_bits)]
                effective_route = '+'.join(x['route'] for x in consulted) or 'rules'
                effective_thresholds = [1.01] * 3
                effective_probabilities = None
                if consulted:
                    if mode == 'base_gc_gbdt_d':
                        by_route = {x['route']: x for x in consulted}
                        b = by_route.get(old_route)
                        d = by_route.get('gbdt_prior_v1')
                        effective_thresholds = [(b['thresholds'][g] if b else 1.01) if g < 2 else d['thresholds'][g] for g in range(3)]
                        if d['probabilities'] is not None:
                            effective_probabilities = [(b['probabilities'][g] if b and b['probabilities'] is not None else 0.) if g < 2 else d['probabilities'][g] for g in range(3)]
                    else:
                        effective_thresholds = list(consulted[0]['thresholds'])
                        effective_probabilities = consulted[0]['probabilities']
                        if mode == 'gbdt_gc_only' or (mode == 'base_validation_recommendation' and base_gc):
                            effective_thresholds[2] = 1.01
                require(model_bits == [int(effective_probabilities is not None and effective_probabilities[g] >= effective_thresholds[g]) for g in range(3)],
                        'Effective exported model decision does not reproduce candidate bits')
                findings = [dict(defectType='MODEL_' + group + '_CANDIDATE', group=group,
                                 certainty='candidate', score_eligible=False,
                                 locator=record['label_path'], pointer='/',
                                 basis='Validation-frozen model flag at sample level; localization/scientific correctness requires review',
                                 observation=dict(policy_variant=mode, models=consulted))
                            for i, group in enumerate(('G', 'C', 'D')) if model_bits[i]]
                result = dict(sample_id=sid, package=package.name, family=family,
                              label_path=record['label_path'], data_path=record['data_path'],
                              input_missing=record['input_missing'], selected_variant=mode,
                              route=effective_route, thresholds=effective_thresholds,
                              probabilities=effective_probabilities,
                              models=consulted, model_candidate_bits=model_bits,
                              current_rule_bits=current, combined_flag_bits=combined,
                              findings=findings, model_findings_score_eligible=False,
                              prior_observations_path=str(prior_path),
                              frozen_extension_sha256=extension_hash,
                              frozen_validation_policy_sha256=extension_hash,
                              ground_truth_available=False, F1=None, precision=None, recall=None,
                              official_score_computed=False)
                for i, group in enumerate(('G', 'C', 'D')):
                    counts[group + '_model_candidate_samples'] += model_bits[i]
                    counts[group + '_combined_flag_samples'] += combined[i]
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        summary = dict(package=package.name, family=family, records=len(inventory), selected_variant=mode,
                       missing_inputs=sum(r['input_missing'] for r in inventory), **counts,
                       cached_prediction_sha256=sources, prior_observations_sha256=sha(prior_path),
                       ground_truth_available=False, F1=None, precision=None, recall=None)
        write_json(folder / 'selected_summary.json', summary)
        package_summaries.append(summary)
        total_records += len(inventory)
    require(len(package_summaries) == 12 and total_records == 24782, 'Incomplete roadshow fusion coverage')
    write_csv(out / 'selected_package_summary.csv', package_summaries)
    write_json(out / 'complete.json', dict(status='complete', packages=12, records=total_records,
                seconds=time.monotonic() - started, frozen_extension_sha256=extension_hash,
                selected_variant='val_selected_gbdt_extension',
                frozen_validation_policy_sha256=extension_hash,
                prior_complete_sha256=sha(args.prior_output / 'prior_complete.json'),
                source_prior=str(args.prior_output), cached_prediction_file_sha256_checks=prediction_checks,
                model_candidates_scored=False, ground_truth_available=False,
                F1=None, precision=None, recall=None, gpu_forward_passes=0,
                summaries=package_summaries,
                interpretation='All 12 roadshow packages; fresh prior observations are referenced separately. Model flags remain candidates. No B training, threshold fitting, scientific truth, or official score.'))
    event('roadshow_extension_fusion_complete', output=str(out), packages=12, records=total_records)


if __name__ == '__main__':
    main()
