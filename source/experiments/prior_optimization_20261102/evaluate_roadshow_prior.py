#!/usr/bin/env python3
"""Fresh current-input prior audit for every roadshow record, with cached experts.

No truth labels are inferred, and no model is trained or forwarded. Current
prior observations and model candidates remain separate. Whole prediction-file
hashes and sample identities are checked; old roadshow predictions do not carry
per-record input hashes. Raw images are not fully rehashed by this script.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from compare_versions import ALLOWED_OUTPUT, event, read_json, rows, sha, require, write_csv, write_json


def selected_choices(frozen):
    name = frozen['recommended_variant']
    if name in {'pure_new_prior', 'new_prior_retained_gc'}:
        return {}, False, False
    for key, value in frozen['recalibrated'].items():
        if name.startswith(key + '_'):
            return value['choices'], value['gc_only'], name.endswith('_cme_ulf_rules_only')
    if name.startswith('old_static_'):
        return frozen['old_static_choices'], '_gc_' in name, name.endswith('_cme_ulf_rules_only')
    raise ValueError('Unrecognized frozen recommended variant: ' + name)


def load_model(package_dir, route, inventory):
    marker = read_json(package_dir / 'models' / (route + '.complete.json'))
    path = package_dir / 'models' / (route + '.jsonl')
    require(marker['status'] == 'complete', 'Incomplete roadshow model cache')
    require(sha(path) == marker['prediction_sha256'], 'Roadshow cached prediction hash mismatch')
    answer = {}
    for record in rows(path):
        require(record['id'] not in answer, 'Duplicate roadshow model ID')
        require(record['status'] in {'success', 'input_missing'}, 'Unexpected roadshow model status')
        answer[record['id']] = record
    require(set(answer) == {r['id'] for r in inventory}, 'Roadshow cached model/inventory IDs differ')
    require(len(answer) == marker['records'], 'Roadshow prediction count mismatch')
    for record in inventory:
        require((answer[record['id']]['status'] == 'input_missing') == bool(record['input_missing']),
                'Roadshow missing-input states differ')
    return answer, dict(route=route, prediction_sha256=marker['prediction_sha256'],
                        cached_inference_seconds=marker['model_load_and_inference_seconds'],
                        cached_hardware=marker['hardware'], per_record_input_hash_present=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roadshow', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101'))
    parser.add_argument('--frozen-policy', type=Path, required=True)
    parser.add_argument('--prior-module', type=Path, default=Path(__file__).with_name('prior_rules.py'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--policy-wait-seconds', type=int, default=1800)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()) and out != ALLOWED_OUTPUT.resolve(), 'Unsafe output root')
    require(not out.exists(), 'Refusing to overwrite roadshow output')
    require(1 <= args.workers <= 32, 'Invalid worker count')
    out.mkdir(parents=True)
    started = time.monotonic()
    module_spec = importlib.util.spec_from_file_location('roadshow_prior_candidate', args.prior_module)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    # Roadshow label-provided origins remain dependent; no clean/reference
    # directory, tier or quality suffix is used to authorize a hard verdict.
    policy = module.Policy(trust_ulf_dateobs=False, ulf_duration_convention='either')
    inventories = {}
    input_provenance = {}
    for folder in sorted((args.roadshow / 'packages').iterdir()):
        path = folder / 'inventory.jsonl'
        if not path.is_file():
            continue
        records = []
        for record in rows(path):
            require(record.get('ground_truth_available') is False, 'Unexpected roadshow truth state')
            records.append({k: record.get(k) for k in ('id', 'package', 'family', 'label_path', 'data_path',
                                                       'data_sha256', 'input_sha256', 'input_missing', 'rule_bits')})
        require(len({r['id'] for r in records}) == len(records), 'Duplicate roadshow inventory ID')
        inventories[folder.name] = records
        input_provenance[folder.name] = dict(inventory_sha256=sha(path), records=len(records))
    require(len(inventories) == 12 and sum(map(len, inventories.values())) == 24782,
            'Expected all 12 roadshow packages and all 24,782 records')

    unique = {}
    for records in inventories.values():
        for record in records:
            if record['family'] == 'ULF_sample' and not record['input_missing']:
                unique.setdefault(record['data_sha256'], record)
    event('roadshow_full_csv_scan', unique_csv=len(unique), records=24782)

    def scan(record):
        require(sha(record['data_path']) == record['data_sha256'], 'Current roadshow CSV changed')
        stats = module.csv_statistics(record['data_path'])
        require(stats.get('full_scan') is True, 'Roadshow CSV scan incomplete')
        write_json(out / 'csv_statistics' / (record['data_sha256'] + '.json'),
                   dict(source_sha256=record['data_sha256'], source_path=record['data_path'], statistics=stats))
        return record['data_sha256'], stats

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        csv_stats = dict(pool.map(scan, unique.values()))
    csv_seconds = time.monotonic() - started

    def audit_package(item):
        name, records = item
        prepared, source_labels = [], {}
        for record in records:
            data = Path(record['label_path']).read_bytes()
            source_labels[record['id']] = hashlib.sha256(data).hexdigest()
            prepared.append(dict(sample_id=record['id'], family=record['family'], package=name,
                                 label=json.loads(data.decode('utf-8-sig')),
                                 label_path=record['label_path'], data_path=record['data_path'],
                                 csv_stats=csv_stats.get(record['data_sha256'])))
        results = module.evaluate_package(prepared, policy=policy)
        folder = out / 'packages' / name
        folder.mkdir(parents=True)
        with (folder / 'prior_observations.jsonl').open('w', encoding='utf-8') as stream:
            for record, result in zip(records, results):
                require(result['sample_id'] == record['id'], 'Roadshow prior sample ID mismatch')
                result.update(package=name, current_label_sha256=source_labels[record['id']],
                              recorded_data_sha256=record['data_sha256'],
                              historical_prepared_input_sha256=record['input_sha256'],
                              input_missing=record['input_missing'], ground_truth_available=False,
                              F1=None, precision=None, recall=None,
                              official_score_computed=False)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        summary = dict(package=name, family=records[0]['family'], records=len(records),
                       missing_inputs=sum(r['input_missing'] for r in records),
                       confirmed_finding_counts=dict(collections.Counter(f['defectType'] for r in results for f in r['findings'] if f['score_eligible'])),
                       candidate_finding_counts=dict(collections.Counter(f['defectType'] for r in results for f in r['findings'] if f['certainty'] == 'candidate')),
                       structural_missing_finding_counts=dict(collections.Counter(f['defectType'] for r in results for f in r['findings'] if f['group'] is None)),
                       confirmed_G=sum(r['confirmed_bits'][0] for r in results),
                       confirmed_C=sum(r['confirmed_bits'][1] for r in results),
                       confirmed_D=sum(r['confirmed_bits'][2] for r in results),
                       ground_truth_available=False, F1=None, precision=None, recall=None,
                       official_score_computed=False)
        write_json(folder / 'prior_summary.json', summary)
        event('roadshow_prior_package_complete', package=name, records=len(records),
              confirmed_counts=summary['confirmed_finding_counts'])
        return name, summary

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        summaries = dict(pool.map(audit_package, inventories.items()))
    write_csv(out / 'prior_package_summary.csv', summaries.values())
    write_json(out / 'prior_complete.json', dict(status='complete', packages=12, records=24782,
                csv_axis_full_scans=len(csv_stats), csv_scan_seconds=csv_seconds,
                seconds=time.monotonic() - started, prior_policy=dataclasses.asdict(policy),
                prior_module_sha256=sha(args.prior_module), inventories=input_provenance,
                label_bytes_hashed_current=True, raw_csv_bytes_rehashed=True,
                raw_image_bytes_rehashed=False, ground_truth_available=False,
                F1=None, precision=None, recall=None, official_score_computed=False))
    event('roadshow_prior_complete', output=str(out))

    deadline = time.monotonic() + args.policy_wait_seconds
    while not args.frozen_policy.is_file():
        require(time.monotonic() < deadline, 'Validation policy not ready; prior outputs are complete and retained')
        time.sleep(5)
    frozen = read_json(args.frozen_policy)
    require(frozen['status'] == 'frozen' and frozen['test_split_parsed'] is False, 'Policy was not frozen before historical test')
    require(frozen['selection_split'] == 'original_validation_only', 'Roadshow must never select its own routes')
    choices, gc_only, force_rules = selected_choices(frozen)
    models_verified = 0
    combined_summaries = []
    for name, records in inventories.items():
        family = records[0]['family']
        choice = choices.get(family, dict(route='rules', thresholds=[1.01] * 3))
        route = 'rules' if force_rules and family in {'CME_C2_sample', 'ULF_sample'} else choice['route']
        model, model_source = ({}, {})
        if route != 'rules':
            model, model_source = load_model(args.roadshow / 'packages' / name, route, records)
            models_verified += 1
        prior_path = out / 'packages' / name / 'prior_observations.jsonl'
        prior_results = list(rows(prior_path))
        counts = collections.Counter()
        with (out / 'packages' / name / 'selected_candidates.jsonl').open('w', encoding='utf-8') as stream:
            for record, prior in zip(records, prior_results):
                probabilities = model.get(record['id'], {}).get('probabilities')
                if probabilities is not None:
                    require(len(probabilities) == 3 and all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in probabilities), 'Invalid cached roadshow probabilities')
                model_bits = [int(probabilities is not None and (not gc_only or group < 2)
                                  and probabilities[group] >= choice['thresholds'][group]) for group in range(3)]
                current_bits = list(prior['confirmed_bits'])
                if frozen['recommended_variant'] != 'pure_new_prior':
                    current_bits = [int(current_bits[g] or record['rule_bits'][g]) if g < 2 else current_bits[g] for g in range(3)]
                combined_bits = [int(a or b) for a, b in zip(current_bits, model_bits)]
                for i, group in enumerate(('G', 'C', 'D')):
                    counts[group + '_model_candidate'] += model_bits[i]
                    counts[group + '_combined_flag'] += combined_bits[i]
                result = dict(sample_id=record['id'], package=name, family=family,
                              label_path=record['label_path'], data_path=record['data_path'],
                              route=route, thresholds=choice['thresholds'], probabilities=probabilities,
                              model_candidate_bits=model_bits, current_rule_bits=current_bits,
                              combined_flag_bits=combined_bits, input_missing=record['input_missing'],
                              verdict='model_candidates_require_scientific_review',
                              model_findings_score_eligible=False,
                              prior_observations_path=str(prior_path),
                              frozen_validation_policy_sha256=sha(args.frozen_policy),
                              ground_truth_available=False, F1=None, precision=None, recall=None)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
        summary = dict(package=name, family=family, route=route, records=len(records), **counts,
                       frozen_variant=frozen['recommended_variant'], ground_truth_available=False,
                       F1=None, precision=None, recall=None, cached_model_source=model_source)
        write_json(out / 'packages' / name / 'selected_summary.json', summary)
        combined_summaries.append(summary)
    write_csv(out / 'selected_package_summary.csv', combined_summaries)
    write_json(out / 'complete.json', dict(status='complete', packages=12, records=24782,
                seconds=time.monotonic() - started, selected_variant=frozen['recommended_variant'],
                frozen_validation_policy_sha256=sha(args.frozen_policy),
                cached_prediction_file_sha256_checks=models_verified,
                gpu_forward_passes=0, model_candidates_scored=False,
                ground_truth_available=False, F1=None, precision=None, recall=None,
                official_score_computed=False, summaries=combined_summaries,
                interpretation='Fresh deterministic current-input findings; model candidates use validation-frozen routes and cached probabilities. No roadshow ground truth or official score.'))
    event('roadshow_selected_candidates_complete', output=str(out), packages=12, records=24782,
          selected_variant=frozen['recommended_variant'])


if __name__ == '__main__':
    main()
