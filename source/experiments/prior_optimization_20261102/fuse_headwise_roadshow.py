#!/usr/bin/env python3
"""Optional headwise review profile on all B packages; no changes to the winner."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

from compare_versions import ALLOWED_OUTPUT, GROUPS, event, read_json, require, rows, sha, write_csv, write_json
from fuse_roadshow_extension import load_cached


def main():
    root = Path('/staff/tangjiarui/OPC/inference_prior_20261102')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roadshow', type=Path, default=Path('/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101'))
    parser.add_argument('--prior-output', type=Path, default=root / 'roadshow_v2')
    parser.add_argument('--frozen-policy', type=Path, default=root / 'headwise_v1/frozen_validation_policy.json')
    parser.add_argument('--profile', choices=('f1', 'high_recall_review'), default='high_recall_review')
    parser.add_argument('--gbdt', type=Path, default=root / 'gbdt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(out.is_relative_to(ALLOWED_OUTPUT.resolve()) and out != ALLOWED_OUTPUT.resolve(), 'Unsafe output directory')
    require(not out.exists(), 'Refusing to overwrite optional profile output')
    out.mkdir(parents=True)
    started = time.monotonic()
    frozen = read_json(args.frozen_policy)
    require(frozen['status'] == 'frozen' and frozen['test_split_parsed'] is False, 'Headwise policy not frozen before test')
    policy_hash = sha(args.frozen_policy)
    selected = frozen['profiles'][args.profile]
    require(read_json(args.prior_output / 'prior_complete.json')['status'] == 'complete', 'Fresh prior incomplete')
    training_hash = sha(args.gbdt / 'training.json')
    summaries, checks = [], 0
    for package in sorted((args.roadshow / 'packages').iterdir()):
        path = package / 'inventory.jsonl'
        if not path.is_file():
            continue
        inventory = [{k: r.get(k) for k in ('id', 'family', 'label_path', 'data_path', 'input_sha256', 'input_missing', 'rule_bits')} for r in rows(path)]
        family = inventory[0]['family']
        choices = selected[family]
        routes = {c['route'] for c in choices.values()} - {'rules'}
        cache, sources = {}, {}
        for route in routes:
            if route == 'gbdt_prior_v1':
                model, marker = load_cached(args.gbdt / 'roadshow_predictions' / (package.name + '.jsonl'),
                                           args.gbdt / 'roadshow_predictions' / (package.name + '.complete.json'), inventory,
                                           inventory_hash=sha(path), training_hash=training_hash)
            else:
                model, marker = load_cached(package / 'models' / (route + '.jsonl'), package / 'models' / (route + '.complete.json'), inventory)
            cache[route], sources[route] = model, marker['prediction_sha256']
            checks += 1
        prior_path = args.prior_output / 'packages' / package.name / 'prior_observations.jsonl'
        prior = {r['sample_id']: r for r in rows(prior_path)}
        require(set(prior) == {r['id'] for r in inventory}, 'Prior/inventory IDs differ')
        folder = out / 'packages' / package.name
        folder.mkdir(parents=True)
        (folder / 'prior_observations.jsonl').symlink_to(prior_path)
        counts = collections.Counter()
        route_label = '|'.join(g + ':' + choices[g]['route'] for g in GROUPS)
        thresholds = [choices[g]['threshold'] for g in GROUPS]
        with (folder / 'selected_candidates.jsonl').open('w', encoding='utf-8') as stream:
            for record in inventory:
                sid = record['id']
                current = list(prior[sid]['confirmed_bits'])
                current = [int(current[g] or record['rule_bits'][g]) if g < 2 else current[g] for g in range(3)]
                probabilities = []
                bits = []
                model_sources = []
                for g, name in enumerate(GROUPS):
                    route = choices[name]['route']
                    p = cache[route][sid].get('probabilities') if route != 'rules' else None
                    probabilities.append(p[g] if p is not None else 0.)
                    bits.append(int(p is not None and p[g] >= thresholds[g]))
                    model_sources.append(dict(group=name, route=route, threshold=thresholds[g], probability=p[g] if p is not None else None))
                if record['input_missing']:
                    probabilities = None
                require(bits == [int(probabilities is not None and probabilities[g] >= thresholds[g]) for g in range(3)], 'Exported headwise probabilities/thresholds do not reconstruct model flags')
                combined = [int(a or b) for a, b in zip(current, bits)]
                findings = [dict(defectType='MODEL_' + name + '_CANDIDATE', group=name,
                                 certainty='candidate', score_eligible=False,
                                 locator=record['label_path'], pointer='/',
                                 basis='High-recall or headwise compact expert flag; requires scientific review and localization',
                                 observation=model_sources[g]) for g, name in enumerate(GROUPS) if bits[g]]
                output = dict(sample_id=sid, package=package.name, family=family,
                              label_path=record['label_path'], data_path=record['data_path'],
                              input_missing=record['input_missing'], route=route_label,
                              thresholds=thresholds, probabilities=probabilities,
                              model_candidate_bits=bits, current_rule_bits=current,
                              combined_flag_bits=combined, findings=findings, model_sources=model_sources,
                              frozen_validation_policy_sha256=policy_hash, selected_variant=args.profile,
                              ground_truth_available=False, F1=None, precision=None, recall=None,
                              model_findings_score_eligible=False, prior_observations_path=str(prior_path))
                stream.write(json.dumps(output, ensure_ascii=False, allow_nan=False) + '\n')
                for g, name in enumerate(GROUPS):
                    counts[name + '_model_candidate_samples'] += bits[g]
                    counts[name + '_combined_flag_samples'] += combined[g]
        summary = dict(package=package.name, family=family, records=len(inventory),
                       missing_inputs=sum(r['input_missing'] for r in inventory), route=route_label,
                       selected_variant=args.profile, **counts, cached_prediction_sha256=sources,
                       ground_truth_available=False, F1=None, precision=None, recall=None)
        write_json(folder / 'selected_summary.json', summary)
        summaries.append(summary)
    require(len(summaries) == 12 and sum(s['records'] for s in summaries) == 24782, 'Incomplete B profile coverage')
    write_csv(out / 'selected_package_summary.csv', summaries)
    write_json(out / 'complete.json', dict(status='complete', packages=12, records=24782,
                seconds=time.monotonic() - started, selected_variant=args.profile,
                frozen_validation_policy_sha256=policy_hash, cached_prediction_file_sha256_checks=checks,
                ground_truth_available=False, F1=None, precision=None, recall=None,
                model_candidates_scored=False, gpu_forward_passes=0, summaries=summaries,
                interpretation='Optional headwise review profile; the previous overall-F1 winner remains unchanged. No B calibration or independent scientific truth.'))
    event('headwise_roadshow_complete', output=str(out), packages=12, records=24782, profile=args.profile)


if __name__ == '__main__':
    main()
