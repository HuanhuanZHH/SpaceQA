"""Recompute six-dimensional evidence bounds from the archived check ledgers."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

BASE = Path('13_六维评分修复_20261110')
FORMAL = BASE / '12包正式验收'
OBSERVED = BASE / '技术六维复算/score.json'
POLICY = BASE / 'approved_weight_policy.json'
SPEC = Path('03_评价规范.yaml')
STATES = ('pass', 'fail', 'undetermined', 'not_applicable')
LEGACY = {
    'C31': 'L0-01', 'C32': 'L2-G', 'C33': 'L2-P', 'C34': 'L2-T',
    'C35': 'L2-F', 'C36': 'L3-06', 'C37': 'L3-07', 'C38': 'L3-MIS',
}
S6 = {
    'coverageDistribution': ('coverage_distribution', ('S6-D1', 'S6-D2', 'S6-D3')),
    'traceability': ('documentation_provenance', ('S6-T1', 'S6-T2', 'S6-T3', 'S6-T4', 'S6-T5')),
    'usageConditions': ('usage_conditions', ('S6-U1', 'S6-U2', 'S6-U3')),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def states(counter):
    return {state: counter[state] for state in STATES}


def sample_checks(root, formal, n, dir_id):
    source = formal['sourceFiles']['samples.jsonl']
    path = (root / source['path']).resolve()
    require(path.is_relative_to(root) and sha256(path) == source['sha256'],
            f'Archived sample ledger hash differs: {dir_id}')
    out = defaultdict(Counter)
    ids = set()
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            require(row['dirId'] == dir_id and row['sampleId'] not in ids,
                    f'Invalid sample identity: {dir_id}/{row["sampleId"]}')
            ids.add(row['sampleId'])
            for check, legacy in LEGACY.items():
                status = row['checks'][legacy]['status']
                require(status in STATES, f'Invalid {check} status: {dir_id}')
                out[check][status] += 1
    require(len(ids) == n, f'Sample ledger denominator differs: {dir_id}')
    for check, counts in out.items():
        require(states(counts) == formal['checkResults'][check]['sampleStatusCounts'],
                f'Archived and formal counts differ: {dir_id}/{check}')
    return out, ids


def s6_checks(card, ids):
    check_ids = {check for _, checks in S6.values() for check in checks}
    out = defaultdict(Counter)
    s6_ids = set()
    for row in card['technicalObservationEvidence']:
        require(row['sampleId'] not in s6_ids, f'Duplicate S6 sample: {row["sampleId"]}')
        s6_ids.add(row['sampleId'])
        observations = row['checks']
        require(len(observations) == len(check_ids)
                and {obs['checkId'] for obs in observations} == check_ids,
                f'Incomplete S6 evidence: {row["sampleId"]}')
        for obs in observations:
            require(obs['status'] in STATES, f'Invalid S6 state: {row["sampleId"]}')
            out[obs['checkId']][obs['status']] += 1
    require(s6_ids == ids, f'S6 roster differs: {card["dirId"]}')
    for old_id, checks in S6.values():
        sums = sum((out[check] for check in checks), Counter())
        require(dict(sums) == card['technicalObservations']['dimensions'][old_id]['scopeCounts'],
                f'S6 aggregate differs: {card["dirId"]}/{old_id}')
    return out


def check_bounds(counts):
    p, f, u, _ = (counts[k] for k in STATES)
    n = p + f + u
    if n == 0:
        return None
    return (100 * Fraction(p, n), 100 * Fraction(p + u, n),
            100 * Fraction(p + f, n))


def check_details(check, unit, original):
    used = original.copy()
    scope_note = None
    if check == 'C33':
        used['undetermined'] += used.pop('pass', 0) + used.pop('fail', 0)
        scope_note = '缺独立类别真值，原词表检查结果不作为科学判定'
    value = check_bounds(used)
    detail = {
        'unit': unit, 'originalStatusCounts': states(original),
        'statusCounts': states(used), 'scopeNote': scope_note,
        'applicableUnits': sum(used[k] for k in STATES[:-1]),
        'lowerBound': float(value[0]) if value is not None else None,
        'upperBound': float(value[1]) if value is not None else None,
        'evidenceCoverage': float(value[2]) if value is not None else None,
    }
    return value, detail


def mean_bounds(values):
    return tuple(sum(col, Fraction()) / len(values) for col in zip(*values))


def calculate_dimension(checks, local_group=None):
    bounds = []
    detail = {}
    for check, unit, original in checks:
        value, detail[check] = check_details(check, unit, original)
        if value is not None:
            bounds.append(value)
    if local_group is not None:
        group_id, group_checks, excluded = local_group
        local_bounds = []
        components = {}
        for check, counts in group_checks:
            value, components[check] = check_details(check, 'sample_check_slot', counts)
            components[check]['diagnosticOnly'] = check in excluded
            if value is not None and check not in excluded:
                local_bounds.append(value)
        if local_bounds:
            grouped = mean_bounds(local_bounds)
            bounds.append(grouped)
            detail[group_id] = {
                'unit': 'scoped_evidence_group', 'components': components,
                'applicableScoredComponents': len(local_bounds),
                'lowerBound': float(grouped[0]), 'upperBound': float(grouped[1]),
                'evidenceCoverage': float(grouped[2]),
            }
        else:
            detail[group_id] = {'unit': 'scoped_evidence_group', 'components': components,
                                'applicableScoredComponents': 0}
    values = mean_bounds(bounds) if bounds else (Fraction(0), Fraction(100), Fraction(0))
    return values, {
        'score': float(values[0]), 'upperBound': float(values[1]),
        'evidenceCoverage': float(values[2]), 'applicableChecks': len(bounds),
        'notApplicableChecks': len(detail) - len(bounds), 'checks': detail,
        'scopeNote': None if bounds else '没有独立适用检查，证据空缺采用 [0,100] 与覆盖率 0，不判失败',
    }


def recompute(root):
    dim_ids = {d['dimId']: d['checkIds'] for d in read(root / SPEC)['dimensions']}
    scoring_ids = {
        dim: [check for check in checks
              if not ((dim == 'annotationQuality' and check == 'C35')
                      or (dim == 'sampleCorrespondence' and check == 'C31'))]
        for dim, checks in dim_ids.items()
    }
    policy = read(root / POLICY)
    require(policy['approvalStatus'] == 'approved'
            and policy['weights'] == dict.fromkeys(dim_ids, 1) and len(dim_ids) == 6,
            'Expected approved six equal weights')
    previous = read(root / OBSERVED)
    cards = sorted((root / '04_评分卡').glob('*.json'))
    require(len(cards) == len(previous['packages']) == 12, 'Expected all 12 packages')
    packages = {}
    s6_total = Counter()
    for card_path in cards:
        card = read(card_path)
        dir_id = card['dirId']
        require(dir_id == card_path.stem, f'Card identity differs: {card_path}')
        report_path = FORMAL / card_path.name
        formal = read(root / report_path)
        n = card['sampleCount']
        require(formal['dirId'] == dir_id and formal['registeredSamples'] == n
                and card['fullQualityScore'] is None and formal['fullQualityScore'] is None,
                f'Formal gate or identity differs: {dir_id}')
        history, ids = sample_checks(root, formal, n, dir_id)
        s6 = s6_checks(card, ids)
        s6_total.update(sum(s6.values(), Counter()))

        dimensions, results = {}, []
        for dim_id, check_ids in scoring_ids.items():
            applicable = []
            for check in check_ids:
                if check in LEGACY:
                    applicable.append((check, 'registered_sample', history[check]))
                else:
                    archived = formal['checkResults'][check]
                    status = archived['status']
                    require(status == 'undetermined',
                            f'Package check status changed; review grading policy: {dir_id}/{check}')
                    applicable.append((check, 'registered_package', Counter({status: 1})))
            local_group = None
            if dim_id in S6:
                group_name, checks = S6[dim_id]
                local_group = ('S6:' + group_name, [(check, s6[check]) for check in checks],
                               {'S6-T5', 'S6-U2'})
            scores, details = calculate_dimension(applicable, local_group)
            results.append(scores)
            dimensions[dim_id] = details
        lo, hi, coverage = (sum(v, Fraction()) / len(results) for v in zip(*results))
        old = previous['packages'][dir_id]
        packages[dir_id] = {
            'datasetId': card['datasetId'], 'sampleCount': n,
            'dimensionScores': {k: d['score'] for k, d in dimensions.items()},
            'dimensionUpperBounds': {k: d['upperBound'] for k, d in dimensions.items()},
            'dimensionCoverage': {k: d['evidenceCoverage'] for k, d in dimensions.items()},
            'dimensions': dimensions,
            'sixDimensionCoverageAdjustedScore': float(lo),
            'sixDimensionUpperBound': float(hi),
            'sixDimensionEvidenceCoverage': float(coverage),
            'previousScopedObservation': {
                'dimensionScores': old['dimensionScores'],
                'sixDimensionObservedScore': old['sixDimensionObservedScore'],
            },
            'confirmedPlusPendingScenario': {
                'affectedSamples': card['foundSamples'],
                'affectedRate': card['foundSamples'] / n,
                'score': 100 * (n - card['foundSamples']) / n,
                'rankGlobal': card['rankGlobal'],
            },
            'fullQualityScore': None, 'formalAcceptance': formal['formalAcceptance'],
            'formalEvidencePath': report_path.as_posix(),
            'archivedSampleLedgerSha256': formal['sourceFiles']['samples.jsonl']['sha256'],
        }
    require(sum(s6_total.values()) == 272602, 'Expected 272,602 S6 slots')
    return {
        'version': 'sixdim-deduplicated-evidence-bounds-20261110',
        'scoreScope': 'registered_checks_evidence_bounds_not_formal_quality_score',
        'weights': policy['weights'],
        'aggregation': 'normalize_each_check_group_S6_subchecks_then_equal_check_or_group_mean_within_dimension_then_equal_dimension_mean',
        'dimensionScoreMeaning': 'lower_bound_with_undetermined_in_applicable_denominator',
        'deduplicatedAssignments': {'C31': 'completeness', 'C35': 'sampleCorrespondence',
                                    'S6-T5': 'diagnostic_only_overlap_C45',
                                    'S6-U2': 'diagnostic_only_overlap_C31'},
        'weightPolicySha256': sha256(root / POLICY),
        'previousScopedObservationPath': OBSERVED.as_posix(),
        'registeredSamples': sum(p['sampleCount'] for p in packages.values()),
        's6StatusCounts': states(s6_total), 'formalFullScoresAvailable': 0,
        'packages': packages,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    require(not out.is_relative_to(root), 'Choose an output directory outside the delivery root')
    result = recompute(root)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'score.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                             allow_nan=False) + '\n', encoding='utf-8')
    dims = tuple(result['weights'])
    with (out / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dirId', 'datasetId', 'sampleCount',
                         *(f'{dim}_{metric}' for dim in dims for metric in ('lower', 'upper', 'coverage')),
                         'sixDimensionCoverageAdjustedScore', 'sixDimensionUpperBound',
                         'sixDimensionEvidenceCoverage', 'sixDimensionObservedScore',
                         'confirmedPlusPendingScenarioScore', 'fullQualityScore'])
        for dir_id, row in result['packages'].items():
            writer.writerow([dir_id, row['datasetId'], row['sampleCount'],
                             *(f'{row[field][dim]:.6f}' for dim in dims for field in
                               ('dimensionScores', 'dimensionUpperBounds', 'dimensionCoverage')),
                             *(f'{row[field]:.6f}' for field in ('sixDimensionCoverageAdjustedScore',
                               'sixDimensionUpperBound', 'sixDimensionEvidenceCoverage')),
                             f'{row["previousScopedObservation"]["sixDimensionObservedScore"]:.6f}',
                             f'{row["confirmedPlusPendingScenario"]["score"]:.6f}', None])
    print(json.dumps({'packages': len(result['packages']), 'registeredSamples': result['registeredSamples'],
                      'formalFullScoresAvailable': result['formalFullScoresAvailable'],
                      's6StatusCounts': result['s6StatusCounts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
