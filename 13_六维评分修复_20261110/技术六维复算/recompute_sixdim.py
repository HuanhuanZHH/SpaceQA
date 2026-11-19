"""Recompute the evidence-scoped six-dimension scores from an existing delivery."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SAMPLE_CHECKS = {
    'completeness': ('C31',),
    'annotationQuality': ('C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38'),
    'sampleCorrespondence': ('C31', 'C35'),
}
S6_CHECKS = {
    'coverageDistribution': ('coverage_distribution', ('S6-D1', 'S6-D2', 'S6-D3')),
    'traceability': ('documentation_provenance', ('S6-T1', 'S6-T2', 'S6-T3', 'S6-T4', 'S6-T5')),
    'usageConditions': ('usage_conditions', ('S6-U1', 'S6-U2', 'S6-U3')),
}
DIMENSIONS = tuple(SAMPLE_CHECKS) + tuple(S6_CHECKS)
STATES = ('pass', 'fail', 'undetermined', 'not_applicable')
FORMAL_EVIDENCE = Path('13_六维评分修复_20261110/12包正式验收')
POLICY = Path('13_六维评分修复_20261110/approved_weight_policy.json')
FINDINGS = Path('06_问题清单/findings.jsonl')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count_status(counter):
    return {state: counter[state] for state in STATES}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def recompute(root):
    weights = read(root / POLICY)
    require(weights['approvalStatus'] == 'approved'
            and weights['weights'] == dict.fromkeys(DIMENSIONS, 1),
            'The approved equal-weight policy is required')
    finding_ids = set()
    by_package = defaultdict(list)
    with (root / FINDINGS).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            fid = row['findingId']
            require(fid not in finding_ids, f'Duplicate finding ID: {fid}')
            finding_ids.add(fid)
            by_package[row['dirId']].append(row)
    cards = sorted((root / '04_评分卡').glob('*.json'))
    require(len(cards) == 12, 'Expected 12 archived scorecards')
    packages = {}
    total_slots = Counter()
    for path in cards:
        card = read(path)
        dir_id = card['dirId']
        n = card['sampleCount']
        require(dir_id == path.stem and n > 0, f'Invalid card identity or sample count: {path}')
        rows = by_package.pop(dir_id, [])
        require({row['findingId'] for row in rows} ==
                {row['findingId'] for row in card['findings']},
                f'Card findings and exported findings differ: {dir_id}')
        source_dims = {d['dimId']: d for d in card['dimensions']}
        require(set(source_dims) == set(DIMENSIONS), f'Unknown dimensions: {dir_id}')
        per_sample = card['technicalObservationEvidence']
        sample_ids = {row['sampleId'] for row in per_sample}
        require(len(per_sample) == len(sample_ids) == n,
                f'Missing or duplicated S6 sample: {dir_id}')
        require(all(row['sampleId'] in sample_ids for row in rows),
                f'Finding sample outside the S6 roster: {dir_id}')

        dims, precise = {}, {}
        for dim_id, checks in SAMPLE_CHECKS.items():
            source = source_dims[dim_id]
            require(tuple(source['checkIds']) == checks and source['denominator'] == n,
                    f'Original dimension scope differs: {dir_id}/{dim_id}')
            confirmed = {r['sampleId'] for r in rows if r['checkId'] in checks
                         and r['findingStage'] == 'confirmed' and r['scoreEligible'] is True}
            candidates = {r['sampleId'] for r in rows if r['checkId'] in checks
                          and r['findingStage'] == 'candidate'}
            affected = len(confirmed | candidates)
            require(len(confirmed) == source['confirmedSamples'],
                    f'Original dimension confirmed sample count differs: {dir_id}/{dim_id}')
            precise[dim_id] = 100 * Fraction(n - affected, n)
            dims[dim_id] = {
                'score': float(precise[dim_id]), 'basis': 'confirmed_eligible_or_candidate_sample_union',
                'checkIds': checks, 'denominator': n, 'numerator': n - affected,
                'confirmedSamples': len(confirmed), 'candidateSamples': len(candidates),
                'intersectionSamples': len(confirmed & candidates), 'affectedSamples': affected,
                'unit': 'registered_sample',
            }

        by_check = defaultdict(Counter)
        check_ids = {check for _, checks in S6_CHECKS.values() for check in checks}
        for row in per_sample:
            checks = row['checks']
            require(len(checks) == len(check_ids) and {c['checkId'] for c in checks} == check_ids,
                    f'Missing or repeated S6 check: {dir_id}/{row["sampleId"]}')
            for c in checks:
                status = c['status']
                require(status in STATES, f'Invalid S6 state: {dir_id}/{c["checkId"]}')
                by_check[c['checkId']][status] += 1
        for dim_id, (old_id, checks) in S6_CHECKS.items():
            counts = sum((by_check[k] for k in checks), Counter())
            total_slots.update(counts)
            old = card['technicalObservations']['dimensions'][old_id]
            require(dict(counts) == old['scopeCounts'],
                    f'S6 aggregate differs from archived per-sample evidence: {dir_id}/{dim_id}')
            # An undetermined check cannot silently contribute a pass to this release.
            require(counts['undetermined'] == 0, f'Undetermined S6 evidence: {dir_id}/{dim_id}')
            denominator = counts['pass'] + counts['fail']
            require(denominator > 0, f'No determinate S6 slots: {dir_id}/{dim_id}')
            precise[dim_id] = 100 * Fraction(counts['pass'], denominator)
            require(abs(float(precise[dim_id]) - old['displayScore']) < 1e-9,
                    f'S6 score differs from archived display value: {dir_id}/{dim_id}')
            dims[dim_id] = {
                'score': float(precise[dim_id]), 'basis': 'determinate_scoped_S6_slots',
                'checkIds': checks, 'denominator': denominator, 'numerator': counts['pass'],
                'statusCounts': count_status(counts),
                'byCheck': {check: count_status(by_check[check]) for check in checks},
                'unit': 'sample_check_slot',
            }
        fpath = FORMAL_EVIDENCE / path.name
        formal = read(root / fpath)
        require(formal['dirId'] == dir_id and formal['fullQualityScore'] == card['fullQualityScore'],
                f'Formal evidence and card disagree: {dir_id}')
        six = sum(precise.values(), Fraction()) / len(DIMENSIONS)
        packages[dir_id] = {
            'datasetId': card['datasetId'],
            'sampleCount': n,
            'sixDimensionObservedScore': float(six),
            'dimensionScores': {key: float(value) for key, value in precise.items()},
            'dimensions': dims,
            'fullQualityScore': formal['fullQualityScore'],
            'formalAcceptance': formal['formalAcceptance'],
            'formalEvidencePath': fpath.as_posix(),
        }
    require(not by_package, f'Findings for unknown packages: {sorted(by_package)}')
    require(sum(total_slots.values()) == 272602, 'Unexpected aggregate S6 slot count')
    return {
        'version': 'sixdim-observed-confirmed-plus-pending-20261110',
        'scoreScope': 'scoped_observation_and_candidate_scenario',
        'weights': weights['weights'],
        'aggregation': 'arithmetic_mean_of_unrounded_dimension_scores',
        'findingsSourceSha256': digest(root / FINDINGS),
        'weightPolicySha256': digest(root / POLICY),
        'findingRecords': len(finding_ids),
        'registeredSamples': sum(p['sampleCount'] for p in packages.values()),
        's6StatusCounts': count_status(total_slots),
        'formalFullScoresAvailable': sum(p['fullQualityScore'] is not None for p in packages.values()),
        'packages': packages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    require(not out.is_relative_to(root), 'Use a separate output directory outside the delivery root')
    result = recompute(root)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'score.json', result)
    with (out / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dirId', 'datasetId', 'sampleCount', *DIMENSIONS,
                         'sixDimensionObservedScore', 'fullQualityScore'])
        for dir_id, entry in result['packages'].items():
            writer.writerow([dir_id, entry['datasetId'], entry['sampleCount'],
                             *(f'{entry["dimensionScores"][k]:.6f}' for k in DIMENSIONS),
                             f'{entry["sixDimensionObservedScore"]:.6f}', entry['fullQualityScore']])
    print(json.dumps({
        'packages': len(result['packages']), 'findingRecords': result['findingRecords'],
        'registeredSamples': result['registeredSamples'],
        's6StatusCounts': result['s6StatusCounts'],
        'formalFullScoresAvailable': result['formalFullScoresAvailable'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
