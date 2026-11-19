"""Compare published per-sample outputs without erasing raw inequalities.

The one documented context migration is validated against actual CSV values,
exact known wording, and the original/new finding-ID derivations.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

PUBLISHED_DETAIL = '八类时段候选已关联到被质疑的输入事件起止记录。CSV仅含数值t轴，未提供独立UTC原点；列出其首末值供复核，不把输入区间认作真实错误时段或独立端点偏差。'
EVENT_DETAIL = '新一轮八类时段候选对应此输入事件记录；区间为被质疑的输入起止时段。CSV坐标信息仅作上下文，数值t轴不提供独立UTC原点。缺独立真实端点，不能把该区间称为已证实错误时段或真实偏差量。'
LEGACY_EVENT_DETAIL = '新一轮八类时段候选对应此输入事件记录；给出被质疑的原始起止时段及CSV实际时间域供复核。缺独立真实端点，不能把该区间称为已证实错误时段或真实偏差量。'
WINDOW_DETAIL = '时段分类候选已关联到输入观测时间窗；这不是独立的事件起止定位，须结合CSV时间域和事件定义复核。'
ABSENT = '<ABSENT>'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read(path, key):
    with path.open(encoding='utf-8-sig') as stream:
        rows = [json.loads(line) for line in stream]
    mapped = {row[key]: row for row in rows}
    assert len(rows) == len(mapped), (str(path), 'duplicate sample IDs')
    return mapped


def indexed_sources(root):
    index = json.loads((root / 'report_index.json').read_text(encoding='utf-8-sig'))
    result = {}
    for report in index['reports']:
        files = {Path(value).name: Path(value) for value in report['sourcefiles']}
        assert 'samples.jsonl' in files and 'predictions.jsonl' in files, report['name']
        assert all(path.resolve().is_relative_to(root.resolve()) for path in files.values()), report['name']
        assert all(path.is_file() for path in files.values()), report['name']
        result[report['name']] = dict(files=files, reportVersion=report.get('reportVersion'))
    assert len(result) == len(index['reports']), 'duplicate package names in published index'
    return result


def semantic_key(finding):
    locator = copy.deepcopy(finding.get('locator') or {})
    locator.pop('time_axis_domain', None)
    return canonical([finding['defectType'], finding['checkId'], finding['ruleVersion'], finding['sourceKind'], locator])


def field_differences(current, reference, prefix=''):
    if isinstance(current, dict) and isinstance(reference, dict):
        for key in sorted(set(current) | set(reference)):
            if key not in current:
                yield prefix + '/' + key, ABSENT, reference[key]
            elif key not in reference:
                yield prefix + '/' + key, current[key], ABSENT
            else:
                yield from field_differences(current[key], reference[key], prefix + '/' + key)
    elif current != reference:
        yield prefix, current, reference


def actual_numeric_domain(path, cache):
    if path in cache:
        return cache[path]
    # Read every row independently instead of reusing the producer's domain.
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        header, first = next(reader), next(reader)
        last = first
        for row in reader:
            if len(row) == len(header):
                last = row
    column = next(i for i, name in enumerate(header) if name.lower() in {'t', 'time', 't_s', 'time_s', 'seconds', 'seconds_elapsed'})
    start, end = float(first[column]), float(last[column])
    assert math.isfinite(start) and math.isfinite(end) and start <= end, path
    name = header[column]
    domain = dict(kind='numeric_axis', column=name, columnIndex=column, firstValue=start, lastValue=end,
                  unit='second' if name.lower() in {'t_s', 'time_s', 'seconds', 'seconds_elapsed'} else 'not declared by CSV header',
                  source='actual CSV first and last rows', independent_absolute_time_mapping_established=False,
                  interpretation='Raw numeric axis only; no independent UTC origin is supplied by this CSV.')
    cache[path] = domain
    return domain


def expected_id(finding, locator):
    payload = [finding['sampleId'], finding['defectType'], locator, finding['candidate_trigger']]
    return 'REFRESH-' + hashlib.sha256(canonical(payload).encode()).hexdigest()[:20]


def documented_metadata_change(current, reference, domain_cache, legacy):
    """Accept only the two observed and independently verified migrations."""
    if current == reference:
        return True
    for finding in (current, reference):
        if (finding['defectType'] != 'L2-T' or finding['sourceKind'] != 'frozen_eight_class_plus_time_context'
                or not finding['sampleId'].startswith('ULF_sample_') or finding['findingStage'] != 'candidate'
                or finding['scoreEligible'] is not False or finding['scientific_confirmation'] is not False):
            return False
    if current['sampleId'] != reference['sampleId'] or current['candidate_trigger'] != reference['candidate_trigger']:
        return False
    try:
        domain = actual_numeric_domain(current['locator']['data_path'], domain_cache)
        a, b = copy.deepcopy(current), copy.deepcopy(reference)
        aloc, bloc = a['locator'], b['locator']
        ameas, bmeas = a['measurements'], b['measurements']
        trigger = a['candidate_trigger']
        if trigger == 'questioned_input_event_interval':
            assert a['detail'] == EVENT_DETAIL
            assert b['detail'] == (LEGACY_EVENT_DETAIL if legacy else PUBLISHED_DETAIL)
            assert aloc['time_axis_domain'] == domain
            assert bloc['time_axis_domain'] == (None if legacy else domain)
            assert ameas['actual_csv_numeric_axis'] == domain
            if legacy:
                assert 'actual_csv_numeric_axis' not in bmeas
            else:
                assert bmeas['actual_csv_numeric_axis'] == domain
            assert a['findingId'] == expected_id(a, aloc)
            original_locator = copy.deepcopy(bloc)
            original_locator['time_axis_domain'] = None
            assert b['findingId'] == expected_id(b, original_locator)
        elif trigger == 'questioned_observation_window':
            assert a['detail'] == WINDOW_DETAIL
            assert b['detail'] == (WINDOW_DETAIL if legacy else PUBLISHED_DETAIL)
            assert 'time_axis_domain' not in aloc
            assert ameas['actual_csv_time_domain'] == domain
            assert 'actual_csv_numeric_axis' not in ameas
            if legacy:
                assert 'time_axis_domain' not in bloc
                assert bmeas['actual_csv_time_domain'] is None
                assert 'actual_csv_numeric_axis' not in bmeas
            else:
                assert bloc['time_axis_domain'] == domain
                assert bmeas['actual_csv_numeric_axis'] == domain
                assert 'actual_csv_time_domain' not in bmeas
            original_locator = copy.deepcopy(bloc)
            original_locator.pop('time_axis_domain', None)
            assert a['findingId'] == b['findingId'] == expected_id(a, original_locator)
        else:
            return False
        for finding in (a, b):
            finding.pop('findingId')
            finding.pop('detail')
            finding['locator'].pop('time_axis_domain', None)
            finding['measurements'].pop('actual_csv_numeric_axis', None)
            finding['measurements'].pop('actual_csv_time_domain', None)
        # Every other field, including probabilities, intervals, hashes, check
        # versions, and scientific/score states must remain exactly equal.
        return a == b
    except (AssertionError, KeyError, ValueError, OSError, StopIteration):
        return False


def compare(root, reference, *, legacy=False):
    current_sources = indexed_sources(root)
    reference_sources = None if legacy else indexed_sources(reference)
    if not legacy:
        assert set(current_sources) == set(reference_sources), 'published package sets differ'
    packages, domain_cache = [], {}
    for name, current_source in sorted(current_sources.items()):
        previous_source = (dict(files={kind: reference / 'packages' / name / kind for kind in ['samples.jsonl', 'predictions.jsonl']},
                                reportVersion='preserved-original-v1') if legacy else reference_sources[name])
        cp = read(current_source['files']['predictions.jsonl'], 'id')
        pp = read(previous_source['files']['predictions.jsonl'], 'id')
        cl = read(current_source['files']['samples.jsonl'], 'sampleId')
        pl = read(previous_source['files']['samples.jsonl'], 'sampleId')
        ids = set(cp)
        flags = dict(sampleIdsEqual=ids == set(pp) == set(cl) == set(pl), eightBitPredictionsEqual=True,
                     eightBitRulesEqual=True, modelCandidateSetsEqual=True, probabilitiesExactlyEqual=True,
                     predictionRecordsExactlyEqual=True, findingIdSetsEqual=True, matchedFindingLocatorsEqual=True,
                     semanticFindingLocationSetsEqual=True, allFindingsExactlyEqual=True,
                     allFindingContentEqualExceptNumericTimeContextEnrichment=True, allLedgerFieldsExceptFindingsEqual=True)
        paths, issues = Counter(), []
        id_samples = changed_records = enriched_records = 0
        for sid in sorted(ids & set(pp) & set(cl) & set(pl)):
            a, b = cp[sid], pp[sid]
            flags['predictionRecordsExactlyEqual'] &= a == b
            flags['eightBitPredictionsEqual'] &= all(a.get(k) == b.get(k) for k in ['classes', 'prediction'])
            flags['eightBitRulesEqual'] &= a.get('rule') == b.get('rule')
            flags['probabilitiesExactlyEqual'] &= a.get('probabilities') == b.get('probabilities')
            flags['modelCandidateSetsEqual'] &= set(a.get('model_candidates', [])) == set(b.get('model_candidates', []))
            flags['allLedgerFieldsExceptFindingsEqual'] &= {k: v for k, v in cl[sid].items() if k != 'findings'} == {k: v for k, v in pl[sid].items() if k != 'findings'}
            af, bf = cl[sid]['findings'], pl[sid]['findings']
            assert len({f['findingId'] for f in af}) == len(af), (name, sid, 'duplicate current finding ID')
            assert len({f['findingId'] for f in bf}) == len(bf), (name, sid, 'duplicate reference finding ID')
            aid, bid = {f['findingId']: f for f in af}, {f['findingId']: f for f in bf}
            if set(aid) != set(bid):
                flags['findingIdSetsEqual'] = False
                id_samples += 1
            flags['matchedFindingLocatorsEqual'] &= all(aid[fid].get('locator') == bid[fid].get('locator') for fid in set(aid) & set(bid))
            flags['allFindingsExactlyEqual'] &= Counter(map(canonical, af)) == Counter(map(canonical, bf))
            ag, bg = {semantic_key(f): f for f in af}, {semantic_key(f): f for f in bf}
            assert len(ag) == len(af) and len(bg) == len(bf), (name, sid, 'ambiguous semantic finding group')
            same_locations = set(ag) == set(bg)
            flags['semanticFindingLocationSetsEqual'] &= same_locations
            if not same_locations:
                flags['allFindingContentEqualExceptNumericTimeContextEnrichment'] = False
                if len(issues) < 10:
                    issues.append(dict(sampleId=sid, reason='semantic finding locations differ'))
            for key in sorted(set(ag) & set(bg)):
                differences = list(field_differences(ag[key], bg[key]))
                if not differences:
                    continue
                changed_records += 1
                paths.update(path for path, _, _ in differences)
                accepted = documented_metadata_change(ag[key], bg[key], domain_cache, legacy)
                flags['allFindingContentEqualExceptNumericTimeContextEnrichment'] &= accepted
                enriched_records += accepted
                if not accepted and len(issues) < 10:
                    issues.append(dict(sampleId=sid, defectType=ag[key]['defectType'],
                                       fieldDifferences=[dict(path=p, current=x, reference=y) for p, x, y in differences]))
        packages.append(dict(package=name, samples=len(ids), **flags, referenceReportVersion=previous_source['reportVersion'],
                             currentSourcefiles={k: str(v) for k, v in current_source['files'].items()},
                             referenceSourcefiles={k: str(v) for k, v in previous_source['files'].items()},
                             findingIdMismatchSamples=id_samples, changedFindingRecords=changed_records,
                             independentlyVerifiedNumericContextRecords=enriched_records,
                             exactChangedFindingFieldPaths=dict(paths), unexpectedDifferences=issues))
    required = ['sampleIdsEqual', 'eightBitPredictionsEqual', 'eightBitRulesEqual', 'modelCandidateSetsEqual',
                'probabilitiesExactlyEqual', 'predictionRecordsExactlyEqual', 'semanticFindingLocationSetsEqual',
                'allFindingContentEqualExceptNumericTimeContextEnrichment']
    checks = required + ['findingIdSetsEqual', 'matchedFindingLocatorsEqual', 'allFindingsExactlyEqual', 'allLedgerFieldsExceptFindingsEqual']
    summary = dict(status='prediction_and_location_pass_with_documented_context_metadata_migration' if all(all(row[k] for k in required) for row in packages) else 'differences_found',
                   benchmarkRoot=str(root), referenceRoot=str(reference),
                   referenceSelection='preserved_original_v1_packages' if legacy else 'published_report_index_sourcefiles',
                   packageCount=len(packages), samples=sum(row['samples'] for row in packages),
                   eightBitSlots=sum(row['samples'] for row in packages) * 8,
                   **{key: all(row[key] for row in packages) for key in checks},
                   findingIdMismatchSamples=sum(row['findingIdMismatchSamples'] for row in packages),
                   changedFindingRecords=sum(row['changedFindingRecords'] for row in packages),
                   independentlyVerifiedNumericContextRecords=sum(row['independentlyVerifiedNumericContextRecords'] for row in packages),
                   independentFullCsvReads=len(domain_cache),
                   documentedDifference='ULF L2-T metadata only: exact known wording, identical raw numeric CSV axis under different context keys, and checked legacy/new locator-derived IDs. Raw inequality is reported. Every other finding field and questioned event interval must remain exact.',
                   packages=packages)
    assert len(packages) == 12 and summary['samples'] == 24782
    return summary


def main(root, reference):
    published = compare(root, reference)
    legacy = compare(root, reference, legacy=True)
    for filename, result in [('per_sample_output_comparison.json', published),
                             ('per_sample_output_comparison_preserved_v1.json', legacy)]:
        (root / filename).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in published.items() if key != 'packages'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reference)
