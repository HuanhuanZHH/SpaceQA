"""Check C31 pairing against independently listed roadshow filenames."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import os
import re
import sys

data, delivery, work = map(Path, sys.argv[1:])
root = data / 'data_unzip'
findings = delivery / '06_问题清单' / 'findings.jsonl'
rules = {
    'AIA_AR_sample': ('label', '.png.json', 'data', '.png'),
    'EPB_seg_sample': ('', '.json', '', '.png'),
    'ULF_sample': ('', '.csv.json', '', '.csv'),
}

def join(folder, name):
    return folder + '/' + name if folder else name

def source_relative(path, package):
    parts = PurePosixPath(path).parts
    if package not in parts:
        raise ValueError(f'Locator outside source package: {path}')
    return '/'.join(parts[parts.index(package) + 1:])

def filenames(path, suffix):
    pattern = re.compile(r'^[A-Za-z0-9_]{20}' + re.escape(suffix) + r'$')
    matched = set()
    other = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.name.endswith(suffix):
                continue
            if entry.is_file(follow_symlinks=False) and pattern.fullmatch(entry.name):
                matched.add(entry.name.removesuffix(suffix))
            else:
                other.append(entry.name)
    return matched, sorted(other)

predicted = defaultdict(lambda: defaultdict(list))
rows = defaultdict(list)
hash_findings = hashlib.sha256()
with findings.open('rb') as f:
    for line in f:
        hash_findings.update(line)
        if not line.strip():
            continue
        item = json.loads(line)
        if item['checkId'] != 'C31':
            continue
        pkg = item['dirId']
        if item['findingStage'] != 'confirmed' or item.get('scoreEligible') is not True:
            rows[pkg].append({'findingId': item['findingId'], 'reason': 'not confirmed and score-eligible'})
            continue
        locator = item['locator']
        pair = (source_relative(locator['path'], pkg), source_relative(locator['expected_data_path'], pkg))
        predicted[pkg][pair].append({'findingId': item['findingId'], 'sampleId': item['sampleId'],
                                     'evidenceRef': item['evidenceRef']})

docs = data / '01_现场展示数据清单.md'
listing = docs.read_text(encoding='utf-8-sig')
expected_dirs = set(re.findall(r'((?:AIA_AR_sample|EPB_seg_sample|ULF_sample)_[A-Za-z0-9]+)\.tar', listing))
actual_dirs = {p.name for p in root.iterdir() if p.is_dir()}
assert len(expected_dirs) == 12 and expected_dirs <= actual_dirs
report = []
position_rows = []
for pkg in sorted(expected_dirs):
    family = next(f for f in rules if pkg.startswith(f + '_'))
    label_dir, label_suffix, data_dir, data_suffix = rules[family]
    label_stems, unusual_labels = filenames(root / pkg / label_dir, label_suffix)
    data_stems, unusual_data = filenames(root / pkg / data_dir, data_suffix)
    labels_missing_data = sorted(label_stems - data_stems)
    label_lookup = {join(label_dir, stem + label_suffix) for stem in label_stems}
    data_lookup = {join(data_dir, stem + data_suffix) for stem in data_stems}
    expected = {(join(label_dir, stem + label_suffix), join(data_dir, stem + data_suffix))
                for stem in labels_missing_data}
    observed = set(predicted.pop(pkg, {}))
    record = {'dirId': pkg, 'datasetId': family, 'labels': len(label_stems), 'dataFiles': len(data_stems),
              'paired': len(label_stems & data_stems), 'unpairedData': len(data_stems - label_stems),
              'unusualLabelNames': unusual_labels, 'unusualDataNames': unusual_data,
              'expectedMissing': len(expected), 'reportedMissing': len(observed),
              'TP': len(expected & observed), 'FP': len(observed - expected), 'FN': len(expected - observed),
              'nonScoringFindings': rows.pop(pkg, [])}
    record['precision'] = record['TP'] / len(observed) if observed else None
    record['recall'] = record['TP'] / len(expected) if expected else None
    report.append(record)
    for kind, pairs in [('TP', expected & observed), ('FP', observed - expected), ('FN', expected - observed)]:
        for label, expected_data in sorted(pairs):
            position_rows.append({'dirId': pkg, 'verdict': kind, 'labelRelative': label,
                                  'expectedDataRelative': expected_data, 'labelExists': label in label_lookup,
                                  'dataExists': expected_data in data_lookup})
assert not predicted and not rows, (predicted.keys(), rows.keys())
totals = Counter()
for r in report:
    for k in ('labels', 'dataFiles', 'paired', 'unpairedData', 'expectedMissing', 'reportedMissing', 'TP', 'FP', 'FN'):
        totals[k] += r[k]
result = {
    'completedUtc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
    'inputDocSha256': hashlib.sha256(docs.read_bytes()).hexdigest(),
    'findingsSha256': hash_findings.hexdigest(), 'method': 'case-sensitive released filename pairing; no data-file contents read',
    'metricScope': 'C31 label-to-data missing-pair positions only; not the official critical-recall threshold',
    'totals': {**totals, 'precision': totals['TP'] / totals['reportedMissing'] if totals['reportedMissing'] else None,
               'recall': totals['TP'] / totals['expectedMissing'] if totals['expectedMissing'] else None},
    'packages': report,
}
(work / 'receipt.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
with (work / 'positions.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=('dirId','verdict','labelRelative','expectedDataRelative','labelExists','dataExists'))
    writer.writeheader()
    writer.writerows(position_rows)
print(json.dumps(result, ensure_ascii=False, indent=2))
