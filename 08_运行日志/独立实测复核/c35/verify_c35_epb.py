"""Independently validate C35 on the four issued EPB directories using labels and PNG IHDR."""
import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IDS = ('EPB_seg_sample_fbbqy66u2d', 'EPB_seg_sample_mc4t7fou91',
       'EPB_seg_sample_qmn05l7r91', 'EPB_seg_sample_z443g9dgh7')
POINTER = '/basic_annotation/imgSize'
DOC = Path('13_六维评分修复_20261110/归档验收依据/发放资料/01_现场展示数据清单.md')
FINDINGS = Path('06_问题清单/findings.jsonl')
SIZE_PATTERN = re.compile(r'\s*([1-9]\d*)\s*[xX×]\s*([1-9]\d*)\s*(?:pixels?)?\s*')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def inspect_label(entry):
    dir_id, label = entry
    pair = label.with_suffix('.png')
    raw = label.read_bytes()
    evidence = {
        'dirId': dir_id, 'sampleId': f'{dir_id}:{label.name}', 'label': label.name,
        'image': pair.name, 'annotationPointer': POINTER, 'labelSha256': sha_bytes(raw),
        'imageHeaderBytesRead': 0,
    }
    try:
        annotation = json.loads(raw.decode('utf-8-sig'))
    except (ValueError, UnicodeError):
        return dict(evidence, status='undetermined', reason='invalid_json')
    basic = annotation.get('basic_annotation')
    if not isinstance(basic, dict):
        basic = {}
    evidence['declaredFileName'] = basic.get('fileName')
    evidence['declaredSizeField'] = basic.get('imgSize')
    try:
        with pair.open('rb') as stream:
            header = stream.read(24)
    except FileNotFoundError:
        return dict(evidence, status='undetermined', reason='missing_png')
    except OSError as exc:
        return dict(evidence, status='undetermined', reason='unreadable_png', errno=exc.errno)
    evidence['imageHeaderBytesRead'] = len(header)
    evidence['imageHeaderSha256'] = sha_bytes(header)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[8:16] != b'\0\0\0\rIHDR':
        return dict(evidence, status='undetermined', reason='invalid_png_header')
    actual = struct.unpack('>II', header[16:24])
    if any(v == 0 or v > 2**31 - 1 for v in actual):
        return dict(evidence, status='undetermined', reason='invalid_png_dimensions')
    evidence['actual'] = list(actual)
    text = basic.get('imgSize')
    if text is None or (isinstance(text, str) and not text.strip()):
        return dict(evidence, status='not_applicable', reason='missing_size_field')
    match = SIZE_PATTERN.fullmatch(text) if isinstance(text, str) else None
    if match is None:
        return dict(evidence, status='undetermined', reason='invalid_size_field')
    declared = tuple(int(v) for v in match.groups())
    errors = [abs(d-a) for d,a in zip(declared, actual)]
    evidence.update(declared=list(declared), maxPixelError=max(errors),
                    maxRelativeError=max(e/a for e,a in zip(errors, actual)))
    return dict(evidence, status='fail' if max(errors) else 'pass',
                reason='size_mismatch' if max(errors) else 'dimensions_match')


def baseline_findings(path):
    rows = []
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['checkId'] == 'C35' and row['dirId'] in IDS and row['findingStage'] == 'confirmed':
                rows.append(row)
    return rows


def score_scope(samples, findings):
    rows = {r['sampleId']: r for r in samples}
    truth = {key for key,r in rows.items() if r['status'] == 'fail'}
    matches = set()
    matched_findings = []
    false_positives = []
    duplicates = []
    not_adjudicable = []
    for finding in findings:
        key = finding['sampleId']
        check = rows.get(key)
        if check is None or check['status'] not in {'pass', 'fail'}:
            not_adjudicable.append(finding['findingId'])
            continue
        locator = finding['locator']
        location_matches = (
            locator.get('type') == 'annotationRecord'
            and locator.get('annotation_pointer') == POINTER
            and Path(locator.get('path') or '').name == check['label']
            and Path(locator.get('path') or '').parent.name == check['dirId']
        )
        if not location_matches or key not in truth:
            false_positives.append(finding['findingId'])
            continue
        if key in matches:
            duplicates.append(finding['findingId'])
            continue
        matches.add(key)
        old = finding['measurements']
        severity = finding['severity']['value']
        raw_declared = [abs(a-b) for a,b in zip(old['declared'], check['declared'])]
        raw_actual = [abs(a-b) for a,b in zip(old['actual'], check['actual'])]
        measurements = {
            'declared': raw_declared, 'actual': raw_actual,
            'maxPixelError': abs(old['max_pixel_error'] - check['maxPixelError']),
            'maxRelativeError': abs(severity - check['maxRelativeError']),
        }
        matched_findings.append({
            'findingId': finding['findingId'], 'sampleId': key,
            'labelSha256Matches': finding['evidence']['label_sha256'] == check['labelSha256'],
            'declared': check['declared'], 'actual': check['actual'],
            'maxPixelError': check['maxPixelError'], 'maxRelativeError': check['maxRelativeError'],
            'measurementDifferences': measurements,
        })

    tp, fp, fn = len(matches), len(false_positives), len(truth - matches)
    counter = Counter(r['status'] for r in samples)
    excluded = Counter(r['reason'] for r in samples if r['status'] in {'undetermined', 'not_applicable'})
    tp_differences = [r['measurementDifferences'] for r in matched_findings]
    max_declared = max((max(r['declared']) for r in tp_differences), default=None)
    max_actual = max((max(r['actual']) for r in tp_differences), default=None)
    max_pixels = max((r['maxPixelError'] for r in tp_differences), default=None)
    max_relative = max((r['maxRelativeError'] for r in tp_differences), default=None)
    return {
        'registeredLabels': len(rows), 'statusCounts': dict(counter),
        'evaluatedSampleCount': counter['pass'] + counter['fail'],
        'excludedByReason': dict(excluded), 'confirmedFindingRows': len(findings),
        'tp': tp, 'fp': fp, 'fn': fn, 'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
        'f1': ratio(2*tp, 2*tp+fp+fn), 'localizedFindingDuplicates': len(duplicates),
        'duplicateFindingIds': duplicates, 'unadjudicableFindingIds': not_adjudicable,
        'falsePositiveFindingIds': false_positives, 'falseNegativeSampleIds': sorted(truth - matches),
        'labelHashMatchesAmongTP': sum(r['labelSha256Matches'] for r in matched_findings),
        'measurementDifferences': {
            'maxDeclaredDimensionPixels': max_declared, 'maxActualDimensionPixels': max_actual,
            'maxAbsoluteErrorPixels': max_pixels, 'maxRelativeErrorRatio': max_relative,
        },
        'matchedFindings': matched_findings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery-root', type=Path, required=True)
    parser.add_argument('--issued-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    root, issued, out = (p.resolve() for p in (args.delivery_root, args.issued_root, args.output))
    if out.is_relative_to(root) or out.is_relative_to(issued):
        raise ValueError('Output must be outside the delivery and issued data roots')
    work = []
    orphan_images = {}
    for dir_id in IDS:
        package = issued / dir_id
        paths = sorted(p for p in package.iterdir() if p.suffix in {'.json', '.png'} and p.is_file())
        labels = [p for p in paths if p.suffix == '.json']
        work.extend((dir_id, p) for p in labels)
        label_stems = {p.stem for p in labels}
        orphan_images[dir_id] = [p.name for p in paths if p.suffix == '.png' and p.stem not in label_stems]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        samples = list(pool.map(inspect_label, work))
    findings = baseline_findings(root / FINDINGS)
    by_sample, by_finding = defaultdict(list), defaultdict(list)
    for row in samples:
        by_sample[row['dirId']].append(row)
    for row in findings:
        by_finding[row['dirId']].append(row)
    report = {
        'version': 'c35-issued-epb-independent-header-check-20261110',
        'checkId': 'C35', 'tolerancePixels': 0, 'scope': IDS, 'dataRoot': str(issued),
        'referenceBasis': {
            'issuedDatasetManifest': DOC.as_posix(), 'sections': ['3.2', '4.3'],
            'sha256': sha_bytes((root / DOC).read_bytes()), 'annotationPointer': POINTER,
            'pairing': 'same directory and stem, with .json and .png extensions',
            'referenceMethod': 'independently parse issued JSON and actual PNG IHDR, first 24 bytes only',
        },
        'findingsSha256': sha_bytes((root / FINDINGS).read_bytes()),
        'imageHeaderBytesRead': sum(r['imageHeaderBytesRead'] for r in samples),
        'fileNameDeclarationMismatchCount': sum(r.get('declaredFileName') not in (None, r['image']) for r in samples),
        'byDirectory': {d: dict(score_scope(by_sample[d], by_finding[d]), orphanImages=orphan_images[d]) for d in IDS},
        'total': score_scope(samples, findings),
        'limitations': [
            'Only C35 and these four issued EPB directories are evaluated.',
            'Missing or invalid images are unadjudicated; absent size fields on readable images are not applicable.',
            'Only the PNG header is read. No image decoding, model inference or official scoring is performed.',
            'Locations are compared by directory identity, label basename and JSON pointer.',
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                      encoding='utf-8')
    with (out / 'sample_evidence.jsonl').open('w', encoding='utf-8') as stream:
        for row in samples:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    compact = dict(report['total'])
    compact.pop('matchedFindings')
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
