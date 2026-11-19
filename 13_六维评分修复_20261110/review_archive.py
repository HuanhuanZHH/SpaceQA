"""Review v2.3 evidence from archived results without reading raw packages or running models."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DIMENSIONS = (
    ('completeness', '完整性', 'C31', '字段齐备、重复标注和取值自洽的全范围检查记录'),
    ('annotationQuality', '标注质量', 'C32 C33 C34 C36 C37 C38', '独立标注参考及已验证的空间、时间容差'),
    ('sampleCorrespondence', '样本对应', 'C31 C35', '受影响样本的图像或测量证据，以及适用的对象身份对照'),
    ('coverageDistribution', '覆盖与分布', 'C41 C42 C43 C44', '目标类别与最低数、目标覆盖网格、独立分布与容差、划分协议'),
    ('traceability', '文档与来源追溯', 'C45 C46 C47', '独立核验的来源和处理链、声明与实测的对照证据'),
    ('usageConditions', '使用条件', 'C48 C49 C50', '验收、入库和选数各用途的实际授权、约束和审核记录'),
)
CODES = dict(zip((f'C{i}' for i in range(31, 39)),
                 ('L0-01', 'L2-G', 'L2-P', 'L2-T', 'L2-F', 'L3-06', 'L3-07', 'L3-MIS')))
STATES = ('pass', 'fail', 'undetermined', 'not_applicable')

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def jsonl(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def sha(path):
    with path.open('rb') as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
        return checksum.hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def collect(root, history, release):
    archive = root / '13_六维评分修复_20261110' / '归档验收依据'
    archive.mkdir()
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ('01_现场展示数据清单.md', '02_选手展示和自测要求.md', 'SHA256SUMS.txt',
                 'data_unzip_verification_20261108T112921Z.json'):
        copy(release / name, archive / '发放资料' / name)
    for name in ('manifest.json', 'icarus0.workspace.sha256', 'icarus1.workspace.sha256'):
        copy(history / 'trace' / name, archive / '原运行' / name)
    for name in ('summary.json', 'comparison.csv'):
        copy(history / 'report' / name, archive / '原运行' / name)
    for path in sorted((root / '04_评分卡').glob('*.json')):
        card = read(path)
        dest = archive / '原运行' / 'packages' / path.stem
        src = history / 'report' / 'packages' / path.stem
        copy(src / 'scorecard.json', dest / 'scorecard.json')
        copy(src / 'findings.jsonl', dest / 'findings.jsonl')
        ledger = Path(card['technicalObservations']['source']['inputLedger'])
        if not ledger.resolve().is_relative_to(history.resolve()) or ledger.parent.name != path.stem:
            raise ValueError(f'Unexpected source sample ledger: {ledger}')
        copy(ledger, dest / 'samples.jsonl')

def review(root, output):
    output.mkdir(parents=True)
    base = root / '13_六维评分修复_20261110'
    evidence = base / '归档验收依据'
    trace = read(evidence / '原运行' / 'manifest.json')
    policy_path = base / 'approved_weight_policy.json'
    policy = read(policy_path)
    if trace['status'] != 'verified' or policy['approvalStatus'] != 'approved':
        raise ValueError('Unverified historical trace or unapproved weights')
    if policy['weights'] != {d[0]: 1 for d in DIMENSIONS}:
        raise ValueError('Unexpected dimension weights')
    official = {}
    for line in (evidence / '发放资料' / 'SHA256SUMS.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            checksum, name = line.split(maxsplit=1)
            official[Path(name.lstrip('*')).name] = checksum.lower()
    cards = sorted((root / '04_评分卡').glob('*.json'))
    if len(cards) != 12 or set(trace['packages']) != {p.stem for p in cards}:
        raise ValueError('Release and archive must identify the same 12 packages')
    start = datetime.now(timezone.utc).isoformat()
    reports = []
    for path in cards:
        card = read(path)
        ident = path.stem
        member = trace['packages'][ident]
        if card['dirId'] != ident or official[member['original_tar']] != member['original_tar_sha256']:
            raise ValueError(f'Package identity mismatch: {ident}')
        src = evidence / '原运行' / 'packages' / ident
        hashes = {
            'scorecard.json': card['sourceHashes']['scorecard'],
            'findings.jsonl': card['sourceHashes']['findings'],
            'samples.jsonl': card['technicalObservations']['source']['inputLedgerSha256'],
        }
        for name, checksum in hashes.items():
            if sha(src / name) != checksum:
                raise ValueError(f'Archive byte mismatch: {ident}/{name}')
        samples = jsonl(src / 'samples.jsonl')
        findings = jsonl(src / 'findings.jsonl')
        if len(samples) != member['samples'] or len(samples) != card['sampleCount']:
            raise ValueError(f'Unexpected sample count: {ident}')
        if {r['findingId'] for r in findings} != {r['findingId'] for r in card['findings']}:
            raise ValueError(f'Unexpected finding IDs: {ident}')
        obs = card['technicalObservationEvidence']
        pairs = {r['sampleId']: r.get('labelSha256') for r in samples}
        if len(obs) != len(samples) or len(pairs) != len(samples) or pairs != {
            r['sampleId']: r['labelSha256'] for r in obs
        }:
            raise ValueError(f'S6 sample identity mismatch: {ident}')
        s6 = {}
        for row in obs:
            for check in row['checks']:
                s6.setdefault(check['checkId'], Counter())[check['status']] += 1
        results = {}
        for check_id, code in CODES.items():
            counts = Counter(s['checks'][code]['status'] for s in samples)
            hits = [r for r in card['findings'] if r['checkId'] == check_id]
            if set(counts) - set(STATES):
                raise ValueError(f'Invalid archived check state: {ident}/{check_id}')
            results[check_id] = {
                'scope': '历史样本检查账本及独立汇总的问题清单',
                'sampleStatusCounts': {k: counts[k] for k in STATES},
                'confirmedFindingCount': sum(r['confirmedOrPending'] == '已确认错误' for r in hits),
                'candidateFindingCount': sum(r['confirmedOrPending'] == '待复核事项' for r in hits),
                'findingIds': [r['findingId'] for r in hits],
            }
        previous = root / 'repair_20261110' / 'formal_replay' / ident
        prior = read(previous / 'results.json')
        if {r['checkId'] for r in prior['checks']} != {f'C{i}' for i in range(41, 51)}:
            raise ValueError(f'Incomplete archived formal checks: {ident}')
        for row in prior['checks']:
            ref = row['evidenceRef']
            proof = previous / ref['path']
            if sha(proof) != ref['contentHash']:
                raise ValueError(f'Extended check evidence mismatch: {ident}/{row["checkId"]}')
            results[row['checkId']] = {
                'scope': '已归档的扩展检查实际运行', 'status': row['status'], 'reason': row['reason'],
                'evidencePath': str(proof.relative_to(root)), 'sha256': ref['contentHash'],
            }
        missing = results['C31']['sampleStatusCounts']['fail']
        if missing != len(samples) - member['ready']:
            raise ValueError(f'File pairing count differs from archive: {ident}')
        tech = {d['dimId']: d for d in card['dimensions']}
        dimensions = []
        for dim, name, group, missing_basis in DIMENSIONS:
            ids = group.split()
            if dim == 'completeness':
                observed = {'scope': 'C31 登记配对存在性', 'verdict': 'fail' if missing else 'pass',
                            'registered': len(samples), 'missing': missing, 'score': tech[dim]['score']}
            elif dim in ('annotationQuality', 'sampleCorrespondence'):
                observed = {'scope': '已执行的规则与候选筛查', 'score': tech[dim]['observedTechnicalScore'],
                            'confirmed': sum(results[k]['confirmedFindingCount'] for k in ids),
                            'candidates': sum(results[k]['candidateFindingCount'] for k in ids)}
            else:
                letter = {'coverageDistribution': 'D', 'traceability': 'T', 'usageConditions': 'U'}[dim]
                observed = {'scope': '归档 S6 限定观测',
                            'checks': {k: dict(v) for k, v in sorted(s6.items()) if k.startswith(f'S6-{letter}')}}
            dimensions.append({
                'dimId': dim, 'name': name, 'verdict': 'undetermined',
                'score': None, 'range': None, 'grade': 'undetermined',
                'meaning': '完整维度证据不足，限定范围的结论见 observed',
                'missingBasis': missing_basis, 'checkIds': ids, 'observed': observed,
            })
        report = {
            'dirId': ident, 'datasetId': card['datasetId'], 'datasetVersion': card['datasetVersion'],
            'registeredSamples': len(samples), 'missingPairedData': missing,
            'operation': 'archive_only_no_new_inference', 'sourceRunId': card['runId'],
            'sourceExportId': card['exportRunId'], 'sourceTraceRun': trace['run'],
            'officialArchiveSha256': member['original_tar_sha256'],
            'sourceFiles': {name: {'path': str((src / name).relative_to(root)), 'sha256': digest}
                            for name, digest in hashes.items()},
            'dimensions': dimensions, 'checkResults': results,
            'formalAcceptance': 'evidence_incomplete', 'fullQualityScore': None,
            'observedTechnicalScore': card['observedTechnicalScore'], 'technicalRankGlobal': card['rankGlobal'],
            'approvedWeightPolicySha256': sha(policy_path),
        }
        write(output / f'{ident}.json', report)
        reports.append(report)
        print(f'{ident}: {len(samples)} registered, {missing} missing', flush=True)
    with (output / '六维验收矩阵.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dirId', 'datasetId', 'dimId', 'name', 'verdict', 'score', 'missingBasis', 'evidence'])
        for item in reports:
            for dim in item['dimensions']:
                writer.writerow([item['dirId'], item['datasetId'], dim['dimId'], dim['name'],
                                 dim['verdict'], dim['score'], dim['missingBasis'], f"{item['dirId']}.json"])
    write(output / '验收回执.json', {
        'status': 'review_complete_full_score_blocked', 'startedUtc': start,
        'completedUtc': datetime.now(timezone.utc).isoformat(), 'packageCount': len(reports),
        'registeredSamples': sum(r['registeredSamples'] for r in reports),
        'missingPairedData': sum(r['missingPairedData'] for r in reports),
        'dimensionReviewCount': sum(len(r['dimensions']) for r in reports),
        'formalFullScoresAvailable': 0, 'newModelRuns': 0,
        'sourceTraceRun': trace['run'], 'weightPolicySha256': sha(policy_path),
        'reports': [f"{r['dirId']}.json" for r in reports],
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--collect', type=Path, nargs=2, metavar=('HISTORICAL_RUN', 'OFFICIAL_RELEASE'))
    args = parser.parse_args()
    if args.collect:
        collect(args.root.resolve(), *args.collect)
    review(args.root.resolve(), args.output.resolve())
