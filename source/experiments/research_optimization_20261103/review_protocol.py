"""Prepare blind grouped review and preserve unknowns until two reviews agree.

Identity/independence fields are declarations, not authentication or official truth.
Only val records are accepted by default. No image, label or prediction is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from functools import lru_cache

VERSION = 'review-protocol-2026.11.08-3'
FAMILIES = ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample', 'EPB_seg_sample', 'DMSP_SSUSI', 'ULF_sample']
DEFECTS = ['L0-01', 'L0-03', 'L2-G', 'L2-P', 'L2-T', 'L2-F', 'L3-06', 'L3-07', 'L3-MIS']


def image_dimensions(path):
    """Return ``[width, height]`` when the image header is readable.

    Review fixtures and some scientific products can have a damaged ancillary
    chunk while their dimensions are still present in the file header.  PIL
    quite correctly refuses to decode those files, but dimension validation
    should still be possible.  We therefore try PIL first and then use small,
    read-only header parsers for PNG/JPEG.  ``None`` means that even the
    header could not be trusted and the visual observation remains unknown.
    """
    path = Path(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.load()
            return list(image.size)
    except (ImportError, OSError, TypeError, ValueError):
        pass

    try:
        payload = path.read_bytes()
    except OSError:
        return None

    # PNG: tolerate a non-canonical signature as long as a valid IHDR is
    # present.  This lets us validate dimensions without treating pixels as
    # scientifically readable evidence.
    marker = payload.find(b'IHDR')
    if marker >= 4 and len(payload) >= marker + 12:
        width = int.from_bytes(payload[marker + 4:marker + 8], 'big')
        height = int.from_bytes(payload[marker + 8:marker + 12], 'big')
        if width > 0 and height > 0:
            return [width, height]

    # JPEG: scan SOF markers (the dimensions are in the frame header).
    if payload[:2] == b'\xff\xd8':
        index = 2
        while index + 9 < len(payload):
            if payload[index] != 0xFF:
                index += 1
                continue
            while index < len(payload) and payload[index] == 0xFF:
                index += 1
            if index >= len(payload):
                break
            marker = payload[index]
            index += 1
            if marker in (0xD8, 0xD9):
                continue
            if index + 2 > len(payload):
                break
            length = int.from_bytes(payload[index:index + 2], 'big')
            if length < 2 or index + length > len(payload):
                break
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if length >= 7:
                    height = int.from_bytes(payload[index + 3:index + 5], 'big')
                    width = int.from_bytes(payload[index + 5:index + 7], 'big')
                    if width > 0 and height > 0:
                        return [width, height]
            index += length
    return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def build_queue(manifest, output, per_family=20, seed=20261103):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if per_family <= 0:
        raise ValueError('per_family must be positive')
    before = sha(manifest)
    rows = read_rows(manifest)
    if not rows or any(row.get('split') != 'val' for row in rows):
        raise ValueError('Only an explicit all-validation manifest is accepted; train/test/B are rejected')
    seen_ids = set()
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        identity = row.get('id')
        group = row.get('training_group_id') or row.get('split_group_id') or row.get('base_sample_id')
        if not identity or not group or identity in seen_ids:
            raise ValueError('Missing identity/group or duplicated sample id')
        seen_ids.add(identity)
        if row.get('family') not in FAMILIES:
            raise ValueError('Unknown family')
        grouped[row['family']][group].append(row)
    if set(grouped) != set(FAMILIES):
        raise ValueError('All six families must be present before review sampling')
    group_families = defaultdict(set)
    for family, groups in grouped.items():
        for group in groups:
            group_families[group].add(family)
    if any(len(v) > 1 for v in group_families.values()):
        raise ValueError('Cross-family duplicate groups need a single cluster design; refusing ambiguous weights')
    rng = random.Random(seed)
    samples, mapping, coverage = [], [], []
    for family in FAMILIES:
        groups = grouped[family]
        keys = sorted(groups)
        count = min(per_family, len(keys))
        selected = rng.sample(keys, count)
        for group in selected:
            members = sorted(groups[group], key=lambda r: r['id'])
            row = rng.choice(members)
            review_id = 'REV-' + hashlib.sha256(f'{seed}|{row["id"]}'.encode()).hexdigest()[:20]
            label_path = Path(row['label_path'])
            data_path = Path(row['data_path']) if row.get('data_path') else None
            actual_label_sha = sha(label_path) if label_path.is_file() else None
            actual_data_sha = sha(data_path) if data_path and data_path.is_file() else None
            for key, actual in [('label_sha256', actual_label_sha), ('data_sha256', actual_data_sha)]:
                if row.get(key) is not None and row[key] != actual:
                    raise ValueError('Current review source differs from its frozen manifest: ' + key)
            samples.append(dict(review_id=review_id, family=family, data_path=str(data_path) if data_path else None,
                                label_path=str(label_path), data_exists=bool(data_path and data_path.is_file()),
                                label_exists=label_path.is_file(), data_sha256=actual_data_sha, label_sha256=actual_label_sha,
                                coordinate_instruction='Use native pixels, frame_id=data_sha256, image_size=[W,H], or an explicit time basis; no guessed units'))
            mapping.append(dict(review_id=review_id, source_id=row['id'], family=family,
                                source_group=group, split='val', tier=row.get('tier'),
                                inclusion_probability=count / len(keys) / len(members),
                                inverse_probability_weight=len(keys) * len(members) / count))
        coverage.append(dict(family=family, eligible_groups=len(keys), eligible_records=sum(map(len, groups.values())), selected=count))
    rng.shuffle(samples)
    if sha(manifest) != before:
        raise RuntimeError('Source manifest changed during sampling')
    output.mkdir(parents=True)
    write_rows(output / 'blind_samples.jsonl', samples)
    write_rows(output / 'private_sampling_map.jsonl', mapping)
    forms = [dict(review_id=s['review_id'], reviewer_id=None, review_complete=False,
                  declared_independent_of_model_predictions=False, applicable_classes=None,
                  defects=None, notes='Unreviewed; null is unknown, [] is only allowed after completed review') for s in samples]
    write_rows(output / 'reviewer_A.jsonl', forms)
    write_rows(output / 'reviewer_B.jsonl', forms)
    result = dict(version=VERSION, seed=seed, source_manifest_sha256=before, status='awaiting_two_independent_reviews',
                  samples=len(samples), families=coverage, model_predictions_exposed=False,
                  protocol='Uniform sample of validation groups by family, then uniform one current observation per selected group; HT weights are private',
                  official_truth=False, metric_status='not_available_without_review',
                  cautions=['Do not show private_sampling_map or old model scores to reviewers',
                            'P0 is not automatically human-reviewed clean',
                            'No automatic defect labels are inserted',
                            'Do not reuse reviewed validation items as a fresh untouched test'])
    write_json(output / 'queue_summary.json', result)
    (output / 'REVIEW_INSTRUCTIONS.md').write_text(
        '# 双人盲审\n\n只向审阅员提供blind_samples及各自表格，不提供private_sampling_map、模型预测或档位。\n'
        '两名审阅员独立查看当前图像/CSV与当前标签。每项选择适用类别并给出缺陷类别、样本、标注记录、原生像素区域或明确时间基准。\n'
        '未完成时defects=null，不能填空列表冒充无缺陷。完成检查且没有发现才填[]；无法判断的类别不列入已完整审查范围。\n'
        '同意独立声明并填写身份后合并；位置或类别不一致转第三人仲裁。现有规则/模型结果不充当真值。\n'
        '审阅身份与独立性只是声明，需团队实际组织核实。此材料不是官方答案。\n', encoding='utf-8')
    return result


def canonical_defects(defects):
    normalized = []
    for defect in defects:
        if defect.get('defect_class') not in DEFECTS or not isinstance(defect.get('locator'), dict) or not defect['locator']:
            raise ValueError('Each reviewed defect needs a recognized defect_class and a nonempty locator')
        if defect.get('status', 'known') != 'known':
            raise ValueError('Unknown/excluded review items cannot be converted into known consensus')
        if defect.get('critical') is not None and not isinstance(defect['critical'], bool):
            raise ValueError('Critical flag must be a boolean or unknown/null')
        # Validate against the same coordinate contract used for actual scoring.
        # Two reviewers repeating the same malformed location is not usable truth.
        locator_validator().locator_metrics(defect['locator'], defect['locator'])
        normalized.append(dict(defect_class=defect['defect_class'], locator=defect['locator'],
                               status='known', critical=defect.get('critical'), subtype=defect.get('subtype')))
    return sorted({json.dumps(d, sort_keys=True, ensure_ascii=False, allow_nan=False) for d in normalized})


@lru_cache(maxsize=1)
def locator_validator():
    candidates = [Path(__file__).resolve().parent / 'instance_audit.py',
                  Path(__file__).resolve().parents[2] / 'src/spaceqa_agent/instance_audit.py']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise ValueError('The instance_audit coordinate validator must be installed beside this script or in repository src')
    spec = importlib.util.spec_from_file_location('review_locator_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def index_forms(path):
    result = {}
    for row in read_rows(path):
        identity = row.get('review_id')
        if not identity or identity in result:
            raise ValueError('Duplicate or missing review identity')
        result[identity] = row
    return result


def valid_review(row):
    if not (row.get('review_complete') is True and isinstance(row.get('reviewer_id'), str) and row['reviewer_id'].strip()
            and row.get('declared_independent_of_model_predictions') is True and isinstance(row.get('defects'), list)):
        return False
    classes = row.get('applicable_classes')
    if not isinstance(classes, list) or not classes or any(c not in DEFECTS for c in classes):
        return False
    try:
        canonical_defects(row['defects'])
    except (ValueError, TypeError, KeyError):
        return False
    if any(d['defect_class'] not in classes for d in row['defects']):
        raise ValueError('Defect outside reviewed class coverage')
    return True


def reconcile(queue_dir, output):
    queue_dir, output = Path(queue_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    samples = index_forms(queue_dir / 'blind_samples.jsonl')
    a, b = index_forms(queue_dir / 'reviewer_A.jsonl'), index_forms(queue_dir / 'reviewer_B.jsonl')
    if set(a) != set(samples) or set(b) != set(samples):
        raise ValueError('Review forms must match the blind queue exactly')
    truth, pending = [], []
    for identity in samples:
        ra, rb = a[identity], b[identity]
        reason = None
        if not valid_review(ra) or not valid_review(rb):
            reason = 'incomplete_or_independence_not_declared'
        elif ra['reviewer_id'].strip() == rb['reviewer_id'].strip():
            reason = 'two_distinct_reviewers_required'
        elif set(ra['applicable_classes']) != set(rb['applicable_classes']):
            reason = 'coverage_disagreement_requires_adjudication'
        elif canonical_defects(ra['defects']) != canonical_defects(rb['defects']):
            reason = 'defect_or_locator_disagreement_requires_adjudication'
        if reason is None:
            sample = samples[identity]
            for kind in ['data', 'label']:
                path = Path(sample[kind + '_path']) if sample.get(kind + '_path') else None
                actual = sha(path) if path and path.is_file() else None
                if kind + '_sha256' not in sample or actual != sample[kind + '_sha256']:
                    reason = 'review_source_version_missing_or_changed'
                    break
            # A missing observation cannot support a negative visual/temporal
            # review. Keep the row, but require coverage to stop at file absence.
            if not sample.get('label_sha256'):
                reason = 'label_unavailable_for_annotation_review'
            elif not sample.get('data_sha256') and not set(ra['applicable_classes']).issubset({'L0-01', 'L0-03'}):
                reason = 'reviewed_classes_require_missing_observation'
            actual_image_size = None
            if (sample.get('data_sha256') and sample.get('family') != 'ULF_sample'
                    and set(ra['applicable_classes']) - {'L0-01', 'L0-03'}):
                actual_image_size = image_dimensions(sample['data_path'])
                if actual_image_size is None:
                    reason = 'reviewed_image_observation_unreadable'
            for defect in ra['defects']:
                loc = defect['locator']
                if loc.get('type') in ['bbox', 'polygon'] and loc.get('frame_id') != sample.get('data_sha256'):
                    reason = 'reviewed_locator_frame_does_not_match_source_version'
                if loc.get('type') in ['bbox', 'polygon']:
                    if actual_image_size is None:
                        actual_image_size = image_dimensions(sample['data_path'])
                        if actual_image_size is None:
                            reason = 'reviewed_image_dimensions_unverified'
                    if actual_image_size is not None and loc.get('image_size') is not None and loc['image_size'] != actual_image_size:
                        reason = 'reviewed_locator_dimensions_do_not_match_source'
                if loc.get('type') in ['file', 'record']:
                    allowed = {str(Path(sample[k])).replace('\\', '/') for k in ['data_path', 'label_path'] if sample.get(k)}
                    if str(loc.get('path')).replace('\\', '/') not in allowed:
                        reason = 'reviewed_file_locator_is_outside_sample'
        if reason:
            pending.append(dict(review_id=identity, status='unknown', reason=reason))
            continue
        truth.append(dict(sample_id=identity, family=samples[identity]['family'], review_status='consensus',
                          reviewed_classes=sorted(set(ra['applicable_classes'])),
                          defects=[json.loads(s) for s in canonical_defects(ra['defects'])],
                          data_sha256=samples[identity].get('data_sha256'), label_sha256=samples[identity].get('label_sha256'),
                          reviewer_ids=[ra['reviewer_id'].strip(), rb['reviewer_id'].strip()],
                          independent_of_model_predictions='declared_by_reviewers_not_authenticated'))
    output.mkdir(parents=True)
    write_rows(output / 'consensus_review.jsonl', truth)
    write_rows(output / 'requires_review_or_adjudication.jsonl', pending)
    result = dict(version=VERSION, status='complete' if not pending else 'incomplete',
                  consensus_samples=len(truth), unknown_samples=len(pending),
                  scientific_metrics_computed=False, official_reference=False,
                  critical_annotations_complete=all(d.get('critical') is not None for row in truth for d in row['defects']) if truth else False,
                  truth_scope='Only consensus samples and explicitly reviewed classes; all other sample-class pairs unknown',
                  independence='Two distinct reviewer identifiers declared; human identity and independence must be organizationally verified')
    write_json(output / 'reconciliation_summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--manifest', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--per-family', type=int, default=20)
    build.add_argument('--seed', type=int, default=20261103)
    merge = sub.add_parser('reconcile')
    merge.add_argument('--queue-dir', required=True)
    merge.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_queue(args.manifest, args.output, args.per_family, args.seed)
    else:
        result = reconcile(args.queue_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
