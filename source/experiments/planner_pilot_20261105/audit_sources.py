"""Independent re-read of every pilot source and mechanical reference."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from PIL import Image

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            block = f.read(256 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

def dictionaries(obj, location=''):
    pending = [(location, obj)]
    while pending:
        p, x = pending.pop()
        if isinstance(x, dict):
            yield p, x
            pending.extend((p + '/' + str(k), v) for k, v in x.items())
        elif isinstance(x, list):
            pending.extend((p + '/' + str(i), v) for i, v in enumerate(x))

def utc(value):
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)).timestamp()
    except ValueError:
        return None

def independently_recompute(task):
    source = task['source']
    label_path, data_path = Path(source['label_path']), Path(source['data_path'])
    label = json.loads(label_path.read_text())
    assert sha(label_path) == source['label_sha256']
    assert (sha(data_path) if data_path.is_file() else None) == source['data_sha256']
    basic = label.get('basic_annotation', {})
    science = label.get('science_annotation', {})
    event = label.get('hek_event_annotation', label.get('ulf_event_annotation', {}))
    records = event.get('matched_events', [])
    existence = data_path.is_file()
    result = {'missing_data': 'pass' if existence else 'fail'}
    names = [label.get('label_header', {}).get('fileName'), basic.get('fileName')]
    names = [n for n in names if isinstance(n, str) and n]
    result['reference_name_conflict'] = ('fail' if any(n != data_path.name for n in names) else 'pass') if names else 'not_applicable'
    if source['family'] in ['AIA_AR_sample', 'HMI_AR_sample', 'CME_C2_sample']:
        mismatches = []
        if 'event_count' in event:
            mismatches.append(event['event_count'] != len(records))
        if 'has_events' in event:
            mismatches.append(bool(event['has_events']) != bool(records))
        if isinstance(event.get('event_types'), list):
            actual = set(str(r.get('event_type', r.get('kind'))) for r in records if r.get('event_type', r.get('kind')) is not None)
            # CME schemas can omit the per-event type. Do not convert an absent
            # optional field to an observed negative category. Count/presence
            # declarations are still independently decidable.
            implicit_cme = source['family'] == 'CME_C2_sample' and bool(records) and any(r.get('event_type', r.get('kind')) is None for r in records)
            if not implicit_cme:
                mismatches.append(set(map(str, event['event_types'])) != actual)
        result['declaration_conflict'] = 'fail' if any(mismatches) else 'pass'
    width = height = None
    if existence and data_path.suffix.lower() == '.png':
        with Image.open(data_path) as image:
            width, height = image.size
        assert (width, height) == (task['facts']['width'], task['facts']['height'])
    if source['family'] in ['EPB_seg_sample', 'DMSP_SSUSI']:
        declared = []
        for p, obj in dictionaries(label):
            if not any(k in p.lower() for k in ['basic_annotation', 'science_annotation', 'metadata']):
                continue
            for wk, hk in [('width', 'height'), ('imageWidth', 'imageHeight'), ('image_width', 'image_height')]:
                if wk in obj and hk in obj and isinstance(obj[wk], (int, float)) and isinstance(obj[hk], (int, float)):
                    declared.append((obj[wk], obj[hk]))
            match = re.match(r'^\s*(\d+)\s*[xX×]\s*(\d+)', str(obj.get('imgSize', '')))
            if match:
                declared.append((int(match[1]), int(match[2])))
        result['declaration_conflict'] = ('fail' if any(pair != (width, height) for pair in declared) else 'pass') if width is not None and declared else 'not_applicable'
    if source['family'] == 'ULF_sample':
        times = []
        if existence:
            with data_path.open(newline='', encoding='utf-8-sig') as f:
                rows = csv.reader(f)
                next(rows)
                for row in rows:
                    if row and row[0].strip():
                        times.append(float(row[0]))
            assert len(times) == task['facts']['n_rows']
        mismatches = []
        if len(times) > 1:
            step = statistics.median([times[i+1]-times[i] for i in range(len(times)-1)])
            rate = 1/step if step > 0 else None
            if rate is not None:
                assert math.isclose(rate, task['facts']['measured_rate_hz'])
                if isinstance(science.get('sampling_rate_hz'), (int, float)):
                    mismatches.append(not math.isclose(science['sampling_rate_hz'], rate, rel_tol=.001, abs_tol=1e-8))
                if isinstance(science.get('n_points'), (int, float)):
                    mismatches.append(science['n_points'] != len(times))
        result['declaration_conflict'] = ('fail' if any(mismatches) else 'pass') if mismatches else 'not_applicable'
        origin = utc(basic.get('startTime'))
        intervals = [(utc(r.get('start_dt', r.get('event_starttime'))), utc(r.get('end_dt', r.get('event_endtime')))) for r in records]
        intervals = [(a, b) for a, b in intervals if a is not None and b is not None]
        known = bool(times) and origin is not None and bool(intervals)
        bad = any(a > b or a < origin + times[0] or b > origin + times[-1] for a, b in intervals) if known else False
        result['record_domain_violation'] = ('fail' if bad else 'pass') if known else 'not_applicable'
    else:
        points = []
        for p, obj in dictionaries(label):
            for key in ['bbox_polygon_pixel', 'points']:
                if key == 'points' and not ('mask/shapes/' in p or 'specialShapes/' in p):
                    continue
                value = obj.get(key)
                if isinstance(value, list) and value and all(isinstance(v, list) and len(v) >= 2 and all(isinstance(a, (int, float)) for a in v[:2]) for v in value):
                    points.extend(v[:2] for v in value)
        known = width is not None and bool(points)
        bad = any(x < 0 or y < 0 or x >= width or y >= height for x, y in points) if known else False
        result['record_domain_violation'] = ('fail' if bad else 'pass') if known else 'not_applicable'
    for key, expected in task['expected'].items():
        assert result[key] == expected, (task['task_id'], key, result[key], expected)
    return {'task_id': task['task_id'], 'source_hashes_verified': True,
            'independently_recomputed': {k: result[k] for k in task['expected']}}

def main():
    out = Path(sys.argv[1])
    tasks = json.loads((out/'tasks_private.json').read_text())
    assert len(tasks) == len({t['source']['training_group_id'] for t in tasks}) == 36
    checks = [independently_recompute(t) for t in tasks]
    result = {'status': 'pass', 'source_tasks': 36, 'independent_recomputed_slots': sum(len(t['expected']) for t in tasks),
              'method': 'Separate raw-JSON traversal, Pillow image decoding, CSV re-read and timestamp arithmetic; no model output used',
              'limitation': 'Independent implementation of mechanical facts, not independent scientific defect annotations',
              'checks': checks}
    (out/'independent_source_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'checks'}))

if __name__ == '__main__':
    main()
