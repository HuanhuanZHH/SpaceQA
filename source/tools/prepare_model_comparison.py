"""Frozen event-day grouped pilot; source files are never modified."""
import copy
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/staff/tangjiarui/OPC')
OUT = ROOT / 'opc-agent/runs/model-comparison-20261031-v2'
PROMPT = ('Review a space-science annotation. The left panel is the raw observation; '
          'the right panel overlays the current annotation in red. No clean reference is supplied. '
          'Choose the main defect: A=none supported; B=position/boundary/coverage error; '
          'C=missing or spurious object annotation; D=wrong category, parameter, or declared image dimensions. '
          'EPB polygons follow plasma bubble dark structures; DMSP boxes mark auroral classes. '
          'unclear is a valid DMSP class. Return exactly one letter A,B,C,D. ')
CLASSES = ['NONE', 'GEOMETRY', 'COUNT', 'LABEL_OR_METADATA']


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def shapes(label):
    sci = label.get('science_annotation', {})
    if isinstance(sci.get('specialShapes'), list):
        return sci['specialShapes']
    mask = sci.get('mask', {})
    if isinstance(mask, dict) and 'shapes' in mask:
        return mask['shapes']
    for value in sci.values():
        if isinstance(value, dict) and isinstance(value.get('shapes'), list):
            return value['shapes']
    return []


def make_view(raw_path, annotation, destination):
    with Image.open(raw_path) as im:
        raw = im.convert('RGB')
    width, height = raw.size
    raw.thumbnail((384, 384), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (768, 384), (24, 24, 24))
    canvas.paste(raw, ((384-raw.width)//2, (384-raw.height)//2))
    overlay = raw.copy()
    draw = ImageDraw.Draw(overlay)
    for shape in shapes(annotation):
        pts = [(float(x)*raw.width/width, float(y)*raw.height/height) for x, y in shape.get('points', [])]
        if len(pts) < 2:
            continue
        if shape.get('shape_type') == 'rectangle' and len(pts) == 2:
            x = sorted([pts[0][0], pts[1][0]])
            y = sorted([pts[0][1], pts[1][1]])
            draw.rectangle((x[0], y[0], x[1], y[1]), outline='red', width=2)
        else:
            draw.line(pts + [pts[0]], fill='red', width=2)
    canvas.paste(overlay, (384+(384-overlay.width)//2, (384-overlay.height)//2))
    canvas.save(destination, quality=92)


def features(label, family):
    ss = shapes(label)
    pts = [p for s in ss for p in s.get('points', [])]
    meta = label.get('science_annotation', {}).get('mask', {})
    basic = label.get('basic_annotation', {})
    # Verified constant PNG sizes for these two pilot families, independently decoded in phase1.
    w,h=(390.0,350.0) if family=='DMSP_SSUSI' else (1024.0,1024.0)
    oob = sum(x < 0 or x > w or y < 0 or y > h for x, y in pts)
    invalid = sum(str(s.get('label', '')).startswith('__') or str(s.get('label', '')).lower() in
                  ('invalid', 'wrong', 'unknown_class') for s in ss)
    areas = []
    for s in ss:
        ps = np.asarray(s.get('points', []), dtype=float)
        if len(ps) > 1:
            areas.append(float(np.prod(np.maximum(ps.max(0)-ps.min(0), 0)) / (w*h)))
    dims=re.search(r'(\d+)\s*[x×]\s*(\d+)',str(basic.get('imgSize','')))
    dim_error=float(bool(dims and (int(dims[1])!=int(w) or int(dims[2])!=int(h))))
    for field,value in [('imageWidth',w),('imageHeight',h)]:
        if field in basic and int(basic[field])!=int(value):dim_error=1.0
        if field in meta and int(meta[field])!=int(value):dim_error=1.0
    return [float(family == 'DMSP_SSUSI'), min(len(ss)/20, 2), invalid/max(len(ss), 1),
            oob/max(len(pts), 1), sum(areas), max(areas, default=0), min(len(pts)/1000, 2),
            float(not ss),dim_error]


def annotation_text(annotation,family):
    basic=annotation.get('basic_annotation',{})
    mask=annotation.get('science_annotation',{}).get('mask',{})
    return (f'Family={family}; labels='+json.dumps([s.get('label') for s in shapes(annotation)])+
            '; declared imgSize='+str(basic.get('imgSize','not specified'))+
            '; annotation image dimensions='+str([mask.get('imageWidth',basic.get('imageWidth')),
                                                   mask.get('imageHeight',basic.get('imageHeight'))])+
            '; decoded original PNG dimensions='+str([390,350] if family=='DMSP_SSUSI' else [1024,1024])+'.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'images').mkdir(exist_ok=True)
    rng = random.Random(20261031)
    sources = [('EPB_seg_sample', 'epb_p0_p1_audit_20261027'),
               ('DMSP_SSUSI', 'dmsp_p0_p1_audit_20261029')]
    all_rows, skipped = [], Counter()
    for family, subdir in sources:
        for line in (ROOT/'phase2_visual_audit_xzj'/subdir/'sample_comparisons.jsonl').open():
            row = json.loads(line)
            codes = {x['code'] for x in row.get('findings', [])}
            if 'L0-01' in codes or row.get('pending'):
                skipped['missing_or_pending'] += 1
                continue
            groups = {1 if x == 'L2-G' else 2 if x in ('L3-06', 'L3-07') else
                      3 if x in ('L2-P','L2-F') else -1 for x in codes}
            if -1 in groups or len(groups) > 1:
                skipped['mixed_or_unsupported'] += 1
                continue
            gold = next(iter(groups), 0)
            p0 = json.load(open(row['paths']['P0']['label']))
            p1 = json.load(open(row['paths']['P1']['label']))
            # Calendar day groups adjacent frames. Duplicate raw content is unioned below.
            basic = p0.get('basic_annotation', {})
            timestamp = basic.get('timestamp') or basic.get('startTime', '')
            if not timestamp:
                timestamp = p0.get('science_annotation', {}).get('dateObs', '')
            day = str(timestamp)[:10]
            if len(day) != 10:
                skipped['no_observation_date'] += 1
                continue
            all_rows.append(dict(source=row, family=family, p0=p0, p1=p1, gold=gold,
                                 day_group=family+':'+day))
    # Join known split_group_id duplicates and repeated raw hashes with event-day groups.
    manifest = ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    split_ids = {}
    for line in manifest.open():
        x = json.loads(line)
        if x['dataset_family'] in dict(sources):
            split_ids[x['base_sample_id']] = x['split_group_id']
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a,b)] = min(a,b)
    for x in all_rows:
        row = x['source']
        union(x['day_group'], split_ids[row['base_sample_id']])
        union(x['day_group'], 'sha:'+row['hashes']['P0']['data'])
    day_roots = sorted({find(x['day_group']) for x in all_rows})
    rng.shuffle(day_roots)
    assignment = {g: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train')
                  for i, g in enumerate(day_roots)}
    records = []
    for x in all_rows:
        row = x['source']
        split = assignment[find(x['day_group'])]
        sid = row['base_sample_id']
        uid = hashlib.sha256(sid.encode()).hexdigest()[:20]
        image = OUT/'images'/f'{uid}_real.jpg'
        make_view(row['paths']['P1']['data'], x['p1'], image)
        feat = features(x['p1'], x['family'])
        record = dict(id=uid+'_real', base_sample_id=sid, split_group=find(x['day_group']),
                      split=split, family=x['family'], source='public_P0_assisted_real_P1',
                      image=str(image), target=x['gold'], codes=sorted({f['code'] for f in row.get('findings', [])}),
                      features=feat, prompt=PROMPT+annotation_text(x['p1'],x['family']),
                      rule_prediction=3 if feat[2] or feat[8] else 1 if feat[3] else 0)
        records.append(record)
        # Synthetic train-only augmentation is never mixed into real-P1 test metrics.
        if split != 'train' or not shapes(x['p0']) or rng.random() > 0.25:
            continue
        for target in range(4):
            ann = copy.deepcopy(x['p0'])
            ss = shapes(ann)
            shape = ss[0]
            if target == 1:
                shift = rng.choice([-1, 1])*rng.uniform(40, 160)
                shape['points'] = [[a+shift, b] for a, b in shape['points']]
            elif target == 2:
                if rng.random() < .5:
                    ss.pop(0)
                else:
                    extra = copy.deepcopy(shape)
                    extra['points'] = [[a+180, b-100] for a,b in extra['points']]
                    ss.append(extra)
            elif target == 3:
                shape['label'] = '__OOV__'
            im = OUT/'images'/f'{uid}_synth{target}.jpg'
            make_view(row['paths']['P0']['data'], ann, im)
            f = features(ann, x['family'])
            records.append(dict(record, id=uid+f'_synth{target}', image=str(im), target=target,
                                source='synthetic_train_only', features=f,
                                prompt=PROMPT+annotation_text(ann,x['family']),
                                rule_prediction=3 if f[2] or f[8] else 1 if f[3] else 0))
    assert not ({r['split_group'] for r in records if r['split']=='train'} &
                {r['split_group'] for r in records if r['split']!='train'})
    with (OUT/'dataset.jsonl').open('w') as f:
        for r in records:
            f.write(json.dumps(r)+'\n')
    summary = dict(classes=CLASSES, records=len(records), observation_day_groups=len(day_roots),
                   split_counts=dict(Counter(r['split'] for r in records)),
                   cells=dict(Counter(f"{r['split']}:{r['source']}:{CLASSES[r['target']]}" for r in records)),
                   skipped=dict(skipped), seed=20261031,
                   limitations=['P0-assisted technical labels, not official hidden scientific truth',
                                'Observation calendar day proxy, not full scientific event IDs',
                                'EPB/DMSP pilot only; no localization or official score conclusion'])
    dump(OUT/'dataset_summary.json', summary)
    print(json.dumps(summary))

if __name__ == '__main__':
    main()
