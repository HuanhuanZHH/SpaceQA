"""Full 30-package inventory, independent technical targets and current-only views."""
import collections
import concurrent.futures
import copy
import hashlib
import json
import io
import math
import os
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from common import ROOT, REPO, OUT, OLD, MANIFEST, AUDIT, FAMILIES, TIERS, oldprep, ann, rules, digest, dump, lines, emit

META = {'imgSize', 'imageWidth', 'imageHeight', 'sampling_rate_hz', 'duration_s', 'n_points', 'components', 'units',
        'startTime', 'endTime', 'cme_time', 'event_starttime', 'event_endtime', 'start_dt', 'end_dt',
        'label', 'event_type', 'eventtype', 'concept', 'event_types', 'kind',
        'central_pa', 'width', 'mpa', 'linear_speed', 'speed_2nd_initial', 'speed_2nd_final', 'speed_2nd_20R',
        'acceleration', 'mass', 'kinetic_energy'}
COLLECTIONS = {'matched_events', 'shapes', 'specialShapes'}
REGISTRATIONS = {}
PRIOR = {}


def numeric_points(obj):
    points = obj['geometry'].get('points')
    if not obj['geometry'].get('valid') or not points:
        return None
    p = np.asarray(points, float)
    if len(p) > 2 and np.allclose(p[0], p[-1], atol=1e-10, rtol=0):
        p = p[:-1]
    return p


def metadata(value):
    result = {}
    def walk(v, prefix=''):
        if isinstance(v, dict):
            for key, x in v.items():
                path = prefix+'/'+key
                if key in COLLECTIONS:
                    continue
                if key in META:
                    result[path] = x
                elif isinstance(x, dict):
                    walk(x, path)
    walk(value)
    return result


def object_pairs(a, b, family):
    if family in ['AIA_AR_sample', 'HMI_AR_sample']:
        # Stable IDs can repeat; exact occurrences first, then within-ID order.
        ids = ['kb_archivid', 'SOL_standard', 'solar_object_locator']
        left = collections.defaultdict(list); right = collections.defaultdict(list)
        for i, x in enumerate(a): left[tuple(str(x['raw'].get(k)) for k in ids)].append(i)
        for j, x in enumerate(b): right[tuple(str(x['raw'].get(k)) for k in ids)].append(j)
        pairs = []; removed = []; added = []
        for key in left.keys() | right.keys():
            li = list(left[key]); ri = list(right[key])
            for i in li[:]:
                j = next((j for j in ri if a[i]['raw'] == b[j]['raw']), None)
                if j is not None: pairs.append((i, j)); li.remove(i); ri.remove(j)
            pairs.extend(zip(li, ri)); removed.extend(li[len(ri):]); added.extend(ri[len(li):])
        return pairs, removed, added, False
    if family in ['EPB_seg_sample', 'DMSP_SSUSI']:
        if not a or not b:
            return [], list(range(len(a))), list(range(len(b))), False
        def box(x):
            p = numeric_points(x)
            return np.r_[p.min(0), p.max(0)] if p is not None else np.zeros(4)
        aa = np.array([box(x) for x in a]); bb = np.array([box(x) for x in b])
        lo = np.maximum(aa[:, None, :2], bb[None, :, :2]); hi = np.minimum(aa[:, None, 2:], bb[None, :, 2:])
        inter = np.maximum(hi-lo, 0).prod(2)
        iou = inter/np.maximum((aa[:, 2:]-aa[:, :2]).prod(1)[:, None]+(bb[:, 2:]-bb[:, :2]).prod(1)[None, :]-inter, 1e-8)
        ii, jj = linear_sum_assignment(1-iou)
        return list(zip(ii.tolist(), jj.tolist())), sorted(set(range(len(a)))-set(ii)), sorted(set(range(len(b)))-set(jj)), False
    if family == 'ULF_sample' and len(a) == len(b):
        return list(zip(range(len(a)), range(len(b)))), [], [], False
    matches, left, right, al, ar, _ = ann._match_objects(a, b, family)
    return [(m['clean_index'], m['variant_index']) for m in matches], list(left), list(right), bool(al or ar)


def target(clean, current, p0, row):
    family = row['dataset_family']; a = ann.extract_annotation(clean, family)['objects']; b = ann.extract_annotation(current, family)['objects']
    pairs, removed, added, ambiguous = object_pairs(a, b, family)
    y = [0, int(bool(len(a) != len(b) or ((removed or added) and not ambiguous))), 0]
    valid = [True, True, True]; notes = []; evidence = []
    if metadata(clean) != metadata(current):
        y[2] = 1
        evidence.append('current_declared_class_time_or_parameter_differs_from_P0')
    for i, j in pairs:
        if metadata(a[i]['raw']) != metadata(b[j]['raw']):
            y[2] = 1
    matrix = np.array([[1., 0.], [0., 1.], [0., 0.]])
    tolerance = .5
    same = bool(p0['data_exists'] and row['data_exists'] and p0['data_sha256'] == row['data_sha256'])
    reg = REGISTRATIONS.get((family, row['tier'], row['source_key']))
    if family in ['AIA_AR_sample', 'HMI_AR_sample'] and not same and (a or b):
        if reg and (reg.get('image_correlation') or 0) >= .98 and reg.get('matrix_3x2'):
            matrix = np.array(reg['matrix_3x2']); tolerance = 3.; notes.append('existing_pixel_validated_registration')
        elif reg and reg.get('image_pixels_equal'):
            same = True
        else:
            valid[0] = False; notes.append('geometry_requires_pixel_validated_registration')
    if family in ['EPB_seg_sample', 'DMSP_SSUSI'] and not same:
        valid[0] = False; notes.append('raw_unavailable_or_changed')
    if ambiguous:
        valid[0] = False; notes.append('object_correspondence_ambiguous')
    if valid[0]:
        for i, j in pairs:
            p = numeric_points(a[i]); q = numeric_points(b[j])
            if p is None or q is None:
                if (p is None) != (q is None): valid[0] = False; notes.append('geometry_availability_changed')
                continue
            expected = np.c_[p, np.ones(len(p))] @ matrix
            if not same and family in ['AIA_AR_sample', 'HMI_AR_sample']:
                expected = np.clip(expected, 0, row['image_width']-1)
            if expected.shape != q.shape:
                y[0] = 1
            else:
                # Point-set assignment ignores cyclic vertex order and winding.
                distances = np.linalg.norm(expected[:, None]-q[None], axis=2)
                ii, jj = linear_sum_assignment(distances)
                if distances[ii, jj].max(initial=0) > tolerance: y[0] = 1
    if row['tier'] == 'P0':
        y = [0, 0, 0]; valid = [True, True, True]; notes = ['reference_baseline_not_independent_truth']
    return dict(target=y, truth_valid=valid, truth_notes=sorted(set(notes)), truth_evidence=evidence,
                object_counts=[len(a), len(b)], truth_kind='P0_derived_technical_proxy',
                geometry_tolerance_native_pixels=tolerance)


def atomic_image(im, destination):
    if destination.exists(): return
    tmp = destination.with_name(destination.name+f'.{os.getpid()}.tmp')
    if destination.suffix=='.png':im.save(tmp,format='PNG')
    else:im.save(tmp, format='JPEG', quality=92)
    tmp.replace(destination)


def raw_panel(row):
    key = row.get('data_sha256') or digest(['missing', row['base_sample_id'], row['tier']])
    destination = OUT/'raw_views'/(key+'.png')
    stats_path = OUT/'raw_views'/(key+'.json')
    if destination.exists() and stats_path.exists():
        previous=json.loads(stats_path.read_text())
        if previous.get('verified_source_sha256')==row.get('data_sha256'):
            return Image.open(destination).convert('RGB'), previous
    stats = {}
    data_bytes=None
    if row['data_exists']:
        data_bytes=Path(row['data_path']).read_bytes()
        observed=hashlib.sha256(data_bytes).hexdigest()
        if observed!=row['data_sha256']:raise RuntimeError('Raw data hash differs from manifest: '+row['data_path'])
    if not row['data_exists']:
        raw = Image.new('RGB', (384, 384), (24, 24, 24)); ImageDraw.Draw(raw).text((20, 180), 'OBSERVATION FILE MISSING', fill='white')
        stats['input_missing'] = True
    elif row['dataset_family'] == 'ULF_sample':
        text=data_bytes.decode('utf-8-sig');header=text.splitlines()[0].strip().split(',')
        data = np.genfromtxt(io.StringIO(text), delimiter=',', skip_header=1, invalid_raise=False)
        data = np.atleast_2d(data)
        raw = Image.new('RGB', (384, 384), (24, 24, 24)); draw = ImageDraw.Draw(raw)
        channels = min(data.shape[1]-1, 4); times = data[:, 0]; dt = np.diff(times); positive = dt[np.isfinite(dt) & (dt > 0)]
        stats = dict(rows=len(data), components=header[1:], first_t=float(times[0]), last_t=float(times[-1]),
                     median_dt=float(np.median(positive)) if len(positive) else None, regular=bool(len(positive)==len(dt) and np.allclose(positive, np.median(positive), atol=1e-5)))
        for k in range(channels):
            values = data[:, k+1]; good = np.isfinite(values) & np.isfinite(times)
            t = times[good]; v = values[good]
            if not len(v): continue
            stride = max(1, len(v)//700); t=t[::stride]; v=v[::stride]
            x = 15+350*(t-t.min())/max(float(t.max()-t.min()), 1e-8)
            y = 30+k*85+65*(1-(v-v.min())/max(float(v.max()-v.min()), 1e-8))
            draw.line(list(zip(x, y)), fill=['cyan','yellow','lime','orange'][k], width=1); draw.text((15, 15+k*85), header[k+1], fill='white')
    else:
        with Image.open(io.BytesIO(data_bytes)) as source:
            raw = source.convert('RGB'); stats = dict(width=raw.width, height=raw.height)
            raw.thumbnail((384, 384), Image.Resampling.LANCZOS)
    stats['verified_source_sha256']=row.get('data_sha256')
    atomic_image(raw, destination)
    tmp = stats_path.with_name(stats_path.name+f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(stats)); tmp.replace(stats_path)
    return raw, stats


def feature_view(label, row):
    family=row['dataset_family']; extracted=ann.extract_annotation(label, family); objects=extracted['objects']
    raw, stats = raw_panel(row); w=row.get('image_width') or 384; h=row.get('image_height') or 384
    canvas=Image.new('RGB',(768,384),(24,24,24));canvas.paste(raw,((384-raw.width)//2,(384-raw.height)//2))
    overlay=raw.copy();draw=ImageDraw.Draw(overlay); allpoints=[];areas=[]
    for obj in objects:
        p=numeric_points(obj)
        if p is not None:
            allpoints.extend(p.tolist());areas.append(float(np.prod(np.maximum(p.max(0)-p.min(0),0))/(w*h)))
            points=[(float(x)*raw.width/w,float(y)*raw.height/h) for x,y in p]
            if obj['geometry']['kind']=='rectangle' and len(points)==2:
                lo=np.min(points,0);hi=np.max(points,0);draw.rectangle([*lo,*hi],outline='red',width=2)
            elif len(points)>=2:draw.line(points+[points[0]],fill='red',width=2)
    if family=='CME_C2_sample':
        for k,obj in enumerate(objects[:8]):
            r=obj['raw'];draw.text((8,8+15*k),f"PA={r.get('central_pa')} width={r.get('width')}",fill='red')
    if family=='ULF_sample':
        basic=label.get('basic_annotation',{});draw.text((8,365),str(basic.get('startTime',''))[-19:],fill='red')
        # Event windows use current declarations relative to dateObs.
        origin=rules.parse_time(label.get('science_annotation',{}).get('dateObs') or basic.get('startTime'))
        duration=max((stats.get('last_t',1)-stats.get('first_t',0)),1)
        for obj in objects:
            event=obj['raw'];a=rules.parse_time(event.get('start_dt'));b=rules.parse_time(event.get('end_dt'))
            if origin and a and b:
                x0=15+350*(a-origin).total_seconds()/duration;x1=15+350*(b-origin).total_seconds()/duration
                draw.rectangle((min(x0,x1),10,max(x0,x1),355),outline='red',width=2)
    canvas.paste(overlay,(384+(384-overlay.width)//2,(384-overlay.height)//2))
    categories=[o['class_label'] or o['raw'].get('kind') for o in objects]
    expected=rules.EXPECTED_EVENT_TYPES.get(family)
    invalid=sum(bool(expected and x and x not in expected) or str(x).startswith('__') for x in categories)
    if family=='ULF_sample':
        declared=set(label.get('ulf_event_annotation',{}).get('event_types',[]))
        invalid=sum(bool(x and declared and x not in declared) for x in categories)
    oob=sum(x<0 or x>w or y<0 or y>h for x,y in allpoints)
    dimerror=False
    for d in extracted['declared_dimensions']['sources']:
        actual=w if d['axis']=='width' else h
        if d['value']!=actual:dimerror=True
    feats=[float(family=='DMSP_SSUSI'),min(len(objects)/20,2),invalid/max(1,len(objects)),oob/max(1,len(allpoints)),
           sum(areas),max(areas,default=0),min(len(allpoints)/1000,2),float(not objects),float(dimerror)]
    if family in ['EPB_seg_sample','DMSP_SSUSI']:
        feats=oldprep.features(label,family);prompt=oldprep.PROMPT+oldprep.annotation_text(label,family)
    else:
        details=dict(family=family,labels=categories,current_metadata=metadata(label),
                     current_event_parameters=[metadata(o['raw']) for o in objects[:12]],decoded_data=stats)
        prompt=(oldprep.PROMPT+' For ULF the panels show current CSV waveforms and current event windows; temporal and numeric parameter errors belong to D. '
                'For AIA/HMI boxes mark Active Regions; for CME annotations include event times and angular parameters. '
                'Current input only: '+json.dumps(details,separators=(',',':'))[:6500])
    # Exact-input reuse is permitted; no tier/name/P0 truth enters the key.
    key=digest([row.get('data_sha256'),label, 'raw-current-v1'])
    path=OUT/'views'/(key+'.jpg');atomic_image(canvas,path)
    image_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    input_sha=digest([image_sha,prompt]);student_sha=digest([image_sha,feats])
    rule=3 if feats[2] or feats[8] else 1 if feats[3] else 0
    diagnostics=[]
    if family=='ULF_sample' and not stats.get('input_missing'):
        science=label.get('science_annotation',{})
        mismatch=bool(science.get('n_points',stats['rows'])!=stats['rows'] or science.get('components',stats['components'])!=stats['components'])
        if stats.get('regular') and stats.get('median_dt'):
            rate=1/stats['median_dt'];duration=stats['last_t']-stats['first_t']+stats['median_dt']
            mismatch=mismatch or abs(float(science.get('sampling_rate_hz',rate))-rate)>1e-6 or abs(float(science.get('duration_s',duration))-duration)>1e-5
        if mismatch:rule=3;diagnostics.append('current_CSV_declared_parameter_contradiction')
        if science.get('units') not in [None,'nT (H/Z/T), deg (D)']:rule=3
        anchor=rules.parse_time(science.get('dateObs'));basic=label.get('basic_annotation',{})
        if anchor:
            for field,offset in [('startTime',stats['first_t']),('endTime',stats['last_t'])]:
                declared=rules.parse_time(basic.get(field))
                if declared and abs((declared-anchor-timedelta(seconds=offset)).total_seconds())>1e-5:
                    rule=3;diagnostics.append('current_CSV_time_anchor_contradiction')
        eventtypes=set(label.get('ulf_event_annotation',{}).get('event_types',[]))
        if any(o['raw'].get('kind') not in eventtypes for o in objects):rule=3
    return dict(features=feats,prompt=prompt,image=str(path),input_sha256=input_sha,student_input_sha256=student_sha,
                rule_prediction=rule,rule_diagnostics=diagnostics,input_missing=not row['data_exists'])


def init_worker(registrations, prior):
    global REGISTRATIONS, PRIOR
    REGISTRATIONS=registrations;PRIOR=prior


def work(task):
    sid, tiers=task;path=OUT/'shards'/(sid.split(':')[-1]+'.json')
    if path.exists():
        cached=json.loads(path.read_text())
        if all(r.get('preparation_version')==2 for r in cached):return cached
    p0=tiers['P0'];clean=json.loads(Path(p0['label_path']).read_text(encoding='utf-8-sig'));result=[]
    for tier in TIERS:
        row=tiers[tier];label_bytes=Path(row['label_path']).read_bytes()
        label=json.loads(label_bytes.decode('utf-8-sig'))
        if hashlib.sha256(label_bytes).hexdigest()!=row['label_sha256']:raise RuntimeError('Manifest label hash changed: '+row['label_path'])
        if Path(row['data_path']).exists()!=row['data_exists']:raise RuntimeError('Manifest existence changed: '+row['data_path'])
        r=dict(id=sid+':'+tier,base_sample_id=sid,family=row['dataset_family'],tier=tier,package_id=row['package_id'],
            data_path=row['data_path'],label_path=row['label_path'],source_key=row['source_key'],
            data_sha256=row['data_sha256'],label_sha256=row['label_sha256'],data_exists=row['data_exists'],
            prior_split=PRIOR.get(sid,{}).get('split','unseen_family_or_excluded_base'),split_group_id=row['split_group_id'])
        r.update(target(clean,label,p0,row));r.update(feature_view(label,row));r['preparation_version']=2;result.append(r)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(path)
    return result


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    for name in ['raw_views','views','shards']: (OUT/name).mkdir(exist_ok=True)
    groups=collections.defaultdict(dict);counts=collections.Counter()
    for r in lines(MANIFEST):
        r['data_path']=r.get('data_path') or r['expected_data_path']
        groups[r['base_sample_id']][r['tier']]=r;counts[(r['dataset_family'],r['tier'],r['package_id'])]+=1
    assert len(groups)==24713 and sum(counts.values())==123565 and len(counts)==30
    inventory=[dict(family=f,tier=t,package_id=p,records=n) for (f,t,p),n in counts.items()]
    dump('inventory.json',dict(packages=inventory,bases=len(groups),records=sum(counts.values()),manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest()))
    prior={r['base_sample_id']:r for r in lines(OLD/'dataset.jsonl') if r['source']=='public_P0_assisted_real_P1'}
    regs={}
    for family,tier,dirname in [('AIA_AR_sample','P1','aia_p0_p1_audit_20261026'),('AIA_AR_sample','P2','aia_p0_p2_audit_20261027'),('HMI_AR_sample','P1','hmi_p0_p1_audit_20261029')]:
        p=AUDIT/dirname/'image_registration.jsonl'
        if p.exists():
            for r in lines(p):regs[(family,tier,r['filename'])]=r
    dump('prepare_state.json',dict(status='running',bases=len(groups),records=123565,workers=12,pid=os.getpid()))
    # Group tasks keep P0 and all five variants together, and support safe resume.
    done=0
    with (OUT/'dataset.jsonl.tmp').open('w') as f:
        with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=init_worker,initargs=(regs,prior)) as pool:
            for items in pool.map(work,sorted(groups.items()),chunksize=8):
                for r in items:f.write(json.dumps(r)+'\n')
                done+=1
                if done%100==0:
                    emit('prepare',bases_done=done,bases_total=len(groups),seconds=time.monotonic()-start)
                    dump('prepare_state.json',dict(status='running',bases_done=done,bases_total=len(groups),records_done=done*5,pid=os.getpid()))
    (OUT/'dataset.jsonl.tmp').replace(OUT/'dataset.jsonl')
    dump('prepare_state.json',dict(status='complete',bases=len(groups),records=123565,seconds=time.monotonic()-start,pid=os.getpid()))
    emit('prepared',records=123565,seconds=time.monotonic()-start)


if __name__=='__main__':main()
