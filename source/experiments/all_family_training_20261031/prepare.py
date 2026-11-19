"""Freeze grouped splits, conservative targets and current-only training inputs."""
import collections
import concurrent.futures
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

from common import *

sys.path.insert(0,str(REPO/'experiments/full_matrix_20261031'))
reference=module('allfamily_reference_common',REPO/'experiments/full_matrix_20261031/common.py')
ann=reference.ann


class Groups:
    def __init__(self):self.parent={}
    def root(self,a):
        self.parent.setdefault(a,a)
        if self.parent[a]!=a:self.parent[a]=self.root(self.parent[a])
        return self.parent[a]
    def join(self,a,b):
        a,b=self.root(a),self.root(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def split_groups(rows):
    uf=Groups();first={};bases={}
    for r in rows:
        b=r['base_sample_id'];uf.root(b);bases[b]=r['family']
        for key in [('manifest',r['split_group_id']),('raw',r.get('data_sha256'))]:
            if key[1] is None:continue
            if key in first:uf.join(b,first[key])
            else:first[key]=b
    answer={};members=collections.defaultdict(list)
    for b in sorted(bases):members[uf.root(b)].append(b)
    for values in members.values():
        group=digest(sorted(values));u=int(digest([20261031,group])[:16],16)/16**16
        split='train' if u<.7 else 'val' if u<.85 else 'test'
        for b in values:answer[b]=(split,group)
    return answer


def stats_for(row):
    p=SOURCE/'raw_views'/(str(row.get('data_sha256'))+'.json')
    return json.loads(p.read_text(),parse_constant=lambda _:None) if p.exists() else {}


def ulf_refine(row,clean,current,stats):
    y=list(row['target']);v=list(row['truth_valid']);notes=list(row['truth_notes'])
    if row['tier']=='P0':return y,v,notes
    # An aggregate positive remains known if one component is confirmed.
    a=clean.get('science_annotation',{});b=current.get('science_annotation',{})
    confirmed=False;uncertain=False;matching=[]
    expected={'n_points':stats.get('rows'),'components':stats.get('components')}
    dt=stats.get('median_dt')
    if dt and stats.get('last_t') is not None:
        expected.update(sampling_rate_hz=1/dt,duration_s=stats['last_t']-stats['first_t']+dt)
    for field in ['n_points','components','sampling_rate_hz','duration_s','units']:
        before=a.get(field);after=b.get(field);changed=before!=after
        if field=='units':
            confirmed|=changed
            continue
        actual=expected.get(field)
        if actual is None:
            if changed:uncertain=True
            continue
        same=(after==actual) if not isinstance(actual,(float,int)) else isinstance(after,(float,int)) and abs(after-actual)<1e-5*max(1,abs(actual))
        if changed and not same:confirmed=True
        elif changed:matching.append(field)
        elif not same:uncertain=True
    aa=clean.get('basic_annotation',{});bb=current.get('basic_annotation',{})
    def stamp(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    for field,key in [('startTime','first_t'),('endTime','last_t')]:
        if aa.get(field)!=bb.get(field):
            try:
                residual=abs(stamp(bb[field])-stamp(b['dateObs'])-stats[key])
                if residual>1e-5:confirmed=True
                else:matching.append(field)
            except (KeyError,TypeError,ValueError):uncertain=True
    ea=clean['ulf_event_annotation'];eb=current['ulf_event_annotation']
    if len(ea['matched_events'])==len(eb['matched_events']):
        for x,z in zip(ea['matched_events'],eb['matched_events']):
            if any(x.get(k)!=z.get(k) for k in ['kind','start_dt','end_dt']):
                # This is an annotation-reference proxy, not physical event validation.
                if row['data_sha256'] and a.get('dateObs')==b.get('dateObs'):confirmed=True
                else:uncertain=True
    else:uncertain=True
    if ea.get('event_types')!=eb.get('event_types'):
        if set(eb.get('event_types',[]))!={e.get('kind') for e in eb['matched_events']}:confirmed=True
        else:uncertain=True
    y[2]=int(confirmed);v[2]=bool(confirmed or not uncertain)
    if matching:notes.append('changed_but_matches_current_data:'+','.join(matching))
    if uncertain:notes.append('ULF_data_label_attribution_or_event_identity_unverified')
    return y,v,notes


def current_input(row,label,stats):
    extracted=ann.extract_annotation(label,row['family'])
    objects=extracted['objects'];science=label.get('science_annotation',{});basic=label.get('basic_annotation',{})
    # Exact whitelists deliberately exclude identifiers, paths and tier/version fields.
    paramkeys=['kind','label','evenType','event_type','eventtype','concept','start_dt','end_dt','cme_time',
        'central_pa','width','mpa','linear_speed','speed_2nd_initial','speed_2nd_final','speed_2nd_20R',
        'acceleration','mass','kinetic_energy']
    geometry=[]
    for o in objects:
        g=o['geometry'];entry={k:g.get(k) for k in ['kind','valid','parameters'] if k in g}
        p=g.get('points')
        if p and g.get('valid'):
            p=np.asarray(p,float);entry.update(points_count=len(p),bounds=[*p.min(0).tolist(),*p.max(0).tolist()],centroid=p.mean(0).tolist())
        geometry.append(entry)
    summary=dict(family=row['family'],decoded={k:v for k,v in stats.items() if k!='verified_source_sha256'},
        dimensions=extracted['declared_dimensions']['sources'],
        declared={k:science.get(k) for k in ['dateObs','sampling_rate_hz','duration_s','n_points','components','units'] if k in science},
        window={k:basic.get(k) for k in ['startTime','endTime'] if k in basic},
        events=[{k:o['raw'][k] for k in paramkeys if k in o['raw']} for o in objects],
        geometry=geometry,
        event_summary={key:{k:label[key].get(k) for k in ['event_types','has_events','event_count'] if k in label[key]}
                       for key in ['hek_event_annotation','ulf_event_annotation'] if key in label})
    f=np.zeros(FEATURES,np.float32);f[:9]=row['features']
    f[9+FAMILIES.index(row['family'])]=1
    def put(key,value):
        h=digest(key);idx=16+int(h[:8],16)%(FEATURES-16)
        sign=1 if int(h[8],16)%2 else -1
        f[idx]+=sign*value
    def walk(x,path=''):
        if isinstance(x,dict):
            for k,v in sorted(x.items()):
                if k in ['pointer','state','warnings']:continue
                walk(v,path+'/'+k)
        elif isinstance(x,list):
            put(path+'/length',math.log1p(len(x)))
            for v in x:walk(v,path+'/*')
        elif isinstance(x,(float,int)) and math.isfinite(float(x)):
            put(path,math.copysign(math.log1p(abs(x)),x))
        elif isinstance(x,str):
            # Time values are current parameters; no source identifiers are present.
            put(path+'='+x,1)
    walk(summary)
    f=np.clip(f,-100,100)
    prompt=('Evaluate only this current observation and its current annotation. Left panel: raw observation; right: current annotation. '
        'ULF is a waveform/time-window view; CME regions are angular declarations. '
        'Find G=boundary/position/coverage defect, C=missing/spurious event count, D=class/dimension/time/parameter contradiction. '
        'Do not assume every record has a defect. Return exactly one code: A=none, B=G, C=C, D=G+C, '
        'E=D, F=G+D, G=C+D, H=G+C+D. Current fields: '+json.dumps(summary,separators=(',',':')))
    return f.tolist(),prompt


def region_mask(label,row,stats,matrix=None):
    # Display coordinates follow the same current raw panel placement as SOURCE views.
    rawp=SOURCE/'raw_views'/(str(row.get('data_sha256'))+'.png')
    if not rawp.exists():return None
    with Image.open(rawp) as im:rw,rh=im.size
    w=stats.get('width',384);h=stats.get('height',384)
    result=Image.new('L',(384,384));draw=ImageDraw.Draw(result)
    ox=(384-rw)//2;oy=(384-rh)//2
    objects=ann.extract_annotation(label,row['family'])['objects'];available=True
    for obj in objects:
        g=obj['geometry'];points=g.get('points')
        if points and g.get('valid'):
            p=np.array(points,float)
            if matrix is not None:p=np.c_[p,np.ones(len(p))]@np.asarray(matrix)
            p=p/[w,h]*[rw,rh]+[ox,oy]
            if g['kind']=='rectangle' and len(p)==2:
                lo=p.min(0);hi=p.max(0);draw.rectangle([*lo,*hi],fill=1)
            elif len(p)>=3:draw.polygon([tuple(v) for v in p],fill=1)
        elif row['family']=='CME_C2_sample':
            params=g.get('parameters',{});angle=params.get('central_pa');width=params.get('width')
            if isinstance(angle,(float,int)) and isinstance(width,(float,int)):
                cx=ox+rw/2;cy=oy+rh/2;radius=min(rw,rh)/2
                draw.pieslice([cx-radius,cy-radius,cx+radius,cy+radius],270-angle-width/2,270-angle+width/2,fill=1)
            else:available=False
        elif row['family']=='ULF_sample':
            try:
                def t(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
                origin=t(label['science_annotation']['dateObs'])
                a=t(obj['raw']['start_dt'])-origin;b=t(obj['raw']['end_dt'])-origin
                lo=stats['first_t'];span=max(stats['last_t']-lo,1)
                x0=ox+15+350*(a-lo)/span;x1=ox+15+350*(b-lo)/span
                draw.rectangle([min(x0,x1),oy+10,max(x0,x1),oy+355],fill=1)
            except (TypeError,KeyError,ValueError):available=False
        else:available=False
    if not available:return None
    panel=Image.new('L',(768,384));panel.paste(result,(0,0));panel.paste(result,(384,0))
    return panel


def process(task):
    row,p0,split,group=task;path=OUT/'rows'/(digest(row['id'])+'.json')
    if path.exists():
        cached=json.loads(path.read_text())
        if cached.get('all_family_preparation_version')==2:return cached
    label=json.loads(Path(row['label_path']).read_text(encoding='utf-8-sig'))
    clean=json.loads(Path(p0['label_path']).read_text(encoding='utf-8-sig'));stats=stats_for(row)
    assert sha(row['label_path'])==row['label_sha256']
    assert sha(p0['label_path'])==p0['label_sha256']
    y=list(row['target']);valid=list(row['truth_valid']);notes=list(row['truth_notes']);matrix=None
    changed=row['data_sha256']!=p0['data_sha256']
    if row['family'] in FAMILIES[:2] and changed:
        regpath=SOURCE/'registration'/(row['id'].replace(':','_')+'.json')
        reg=json.loads(regpath.read_text()) if regpath.exists() else {}
        expected=[p0['data_sha256'],row['data_sha256'],p0['label_sha256'],row['label_sha256']]
        trusted=reg.get('source_hashes_v2')==expected and reg.get('valid')
        valid[0]=bool(trusted)
        if trusted:
            y[0]=reg['geometry_target'];matrix=reg.get('registration',{}).get('matrix_3x2')
            notes.append('hash_verified_pixel_registration_v2')
        else:notes.append('geometry_unverified_after_full_hash_check')
    if row['family']=='ULF_sample':y,valid,notes=ulf_refine(dict(row,target=y,truth_valid=valid,truth_notes=notes),clean,label,stats)
    features,prompt=current_input(row,label,stats)
    mask=None
    if not row['input_missing'] and (not changed or row['family'] not in FAMILIES[:2] or valid[0]):
        mask=region_mask(clean,row,stats,matrix)
    maskpath=OUT/'masks'/(digest(row['id'])+'.png')
    if mask is not None:mask.save(maskpath)
    if not row['input_missing']:
        imagepath=OUT/'images'/(digest(row['id'])+'.png')
        if not imagepath.exists():
            with Image.open(row['image']) as im:im.resize((SIZE[1],SIZE[0]),Image.Resampling.BILINEAR).save(imagepath)
    else:imagepath=None
    r=dict(row,split=split,training_group_id=group,target=y,truth_valid=valid,truth_notes=sorted(set(notes)),
        features=features,prompt=prompt,training_image=str(imagepath) if imagepath else None,
        aux_mask=str(maskpath) if mask is not None else None,aux_mask_valid=mask is not None,
        input_profile='all-family-current-only-v1',all_family_preparation_version=2)
    r['input_sha256']=digest([sha(row['image']),prompt,features])
    write(path,r);return r


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    for name in ['rows','masks','images','logs','checkpoints','training']: (OUT/name).mkdir(exist_ok=True)
    while True:
        ps=json.loads((SOURCE/'prepare_state.json').read_text())
        rp=SOURCE/'registration_state.json';rs=json.loads(rp.read_text()) if rp.exists() else {}
        write(OUT/'prepare_state.json',dict(status='waiting_source_and_registration',source=ps,registration=rs,pid=os.getpid()))
        if ps['status']=='complete' and rs.get('status')=='complete':break
        if time.monotonic()-start>8*3600:raise RuntimeError('Source preparation/registration timeout')
        time.sleep(20)
    rows=lines(SOURCE/'dataset.jsonl');assert len(rows)==123565
    assignment=split_groups(rows);refs={r['base_sample_id']:r for r in rows if r['tier']=='P0'}
    assert len(refs)==24713
    tasks=[(r,refs[r['base_sample_id']],*assignment[r['base_sample_id']]) for r in rows]
    result=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as pool:
        for r in pool.map(process,tasks,chunksize=20):
            result.append(r)
            if len(result)%1000==0:
                emit('allfamily_prepare',records=len(result),total=len(rows))
                write(OUT/'prepare_state.json',dict(status='preparing_inputs',records=len(result),total=len(rows),pid=os.getpid()))
    assert_coverage(result)
    for split in ['train','val','test']:
        rr=[r for r in result if r['split']==split];assert_coverage(rr);save_lines(OUT/(split+'.jsonl'),rr)
    save_lines(OUT/'inventory.jsonl',result)
    train=[r for r in result if r['split']=='train' and not r['input_missing'] and any(r['truth_valid'])]
    validation=[r for r in result if r['split']=='val' and not r['input_missing'] and any(r['truth_valid'])]
    assert_coverage(train);assert_coverage(validation)
    feature=np.array([r['features'] for r in train],np.float64)
    summary=dict(status='complete',records=len(result),bases=len(refs),packages=30,
        split_counts={s:sum(r['split']==s for r in result) for s in ['train','val','test']},
        train_coverage=assert_coverage(train),validation_coverage=assert_coverage(validation),
        missing_input=sum(r['input_missing'] for r in result),
        groups=len({r['training_group_id'] for r in result}),
        split_hashes={s:sha(OUT/(s+'.jsonl')) for s in ['train','val','test']},
        mean=feature.mean(0).tolist(),std=np.maximum(feature.std(0),.01).tolist(),
        seconds=time.monotonic()-start,source_manifest_hash=sha(MANIFEST))
    write(OUT/'preparation.json',summary);write(OUT/'prepare_state.json',summary)
    emit('allfamily_prepared',records=len(result),splits=summary['split_counts'])


if __name__=='__main__':main()
