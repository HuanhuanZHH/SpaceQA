"""Conservative, masked eight-class P0-derived TECHNICAL supervision.

This module never supplies reference differences to inference features. Ambiguous
identity and changed observation pixels mask visual-reference comparisons. A
change relative to P0 is weak supervision, not an independent scientific verdict.
"""
from __future__ import annotations
import collections, datetime as dt, hashlib, json, math
from pathlib import Path

CLASSES=('L0-01','L2-G','L2-P','L2-T','L2-F','L3-06','L3-07','L3-MIS')
NAMES=('missing_data','boundary_position','class_error','event_time','dimensions','missing_annotation','spurious_annotation','coverage')
PIXEL_FAMILIES={'AIA_AR_sample','HMI_AR_sample','EPB_seg_sample','DMSP_SSUSI'}


def area(points):
    if not isinstance(points,list) or len(points)<2:return None
    try:
        pts=[(float(a),float(b)) for a,b in points]
        if any(not math.isfinite(v) for p in pts for v in p):return None
        if len(pts)==2:return abs((pts[1][0]-pts[0][0])*(pts[1][1]-pts[0][1]))
        return abs(sum(a*d-c*b for (a,b),(c,d) in zip(pts,pts[1:]+pts[:1])))/2
    except (ValueError,TypeError):return None


def canonical_polygon(points):
    pts=[tuple(p) for p in points]
    if pts and pts[0]==pts[-1]:pts.pop()
    if not pts:return ()
    # Polygon vertex starting index / winding are representation differences.
    options=[]
    for p in [pts,list(reversed(pts))]:
        k=min(range(len(p)),key=lambda i:p[i]);options.append(tuple(p[k:]+p[:k]))
    return min(options)


def geometry_difference(before,after):
    a,b=area(before),area(after)
    if a is None or b is None or a<=0 or b<=0:return None
    if canonical_polygon(before)==canonical_polygon(after):return dict(geometry=False,coverage=False,area_relative_error=0.,centroid_shift_pixels=0.)
    def center(p):return ((min(x for x,y in p)+max(x for x,y in p))/2,(min(y for x,y in p)+max(y for x,y in p))/2)
    ca,cb=center(before),center(after);distance=math.hypot(ca[0]-cb[0],ca[1]-cb[1]);ratio=abs(b-a)/a
    cov=ratio>.02
    return dict(geometry=distance>1. or not cov,coverage=cov,area_relative_error=ratio,centroid_shift_pixels=distance)


def blank():return dict(target=[False]*8,valid=[False]*8,evidence=[],exclusions=[])


def image_diff_label(row):
    result=blank();y,v=result['target'],result['valid'];f=row['family'];counts=row['counts']
    identical=(row.get('image_comparison') or {}).get('status') in {'identical_file_bytes','identical_decoded_pixels'}
    ambiguous=bool(counts.get('ambiguous_clean',0) or counts.get('ambiguous_variant',0))
    weak=any(m.get('confidence')!='strong' for m in row.get('object_matches',[]))
    complete=not ambiguous and not weak
    # Only complete unambiguous collections provide known negatives.
    for b in [2,3]:v[b]=complete
    if identical and complete:
        for b in [5,6]:v[b]=True
        if f in PIXEL_FAMILIES:v[1]=v[7]=True
    if f=='CME_C2_sample' and complete:v[1]=v[7]=True # angular position / width, NOT pixel masks
    if not identical and f in PIXEL_FAMILIES:result['exclusions'].append('changed_or_missing_pixels: spatial/count reference masked')
    if ambiguous or weak:result['exclusions'].append('ambiguous_or_candidate_object_identity: unknown negatives retained')
    def positive(bit,c,**measurements):
        y[bit]=True;v[bit]=True
        result['evidence'].append(dict(code=CLASSES[bit],pointer=c.get('variant_pointer') or c.get('clean_pointer'),
            reference_pointer=c.get('clean_pointer'),measurements=measurements,kind='P0_derived_technical_proxy'))
    for c in row.get('changes',[]):
        category=c['category'];strong=c.get('object_match_confidence')=='strong'
        pointer=c.get('variant_pointer') or c.get('clean_pointer') or ''
        if category=='class_changed' and strong and any(k in pointer for k in ['/matched_events/','/shapes/','/specialShapes/']):positive(2,c)
        if category=='time_changed' and strong:positive(3,c)
        if identical and category in {'annotation_removed','annotation_added'}:
            # These entries are emitted only for unmatched nonambiguous objects.
            positive(5 if category=='annotation_removed' else 6,c)
        if identical and category=='geometry_changed' and strong:
            diff=geometry_difference(c.get('before'),c.get('after'))
            if diff:
                if diff['geometry']:positive(1,c,**diff)
                if diff['coverage']:positive(7,c,**diff)
            else:
                v[1]=v[7]=False;result['exclusions'].append('invalid geometry reference')
        if f=='CME_C2_sample' and strong and category=='metadata_changed':
            key=pointer.rsplit('/',1)[-1]
            if key in {'central_pa','mpa','width'}:
                try:a,b=float(c['before']),float(c['after'])
                except (TypeError,ValueError):continue
                if not math.isfinite(a+b):continue
                delta=abs((a-b+180)%360-180) if key!='width' else abs(a-b)
                if delta>1e-3:positive(7 if key=='width' else 1,c,angular_error_degrees=delta)
    # Pixel changes invalidate spatial/count positives as well as negatives.
    if not identical and f in PIXEL_FAMILIES:
        for b in [1,5,6,7]:y[b]=False;v[b]=False
        result['evidence']=[e for e in result['evidence'] if e['code'] not in {'L2-G','L3-06','L3-07','L3-MIS'}]
    return result


def ulf_diff_label(clean,current,identical):
    result=blank();y,v=result['target'],result['valid']
    left=(clean.get('ulf_event_annotation') or {}).get('matched_events')
    right=(current.get('ulf_event_annotation') or {}).get('matched_events')
    if not isinstance(left,list) or not isinstance(right,list):
        result['exclusions'].append('unsupported_ulf_events');return result
    if left==right:
        for b in [2,3,5,6]:v[b]=True
        return result
    # Match unchanged events, then unique stable source or unique interval.
    used_l=set();used_r=set();pairs=[]
    for i,a in enumerate(left):
        choices=[j for j,b in enumerate(right) if j not in used_r and a==b]
        if choices:used_l.add(i);used_r.add(choices[0]);pairs.append((i,choices[0]))
    for fields in [('source',),('start_dt','end_dt')]:
        lm=collections.defaultdict(list);rm=collections.defaultdict(list)
        for i,a in enumerate(left):
            key=tuple(a.get(k) for k in fields)
            if i not in used_l and all(key):lm[key].append(i)
        for j,b in enumerate(right):
            key=tuple(b.get(k) for k in fields)
            if j not in used_r and all(key):rm[key].append(j)
        for key,ids in lm.items():
            if len(ids)==1 and len(rm.get(key,[]))==1:
                i,j=ids[0],rm[key][0];used_l.add(i);used_r.add(j);pairs.append((i,j))
    for i,j in pairs:
        a,b=left[i],right[j]
        for bit,fields in [(2,['kind']),(3,['start_dt','end_dt'])]:
            changed=[k for k in fields if a.get(k)!=b.get(k)]
            if changed:
                y[bit]=v[bit]=True;result['evidence'].append(dict(code=CLASSES[bit],pointer=f'/ulf_event_annotation/matched_events/{j}/{changed[0]}',kind='P0_derived_technical_proxy'))
    lc=len(left)-len(used_l);rc=len(right)-len(used_r)
    complete=not(lc and rc)
    if complete:
        for b in [2,3]:v[b]=True
        if identical:
            v[5]=v[6]=True;y[5]=lc>0;y[6]=rc>0
    else:result['exclusions'].append('ambiguous_ulf_event_identity')
    if not identical:
        v[3]=v[5]=v[6]=False;y[3]=y[5]=y[6]=False
        result['exclusions'].append('changed_or_missing_csv_reference')
    return result


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build(root,out):
    """All source splits, no reassignment. P0 is confined to target construction."""
    from spaceqa_agent import quality_ledger as q
    from spaceqa_agent.engine import png_header
    from concurrent.futures import ThreadPoolExecutor
    base=root/'opc-agent/runs/all-family-training-20261031'
    diffs=root/'phase2_visual_audit_xzj/outputs/run_20261025T161017Z/annotation_differences.jsonl'
    proxies={}
    with diffs.open() as f:
        for line in f:
            r=json.loads(line);proxies[(r['base_sample_id'],r['tier'])]=image_diff_label(r)
    out.mkdir(parents=True,exist_ok=True)
    contracts={'AIA_AR_sample':['AR'],'HMI_AR_sample':['AR'],'CME_C2_sample':['CME'],'EPB_seg_sample':['EPB'],'DMSP_SSUSI':['PCA','TPA','unclear']}
    summary={};sources={};all_groups={}
    def read_current(r):
        raw=Path(r['label_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=r['label_sha256']:raise ValueError('Current label hash changed: '+r['id'])
        ann=json.loads(raw.decode('utf-8-sig'));data=Path(r['data_path']);exists=data.is_file();wh=None
        if exists and data.suffix.lower()=='.png':
            size=png_header(data);wh=[size['width'],size['height']]
        return ann,exists,wh
    def prepared(records):
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(records),512):
                block=records[start:start+512]
                yield from zip(block,pool.map(read_current,block))
    for split in ['train','val','test']:
        source=base/(split+'.jsonl');records=[json.loads(l) for l in source.open()]
        p0={r['base_sample_id']:r for r in records if r['tier']=='P0'}
        ulf_clean={r['base_sample_id']:json.loads(Path(r['label_path']).read_text()) for r in p0.values() if r['family']=='ULF_sample'}
        groups={r['training_group_id'] for r in records}
        assert all(not groups.intersection(previous) for previous in all_groups.values())
        all_groups[split]=groups;counts=collections.defaultdict(collections.Counter);exceptions=collections.Counter()
        with (out/(split+'.jsonl')).open('w') as fp:
            for completed,(r,(ann,data_exists,wh)) in enumerate(prepared(records),1):
                if r['tier']=='P0':
                    proxy=blank()
                    for bit in [1,2,3,5,6,7]:proxy['valid'][bit]=True
                    if r['family']=='ULF_sample':proxy['valid'][1]=proxy['valid'][7]=False
                elif r['family']=='ULF_sample':
                    ref=p0[r['base_sample_id']]
                    proxy=ulf_diff_label(ulf_clean[r['base_sample_id']],ann,bool(r.get('data_sha256') and r['data_sha256']==ref.get('data_sha256')))
                else:proxy=json.loads(json.dumps(proxies[(r['base_sample_id'],r['tier'])]))
                sample=dict(id=r['id'],package=r['package_id'],family=r['family'],label_path=r['label_path'],
                    data_path=r['data_path'],data_exists=data_exists,data_sha256=r.get('data_sha256'),label_sha256=r['label_sha256'])
                if wh:sample['image_size']=wh
                mechanical=q.analyze_sample(sample,ann,allowed_labels=contracts.get(r['family']),
                    contract_source='Public task family vocabulary; within-vocabulary correctness remains P0-derived' if r['family'] in contracts else None)
                rule=[False]*8
                for finding in mechanical['findings']:
                    bit=CLASSES.index(finding['defectType']);rule[bit]=True;proxy['target'][bit]=True;proxy['valid'][bit]=True
                for bit in [0,4]:
                    ch=mechanical['checks'][CLASSES[bit]]
                    proxy['valid'][bit]=ch['scopeStatus'] in {'pass','fail'};proxy['target'][bit]=ch['scopeStatus']=='fail'
                if not sample['data_exists']:
                    for bit in [1,3,5,6,7]:proxy['valid'][bit]=False;proxy['target'][bit]=False
                    proxy['exclusions'].append('missing_observation')
                item=dict(id=r['id'],family=r['family'],tier=r['tier'],training_group_id=r['training_group_id'],
                    reference_kind='P0_derived_technical_proxy_NOT_official',rule=rule,**proxy)
                fp.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
                for bit,c in enumerate(CLASSES):
                    for key,ok in [('valid',proxy['valid'][bit]),('positive',proxy['valid'][bit] and proxy['target'][bit]),('unknown',not proxy['valid'][bit])]:
                        if ok:counts[r['family']+'|'+r['tier']+'|'+c][key]+=1
                exceptions.update(proxy['exclusions'])
                if completed%1024==0:
                    state=dict(stage='preparing_targets',split=split,completed=completed,total=len(records))
                    write(out/'state.json',state);print(json.dumps(state),flush=True)
        summary[split]=dict(records=len(records),groups=len(groups),coverage=dict(counts),exclusions=dict(exceptions))
        h=hashlib.sha256();
        with source.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        sources[split]=h.hexdigest()
        print(json.dumps(dict(stage='fine_targets_prepared',split=split,records=len(records),exclusions=dict(exceptions))),flush=True)
    write(out/'protocol.json',dict(classes=CLASSES,names=NAMES,source_split_hashes=sources,summary=summary,
        geometry_tolerance='1 pixel bbox-centre displacement; otherwise changed equal-area shape; polygon winding/start index invariant',
        coverage_tolerance='absolute relative polygon area difference > 2%; CME width angular difference > .001 degree',
        time='matched event timestamp/endpoint difference; CME event anchor is not an interval endpoint',
        caveats=['P0-relative technical supervision, not independent scientific defect truth','Changed pixels mask geometric/coverage/count targets',
                 'Ambiguous/candidate matching never establishes negatives','Mechanical vocabulary checking is narrower than full semantic class truth',
                 'Eight labels may co-occur; classifier labels do not establish defect location'],
        groups_disjoint=True,reference_inputs_used_for_inference=False))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/staff/tangjiarui/OPC'));ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();build(args.root,args.output)
