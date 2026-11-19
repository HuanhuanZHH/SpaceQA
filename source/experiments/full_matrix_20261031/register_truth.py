"""Pixel-validated geometry truth; never used as a model input or prediction."""
import collections
import concurrent.futures
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import OUT, MANIFEST, AUDIT, ann, lines, dump, emit
from prepare import numeric_points, object_pairs

sys.path.insert(0,str(AUDIT/'src'))
from analyze_aia_registration_20261026 import register as aia_register
from audit_hmi_p1_20261029 import register as hmi_register


def work(task):
    row,p0=task;name=row['id'].replace(':','_')+'.json';destination=OUT/'registration'/name
    hashes=[p0['data_sha256'],row['data_sha256'],p0['label_sha256'],row['label_sha256']]
    if destination.exists():
        cached=json.loads(destination.read_text())
        if cached.get('source_hashes_v2')==hashes:return cached
    result=dict(id=row['id'],valid=False,reason='image_registration_unverified',source_hashes=[p0['data_sha256'],row['data_sha256']],source_hashes_v2=hashes)
    try:
        a=ann.extract_annotation(json.loads(open(p0['label_path']).read()),row['family'])['objects']
        b=ann.extract_annotation(json.loads(open(row['label_path']).read()),row['family'])['objects']
        pairs,_,_,ambiguous=object_pairs(a,b,row['family'])
        pp=[];qq=[]
        for i,j in pairs:
            p=numeric_points(a[i]);q=numeric_points(b[j])
            if p is not None and q is not None and p.shape==q.shape:pp.extend(p);qq.extend(q)
        fit=None
        if len(pp)>=4:
            matrix,inliers=cv2.estimateAffine2D(np.array(pp),np.array(qq),method=cv2.RANSAC,ransacReprojThreshold=3.,maxIters=2000,confidence=.999,refineIters=10)
            if matrix is not None and .2<abs(np.linalg.det(matrix[:,:2]))<3:
                fit=dict(matrix_3x2=matrix.T.tolist())
        request=dict(filename=row['id'],p0_data_exists=True,p1_data_exists=True,image_bytes_equal=False,
            p0_data=p0['data_path'],p1_data=row['data_path'],label_affine_fit=fit,
            paths={'P0':{'data':p0['data_path']},'P1':{'data':row['data_path']}},
            p0_events=len(a),p0_null_geometries=sum(numeric_points(x) is None for x in a))
        reg=(hmi_register if row['family']=='HMI_AR_sample' else aia_register)(request)
        result['registration']=reg
        if reg.get('image_pixels_equal'):
            matrix=np.array([[1.,0.],[0.,1.],[0.,0.]]);tolerance=.5
        elif (reg.get('image_correlation') or 0)>=.98 and reg.get('matrix_3x2'):
            matrix=np.array(reg['matrix_3x2']);tolerance=3.
        else:matrix=None
        if matrix is not None and not ambiguous:
            geom=False;valid=True
            for i,j in pairs:
                p=numeric_points(a[i]);q=numeric_points(b[j])
                if p is None or q is None:
                    if (p is None)!=(q is None):valid=False
                    continue
                expected=np.c_[p,np.ones(len(p))]@matrix
                from PIL import Image
                with Image.open(row['data_path']) as source: dims=np.array(source.size)-1
                expected=np.clip(expected,0,dims)
                if expected.shape!=q.shape:geom=True
                else:
                    d=np.linalg.norm(expected[:,None]-q[None],axis=2);ii,jj=linear_sum_assignment(d)
                    geom=geom or bool(d[ii,jj].max(initial=0)>tolerance)
            result.update(valid=valid,geometry_target=int(geom),tolerance_native_pixels=tolerance,
                reason='pixel_validated_transform' if valid else 'geometry_availability_changed')
    except Exception as exc:
        result.update(reason='registration_exception',error=str(exc)[:400])
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(result));tmp.replace(destination)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'registration').mkdir(exist_ok=True)
    refs={r['base_sample_id']:r for r in lines(MANIFEST) if r['tier']=='P0'}
    start=time.monotonic();seen=set();counts=collections.Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        while time.monotonic()-start<5*3600:
            paths=[p for p in (OUT/'shards').glob('*.json') if p.name not in seen]
            tasks=[]
            for path in paths[:500]:
                rows=json.loads(path.read_text())
                if not all(r.get('preparation_version')==2 for r in rows):continue
                seen.add(path.name)
                for r in rows:
                    if r['family'] in ['AIA_AR_sample','HMI_AR_sample'] and r['data_sha256']!=refs[r['base_sample_id']]['data_sha256'] and r['data_exists'] and refs[r['base_sample_id']]['data_exists']:
                        tasks.append((r,refs[r['base_sample_id']]))
            for result in pool.map(work,tasks,chunksize=1):counts[result['reason']]+=1
            prep=json.loads((OUT/'prepare_state.json').read_text())
            state=dict(status='running',pid=os.getpid(),shards_seen=len(seen),registration_tasks=sum(counts.values()),
                by_status=dict(counts),seconds=time.monotonic()-start)
            dump('registration_state.json',state);emit('registration',**state)
            if prep['status']=='complete' and len(seen)==24713:break
            if not paths:time.sleep(10)
        else:raise RuntimeError('Registration 5-hour budget reached; derived cache retained')
    state['status']='complete';dump('registration_state.json',state)


if __name__=='__main__':main()
