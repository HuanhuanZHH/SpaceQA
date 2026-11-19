"""How much verified technical P1 geometry is lost by global thumbnails?"""
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path('/staff/tangjiarui/OPC')
OUT=ROOT/'opc-agent/runs/spatial-research-20261031'
SOURCES={'AIA':'aia_p0_p1_audit_20261026','HMI':'hmi_p0_p1_audit_20261029',
         'EPB':'epb_p0_p1_audit_20261027','DMSP':'dmsp_p0_p1_audit_20261029'}


def polygon(obj):
    if isinstance(obj,list):return np.array(obj,dtype=float)
    points=obj.get('bbox_polygon_pixel',obj.get('points',[]))
    p=np.array(points,dtype=float)
    if obj.get('shape_type')=='rectangle' and len(p)==2:
        lo=p.min(0);hi=p.max(0);p=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
    return p


def raster(points,size,origin,extent):
    p=(points-np.array(origin))*size/np.array(extent)
    im=Image.new('L',(size,size),0);ImageDraw.Draw(im).polygon([tuple(v) for v in p],fill=1)
    return np.array(im,dtype=np.uint8)


def main():
    rows=[]
    for family,folder in SOURCES.items():
        for line in (ROOT/'phase2_visual_audit_xzj'/folder/'findings.jsonl').open():
            f=json.loads(line)
            if f['code']!='L2-G':continue
            path=f.get('p1_data') or f['paths']['P1']['data']
            with Image.open(path) as im:w,h=im.size
            p=polygon(f.get('expected_in_p1_coordinates',f['before']))
            q=polygon(f.get('aligned_p1_coordinates',f['after']))
            if len(p)>1 and np.array_equal(p[0],p[-1]):p=p[:-1]
            if len(q)>1 and np.array_equal(q[0],q[-1]):q=q[:-1]
            shift=f.get('metrics',{}).get('max_vertex_shift_px',f.get('metrics',{}).get('max_corner_shift_px'))
            both=np.concatenate([p,q]);lo=both.min(0);hi=both.max(0)
            center=(lo+hi)/2;side=max(64,float((hi-lo).max())*1.4)
            origin=np.maximum(0,center-side/2);extent=np.minimum([w,h],origin+side)-origin
            r=dict(family=family,filename=f['filename'],pointer=f['pointer'],max_vertex_shift_native_px=shift,
                   shift_global224_px=None if shift is None else shift*224/max(w,h),raster={})
            for size in (224,448,1024):
                a=raster(p,size,[0,0],[w,h]);b=raster(q,size,[0,0],[w,h])
                r['raster'][str(size)]=dict(changed_pixels=int((a^b).sum()),
                    iou=float((a&b).sum()/max(1,(a|b).sum())))
            a=raster(p,224,origin,extent);b=raster(q,224,origin,extent)
            r['crop224']=dict(native_crop_xywh=[*origin,*extent],changed_pixels=int((a^b).sum()),
                              iou=float((a&b).sum()/max(1,(a|b).sum())))
            rows.append(r)
    (OUT/'resolution_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={}
    for family in ['ALL',*SOURCES]:
        rr=[r for r in rows if family=='ALL' or r['family']==family]
        summary[family]=dict(findings=len(rr),
            native_shift_below1px=sum(r['max_vertex_shift_native_px'] is not None and r['max_vertex_shift_native_px']<1 for r in rr),
            projected_shift_below1px_at224=sum(r['shift_global224_px'] is not None and r['shift_global224_px']<1 for r in rr),
            identical_raster_at_global={str(s):sum(r['raster'][str(s)]['changed_pixels']==0 for r in rr) for s in (224,448,1024)},
            identical_raster_at_crop224=sum(r['crop224']['changed_pixels']==0 for r in rr))
    summary['_limitations']=['Replays earlier clean-derived technical proposals, no independent scientific labels.',
        'Raster visibility is not model accuracy. A raster difference does not prove a physical mistake.',
        'Subpixel differences may be coordinate conventions; official geometric matching tolerance needed.',
        'AIA/HMI expected polygons use earlier validated image transforms. No new registration claimed.']
    (OUT/'resolution_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
