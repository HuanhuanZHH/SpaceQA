"""Reuse the frozen dataset verbatim and prepare train/validation YOLO views."""
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from common import OUT,PREVIOUS,SIZE,lines,shapes,dump,annotation_boxes


def polygon(s,w,h):
    points=np.array(s.get('points',[]),dtype=float)
    if len(points)<2:return None
    if s.get('shape_type')=='rectangle' and len(points)==2:
        lo=points.min(0);hi=points.max(0);points=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]])
    if len(points)<3:return None
    points=np.clip(points/[w,h],0,1)
    # Simplify only subpixel detail at the training-view resolution.
    points=cv2.approxPolyDP(points.astype(np.float32).reshape(-1,1,2),.25/SIZE,True).reshape(-1,2)
    return points if len(points)>=3 else None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['dataset.jsonl','bases.jsonl','dataset_summary.json']:
        shutil.copyfile(PREVIOUS/name,OUT/name)
    with np.load(PREVIOUS/'spatial_data.npz') as z:
        for key in ['raw','clean','current','truth']:np.save(OUT/(key+'.npy'),z[key])
    bases=lines(OUT/'bases.jsonl');raw=np.load(OUT/'raw.npy',mmap_mode='r');counts={}
    for task in ['detect','segment']:
        for split in ['train','val']:
            (OUT/'yolo_data'/task/'images'/split).mkdir(parents=True,exist_ok=True)
            (OUT/'yolo_data'/task/'labels'/split).mkdir(parents=True,exist_ok=True)
    image_dir=OUT/'yolo_images_all';image_dir.mkdir(exist_ok=True)
    for i,b in enumerate(bases):
        name=f'{i:05d}.png';image_path=image_dir/name
        Image.fromarray(raw[i]).save(image_path)
        if b['split']=='test':continue
        ann=json.loads(Path(b['reference_label_path']).read_text())
        w,h=(390,350) if b['family']=='DMSP_SSUSI' else (1024,1024)
        det=[];seg=[]
        for box in annotation_boxes(ann,w,h):
            center=(box[:2]+box[2:])/2;size=box[2:]-box[:2]
            det.append('0 '+' '.join(f'{v:.8f}' for v in [*center,*size]))
        for s in shapes(ann):
            p=polygon(s,w,h)
            if p is not None:seg.append('0 '+' '.join(f'{v:.8f}' for v in p.ravel()))
        for task,text in [('detect',det),('segment',seg)]:
            dst=OUT/'yolo_data'/task/'images'/b['split']/name
            if not dst.exists():dst.symlink_to(image_path)
            (OUT/'yolo_data'/task/'labels'/b['split']/(Path(name).stem+'.txt')).write_text('\n'.join(text)+'\n')
            counts[task+':'+b['split']]=counts.get(task+':'+b['split'],0)+len(text)
    for task in ['detect','segment']:
        cfg=dict(path=str(OUT/'yolo_data'/task),train='images/train',val='images/val',names={0:'space_object'})
        (OUT/f'yolo_{task}.yaml').write_text(json.dumps(cfg,indent=2))
    with np.load(PREVIOUS/'unet_predictions.npz') as z:
        (OUT/'predictions').mkdir(exist_ok=True);np.save(OUT/'predictions/baseline_unet.npy',z['matched'])
    summary=json.loads((OUT/'dataset_summary.json').read_text());summary.update(yolo_object_counts=counts,
        original_dataset_sha256=hashlib.sha256((OUT/'dataset.jsonl').read_bytes()).hexdigest(),
        test_images_excluded_from_yolo_fit=True,all_classes_collapsed_to_foreground=True)
    dump('preparation.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
