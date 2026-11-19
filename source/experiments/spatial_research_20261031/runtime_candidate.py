"""Actual P1 package candidate pass including raw decode, masks and evidence.

This measures the spatial candidate branch, not the full official score run.
"""
import collections
import json
import os
import pickle
import time
from pathlib import Path

os.environ['OPC_SPATIAL_SIZE']='448'
os.environ['OPC_SPATIAL_OUT']='/staff/tangjiarui/OPC/opc-agent/runs/spatial-research-448-20261031'

import cv2
import numpy as np
import torch
from PIL import Image

from spatial import ROOT,OUT,SIZE,UNet,annotation_features,mask,residual_features,seed


def main():
    seed();start=time.monotonic();model=UNet().cuda().eval()
    model.load_state_dict(torch.load(OUT/'unet.pt',map_location='cpu',weights_only=False)['state'])
    with (OUT/'unet_detectors.pkl').open('rb') as f:detector=pickle.load(f)
    groups=collections.defaultdict(list)
    manifest=ROOT/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    for s in manifest.open():
        r=json.loads(s)
        if r['tier']=='P1' and r['dataset_family'] in ('EPB_seg_sample','DMSP_SSUSI'):groups[r['dataset_family']].append(r)
    startup=time.monotonic()-start;result=dict(startup_seconds=startup,batch_sweep={},full_packages={},hardware='icarus0 A100-PCIE-40GB GPU1',
        official_A10_tested=False,formal_full_evaluation=False,component='U-Net448 spatial candidate branch',includes=['raw PNG read/decode','JSON parse','annotation rasterization',
        'GPU mask prediction','residual features','forest predictions','region proposals','JSONL evidence write'],
        excludes=['formal rule evaluation','quality score/ranking','final output audit'],model_weights_bytes=(OUT/'unet.pt').stat().st_size)
    def run(rows,batch,path):
        torch.cuda.reset_peak_memory_stats();ts=time.monotonic();counts=collections.Counter()
        with path.open('w') as output:
            for offset in range(0,len(rows),batch):
                prepared=[]
                for r in rows[offset:offset+batch]:
                    if not r['data_exists']:
                        counts['missing_raw']+=1;output.write(json.dumps(dict(sample_id=r['sample_id'],status='missing_raw_no_visual_claim'))+'\n');continue
                    ann=json.loads(Path(r['label_path']).read_text())
                    with Image.open(r['data_path']) as im:
                        w,h=im.size;raw=np.array(im.convert('RGB').resize((SIZE,SIZE),Image.Resampling.BILINEAR))
                    m=mask(ann,w,h);af=annotation_features(ann,r['dataset_family'],w,h)
                    prepared.append((r,ann,w,h,raw,m,af))
                if not prepared:continue
                x=torch.tensor(np.array([r[4] for r in prepared]).transpose(0,3,1,2),dtype=torch.float32,device='cuda')/255
                with torch.inference_mode():pp=model(x).sigmoid()[:,0].cpu().numpy()
                fs=np.array([r[6]+residual_features(p,r[5],r[1],r[2],r[3]) for r,p in zip(prepared,pp)])
                probs=np.zeros((len(prepared),3))
                for k,clf in enumerate(detector['classifiers']):
                    cl=list(clf.classes_)
                    if 1 in cl:probs[:,k]=clf.predict_proba(fs)[:,cl.index(1)]
                pred=np.round(probs,12)>=detector['multi_tier_thresholds']
                for r,p,flags in zip(prepared,pp,pred):
                    res=(p>.5)^(r[5]>0);n,_,stats,_=cv2.connectedComponentsWithStats(res.astype(np.uint8),8)
                    regions=sorted(stats[1:].tolist(),key=lambda a:a[4],reverse=True)[:6] if flags.any() else []
                    record=dict(sample_id=r[0]['sample_id'],label_path=r[0]['label_path'],raw_path=r[0]['data_path'],
                        proposed_codes=[detector['codes'][k] for k in range(3) if flags[k]],
                        candidate_regions_xywh=[[a[0]*r[2]/SIZE,a[1]*r[3]/SIZE,a[2]*r[2]/SIZE,a[3]*r[3]/SIZE] for a in regions],
                        status='review_only_no_formal_score',rule_version='spatial-research-20261031')
                    output.write(json.dumps(record)+'\n');counts['image_inferences']+=1
        torch.cuda.synchronize();elapsed=time.monotonic()-ts
        return dict(seconds=elapsed,records=len(rows),counts=dict(counts),records_per_second=len(rows)/elapsed,
            batch=batch,peak_reserved_bytes=torch.cuda.max_memory_reserved(),output_bytes=path.stat().st_size)
    for family,rows in groups.items():
        sample=[r for r in rows if r['data_exists']][:64];result['batch_sweep'][family]={}
        for batch in [1,8,16,32]:
            result['batch_sweep'][family][str(batch)]=run(sample,batch,OUT/f'runtime_{family}_b{batch}.jsonl')
        result['full_packages'][family]=run(rows,16,OUT/f'runtime_full_{family}.jsonl')
        (OUT/'candidate_runtime.json').write_text(json.dumps(result,indent=2));print(json.dumps({family:result['full_packages'][family]}),flush=True)
    result['total_seconds']=time.monotonic()-start
    (OUT/'candidate_runtime.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
