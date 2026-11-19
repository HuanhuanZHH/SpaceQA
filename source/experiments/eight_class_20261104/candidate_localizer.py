"""Pixel-Dice calibration for investigation-only masks, not confirmed defects.

The conservative empty-sample cap is retained in the original comparison. This
additional, explicitly labelled policy maximizes foreground pixel Dice on val
and discloses empty-sample FPR rather than hiding trivial all-zero predictions.
"""
from __future__ import annotations
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import localizer as l

ROUTES=['presence_gated','no_presence_loss','legacy_recalibrated']


def choose(items):
    truth_size=np.array([r['truth'].sum() for r in items]);positive=truth_size>0
    presence=np.array([r['presence'] for r in items]);best=None
    for pixel in [.2,.35,.5,.65,.8,.9]:
        sufficient=[]
        for r in items:
            mask=r['prob']>=pixel
            sufficient.append((np.logical_and(mask,r['truth']).sum(),mask.sum()))
        tp,predicted=np.array(sufficient).T
        for gate in [0.,.1,.3,.5,.7,.9,.95,.99]:
            keep=presence>=gate;t=tp*keep;p=predicted*keep
            dice=2*t.sum()/(p.sum()+truth_size.sum()) if p.sum()+truth_size.sum() else 0.
            sample_dice=np.divide(2*t,p+truth_size,out=np.zeros(len(items)),where=p+truth_size>0)
            fp=float(np.mean(p[~positive]>0)) if (~positive).any() else 0.
            key=(float(dice),float(sample_dice[positive].mean()) if positive.any() else 0.,-fp,pixel,gate)
            if best is None or key>best[0]:best=(key,pixel,gate)
    return dict(pixel=best[1],presence=best[2],validation_objective='maximize foreground micro pixel Dice, then positive mean Dice, then lower empty-sample FPR; investigation candidates only')


def pixel_dice(metrics):
    denominator=2*metrics['tp']+metrics['fp']+metrics['fn']
    return 2*metrics['tp']/denominator if denominator else 0.


def main(args):
    import torch
    from PIL import Image
    from torch.nn import functional as F
    out=args.output;out.mkdir(parents=True,exist_ok=False);h=l.helper();torch.set_num_threads(6)
    started=time.monotonic();thresholds={};metrics={};selected=None
    for split in ['val','test']:
        # The source inventories and hashes were verified in the preceding run.
        sources=json.loads((args.localizer/(split+'_actual_sources.json')).read_text())
        assert sources['status']=='pass'
        h.check_sources_after(sources['records'],out/(split+'_source_stability_before.json'))
        all_rows=json.loads((args.localizer/(split+'_inventory.json')).read_text());rows=[r for r in all_rows if r['usable']]
        for route in ROUTES:
            legacy=route=='legacy_recalibrated'
            path=(l.ROOT/'research_optimization_20261103/localizer/pilot_v1/weighted_bce_dice.pt') if legacy else args.localizer/(route+'.pt')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            model=h.torch_model() if legacy else l.network(saved['width'])
            model.load_state_dict(saved['state_dict']);model=model.cuda().eval();size=saved['size'];per=collections.defaultdict(list)
            class Images(torch.utils.data.Dataset):
                def __len__(self):return len(rows)
                def __getitem__(self,index):
                    row=rows[index]
                    with Image.open(row['data_path']) as im:x=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
                    truth=h.region_mask(row['geometry'],row['width'],row['height'],384).astype(bool)
                    return torch.from_numpy(x.transpose(2,0,1).copy()),truth,index
            loader=torch.utils.data.DataLoader(Images(),batch_size=16,num_workers=6,pin_memory=True);count=0
            for x,truth,indices in loader:
                indices=indices.tolist();fi=torch.tensor([h.FAMILIES.index(rows[i]['family']) for i in indices],device='cuda')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(x.cuda(non_blocking=True))
                    if legacy:z=output;present=np.ones(len(x))
                    else:
                        z,pr=output;present=pr[torch.arange(len(x),device='cuda'),fi].sigmoid().float().cpu().numpy() if route=='presence_gated' else np.ones(len(x))
                    prob=z[torch.arange(len(x),device='cuda'),fi].sigmoid().float()
                    prob=F.interpolate(prob[:,None],size=(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
                for k,index in enumerate(indices):
                    row=rows[index];per[row['family']].append(dict(id=row['id'],prob=prob[k],presence=float(present[k]),truth=truth[k].numpy()))
                count+=len(x)
                if count%128==0 or count==len(rows):print(json.dumps(dict(stage='candidate_policy_evaluation',split=split,route=route,completed=count,total=len(rows),seconds=time.monotonic()-started)),flush=True)
            if split=='val':
                thresholds[route]={f:choose(items) for f,items in per.items()}
                h.write(out/(route+'_frozen_policy.json'),dict(thresholds=thresholds[route],checkpoint_sha256=h.sha(path),test_read=False,
                    scientific_confirmation=False,source='same grouped validation; foreground pixel Dice candidate policy'))
            result={};predictions=[]
            for family,items in per.items():
                th=thresholds[route][family];stats=[l.mask_stats((r['prob']>=th['pixel'])&(r['presence']>=th['presence']),r['truth']) for r in items]
                result[family]=l.summarize(stats);result[family]['foreground_pixel_dice']=pixel_dice(result[family])
                predictions.extend(dict(id=r['id'],family=family,**s) for r,s in zip(items,stats))
            metrics[route+'|'+split]=result
            with (out/(route+'_'+split+'_predictions.jsonl')).open('w') as fp:
                for row in predictions:fp.write(json.dumps(row)+'\n')
            del model,loader;torch.cuda.empty_cache()
        if split=='val':
            selected={f:max(ROUTES,key=lambda route:(metrics[route+'|val'][f]['foreground_pixel_dice'],
                metrics[route+'|val'][f]['positive_mean_dice'] or 0.,-(metrics[route+'|val'][f]['empty_prediction_rate'] or 0.))) for f in h.FAMILIES}
            h.write(out/'deployment_policy.json',dict(status='frozen',selected=selected,
                legacy_thresholds=thresholds['legacy_recalibrated'],
                localizer_threshold_files={route:str(out/(route+'_frozen_policy.json')) for route in ROUTES if route!='legacy_recalibrated'},
                scientific_confirmation=False,test_used_for_selection=False,
                selection='validation foreground micro pixel Dice; investigation candidates only; no empty-sample FPR claim'))
        h.check_sources_after(sources['records'],out/(split+'_source_stability_after.json'))
    h.write(out/'comparison.json',dict(status='complete',metrics=metrics,selected_on_validation=selected,seconds=time.monotonic()-started,
        scope='Additional candidate-only threshold policy; all usable held-out P0 weak masks; not official defect F1',
        caveat='Candidates may have a high empty-sample false positive rate. They cannot become confirmed defects without independent evidence.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--localizer',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    main(ap.parse_args())
