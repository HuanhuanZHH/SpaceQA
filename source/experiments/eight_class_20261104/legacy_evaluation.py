"""Compare previous tiny raw-RGB localizer on the same full held-out images.

Only validation chooses a deployment candidate. Even the selected localizer is a
weak reference: no prediction is promoted to confirmed scientific truth.
"""
from __future__ import annotations
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import localizer as l


def main(args):
    import torch
    from PIL import Image
    from torch.nn import functional as F
    h=l.helper();out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=l.ROOT/'research_optimization_20261103/localizer/pilot_v1/weighted_bce_dice.pt'
    saved=torch.load(source,map_location='cpu',weights_only=False)
    torch.set_num_threads(6);model=h.torch_model().cuda();model.load_state_dict(saved['state_dict']);model.eval()
    thresholds={};metrics={};train=json.loads((args.localizer/'train_inventory.json').read_text())
    summaries=json.loads((args.localizer/'comparison.json').read_text())['metrics']
    old=json.loads((source.parent/'weighted_bce_dice_frozen_thresholds.json').read_text())['thresholds']
    for split in ['val','test']:
        rows=json.loads((args.localizer/(split+'_inventory.json')).read_text());usable=[r for r in rows if r['usable']]
        h.assert_disjoint(train,usable)
        if split=='test':h.assert_disjoint(validation,usable)
        else:validation=usable
        per=collections.defaultdict(list)
        class Images(torch.utils.data.Dataset):
            def __len__(self):return len(usable)
            def __getitem__(self,index):
                r=usable[index]
                with Image.open(r['data_path']) as im:x=np.array(im.convert('RGB').resize((saved['size'],saved['size']),Image.Resampling.BILINEAR),dtype=np.float32)/255
                truth=h.region_mask(r['geometry'],r['width'],r['height'],384).astype(bool)
                return torch.from_numpy(x.transpose(2,0,1).copy()),truth,index
        loader=torch.utils.data.DataLoader(Images(),batch_size=16,num_workers=6,pin_memory=True)
        count=0
        for x,truth,indices in loader:
            indices=indices.tolist();fi=torch.tensor([h.FAMILIES.index(usable[i]['family']) for i in indices],device='cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                p=model(x.cuda(non_blocking=True))[torch.arange(len(x),device='cuda'),fi].sigmoid().float()
            p=F.interpolate(p[:,None],size=(384,384),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
            for k,index in enumerate(indices):
                r=usable[index];per[r['family']].append(dict(id=r['id'],prob=p[k],presence=1.,truth=truth[k].numpy()))
            count+=len(x)
            if count%128==0 or count==len(usable):print(json.dumps(dict(stage='legacy_comparison',split=split,completed=count,total=len(usable),batch=16)),flush=True)
        if split=='val':
            thresholds={f:l.choose_thresholds(items) for f,items in per.items()}
            h.write(out/'legacy_frozen_policy.json',dict(checkpoint_sha256=h.sha(source),thresholds=thresholds,source='same grouped P0 validation; test not read'))
        for route in ['legacy_original_threshold','legacy_recalibrated']:
            result={}
            for family,items in per.items():
                th=thresholds[family] if route=='legacy_recalibrated' else dict(pixel=old[family],presence=0.)
                result[family]=l.summarize([l.mask_stats((r['prob']>=th['pixel'])&(r['presence']>=th['presence']),r['truth']) for r in items])
            metrics[route+'|'+split]=result
        if split=='val':
            pool={**summaries,**metrics};selected={}
            for family in h.FAMILIES:
                choices=[]
                for route in ['presence_gated','no_presence_loss','legacy_recalibrated']:
                    m=pool[route+'|val'][family];fp=m['empty_prediction_rate'];n=m['empty_records'];cap=max(.05,1/n) if n else 1.
                    if fp is not None and fp>cap+1e-12:continue
                    choices.append((m['positive_mean_dice'] or 0.,-(fp or 0.),route))
                selected[family]=max(choices)[-1] if choices and max(choices)[0]>0 else 'abstain'
            # Sources already have fixed validation metrics. No test metric enters selection.
            h.write(out/'deployment_policy.json',dict(status='frozen',selected=selected,legacy_thresholds=thresholds,
                localizer_threshold_files={r:str(args.localizer/(r+'_frozen_policy.json')) for r in ['presence_gated','no_presence_loss']},
                scientific_confirmation=False,source='validation only; all models trained before evaluation',
                test_used_for_selection=False,selection='positive mean Dice subject to same empty-mask FPR cap'))
    h.write(out/'comparison.json',dict(status='complete',metrics={**summaries,**metrics},selected_on_validation=selected,
        legacy_checkpoint_sha256=h.sha(source),new_scope='same all usable held-out P0 images, 384px weak target masks',
        caveat='Configuration comparison: training sample count, epochs, resolution and architecture differ; not an isolated architecture ablation'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--localizer',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    main(ap.parse_args())
