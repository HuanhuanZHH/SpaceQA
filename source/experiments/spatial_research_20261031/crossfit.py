"""Group-respecting out-of-fold raw segmentation for residual calibration."""
import json
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.nn import functional as F

from spatial import OUT,UNet,dice,dump,emit,lines,seed


def main():
    seed();start=time.monotonic()
    with np.load(OUT/'spatial_data.npz') as z:raw=z['raw'];clean=z['clean']
    bases=lines(OUT/'bases.jsonl');tr=np.array([i for i,b in enumerate(bases) if b['split']=='train'])
    groups=[bases[i]['split_group'] for i in tr]
    xx=torch.tensor(raw.transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255
    yy=torch.tensor(clean[:,None],device='cuda',dtype=torch.float32)
    with np.load(OUT/'unet_predictions.npz') as z:pred=z['matched'].astype(np.float32)
    trace=[];covered=set()
    for fold,(train_idx,heldout_idx) in enumerate(GroupKFold(3).split(tr,groups=groups)):
        a,b=tr[train_idx],tr[heldout_idx];assert not set(groups[i] for i in train_idx)&set(groups[i] for i in heldout_idx)
        torch.manual_seed(20261031+fold);rng=np.random.RandomState(20261031+fold)
        model=UNet().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
        posweight=((1-yy[a].mean())/yy[a].mean().clamp_min(.01)).clamp(1,10)
        ts=time.monotonic()
        for epoch in range(24):
            model.train();perm=rng.permutation(a)
            for i in range(0,len(perm),16):
                ii=perm[i:i+16];opt.zero_grad(set_to_none=True);p=model(xx[ii])
                loss=F.binary_cross_entropy_with_logits(p,yy[ii],pos_weight=posweight)+1-dice(p.sigmoid(),yy[ii])
                loss.backward();opt.step()
        model.eval()
        with torch.inference_mode():
            pp=torch.cat([model(xx[ii]).sigmoid() for ii in (b[i:i+24] for i in range(0,len(b),24))])
        pred[b]=pp[:,0].cpu().numpy();covered.update(b.tolist())
        trace.append(dict(fold=fold,train_bases=len(a),heldout_bases=len(b),seconds=time.monotonic()-ts,
                          heldout_dice=float(dice((pp>.5).float(),yy[b]))))
        emit('crossfit',**trace[-1]);del model,opt;torch.cuda.empty_cache()
    assert covered==set(tr.tolist())
    perm=np.arange(len(bases));rng=np.random.RandomState(20261031)
    for family in sorted({b['family'] for b in bases}):
        for split in ('train','val','test'):
            ii=np.array([i for i,b in enumerate(bases) if b['family']==family and b['split']==split]);perm[ii]=rng.permutation(ii)
    np.savez_compressed(OUT/'unet_oof_predictions.npz',matched=pred.astype(np.float16),permuted=pred[perm].astype(np.float16))
    dump('crossfit_training.json',dict(folds=trace,seconds=time.monotonic()-start,training_bases=len(tr),
        all_training_predictions_out_of_fold=True,validation_and_test_masks_unchanged=True,
        final_inference_model='unet.pt',additional_shipped_parameters=0))


if __name__=='__main__':main()
