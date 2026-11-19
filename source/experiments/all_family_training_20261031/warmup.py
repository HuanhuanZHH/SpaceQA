"""Train a useful raw-view student encoder while independent CPU registration runs."""
import collections
import os
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader

from common import *
from prepare import split_groups
from models import Student


class RawViews(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image']) as im:
            # Only current raw observation; discard the annotated right panel entirely.
            im=im.crop((0,0,384,384)).resize((128,128),Image.Resampling.BILINEAR)
            x=np.asarray(im,dtype=np.float32).transpose(2,0,1)/255
        return x,FAMILIES.index(self.rows[i]['family']),i


class MaskedEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Student().convs
        self.decoder=nn.ModuleList([nn.Conv2d(128,64,3,padding=1),nn.Conv2d(64,32,3,padding=1),nn.Conv2d(32,3,3,padding=1)])
    def forward(self,x):
        for conv in self.encoder:x=F.max_pool2d(F.gelu(conv(x)),2)
        for i,conv in enumerate(self.decoder):
            x=conv(F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False))
            if i<2:x=F.gelu(x)
        return x.sigmoid()


def main():
    torch.set_num_threads(4);torch.manual_seed(SEEDS[0]);np.random.seed(SEEDS[0])
    folder=OUT/'pretraining';folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    write(folder/'state.json',dict(status='loading_grouped_training_scope',pid=os.getpid(),gpu=1,test_read=False))
    # No registration or annotation parsing is needed to learn from raw current observations.
    source=lines(SOURCE/'dataset.jsonl');assignment=split_groups(source)
    rows=[{k:r[k] for k in ['id','base_sample_id','family','tier','image','data_sha256']}|
          {'training_group_id':assignment[r['base_sample_id']][1]}
          for r in source if assignment[r['base_sample_id']][0]=='train' and not r['input_missing']]
    excluded_groups={assignment[r['base_sample_id']][1] for r in source if assignment[r['base_sample_id']][0]!='train'}
    assert not excluded_groups&{r['training_group_id'] for r in rows};del source
    coverage=assert_coverage(rows);save_lines(folder/'train_scope.jsonl',rows)
    scope_hash=sha(folder/'train_scope.jsonl');model=MaskedEncoder().cuda()
    generator=torch.Generator().manual_seed(SEEDS[0])
    loader=DataLoader(RawViews(rows),batch_size=128,shuffle=True,generator=generator,num_workers=4,
        pin_memory=True,persistent_workers=True,drop_last=False)
    counts=collections.Counter(r['family'] for r in rows)
    weights=torch.tensor([len(rows)/(6*counts[f]) for f in FAMILIES],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.001)
    resume=folder/'resume.pt';first=0;updates=0;trace=[];best=float('inf');best_encoder=None
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['scope_hash']==scope_hash
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);first=state['epoch']
        updates=state['updates'];trace=state['trace'];best=state['best'];best_encoder=state['best_encoder']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['loader_rng'])
    raw_example=rows[0]['image'];actual=collections.Counter()
    for epoch in range(first,60):
        model.train();seen=set();epoch_coverage=collections.Counter();losses=[]
        for x,family,indices in loader:
            x=x.cuda(non_blocking=True);family=family.cuda(non_blocking=True)
            mask=F.interpolate((torch.rand(len(x),1,16,16,device='cuda')<.4).float(),size=(128,128),mode='nearest')
            corrupted=x*(1-mask)+.5*mask
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(corrupted)
            pred=pred.float();mse=((pred-x).square()*(mask+.1)).mean((1,2,3))
            dx=(pred[:,:,:,1:]-pred[:,:,:,:-1])-(x[:,:,:,1:]-x[:,:,:,:-1])
            dy=(pred[:,:,1:,:]-pred[:,:,:-1,:])-(x[:,:,1:,:]-x[:,:,:-1,:])
            edge=.2*(dx.abs().mean((1,2,3))+dy.abs().mean((1,2,3)))
            loss=((mse+edge)*weights[family]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite raw-encoder pretraining loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
            losses.append(float(loss.detach()))
            for i in indices.tolist():
                r=rows[i];seen.add(r['id']);epoch_coverage[r['family']+'|'+r['tier']]+=1
            if updates%25==0:
                state=dict(status='training',phase='all_family_raw_encoder_pretraining',pid=os.getpid(),gpu=1,
                    epoch=epoch+1,max_epochs=60,updates=updates,records_in_epoch=len(seen),train_records=len(rows),
                    loss=losses[-1],elapsed_seconds=time.monotonic()-start,actual_train_coverage=dict(epoch_coverage),
                    test_read=False,registration_independent=True)
                write(folder/'state.json',state);emit('raw_encoder_train',**{k:v for k,v in state.items() if k!='actual_train_coverage'})
        assert seen=={r['id'] for r in rows} and dict(epoch_coverage)==coverage
        actual.update(epoch_coverage);value=float(np.mean(losses))
        if value<best:
            best=value;best_encoder={k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
        trace.append(dict(epoch=epoch+1,reconstruction_loss=value,unique_train_records=len(seen),actual_train_coverage=dict(epoch_coverage)))
        checkpoint=folder/'raw_encoder.pt';tmp=checkpoint.with_suffix('.tmp')
        torch.save(dict(encoder=best_encoder,source='all-six-family train-only masked reconstruction',scope_sha256=scope_hash,
            ids_sha256=digest(sorted(seen)),epochs=epoch+1,trace=trace),tmp);tmp.replace(checkpoint)
        tmp=resume.with_suffix('.tmp');torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,
            updates=updates,trace=trace,best=best,best_encoder=best_encoder,scope_hash=scope_hash,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state()),tmp);tmp.replace(resume)
        write(folder/'state.json',dict(status='epoch_complete',phase='all_family_raw_encoder_pretraining',pid=os.getpid(),gpu=1,
            epoch=epoch+1,updates=updates,records_in_epoch=len(seen),train_records=len(rows),loss=value,test_read=False,
            elapsed_seconds=time.monotonic()-start,actual_train_coverage=dict(epoch_coverage)))
        prep=OUT/'preparation.json';ready=prep.exists() and json.loads(prep.read_text()).get('status')=='complete'
        if epoch+1>=3 and (ready or time.monotonic()-start>=2*3600):break
    result=dict(status='complete',phase='all_family_raw_encoder_pretraining',epochs=len(trace),updates=updates,
        train_records=len(rows),train_ids_sha256=digest(sorted(r['id'] for r in rows)),train_scope_sha256=scope_hash,
        coverage=coverage,trace=trace,checkpoint_sha256=sha(folder/'raw_encoder.pt'),test_read=False,
        elapsed_seconds=time.monotonic()-start,stopping='min3/max60 epochs; stop at epoch boundary when supervised inputs ready or 2h elapsed',
        use='initialize CNN encoder in supervised and distilled students; not a defect F1 result')
    write(folder/'complete.json',result);write(folder/'state.json',result);emit('raw_encoder_complete',epochs=len(trace),updates=updates,records=len(rows))


if __name__=='__main__':main()
