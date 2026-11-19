"""Resume completed localizer checkpoints with batched, all-family evaluation.

No training is repeated. Validation alone freezes thresholds; all held-out
source files are checked against inventory hashes before inference.
"""
from __future__ import annotations
import argparse,collections,concurrent.futures,json,time
from pathlib import Path
import numpy as np
import localizer as l


def main(args):
    import torch
    from PIL import Image
    out=args.output;h=l.helper();started=time.monotonic()
    completed=json.loads((out/'training_complete.json').read_text())
    assert completed['all_supported_families_trained']
    for record in completed['models']:
        assert len(record['history'])==12 and record['history'][-1]['epoch']==12
        assert h.sha(out/(record['route']+'.pt'))==record['sha256']
    torch.set_num_threads(6);train=json.loads((out/'train_inventory.json').read_text())
    def state(stage,**kw):
        value=dict(stage=stage,evaluation_seconds=time.monotonic()-started,batch=args.batch,**kw)
        h.write(out/'state.json',value);print(json.dumps(value),flush=True)
    ann,_=h.annotation_module(l.ROOT);inventories={};snapshots={}
    for split in ['val','test']:
        state('verifying_evaluation_sources',split=split)
        raw,_=h.load_rows(l.ROOT,split)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            inspected=list(pool.map(lambda row:h.inspect_row(row,ann),raw))
        rows=[r for r in inspected if r['usable']];h.assert_disjoint(train,rows)
        if split=='test':h.assert_disjoint(inventories['val'],rows)
        h.write(out/(split+'_inventory.json'),inspected)
        def verify(row):
            result=dict(id=row['id'])
            for kind in ['data','label']:
                path=row[kind+'_path'];before=h.fingerprint(path);actual=h.sha(path);after=h.fingerprint(path)
                assert actual==row[kind+'_sha256'] and before==after,(row['id'],kind)
                result[kind]=dict(path=path,actual_sha256=actual,expected_sha256=row[kind+'_sha256'],fingerprint_after=after)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            snapshots[split]=list(pool.map(verify,rows))
        h.write(out/(split+'_actual_sources.json'),dict(status='pass',records=snapshots[split]))
        inventories[split]=rows;state('evaluation_inputs',split=split,families=h.summarize(inspected))
    metrics={};thresholds={}
    for split in ['val','test']:
        rows=inventories[split]
        for route in ['presence_gated','no_presence_loss']:
            saved=torch.load(out/(route+'.pt'),map_location='cpu',weights_only=False)
            model=l.network(saved['width']).cuda();model.load_state_dict(saved['state_dict']);model.eval()
            size=saved['size'];per=collections.defaultdict(list)
            class Images(torch.utils.data.Dataset):
                def __len__(self):return len(rows)
                def __getitem__(self,index):
                    row=rows[index]
                    with Image.open(row['data_path']) as im:
                        x=np.array(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
                    truth=h.region_mask(row['geometry'],row['width'],row['height'],size).astype(bool)
                    return torch.from_numpy(x.transpose(2,0,1).copy()),truth,index
            loader=torch.utils.data.DataLoader(Images(),batch_size=args.batch,num_workers=6,pin_memory=True)
            count=0
            for x,truth,indices in loader:
                indices=indices.tolist();families=[h.FAMILIES.index(rows[i]['family']) for i in indices]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    z,pr=model(x.cuda(non_blocking=True))
                ix=torch.arange(len(x),device='cuda');fi=torch.tensor(families,device='cuda')
                probabilities=z[ix,fi].sigmoid().float().cpu().numpy()
                presence=pr[ix,fi].sigmoid().float().cpu().numpy() if route=='presence_gated' else np.ones(len(x))
                for k,index in enumerate(indices):
                    row=rows[index];per[row['family']].append(dict(id=row['id'],prob=probabilities[k],presence=float(presence[k]),truth=truth[k].numpy()))
                count+=len(x)
                if count%128==0 or count==len(rows):state('evaluating',split=split,route=route,completed=count,total=len(rows))
            if split=='val':
                thresholds[route]={family:l.choose_thresholds(items) for family,items in per.items()}
                h.write(out/(route+'_frozen_policy.json'),dict(thresholds=thresholds[route],checkpoint_sha256=h.sha(out/(route+'.pt')),
                    source='all usable grouped P0 validation; independent scientific truth unavailable',test_read=False,
                    batch=args.batch))
            result={};predictions=[]
            for family,items in per.items():
                th=thresholds[route][family]
                stats=[l.mask_stats((r['prob']>=th['pixel'])&(r['presence']>=th['presence']),r['truth']) for r in items]
                result[family]=l.summarize(stats)
                predictions.extend(dict(id=r['id'],family=family,presence=r['presence'],**s) for r,s in zip(items,stats))
            h.write(out/(route+'_'+split+'_metrics.json'),result)
            with (out/(route+'_'+split+'_predictions.jsonl')).open('w') as fp:
                for row in predictions:fp.write(json.dumps(row)+'\n')
            metrics[route+'|'+split]=result;del model,loader;torch.cuda.empty_cache()
        h.check_sources_after(snapshots[split],out/(split+'_source_stability_after.json'))
    selected={family:max(['presence_gated','no_presence_loss'],key=lambda route:
        (metrics[route+'|val'][family]['positive_mean_dice'] or 0.,-(metrics[route+'|val'][family]['empty_prediction_rate'] or 0.))) for family in h.FAMILIES}
    h.write(out/'comparison.json',dict(status='complete',selected_on_validation=selected,metrics=metrics,
        independent_fine_defect_F1=None,scope='P0 weak-mask localization; raw RGB-only global inference; no official defect truth',
        batch=args.batch,seconds=time.monotonic()-started,
        excluded={'CME_C2_sample':'angular-sector/time branch, not pixel masks','ULF_sample':'time-series branch, not pixel masks'}))
    state('complete',selected=selected)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--batch',type=int,default=16)
    args=ap.parse_args()
    try:main(args)
    except BaseException as exc:
        h=l.helper();h.write(args.output/'failure.json',dict(type=type(exc).__name__,message=str(exc)));raise
