"""Add explicitly scoped time supervision, preserving the strict-mask baseline.

ULF observation-window changes are not event-onset truth. CME uniquely unchanged
speed triplets provide secondary weak identity, not strict scientific identity.
"""
from __future__ import annotations
import argparse,collections,datetime as dt,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def seconds(a,b):
    try:
        def parse(x):
            t=dt.datetime.fromisoformat(x.replace('Z','+00:00'))
            return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t
        return abs((parse(a)-parse(b)).total_seconds())
    except (ValueError,TypeError,AttributeError):return None


def observation_window_changes(clean,current):
    result=[]
    for field in ['startTime','endTime']:
        a=(clean.get('basic_annotation') or {}).get(field);b=(current.get('basic_annotation') or {}).get(field)
        delta=seconds(a,b)
        if delta is not None and delta>0:result.append(dict(code='L2-T',pointer='/basic_annotation/'+field,
            reference_pointer='/basic_annotation/'+field,kind='P0_derived_technical_proxy',
            time_scope='observation_window_endpoint_not_event_onset',measurements=dict(error_seconds=delta)))
    return result


def main(args):
    root=Path('/staff/tangjiarui/OPC');args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((args.input/'protocol.json').read_text());cme=collections.defaultdict(list)
    diffs=root/'phase2_visual_audit_xzj/outputs/run_20261025T161017Z/annotation_differences.jsonl'
    for line in diffs.open():
        r=json.loads(line)
        if r['family']!='CME_C2_sample' or r['image_comparison'].get('status')!='identical_file_bytes':continue
        for c in r['changes']:
            if c['category']=='time_changed' and c.get('object_match_method')=='unique_unchanged_speed_triplet' and c.get('object_match_confidence')=='candidate':
                error=seconds(c.get('before'),c.get('after'))
                if error:
                    cme[(r['base_sample_id'],r['tier'])].append(dict(code='L2-T',pointer=c['variant_pointer'],reference_pointer=c['clean_pointer'],
                        time_scope='event_timestamp_not_interval_endpoint',kind='secondary_weak_identity_P0_proxy',
                        measurements=dict(error_seconds=error,matching='unique unchanged speed triplet')))
    summary={}
    for split in ['train','val','test']:
        source=root/'opc-agent/runs/all-family-training-20261031'/(split+'.jsonl')
        records=[json.loads(l) for l in source.open()];ulf=[r for r in records if r['family']=='ULF_sample']
        def read(r):
            raw=Path(r['label_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['label_sha256']
            return r['id'],json.loads(raw.decode('utf-8-sig'))
        annotations={}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0,len(ulf),512):annotations.update(pool.map(read,ulf[start:start+512]))
        refs={r['base_sample_id']:r for r in ulf if r['tier']=='P0'};meta={r['id']:r for r in ulf};counts=collections.Counter();coverage=collections.defaultdict(collections.Counter)
        with (args.output/(split+'.jsonl')).open('w') as out:
            for line in (args.input/(split+'.jsonl')).open():
                row=json.loads(line);row['strict_valid']=row['valid'].copy()
                base=row['id'].rsplit(':',1)[0]
                if row['family']=='ULF_sample' and row['tier']!='P0':
                    r=meta[row['id']];p0=refs[r['base_sample_id']]
                    if r.get('data_sha256') and r.get('data_sha256')==p0.get('data_sha256'):
                        changes=observation_window_changes(annotations[p0['id']],annotations[r['id']])
                        if changes:
                            row['target'][3]=row['valid'][3]=row['strict_valid'][3]=True;row['evidence'].extend(changes);counts['ULF_observation_endpoint_positive']+=1
                secondary=cme.get((base,row['tier']),[])
                if secondary and 'missing_observation' not in row['exclusions']:
                    row['target'][3]=row['valid'][3]=True;row['strict_valid'][3]=False;row['evidence'].extend(secondary);counts['CME_secondary_weak_time_positive']+=1
                for bit,code in enumerate(protocol['classes']):
                    key=row['family']+'|'+row['tier']+'|'+code
                    if row['valid'][bit]:coverage[key]['valid']+=1
                    else:coverage[key]['unknown']+=1
                    if row['valid'][bit] and row['target'][bit]:coverage[key]['positive']+=1
                out.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        summary[split]=dict(counts=counts,coverage=coverage);print(json.dumps(dict(stage='time_targets_enriched',split=split,counts=counts)),flush=True)
    protocol['parent_protocol_sha256']=hashlib.sha256((args.input/'protocol.json').read_bytes()).hexdigest()
    protocol['summary_before_time_enrichment']=protocol['summary'];protocol['summary']=summary
    protocol['time_supervision']={'ULF':'same-CSV P0-relative observation start/end declaration; not event onset',
        'CME':'same-image unique unchanged speed-triplet candidate identity; excluded from strict-mask metrics',
        'scope':'Fine L2-T is reported as scoped time-declaration technical proxy; independent event endpoint recall unavailable'}
    protocol['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);main(ap.parse_args())
