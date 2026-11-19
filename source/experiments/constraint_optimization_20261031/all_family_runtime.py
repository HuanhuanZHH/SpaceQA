"""Sampled all-image-family runtime, fresh raw decoding and overlay rendering.

This is a throughput experiment. Responses are not a defect accuracy benchmark.
"""
import base64
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/constraint-optimization-20261031'
MANIFEST=ROOT.parent/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
REGEX=r'\{"decision":"(confirm|reject|uncertain)","defect_class_id":"(NONE|L2-G|L2-P|L2-T|L2-F|L2-M|L3-06|L3-07|L3-MIS)","confidence":(0\.0|0\.25|0\.5|0\.75|1\.0),"evidence":"(insufficient_context|no_supported_defect|boundary_mismatch|label_mismatch|unlabeled_candidate|unsupported_label|coverage_mismatch)","needs_human_review":(true|false)\}'


def prepare(task):
    row,size=task; start=time.monotonic()
    value=json.loads(Path(row['label_path']).read_text())
    with Image.open(row['data_path']) as im: raw=im.convert('RGB')
    original=raw.size; raw.thumbnail((size//2,size//2),Image.Resampling.LANCZOS)
    over=raw.copy();draw=ImageDraw.Draw(over)
    entries=[]
    sci=value.get('science_annotation',{})
    entries.extend(sci.get('specialShapes',[]));entries.extend(sci.get('mask',{}).get('shapes',[]))
    for k,v in value.items():
        if k.endswith('event_annotation') and isinstance(v,dict):entries.extend(v.get('matched_events',[]))
    for item in entries:
        pp=item.get('points',item.get('bbox_polygon_pixel'))
        if not isinstance(pp,list) or len(pp)<2: continue
        pp=[(float(p[0])*raw.width/original[0],float(p[1])*raw.height/original[1]) for p in pp]
        if item.get('shape_type')=='rectangle' and len(pp)==2:
            xs=sorted(p[0] for p in pp);ys=sorted(p[1] for p in pp)
            draw.rectangle((xs[0],ys[0],xs[1],ys[1]),outline='red',width=2)
        else: draw.line(pp+[pp[0]],fill='red',width=2)
    canvas=Image.new('RGB',(size,size//2),(24,24,24))
    canvas.paste(raw,((size//2-raw.width)//2,(size//2-raw.height)//2))
    canvas.paste(over,(size//2+(size//2-over.width)//2,(size//2-over.height)//2))
    buf=io.BytesIO();canvas.save(buf,format='JPEG',quality=92);payload=buf.getvalue()
    labels=[e.get('label',e.get('event_type',e.get('kind'))) for e in entries]
    basic=value.get('basic_annotation',{})
    prompt=(f"Review {row['dataset_family']} annotation. Left=raw observation, right=current red overlay. "
            f"Image original dimensions={original}; declared imgSize={basic.get('imgSize')}; "
            f"current classes={json.dumps(labels)}. No trusted clean image is given. "
            "Return JSON with decision, defect_class_id, confidence, evidence, needs_human_review. "
            "Do not infer time or metadata mismatches from image appearance. Use uncertain when evidence is insufficient.")
    return dict(id=row['sample_id'],family=row['dataset_family'],image=base64.b64encode(payload).decode(),
                image_sha256=hashlib.sha256(payload).hexdigest(),prompt=prompt,
                preprocessing_seconds=time.monotonic()-start)


def request(row,key):
    start=time.monotonic()
    body=dict(model='Qwen3-VL-8B-Instruct',temperature=0,max_tokens=128,
              structured_outputs={'regex':REGEX},messages=[{'role':'user','content':[
                  {'type':'text','text':row['prompt']},
                  {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+row['image']}}]}])
    req=Request('http://127.0.0.1:8002/v1/chat/completions',data=json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=180) as f:value=json.load(f)
        result=json.loads(value['choices'][0]['message']['content'])
        assert isinstance(result,dict) and 'decision' in result
        return dict(id=row['id'],success=True,seconds=time.monotonic()-start,
                    usage=value.get('usage'),response=result)
    except Exception as exc:
        return dict(id=row['id'],success=False,seconds=time.monotonic()-start,error=str(exc)[:250])


def main():
    key=(ROOT.parent/'.qwen3vl-api-key').read_text().strip()
    counts=collections.Counter();pool=collections.defaultdict(list)
    for line in MANIFEST.open():
        r=json.loads(line); counts[r['package_id']]+=1
        if r['tier']=='P1' and r['data_exists'] and r['dataset_family']!='ULF_sample':
            pool[r['dataset_family']].append(r)
    stop=threading.Event();memory=[]
    def monitor():
        while not stop.is_set():
            memory.append(int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    results=[]
    try:
        for size in (768,1536):
            for family,rows in sorted(pool.items()):
                # 64 distinct raw samples per family; no synthetic/cached preview input.
                rows=sorted(rows,key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())[:64]
                t=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                    prepared=list(ex.map(prepare,[(r,size) for r in rows]))
                prep=time.monotonic()-t
                t=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
                    responses=list(ex.map(lambda r:request(r,key),prepared))
                inference=time.monotonic()-t
                row=dict(family=family,composite_size=[size,size//2],requests=len(rows),
                         unique_input_bytes=len({r['image_sha256'] for r in prepared}),
                         preprocessing_wall_seconds=prep,inference_wall_seconds=inference,
                         combined_seconds=prep+inference,requests_per_second=len(rows)/inference,
                         end_to_end_samples_per_second=len(rows)/(prep+inference),
                         success=sum(r['success'] for r in responses),peak_memory_mib=max(memory),
                         latency_p95_seconds=float(sorted(r['seconds'] for r in responses)[60]),responses=responses)
                results.append(row)
                (OUT/'all_family_vlm_runtime.json').write_text(json.dumps(dict(results=results,
                      scope='64 P1 images/family/size; raw read+decode+render+API inference; no full-package model run',
                      hardware='A100 40GB; original pretrained NF4 runtime, max_num_seqs32, memory utilization0.35',
                      notes=['Not an accuracy benchmark.','Default caches enabled; first cohorts may include grammar/cache warmup.',
                             'Largest-package budget must also include cold model load, full rules, evidence output and audit.',
                             'No A10 hardware validation. ULF uses deterministic CSV checks, not a VLM per CSV.']),indent=2))
                print(json.dumps({k:v for k,v in row.items() if k!='responses'}),flush=True)
    finally:stop.set();thread.join(timeout=2)


if __name__=='__main__':main()
