"""Constrained class inference, real-input load sweeps and teacher outputs."""
import argparse
import base64
import concurrent.futures
import json
import math
import statistics
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path('/staff/tangjiarui/OPC')
OUT = ROOT/'opc-agent/runs/model-comparison-20261031-v2'
KEY = (ROOT/'.qwen3vl-api-key').read_text().strip()


def request(row, port, review=False):
    body = dict(model='Qwen3-VL-8B-Instruct', temperature=0, max_tokens=1,
                logprobs=True, top_logprobs=5,
                structured_outputs={'regex': '[ABCD]'},
                messages=[{'role':'user','content':[
                    {'type':'text','text':row['prompt']},
                    {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+
                        base64.b64encode(Path(row['image']).read_bytes()).decode()}}]}])
    if review:
        body['max_tokens']=128
        body['structured_outputs']={'regex':r'\{"decision":"(confirm|reject|uncertain)","defect_class_id":"(NONE|L2-G|L2-P|L3-06|L3-07|L3-MIS)","confidence":(0\.0|0\.25|0\.5|0\.75|1\.0),"evidence":"(insufficient_context|no_supported_defect|boundary_mismatch|label_mismatch|unlabeled_candidate|unsupported_label|coverage_mismatch)","needs_human_review":(true|false)\}'}
        body['messages'][0]['content'][0]['text']=(row['prompt']+
            ' Instead of a letter, return the required JSON review fields with an evidence code.')
    started = time.monotonic()
    req = Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(),
                  headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
    try:
        with urlopen(req, timeout=180) as response:
            result = json.load(response)
        choice = result['choices'][0]
        letter = choice['message']['content'].strip()
        lp = choice.get('logprobs',{}).get('content',[])
        probabilities = [1e-8]*4
        if lp:
            for candidate in lp[0].get('top_logprobs',[]):
                token = candidate['token'].strip()
                if token in ('A','B','C','D'):
                    probabilities['ABCD'.index(token)] = math.exp(candidate['logprob'])
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        valid=isinstance(json.loads(letter),dict) if review else letter in ('A','B','C','D')
        return dict(id=row['id'], target=row['target'], prediction='ABCD'.find(letter),
                    probabilities=probabilities, success=valid,
                    elapsed=time.monotonic()-started, usage=result.get('usage'))
    except Exception as exc:
        return dict(id=row['id'], success=False, error=type(exc).__name__, elapsed=time.monotonic()-started)


def run_teacher(port,workers=2):
    rows = [json.loads(s) for s in (OUT/'dataset.jsonl').open()]
    # Keep all real test records. Teacher soft targets only use a fixed class-balanced training subset.
    import random
    rng = random.Random(20261031)
    selected = [r for r in rows if r['split'] in ('test','val')]
    for label in range(4):
        pool = [r for r in rows if r['split']=='train' and r['target']==label]
        rng.shuffle(pool)
        selected.extend(pool[:100])
    (OUT/'teacher_selection.json').write_text(json.dumps([r['id'] for r in selected]))
    path = OUT/'teacher_predictions.jsonl'
    done = {json.loads(s)['id'] for s in path.open()} if path.exists() else set()
    with path.open('a') as f, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(lambda r:request(r,port),[r for r in selected if r['id'] not in done])):
            f.write(json.dumps(result)+'\n'); f.flush()
            if i % 50 == 0:
                print('teacher',i,result['success'],flush=True)


def sweep(port, gpu, tag, review=False, unique=False):
    import random
    poolrows = [json.loads(s) for s in (OUT/'dataset.jsonl').open()]
    rows = [r for r in poolrows if r['split']=='test']
    # Fixed 32 real inputs per concurrency level, repeated to estimate variability.
    rows = sorted(rows,key=lambda r:r['id'])[:32]
    random.Random(20261031).shuffle(poolrows)
    request(rows[0],port,review)
    path = OUT/f'batch_{tag}.jsonl'
    stop = threading.Event()
    samples = []
    def monitor():
        while not stop.is_set():
            result = subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                    '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples.append([time.monotonic()]+[float(x) for x in result.strip().split(',')])
            stop.wait(.5)
    thread = threading.Thread(target=monitor,daemon=True); thread.start()
    try:
        with path.open('w') as f:
            for repeat in range(2):
                for concurrency in [1,2,4,8,16,32]:
                    if unique:
                        offset=(repeat*6+[1,2,4,8,16,32].index(concurrency))*32
                        rows=[dict(r,prompt=f'Load-case {tag}-{repeat}-{concurrency}-{r["id"]}. '+r['prompt'])
                              for r in poolrows[offset:offset+32]]
                    begin = time.monotonic()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        results = list(pool.map(lambda r:request(r,port,review),rows))
                    wall = time.monotonic()-begin
                    current = [s for s in samples if s[0]>=begin]
                    elapsed = sorted(r['elapsed'] for r in results)
                    value = dict(port=port,tag=tag,review=review,unique_inputs=unique,repeat=repeat,concurrency=concurrency,requests=len(rows),
                                 success=sum(r['success'] for r in results),wall_seconds=wall,
                                 requests_per_second=len(rows)/wall,
                                 latency_p50=statistics.median(elapsed),latency_p95=elapsed[int(.95*(len(elapsed)-1))],
                                 peak_memory_mib=max((s[1] for s in current),default=None),
                                 peak_gpu_utilization=max((s[2] for s in current),default=None),
                                 results=results)
                    f.write(json.dumps(value)+'\n'); f.flush()
                    print(json.dumps({k:v for k,v in value.items() if k!='results'}),flush=True)
    finally:
        stop.set(); thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['teacher','sweep','launch_batch_server'])
    parser.add_argument('--port',type=int,default=8001)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--tag',default='existing_seq2')
    parser.add_argument('--review',action='store_true')
    parser.add_argument('--unique',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='teacher':run_teacher(args.port,args.workers)
    elif args.mode=='sweep':sweep(args.port,args.gpu,args.tag,args.review,args.unique)
    else:
        import os
        env=os.environ.copy()
        env.update(QWEN_GPU='1',QWEN_PORT='8002',QWEN_GPU_MEMORY_UTILIZATION='0.52',
                   QWEN_MAX_NUM_SEQS='32',QWEN_MIN_FREE_MIB='22000')
        with (OUT/'batch_server.log').open('w') as f:
            process=subprocess.Popen(['bash',str(ROOT/'opc-agent/deploy/qwen3_vl/serve.sh')],
                                     env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        (OUT/'batch_server.pid').write_text(str(process.pid))
        print('isolated server PID',process.pid)

if __name__=='__main__':main()
