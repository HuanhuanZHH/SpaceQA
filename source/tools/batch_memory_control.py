"""Same seq32 payloads, original 35% memory budget: one-factor memory control."""
import concurrent.futures,json,os,random,subprocess,threading,time
from pathlib import Path
from batch_and_teacher import ROOT,OUT,request


def launch():
    env=os.environ.copy();env.update(QWEN_GPU='1',QWEN_PORT='8002',QWEN_GPU_MEMORY_UTILIZATION='0.35',QWEN_MAX_NUM_SEQS='32',QWEN_MIN_FREE_MIB='14000')
    with (OUT/'batch_mem35_server.log').open('w') as f:
        p=subprocess.Popen(['bash',str(ROOT/'opc-agent/deploy/qwen3_vl/serve.sh')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (OUT/'batch_mem35_server.pid').write_text(str(p.pid));print('memory control server',p.pid)


def run():
    runtime=OUT.parent/'model-comparison-20261031'
    poolrows=[json.loads(s) for s in (runtime/'dataset.jsonl').open()]
    warmup=sorted((r for r in poolrows if r['split']=='test'),key=lambda r:r['id'])[0]
    assert request(warmup,8002,True)['success']
    random.Random(20261031).shuffle(poolrows)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            mem=int(subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            samples.append((time.monotonic(),mem));stop.wait(.1)
    t=threading.Thread(target=monitor,daemon=True);t.start()
    try:
        with (runtime/'batch_seq32_mem35_review_control.jsonl').open('w') as f:
            for repeat in range(2):
                offset=(repeat*6+5)*32
                rows=[dict(r,prompt=f'Load-case seq32_review_unique-{repeat}-32-{r["id"]}. '+r['prompt']) for r in poolrows[offset:offset+32]]
                begin=time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
                    results=list(pool.map(lambda row:request(row,8002,True),rows))
                wall=time.monotonic()-begin;elapsed=sorted(r['elapsed'] for r in results)
                value=dict(tag='seq32_mem35_review_control',review=True,unique_inputs=True,repeat=repeat,
                    concurrency=32,requests=32,success=sum(r['success'] for r in results),wall_seconds=wall,
                    requests_per_second=32/wall,latency_p50=(elapsed[15]+elapsed[16])/2,latency_p95=elapsed[29],
                    peak_memory_mib=max((m for tm,m in samples if tm>=begin),default=None),results=results)
                f.write(json.dumps(value)+'\n');f.flush();print(json.dumps({k:v for k,v in value.items() if k!='results'}),flush=True)
    finally:stop.set();t.join(timeout=2)

if __name__=='__main__':
    import sys
    launch() if sys.argv[1]=='launch' else run()
