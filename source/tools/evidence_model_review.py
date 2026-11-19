"""Evidence before classification: stronger zero-shot prompt control."""
import base64,concurrent.futures,json,time,re
from urllib.request import Request,urlopen
from learn_model_comparison import OUT,load,metrics,write
from batch_and_teacher import KEY
from pathlib import Path


def invoke(row):
    prompt=(row['prompt']+' For this review, first state one short visible observation in English (at most 20 words), '
        'then choose the class in the required JSON. Do not infer omission from ordinary unannotated texture. '
        'An empty annotation is not by itself proof of an omitted scientific event. A claimed false annotation '
        'requires image evidence of absence. If scientific evidence is insufficient choose A. '
        'Compare declared versus decoded dimensions directly for D. Do not use an absent clean reference. '
        'Output JSON only with keys evidence (one short sentence), class (one letter A,B,C,D).')
    body=dict(model='Qwen3-VL-8B-Instruct',temperature=0,max_tokens=192,
              messages=[{'role':'user','content':[{'type':'text','text':prompt},
                  {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(Path(row['image']).read_bytes()).decode()}}]}])
    began=time.monotonic()
    try:
        req=Request('http://127.0.0.1:8002/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
        with urlopen(req,timeout=180) as response:data=json.load(response)
        content=data['choices'][0]['message']['content'];matched=re.findall(r'"class"\s*:\s*"([ABCD])"',content)
        if not matched and content.strip() in ('A','B','C','D'):matched=[content.strip()]
        if not matched:raise ValueError('missing_class')
        prediction='ABCD'.index(matched[-1])
        return dict(id=row['id'],target=row['target'],prediction=prediction,success=True,evidence=content,seconds=time.monotonic()-began)
    except Exception as exc:return dict(id=row['id'],success=False,error=type(exc).__name__)


def main():
    _,val,test=load();extra=[json.loads(s) for s in (OUT/'extended_test.jsonl').open()]
    results=[]
    with (OUT/'evidence_review_predictions.jsonl').open('w') as f,concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        for i,r in enumerate(pool.map(invoke,val+test+extra)):
            results.append(r);f.write(json.dumps(r)+'\n');f.flush()
            if i%100==0:print('evidence review',i,flush=True)
    predictions={r['id']:r['prediction'] for r in results if r['success']}
    write(OUT/'evidence_review_summary.json',dict(validation=metrics(val,predictions),metrics=metrics(test,predictions),extended_metrics=metrics(extra,predictions),failures=sum(not r['success'] for r in results)))

if __name__=='__main__':main()
