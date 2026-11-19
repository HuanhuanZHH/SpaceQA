"""Offline release viewer grounded in native immutable evaluation artifacts."""
import argparse,csv,hashlib,http.server,json,threading
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
VERSION='0.6.1-release3'
PAGE="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SpaceQA 交付结果核查</title><style>body{font:17px system-ui;background:#f4f7fa;color:#16324f;max-width:1200px;margin:35px auto;padding:0 20px}h1{font-size:30px}button,select,input{font:inherit;padding:8px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #c7d4df;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:18px}a{color:#0069a6}small{color:#485a70}</style><h1>SpaceQA · 交付结果核查</h1><p>版本 0.6.1-release3。页面读取已完成的新输入评价；确认规则缺陷与模型候选分别展示。</p><p><select id="runs"></select> <button id="load">读取运行</button> <select id="stage"><option value="">全部问题</option><option value="confirmed">规则确认</option><option value="candidate">模型候选</option></select></p><pre id="summary">正在读取运行清单</pre><table><thead><tr><th>检查</th><th>样本</th><th>结论边界</th><th>证据</th></tr></thead><tbody id="rows"></tbody></table><h2>根据本轮记录问答</h2><p><input id="question" size="65" placeholder="例如：本轮哪些结论仍待独立复核？"><button id="ask">查询记录</button></p><small>这是可追溯的记录摘要，不调用外部大模型，也不把候选判断为科学真值。</small><pre id="answer"></pre><h2>证据内容</h2><pre id="evidence">点击问题记录的证据链接。</pre><script src="/release.js"></script></html>"""
JS="""let current='';const el=id=>document.getElementById(id);async function get(p){let r=await fetch(p);if(!r.ok)throw Error(await r.text());return r.json()}async function show(){current=el('runs').value;if(!current)return;let c=await get('/api/runs/'+current);el('summary').textContent=JSON.stringify({runId:c.runId,dirId:c.dirId,sampleCount:c.sampleCount,observedTechnicalScore:c.observedTechnicalScore,fullQualityScore:c.fullQualityScore,reason:c.fullQualityScoreReason},null,2);let f=await get('/api/runs/'+current+'/findings?stage='+el('stage').value);el('rows').replaceChildren();for(let x of f.items.slice(0,100)){let tr=document.createElement('tr');for(let v of [x.checkId,x.sampleId,x.confirmedOrPending==='confirmed'?'规则确认（非独立科学真值）':'候选，待复核']){let td=document.createElement('td');td.textContent=v;tr.append(td)}let td=document.createElement('td');let a=document.createElement('button');a.textContent=x.findingId;a.onclick=async()=>el('evidence').textContent=JSON.stringify(await get('/api/runs/'+current+'/evidence/'+x.evidenceRef[0]),null,2);td.append(a);tr.append(td);el('rows').append(tr)}}el('load').onclick=show;el('stage').onchange=show;el('ask').onclick=async()=>{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runId:current,question:el('question').value})});el('answer').textContent=JSON.stringify(await r.json(),null,2)};get('/api/runs').then(x=>{for(let r of x.items){let o=document.createElement('option');o.value=r.id;o.textContent=r.dirId+' · '+r.runId;el('runs').append(o)}if(x.items.length)show();else el('summary').textContent='没有已完成运行。先执行 evaluate，随后将输出目录通过 --run-root 挂载。'}).catch(e=>el('summary').textContent=e.message);"""
def create_server(host,port,roots):
 if host not in ('127.0.0.1','localhost','::1'):raise ValueError('Loopback binding required')
 roots=[Path(r).resolve() for r in roots];runs={}
 for root in roots:
  for p in sorted(root.rglob('complete.json')):
   if not (p.parent/'evidence_index.csv').is_file() or not (p.parent/'scorecard.json').is_file():continue
   c=json.loads(p.read_text())
   if c.get('status')!='complete':continue
   ident=hashlib.sha256(str(p).encode()).hexdigest()[:20];runs[ident]=p.parent
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,code,body,content='application/json'):
   raw=json.dumps(body,ensure_ascii=False).encode() if content=='application/json' else body.encode();self.send_response(code);self.send_header('Content-Type',content+'; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
  def do_GET(self):
   try:self.get()
   except (OSError,ValueError,KeyError):self.send(400,{'error':'Invalid or changed release artifact'})
  def get(self):
   u=urlsplit(self.path);parts=u.path.strip('/').split('/');query=parse_qs(u.query)
   if u.path in ('/','/chat.html','/evaluation.html','/demo.html'):return self.send(200,PAGE,'text/html')
   if u.path=='/release.js':return self.send(200,JS,'application/javascript')
   if u.path=='/api/health':return self.send(200,{'version':VERSION,'mode':'offline immutable result viewer','externalApiCalls':0})
   if u.path=='/api/runs':return self.send(200,{'items':[dict(id=k,runId=json.loads((v/'complete.json').read_text())['runId'],dirId=json.loads((v/'scorecard.json').read_text())['dirId']) for k,v in runs.items()]})
   if len(parts)>=3 and parts[:2]==['api','runs'] and parts[2] in runs:
    root=runs[parts[2]]
    if len(parts)==3:return self.send(200,json.loads((root/'scorecard.json').read_text()))
    if len(parts)==4 and parts[3]=='findings':
     rows=[json.loads(x) for x in (root/'findings.jsonl').read_text().splitlines()]
     for key,field in [('stage','findingStage'),('checkId','checkId'),('defectType','defectType')]:
      if query.get(key):rows=[r for r in rows if r.get(field)==query[key][0]]
     return self.send(200,{'total':len(rows),'items':rows})
    if len(parts)==5 and parts[3]=='evidence':
     index={e['evidenceId']:e for e in csv.DictReader((root/'evidence_index.csv').open())};e=index[parts[4]];p=(root/e['path']).resolve()
     if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest()!=e['contentHash']:return self.send(409,{'error':'Evidence hash validation failed'})
     return self.send(200,json.loads(p.read_text()))
   self.send(404,{'error':'Unknown release endpoint'})
  def do_POST(self):
   if self.path!='/api/chat':return self.send(404,{'error':'Unknown operation'})
   host=self.headers.get('Host','');origin=self.headers.get('Origin')
   if origin and origin not in ('http://'+host,'https://'+host):return self.send(403,{'error':'Cross-origin request rejected'})
   try:
    n=int(self.headers.get('Content-Length','0'))
    if not 0<n<=16384:return self.send(413,{'error':'Invalid request size'})
    body=json.loads(self.rfile.read(n));root=runs[body['runId']];c=json.loads((root/'scorecard.json').read_text());rows=[json.loads(x) for x in (root/'findings.jsonl').read_text().splitlines()];confirmed=sum(bool(r.get('scoreEligible')) for r in rows)
    return self.send(200,{'runId':c['runId'],'question':str(body.get('question',''))[:1000],'answer':'本轮共 '+str(c['sampleCount'])+' 个登记样本，'+str(confirmed)+' 条规则确认记录和 '+str(len(rows)-confirmed)+' 条候选记录。技术分为 '+str(c['observedTechnicalScore'])+'。完整六维分为 '+str(c.get('fullQualityScore'))+'。候选与类别契约检查不能替代独立科学真值。','source':'scorecard.json + findings.jsonl','mode':'deterministic grounded summary','evidenceLinks':['/api/runs/'+body['runId']+'/evidence/'+r['evidenceRef'][0] for r in rows[:3]],'tokenUsage':0})
   except (OSError,ValueError,KeyError,TypeError):self.send(400,{'error':'Invalid run or request'})
 return http.server.ThreadingHTTPServer((host,port),Handler)
def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8765);ap.add_argument('--run-root',action='append',required=True);a=ap.parse_args(argv);server=create_server(a.host,a.port,a.run_root)
 try:server.serve_forever()
 finally:server.server_close()
if __name__=='__main__':main()
