"""Download the paper's generic base from the official ModelScope repository."""
import concurrent.futures,hashlib,json,time,urllib.parse,urllib.request
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC');WORK=ROOT/'jwvl_optimization_20261104'
DEST=ROOT/'models/Qwen2.5-VL-32B-Instruct';DEST.mkdir(parents=True,exist_ok=True)
META=WORK/'modelscope32_metadata.json'
files=json.loads(META.read_text())['Data']['Files']
selected=[r for r in files if r['Type']=='blob' and (r['Path'].endswith(('.safetensors','.json','.txt','.jinja','.model')) or r['Path'] in ['README.md','LICENSE'])]
# Some ModelScope responses use Type='file'; keep a documented allowlist.
if not selected:selected=[r for r in files if r.get('Size',0)>0 and r['Path'].endswith(('.safetensors','.json','.txt','.jinja','.model','.md'))]
assert len([r for r in selected if r['Path'].endswith('.safetensors')])==18

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def fetch(r):
 rel=Path(r['Path']);assert not rel.is_absolute() and '..' not in rel.parts
 p=DEST/rel;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
 expected=r['Sha256'];size=r['Size']
 if p.exists() and p.stat().st_size==size and sha(p)==expected:return {'file':r['Path'],'status':'already_verified','bytes':size}
 url='https://www.modelscope.cn/api/v1/models/Qwen/Qwen2.5-VL-32B-Instruct/repo?'+urllib.parse.urlencode({'Revision':'master','FilePath':r['Path']})
 last=None
 for attempt in range(4):
  try:
   opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));offset=tmp.stat().st_size if tmp.exists() else 0
   req=urllib.request.Request(url,headers={'User-Agent':'SpaceQA-research/1',**({'Range':f'bytes={offset}-'} if offset else {})})
   with opener.open(req,timeout=90) as response:
    if offset and response.status!=206:offset=0
    if offset:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
    with tmp.open('ab' if offset else 'wb') as out:
     while True:
      b=response.read(4194304)
      if not b:break
      out.write(b)
   assert tmp.stat().st_size==size,(tmp.stat().st_size,size)
   assert sha(tmp)==expected,'Hash mismatch'
   tmp.replace(p);print(json.dumps({'file':r['Path'],'status':'verified','bytes':size}),flush=True)
   return {'file':r['Path'],'status':'verified','bytes':size,'sha256':expected}
  except Exception as ex:
   last=ex;print(json.dumps({'file':r['Path'],'attempt':attempt+1,'error':type(ex).__name__+': '+str(ex)}),flush=True)
   if tmp.exists() and (tmp.stat().st_size>=size):tmp.unlink()
   time.sleep(3)
 raise last

start=time.time()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(fetch,selected))
 (WORK/'qwen32_download.json').write_text(json.dumps({'status':'complete','repository':'Qwen/Qwen2.5-VL-32B-Instruct','provider':'official ModelScope','destination':str(DEST),'bytes':sum(r['bytes'] for r in result),'seconds':time.time()-start,'metadata_sha256':sha(META),'files':result},indent=2))
except BaseException as ex:
 (WORK/'qwen32_download.json').write_text(json.dumps({'status':'failed','reason':type(ex).__name__+': '+str(ex),'seconds':time.time()-start}));raise
