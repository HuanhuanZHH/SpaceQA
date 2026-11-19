"""Public immutable I-JEPA checkpoint, remote only, with download hashes."""
import hashlib
import json
from pathlib import Path

import requests

ROOT=Path('/staff/tangjiarui/OPC')
MODEL=ROOT/'models/ijepa_vith14_1k'
MODEL_ID='facebook/ijepa_vith14_1k'

def main():
    MODEL.mkdir(parents=True,exist_ok=True);s=requests.Session();s.trust_env=False
    info=s.get('https://hf-mirror.com/api/models/'+MODEL_ID+'?blobs=true',timeout=45);info.raise_for_status();info=info.json()
    revision=info['sha'];manifest=[]
    upstream={r['rfilename']:r for r in info['siblings']}
    mirror=s.get('https://modelscope.cn/api/v1/models/'+MODEL_ID+'/repo/files?Revision=master&Root=',timeout=30)
    mirror.raise_for_status();mirror={r['Name']:r for r in mirror.json()['Data']['Files']}
    assert upstream['model.safetensors']['lfs']['sha256']==mirror['model.safetensors']['Sha256']
    for name in ['config.json','preprocessor_config.json','model.safetensors']:
        p=MODEL/name
        if not p.exists():
            # The domestic mirror exposes an independently verifiable SHA256
            # identical to the official HF LFS object. No gated models bypassed.
            mirror_revision=mirror[name]['Revision']
            with s.get(f'https://modelscope.cn/models/{MODEL_ID}/resolve/{mirror_revision}/{name}',stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with p.with_suffix(p.suffix+'.partial').open('wb') as f:
                    for chunk in r.iter_content(4*1024*1024):f.write(chunk)
            p.with_suffix(p.suffix+'.partial').replace(p)
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==mirror[name]['Sha256'],name+' mirror hash mismatch'
        manifest.append(dict(file=name,bytes=p.stat().st_size,sha256=h.hexdigest(),mirror_revision=mirror[name]['Revision']));print(json.dumps(manifest[-1]),flush=True)
    (MODEL/'download_manifest.json').write_text(json.dumps(dict(model_id=MODEL_ID,revision=revision,
        license=info.get('cardData',{}).get('license'),files=manifest),indent=2))

if __name__=='__main__':main()
