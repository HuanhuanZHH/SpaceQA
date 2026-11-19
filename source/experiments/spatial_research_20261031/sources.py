"""Fetch official sources and public small weights directly on the server."""
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

import requests

ROOT = Path('/staff/tangjiarui/OPC')
OUT = ROOT/'opc-agent/runs/spatial-research-20261031'
MODEL = ROOT/'models/dinov2-small'
SOURCES = {
    'ijepa': 'https://raw.githubusercontent.com/facebookresearch/ijepa/main/README.md',
    'vjepa2': 'https://raw.githubusercontent.com/facebookresearch/vjepa2/main/README.md',
    'dinov2': 'https://raw.githubusercontent.com/facebookresearch/dinov2/main/README.md',
    'dinov3': 'https://raw.githubusercontent.com/facebookresearch/dinov3/main/README.md',
    'sam2': 'https://raw.githubusercontent.com/facebookresearch/sam2/main/README.md',
    'efficientad': 'https://raw.githubusercontent.com/nelson1425/EfficientAD/main/README.md',
    'dinov2_model_card': 'https://huggingface.co/facebook/dinov2-small/raw/main/README.md',
    'ijepa_model_card': 'https://huggingface.co/facebook/ijepa_vith14_1k/raw/main/README.md',
    'dinov3_model_card': 'https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/raw/main/README.md',
    'vjepa2_model_card': 'https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/raw/main/README.md',
}


def session():
    s=requests.Session();s.trust_env=False
    return s


def get(s,url,**kwargs):
    try:
        r=s.get(url,**kwargs);r.raise_for_status();return r
    except requests.RequestException:
        if not url.startswith('https://huggingface.co/'):raise
        r=s.get(url.replace('https://huggingface.co/','https://hf-mirror.com/',1),**kwargs)
        r.raise_for_status();return r


def main():
    (OUT/'literature').mkdir(parents=True,exist_ok=True)
    MODEL.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,url=item; start=time.monotonic()
        try:
            r=get(session(),url,timeout=(10,30))
            p=OUT/'literature'/(name+'.md');p.write_text(r.text)
            return dict(name=name,url=url,resolved_url=r.url,status='downloaded',bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),seconds=time.monotonic()-start)
        except Exception as e:return dict(name=name,url=url,status='failed',error=str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        records=list(pool.map(fetch,SOURCES.items()))
    (OUT/'literature/sources.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(dict(source_results=records)),flush=True)
    # Immutable resolved revision, streamed download, and recorded hashes.
    s=session()
    api=get(s,'https://huggingface.co/api/models/facebook/dinov2-small',timeout=30)
    api.raise_for_status();info=api.json();revision=info['sha']
    (MODEL/'source_metadata.json').write_text(json.dumps(dict(model_id=info['id'],revision=revision,
        license=info.get('cardData',{}).get('license'),downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
    names={x['rfilename'] for x in info['siblings']}
    weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
    files=['config.json','preprocessor_config.json',weight]
    manifest=[]
    for name in files:
        path=MODEL/name
        if not path.exists():
            url=f'https://huggingface.co/facebook/dinov2-small/resolve/{revision}/{name}'
            with get(s,url,timeout=(15,90),stream=True) as r:
                r.raise_for_status()
                with path.with_suffix(path.suffix+'.partial').open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
            path.with_suffix(path.suffix+'.partial').replace(path)
        manifest.append(dict(file=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps(manifest[-1]),flush=True)
    (MODEL/'download_manifest.json').write_text(json.dumps(dict(revision=revision,files=manifest),indent=2))
    (OUT/'model_download.json').write_text(json.dumps(dict(path=str(MODEL),revision=revision,files=manifest),indent=2))


if __name__=='__main__':main()
