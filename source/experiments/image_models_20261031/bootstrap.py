"""Isolate image-model dependencies and download only public remote weights."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT=Path('/staff/tangjiarui/OPC')
OUT=ROOT/'opc-agent/runs/image-models-20261031'
VENV=ROOT/'.venv-image-models'
PY=VENV/'bin/python'
MODELS=ROOT/'models/image-model-pilot'
URLS={
 'yolo11n.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
 'yolo11n-seg.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt',
 'yolov8n-seg.pt':'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt',
 'resnet18-f37072fd.pth':'https://download.pytorch.org/models/resnet18-f37072fd.pth',
 'deeplabv3_mobilenet_v3_large-fc3c493d.pth':'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth',
}


def main():
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    if not PY.exists():subprocess.run([sys.executable,'-m','venv',str(VENV)],check=True)
    site=next((VENV/'lib').glob('python*/site-packages'))
    qsite=next((ROOT/'.venv-qwen3vl/lib').glob('python*/site-packages'))
    (site/'opc_existing_torch.pth').write_text(str(qsite)+'\n')
    env={k:v for k,v in os.environ.items() if 'proxy' not in k.lower()}
    env.update(YOLO_CONFIG_DIR=str(OUT/'ultralytics_config'),YOLO_AUTOINSTALL='false')
    packages=['ultralytics==8.3.228','ultralytics-thop','matplotlib','pandas','contourpy','cycler',
              'fonttools','kiwisolver','pyparsing','python-dateutil','pytz','tzdata','six','polars','polars-runtime-32']
    subprocess.run([str(ROOT.parent/'.local/bin/uv'),'pip','install','--python',str(PY),'--no-deps',
        '--index-url','https://pypi.tuna.tsinghua.edu.cn/simple',*packages],env=env,check=True,timeout=300)
    subprocess.run([str(PY),'-c','import ultralytics,torch,torchvision,pandas,matplotlib;print(ultralytics.__version__,torch.__version__,torchvision.__version__)'],check=True,env=env)
    session=requests.Session();session.trust_env=False;manifest=[]
    for name,url in URLS.items():
        p=MODELS/name
        if not p.exists():
            with session.get(url,stream=True,timeout=(15,90)) as r:
                r.raise_for_status()
                with p.with_suffix(p.suffix+'.partial').open('wb') as f:
                    for chunk in r.iter_content(1024*1024):f.write(chunk)
            p.with_suffix(p.suffix+'.partial').replace(p)
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        if 'f37072fd' in name:assert sha.startswith('f37072fd')
        if 'fc3c493d' in name:assert sha.startswith('fc3c493d')
        manifest.append(dict(file=name,source_url=url,bytes=p.stat().st_size,sha256=sha))
        print(json.dumps(manifest[-1]),flush=True)
    (OUT/'downloads.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'bootstrap_complete.json').write_text(json.dumps(dict(python=str(PY),packages=packages,weight_files=len(manifest)),indent=2))


if __name__=='__main__':main()
