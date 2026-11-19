import json
from pathlib import Path
import subprocess

HERE=Path(__file__).parent
DEST='/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105'
subprocess.run(['ssh','icarus0','mkdir','-p',DEST],check=True)
subprocess.run(['scp',str(HERE/'ground_all_refresh.py'),str(HERE/'run_refresh.py'),'icarus0:'+DEST+'/'],check=True)
script='''import json,os,subprocess
from pathlib import Path
base=Path('/staff/tangjiarui/OPC/eight_class_20261104')
out=base/'refresh_20261105'
assert not out.exists(), 'New run already exists; never overwrite an earlier run'
log=base/'refresh_20261105.log'
env=dict(os.environ)
for key in list(env):
    if 'API_KEY' in key or key.startswith('SPACEQA_CHAT_LLM_'):env.pop(key,None)
env.update(CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1')
with log.open('xb') as stream:
    process=subprocess.Popen(['/staff/tangjiarui/OPC/.venv-image-models/bin/python','-u',str(base/'refresh_code_20261105/run_refresh.py')],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
value={'pid':process.pid,'log':str(log),'output':str(out),'gpu':1}
(base/'refresh_20261105_launch.json').write_text(json.dumps(value,indent=2))
print(json.dumps(value))
'''
result=subprocess.run(['ssh','icarus0','/usr/bin/python3','-'],input=script,capture_output=True,text=True,encoding='utf-8',check=True)
print(result.stdout)
Path('D:/OPC/outputs/chat_insights_20261105/fresh_inference_launch.json').write_text(result.stdout,encoding='utf-8')
