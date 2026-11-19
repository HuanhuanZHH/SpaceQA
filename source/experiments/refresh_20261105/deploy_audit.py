from pathlib import Path
import subprocess
here=Path(__file__).parent
dest='/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105/'
subprocess.run(['scp',str(here/'audit_refresh.py'),'icarus0:'+dest],check=True)
subprocess.run(['ssh','icarus0','/staff/tangjiarui/OPC/.venv-image-models/bin/python',dest+'audit_refresh.py'],check=True)
out=Path('D:/OPC/outputs/chat_insights_20261105/fresh_12_packages');out.mkdir(parents=True,exist_ok=True)
base='/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105/'
for name in ['report_index.json','complete.json','independent_validation.json','ranking_detected.csv','classification/complete.json','grounded/complete.json']:
    filename=name.replace('/','_')
    subprocess.run(['scp','icarus0:'+base+name,str(out/filename)],check=True)
print(str(out))
