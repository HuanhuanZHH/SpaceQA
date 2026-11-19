from pathlib import Path
import subprocess
here=Path(__file__).parent;dest='/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105/'
subprocess.run(['scp',str(here/'export_fresh_cases.py'),'icarus0:'+dest],check=True)
subprocess.run(['ssh','icarus0','/staff/tangjiarui/OPC/.venv-image-models/bin/python',dest+'export_fresh_cases.py'],check=True)
subprocess.run(['scp','icarus0:/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105/eight_class_case_examples.json','D:/OPC/outputs/chat_insights_20261105/fresh_12_packages/eight_class_case_examples.json'],check=True)
