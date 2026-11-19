from pathlib import Path
import subprocess
here=Path(__file__).parent;dest='/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105/'
subprocess.run(['scp',str(here/'finish_refresh.py'),str(here/'upgrade_context.py'),'icarus0:'+dest],check=True)
command='PYTHONPATH=/staff/tangjiarui/OPC/eight_class_20261104/runtime/src /staff/tangjiarui/OPC/.venv-image-models/bin/python '+dest+'upgrade_context.py'
subprocess.run(['ssh','icarus0',command],check=True)
