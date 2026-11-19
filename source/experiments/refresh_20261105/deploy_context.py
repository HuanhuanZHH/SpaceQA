from pathlib import Path
import subprocess
here=Path(__file__).parent;dest='/staff/tangjiarui/OPC/eight_class_20261104/refresh_code_20261105/'
subprocess.run(['scp',str(here/'inspect_fresh_context.py'),'icarus0:'+dest],check=True)
subprocess.run(['ssh','icarus0','/staff/tangjiarui/OPC/.venv-image-models/bin/python',dest+'inspect_fresh_context.py'],check=True)
