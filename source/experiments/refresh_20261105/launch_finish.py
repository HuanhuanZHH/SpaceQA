from pathlib import Path
import os
import subprocess
BASE=Path('/staff/tangjiarui/OPC/eight_class_20261104')
env=dict(os.environ)
env.update(PYTHONPATH=str(BASE/'runtime/src'),OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1')
subprocess.run(['/staff/tangjiarui/OPC/.venv-image-models/bin/python','-u',str(BASE/'refresh_code_20261105/finish_refresh.py')],check=True,env=env)
