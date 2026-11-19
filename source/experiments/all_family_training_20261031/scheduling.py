"""Shared execution locks; no changes to fitting, splits, or test policy."""
import fcntl
import json
import os
import socket
import subprocess
import time
from common import CODE, OUT, sha


def lock(name, blocking=False):
    folder=OUT/'locks';folder.mkdir(parents=True,exist_ok=True)
    handle=(folder/(name+'.lock')).open('a+')
    try:fcntl.flock(handle,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
    except BlockingIOError:handle.close();return None
    os.set_inheritable(handle.fileno(),True)
    return handle


def unlock(handle):
    if handle is not None:
        fcntl.flock(handle,fcntl.LOCK_UN);handle.close()


def complete(path):
    try:return json.loads(path.read_text()).get('status') in ['complete','pass']
    except (OSError,ValueError):return False


def verify_code():
    expected=json.loads((OUT/'code_manifest.json').read_text())
    actual={p.name:sha(p) for p in CODE.iterdir() if p.suffix in ['.py','.md']}
    assert actual==expected,'Frozen training/execution source manifest changed'


def process_alive(pid,script):
    """Adopt only this user's exact script on the current machine, not a reused PID."""
    from pathlib import Path
    try:
        proc=Path('/proc')/str(int(pid))
        if proc.stat().st_uid!=os.getuid():return False
        if (proc/'stat').read_text().split(') ',1)[1].startswith('Z'):return False
        args=(proc/'cmdline').read_bytes().split(b'\0')
        return os.fsencode(str(CODE/script)) in args
    except (OSError,ValueError,TypeError):return False


def available_gpu(uuid):
    """No compute processes AND low allocated memory; utilization alone is insufficient."""
    memory=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=15)
    jobs=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    return float(memory.strip())<512 and not jobs.strip()


def identity():return dict(host=socket.gethostname(),pid=os.getpid(),time=time.time())
