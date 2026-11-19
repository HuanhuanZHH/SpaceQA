from pathlib import Path
import datetime, json, os, shutil, signal, subprocess, time
ROOT=Path('/staff/tangjiarui/OPC/chat_console_20261105')
STAGE=ROOT/'polish_stage_20261106'
BACKUP=ROOT/('backup-polish-'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
BACKUP.mkdir()
files=['chat_service.py','chat_insights.py','chat_reporting.py','chat_demo.py','demo_run_package.py',
       'web/chat.html','web/chat.js','web/chat.css']
for name in files:
    destination=ROOT/name
    if destination.exists():
        saved=BACKUP/name;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(destination,saved)
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(STAGE/name,destination)
pid=int((ROOT/'chat.pid').read_text())
cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
assert str(ROOT/'chat_service.py').encode() in cmd
os.kill(pid,signal.SIGTERM)
for _ in range(50):
    if not Path(f'/proc/{pid}').exists():break
    time.sleep(.1)
subprocess.run(['/usr/bin/python3',str(ROOT/'manage.py'),'start'],check=True)
print(json.dumps({'deployed':files,'backup':str(BACKUP)}))
