import os,sys,socket,json,subprocess,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parent;assert socket.gethostname().split('.')[0]=='icarus1'
while not (R/'final_export_receipt.json').exists():time.sleep(5)
receipt=json.load(open(R/'final_export_receipt.json'));assert receipt['tag']=='localhost/spaceqa-agent:0.6.1-final-20261110'
store=Path('/tmp/spaceqa-final-icarus1-20261110');store.mkdir(exist_ok=False)
for name in ['runtime','config','cache','tmp','storage','runroot']:(store/name).mkdir(mode=0o700)
base=Path('/staff/tangjiarui/OPC/compact_release_20261107/build/runtime/crun');crun=R/'crun_icarus1';crun.write_text(base.read_text().replace("Path('/tmp/spaceqa-rootless-20261102')",'Path('+repr(str(store))+')').replace("ROOT = Path('/staff/tangjiarui/OPC/compact_release_20261107/build')",'ROOT = Path('+repr(str(R/'icarus1_runtime'))+')'));crun.chmod(0o755)
env={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']};update=dict(XDG_RUNTIME_DIR=str(store/'runtime'),XDG_CONFIG_HOME=str(store/'config'),XDG_CACHE_HOME=str(store/'cache'),TMPDIR=str(store/'tmp'));env.update(update)
cmd=['podman','--root',str(store/'storage'),'--runroot',str(store/'runroot'),'--storage-driver=vfs','--storage-opt','vfs.ignore_chown_errors=true','--cgroup-manager=cgroupfs','--events-backend=file','--runtime='+str(crun)]
t=time.time()
with (R/'icarus1_load.log').open('w') as f:p=subprocess.run(cmd+['load','--input',str(R.parents[1]/'image/LTH_image.tar')],env=env,stdout=f,stderr=subprocess.STDOUT)
if p.returncode:raise RuntimeError('clean load failed')
info=json.loads(subprocess.check_output(cmd+['image','inspect',receipt['tag']],env=env))[0];assert info['Id']==receipt['imageId']
drivers=R/'host-driver-icarus1';drivers.mkdir(exist_ok=True);ld=subprocess.check_output(['/sbin/ldconfig','-p'],text=True)
for name in ['libcuda.so.1','libnvidia-ml.so.1','libnvidia-ptxjitcompiler.so.1','libnvidia-nvvm.so.4']:
 line=next((l for l in ld.splitlines() if l.strip().startswith(name+' ')),None)
 if line:shutil.copyfile(line.split('=>')[-1].strip(),drivers/name)
cfg=dict(command=cmd,environment=update,gpuUuid='GPU-72e2221a-83b6-60bf-ee18-7587ccd188ca',gpuIndex=1,driverPath=str(drivers),imageId=info['Id'],loadSeconds=time.time()-t,storeInitiallyEmpty=True)
(R/'host_config_icarus1.json').write_text(json.dumps(cfg,indent=2));print('READY',json.dumps(cfg),flush=True)
