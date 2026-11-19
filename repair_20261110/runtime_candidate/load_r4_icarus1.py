import os,json,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'r4_export_receipt.json').exists() or not (R/'host_config_icarus1.json').exists():time.sleep(5)
cfg=json.load(open(R/'host_config_icarus1.json'));receipt=json.load(open(R/'r4_export_receipt.json'));env=dict(os.environ,**cfg['environment']);cmd=cfg['command']
with (R/'icarus1_r4_load.log').open('w') as f:p=subprocess.run(cmd+['load','--input',str(R.parents[1]/'image/LTH_image.tar')],env=env,stdout=f,stderr=subprocess.STDOUT)
assert p.returncode==0
info=json.loads(subprocess.check_output(cmd+['image','inspect',receipt['tag']],env=env))[0];assert info['Id']==receipt['imageId'];cfg['imageId']=info['Id'];cfg['tag']=receipt['tag'];(R/'host_config_icarus1_r4.json').write_text(json.dumps(cfg,indent=2));print('READY R4',flush=True)
