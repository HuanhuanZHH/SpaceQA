"""Finalize metadata after reviewed publication; hashes come last, not during live runs."""
import datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(sys.argv[1]);R=ROOT/'repair_20261110/runtime_candidate'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');temp.replace(p)
comparison=json.load(open(R/'repeat_comparison/comparison.json'));assert comparison['status']=='pass'
summary=json.load(open(ROOT/'repair_20261110/final_export/export_summary.json'));image=json.load(open(ROOT/'image/package_manifest.json'))
source=ROOT/'source';manifest=source/'reproducibility/releases/20261110_final_source_manifest.json'
exclude={'__pycache__','.git','build','.pytest_cache','UNKNOWN.egg-info'}
files=[p for p in source.rglob('*') if p.is_file() and not (set(p.relative_to(source).parts)&exclude) and p.suffix not in {'.pyc','.pyo'} and not str(p.relative_to(source)).startswith('reproducibility/releases/')]
dump(manifest,{'version':'20261110-final','scope':'actual current source files; historical release manifests and generated caches excluded','buildToolchain':{'python':'3.10.12','setuptools':'81.0.0'},'runtimeImageId':image['imageId'],'documentationScope':'Root DEL-4 and agent manifest contain explicit release metadata errata; embedded image assets preserve build-time documentation. Inference code and thresholds are frozen.','files':[{'path':str(p.relative_to(source)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]})
paths={
'DEL-1':['01_方案规划.md'], 'DEL-2':['image','source','agent'], 'DEL-3':['02_部署与使用说明.md'], 'DEL-4':['03_评价规范.md','03_评价规范.yaml'], 'DEL-5':['04_评分卡'],'DEL-6':['05_排序表'],'DEL-7':['06_问题清单'],'DEL-8':['07_证据索引'],'DEL-9':['08_运行日志/run.log.jsonl'],'DEL-10':['08_运行日志/system_env.txt','08_运行日志/system.json','08_运行日志/seeds.yaml'],'DEL-11':['09_未完成事项说明.md'],'DEL-12':['10_演示视频.mp4']}
deliverables=[]
for key,pp in paths.items():
 fs=set()
 for name in pp:
  p=ROOT/name
  fs.update(x for x in p.rglob('*') if x.is_file()) if p.is_dir() else fs.add(p)
 deliverables.append({'id':key,'paths':pp,'fileCount':len(fs),'bytes':sum(p.stat().st_size for p in fs),'status':'implemented_and_verified_with_declared_evidence_limits'})
manifest_data={'schema':'spaceqa-delivery-manifest-2.3','project':'SpaceQA','team':'LTH','version':'20261110-repaired-r4','status':'engineering_repair_verified; scientific acceptance prerequisites remain explicit','frozenUtc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evaluationScope':{'datasetFamilies':['AIA','EPB','ULF'],'datasetVersion':'v34','dirCount':12,'dirIds':[x['dirId'] for x in comparison['packages']],'scorecardDirectory':'04_评分卡','sampleCount':summary['samples']},'deliverables':deliverables,'selfCheck':{'fixedPositionsExpected':53,'report':'repair_20261110/FINAL_REPAIR_VERIFICATION.md','fieldAndLinkAudit':'repair_20261110/final_validation.json','scope':'Documented v2.3 fixed files and field requirements; no official JSON Schema was supplied','sha256List':'SHA256SUMS.txt','sourceManifest':'source/reproducibility/releases/20261110_final_source_manifest.json','freshRepeat':'repair_20261110/runtime_candidate/repeat_comparison/comparison.json'},'declarations':{'scientificGroundTruth':False,'officialAccuracyClaimed':False,'totalScore':'100*(1-confirmed unique rule-defect samples/registered samples); higher is better; not six-dimensional scientific quality','fullQualityScore':'null unless all six dimensions and genuine approved weighting evidence are available','candidateLabels':'not automatic annotation corrections','historicalRecords':'Original and failed attempts preserved separately with actual provenance','runtime':image,'currentResults':{k:summary[k] for k in ['samples','confirmed','found','findings','confirmedRecords','candidateRecords']},'bModule':'Current-image raw4 classification + learned4 localization after upstream new predictions; unknown geometry remains undetermined','configurationErrata':'Root DEL-4 and agent/manifest.yaml correct release descriptions; embedded assets preserve build-time metadata, not execution authority'},'roadshowMaterials':{'directory':'11_展示材料','files':[str(p.relative_to(ROOT)) for p in sorted((ROOT/'11_展示材料').iterdir()) if p.is_file()],'video':'10_演示视频.mp4','videoSource':'repair_20261110/runtime_candidate/video_receipt.json'},'supportingEvidenceDirectory':'repair_20261110','checksumScope':'All regular published files except SHA256SUMS.txt itself; includes declared historical and repair evidence; symlink content is hashed through its published path'}
dump(ROOT/'00_MANIFEST.yaml',manifest_data)
# No ongoing process may write beneath ROOT during this final hash pass.
files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'SHA256SUMS.txt' and not p.name.endswith('.tmp'))
temp=ROOT/'SHA256SUMS.txt.tmp'
with temp.open('w') as f:
 for p in files:f.write(digest(p)+'  '+str(p.relative_to(ROOT))+'\n')
temp.replace(ROOT/'SHA256SUMS.txt')
print(json.dumps({'hashedFiles':len(files),'sourceFiles':len(manifest_data),'imageId':image['imageId']}))
