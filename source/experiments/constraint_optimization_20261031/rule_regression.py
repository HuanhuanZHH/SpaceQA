"""Fixed-field reference agreement, with a sample+class match granularity.

This diagnostic uses manifest PNG dimensions, actual labels and existing audited
P1 references. It is not a full run or the official class+location metric.
"""
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/opc-agent')
OUT=ROOT/'runs/constraint-optimization-20261031'
SRC=ROOT/'experiments/constraint_optimization_20261031'


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def main():
    start=time.monotonic()
    old=load_module('opc_before',SRC/'opc_agent_before.py')
    new=load_module('opc_after',ROOT/'opc_agent.py')
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    manifest=ROOT.parent/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    for s in manifest.open():
        r=json.loads(s)
        if r['tier'] in ('P0','P1'):groups[r['dataset_family']][r['tier']].append(r)
    refs={'AIA_AR_sample':'aia_p0_p1_audit_20261026','HMI_AR_sample':'hmi_p0_p1_audit_20261029',
          'CME_C2_sample':'cme_p0_p1_audit_20261028','EPB_seg_sample':'epb_p0_p1_audit_20261027',
          'DMSP_SSUSI':'dmsp_p0_p1_audit_20261029','ULF_sample':'ulf_p0_p1_audit_20261029'}
    summary={};models={}
    for family,tiers in sorted(groups.items()):
        truth=collections.defaultdict(set)
        for s in (ROOT.parent/'phase2_visual_audit_xzj'/refs[family]/'findings.jsonl').open():
            r=json.loads(s)
            if r['code'] in ('L2-P','L2-T','L2-F','L2-M'):truth[r['code']].add(r['filename'])
        result={}
        for tag,module in [('before',old),('after',new)]:
            datasets={}
            for tier,rows in tiers.items():
                root=Path(rows[0]['label_path']).parent
                if root.name=='label':root=root.parent
                d=module.Dataset(root,rows[0]['package_id'],tier='clean' if tier=='P0' else 'P1')
                for r in rows:
                    label=Path(r['label_path']); d.label_files[label.name]=label
                    d.labels[label.name]=json.loads(label.read_text(encoding='utf-8-sig'))
                    if r['data_exists']:
                        data=Path(r['data_path']); d.data_files[data.name]=data
                        if r.get('image_width'):
                            d.png_meta[data.name]={'width':r['image_width'],'height':r['image_height']}
                datasets[tier]=d
            current=datasets['P1'];module.check_c06(current);module.check_c09(current)
            if hasattr(module,'reconcile_ulf_metadata_attribution'):
                module.reconcile_ulf_metadata_attribution(current,datasets['P0'])
            ff,shared=module.compare_to_reference(current,datasets['P0'])
            predictions=collections.defaultdict(set)
            locator_coverage=collections.Counter()
            for f in current.findings+ff:
                code=f['defectType']
                if code not in truth or f['status']!=module.STATUS_FAIL:continue
                lp=Path(f['_locator'].split('#')[0]); value=current.labels.get(lp.name)
                if value is None:continue
                name=value['basic_annotation']['fileName'];predictions[code].add(name)
                if '#' in f['_locator']:locator_coverage[code]+=1
            cells={}
            for code,gold in sorted(truth.items()):
                pred=predictions[code];tp=len(gold&pred);fp=len(pred-gold);fn=len(gold-pred)
                cells[code]=dict(reference_samples=len(gold),predicted_samples=len(pred),tp=tp,fp=fp,fn=fn,
                                 precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
                                 located_records=locator_coverage[code])
            result[tag]=cells
        summary[family]=result;print(json.dumps({family:result}),flush=True)
    summary['_protocol']=dict(seconds=time.monotonic()-start,scope='All P0/P1 labels for six families; fixed-field types only',
        matching='sample filename + class; not official fine-grained defect localization',
        dimensions='Reused independently verified manifest PNG dimensions; no model calls',
        limitations=['Public trusted clean/audit agreement, not independent hidden accuracy.',
                     'L2-G/L3 classes excluded from this diagnostic; transformed geometry is not solved here.'])
    (OUT/'fixed_field_regression.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
