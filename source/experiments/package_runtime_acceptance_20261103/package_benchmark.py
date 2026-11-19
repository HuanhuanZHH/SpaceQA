"""Whole-package rules/report/audit plus fresh default experts, without old probabilities.

Rules scorecards and fresh expert candidates remain separate artifacts. This is an
A100 full-package execution benchmark, not A10 or fully integrated delivery acceptance.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path('/staff/tangjiarui/OPC')
OUT=ROOT/'package_runtime_acceptance_20261103'
FAMILIES=('AIA_AR_sample','HMI_AR_sample','CME_C2_sample','EPB_seg_sample','DMSP_SSUSI','ULF_sample')
sys.dont_write_bytecode=True


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));temp.replace(path)


def inventory(path):
    files=[];counts=collections.Counter()
    for parent,dirs,names in os.walk(path):
        for name in names:
            p=Path(parent)/name
            if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()):raise ValueError('Symlink/escaped package input')
            s=p.stat();files.append(dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns))
            counts['files']+=1;counts['bytes']+=s.st_size;counts[p.suffix.lower()+'_files']+=1
    if not files:raise ValueError('Empty input package')
    return dict(path=str(path),counts=dict(counts),files=sorted(files,key=lambda r:r['path']))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def io_counters():
    try:return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
    except OSError:return {}


def prepare(root,out):
    out.mkdir(parents=True,exist_ok=True)
    planpath=out/'plan_v1.json'
    if planpath.exists():raise ValueError('Refuse existing plan')
    public=root/'opc1018/newpublish';manifest=root/'phase1_manifest_audit_xzj_20261025/outputs/run_20261025T152348Z/manifest.jsonl'
    packages={}
    for line in manifest.open():
        r=json.loads(line)
        if r['tier']=='P4':packages[r['dataset_family']]=r['package_id']
    selected=[]
    for family in FAMILIES:
        package=packages[family];info=inventory(public/package)
        selected.append(dict(package=package,family=family,tier='P4',release_profile='public-v1',path=str(public/package),counts=info['counts']))
    archives=json.loads((root/'opc-agent/runs/roadshow-evaluation-20261101/archive_integrity.json').read_text())
    road=[]
    for item in archives['packages']:
        path=root/'opc-agent/runs/roadshow-evaluation-20261101/extracted'/item['actual_sha256']/item['package']
        if not path.is_dir():raise ValueError('Roadshow extraction layout not as verified: '+str(path))
        info=inventory(path)
        road.append(dict(package=item['package'],family=item['family'],tier='unknown',release_profile='roadshow-b-v1',path=str(path),counts=info['counts']))
    for family in ('AIA_AR_sample','EPB_seg_sample','ULF_sample'):
        selected.append(max((p for p in road if p['family']==family),key=lambda p:p['counts']['bytes']))
    # Independent immutable source snapshot; no edits in the prior release tree.
    code=out/'code';code.mkdir(exist_ok=True)
    src=root/'spaceqa-agent/src/spaceqa_agent';dest=code/'engine_snapshot/spaceqa_agent'
    if dest.exists():raise ValueError('Refuse existing code snapshot')
    shutil.copytree(src,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    source=root/'research_optimization_20261103/portable_experts.py'
    portable=code/'portable_experts_frozen.py';shutil.copyfile(source,portable)
    lock=root/'research_optimization_20261103/portable_lock_v1/lock.json'
    snapshot={str(p):sha(p) for p in dest.rglob('*') if p.is_file()}
    plan=dict(version=1,root=str(root),output_root=str(out),packages=selected,all_roadshow_packages=road,
        engine_snapshot=str(dest.parent),engine_source_hashes=snapshot,portable=str(portable),portable_sha256=sha(portable),
        lock=str(lock),lock_sha256=sha(lock),manifest_sha256=sha(manifest),runner_sha256=sha(__file__),
        scope='Six public P4 + largest extracted roadshow package per present family by bytes; no severity inferred from opaque IDs.',
        official_limit_seconds=10800,batch_size=4,gpu_memory_gib=4,
        limitations=['Rules scorecards and fresh default expert candidates are separate, not an integrated delivery verdict.',
                     'Host is A100, not A10; no optional LLM or new localization network.'])
    write(planpath,plan);print(json.dumps(dict(plan=str(planpath),packages=len(selected),input_bytes=sum(p['counts']['bytes'] for p in selected))))


def run(planpath,package,device):
    import fcntl
    plan=json.loads(planpath.read_text());root=Path(plan['root']);outroot=Path(plan['output_root']);started=time.monotonic()
    item=next(p for p in plan['packages'] if p['package']==package)
    if device not in ('cpu','cuda:0'):raise ValueError('Only explicit CPU or GPU0')
    gpu_family=item['family'] in ('AIA_AR_sample','EPB_seg_sample','DMSP_SSUSI')
    if gpu_family!=(device=='cuda:0'):raise ValueError('Use GPU0 for image default experts, CPU for tree experts')
    path=Path(item['path']).resolve();out=outroot/'runs'/package
    if out.exists() or out.is_relative_to(path) or path.is_relative_to(out):raise ValueError('Refuse existing/overlapping output')
    lockfile=(outroot/('gpu0-benchmark.lock' if gpu_family else 'cpu-benchmark.lock')).open('a')
    fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out.mkdir(parents=True);stages=[];initial_io=io_counters();stop=threading.Event();hardware=[]
    os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
    def stage(name,fn):
        at=time.monotonic();write(out/'state.json',dict(status='running',stage=name,elapsed_seconds=at-started))
        try:return fn()
        finally:
            r=dict(stage=name,seconds=time.monotonic()-at);stages.append(r);write(out/'stage_times.json',stages);print(json.dumps(r),flush=True)
    def monitor():
        while not stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                own=subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=5).strip()
                mine=[x for x in own.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
                hardware.append(dict(elapsed_seconds=time.monotonic()-started,gpu0_global=raw,own_process_gpu_mib=mine))
            except Exception as e:hardware.append(dict(error=str(e)))
            stop.wait(2)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        write(out/'request.json',dict(package=item,device=device,pid=os.getpid(),runner_sha256=sha(__file__),plan_sha256=sha(planpath),
            platform=platform.platform(),gpu_shared=True,qwen_service_kept=True,gpu1_used=False,probability_cache_reuse=False))
        before=stage('full_input_inventory_before',lambda:inventory(path));write(out/'input_inventory_before.json',before)
        if sha(plan['portable'])!=plan['portable_sha256'] or sha(plan['lock'])!=plan['lock_sha256']:raise ValueError('Frozen portable/lock changed')
        for p,h in plan['engine_source_hashes'].items():
            if sha(p)!=h:raise ValueError('Engine snapshot changed')
        sys.path.insert(0,plan['engine_snapshot'])
        from spaceqa_agent import engine
        from spaceqa_agent.audit import audit
        for name in ('inventory_dataset','base_analysis','apply_reference_checks','score_dataset','write_outputs'):
            fn=getattr(engine,name)
            def timed(*a,_fn=fn,_name=name,**kw):return stage('rules.'+_name,lambda:_fn(*a,**kw))
            setattr(engine,name,timed)
        rule_summary=stage('rules_full_scan_report_evidence',lambda:engine.run(path,out/'rules_report',20261103,set(engine.ALL_CHECKS)))
        rules_audit=stage('rules_output_audit',lambda:audit(out/'rules_report'))
        if rules_audit['status']!='pass':raise ValueError('Rules audit failed')
        portable=module('package_portable_frozen',Path(plan['portable']))
        # Instrumentation only: retain the exact frozen preprocessing/model functions.
        portable.RESEARCH_ROOT=outroot
        for name in ('validate_lock','prepare','fresh_probabilities','sources_unchanged'):
            fn=getattr(portable,name)
            def timed(*a,_fn=fn,_name=name,**kw):return stage('fresh.'+_name,lambda:_fn(*a,**kw))
            setattr(portable,name,timed)
        args=portable.parser().parse_args(['run','--input',str(path),'--family',item['family'],'--output',str(out/'fresh_experts'),
            '--lock',plan['lock'],'--device',device,'--batch-size','4','--gpu-memory-gb','4','--release-profile',item['release_profile']])
        status=stage('fresh_expert_whole_package',lambda:portable.run(args))
        complete=json.loads((out/'fresh_experts/complete.json').read_text())
        records=list(portable.rows(out/'fresh_experts/inventory.jsonl'))
        expected_labels={r['path'] for r in before['files'] if Path(r['path']).suffix.lower()=='.json'}
        actual_labels={r['label_path'] for r in records if r['label_path'] is not None}
        expected_data={r['path'] for r in before['files'] if Path(r['path']).suffix.lower() in ('.png','.csv')}
        actual_existing_data={r['data_path'] for r in records if Path(r['data_path']).is_file()}
        present=sum(not r['input_missing'] for r in records)
        coverage=dict(all_labels=actual_labels==expected_labels,all_existing_data=actual_existing_data==expected_data,
            expected_labels=len(expected_labels),records=len(records),existing_data=len(expected_data),forward_eligible=present,
            fresh_forward_samples=complete['fresh_forward_samples'],probability_cache_reuse=complete['cached_probability_reuse'],
            missing_records_retained=complete['abstentions'],no_limit_or_selection=True)
        if status or not all(coverage[k] for k in ('all_labels','all_existing_data')):raise ValueError('Full-package coverage or inference failed')
        if any(n!=present for n in complete['fresh_forward_samples'].values()) or complete['cached_probability_reuse']!=0:raise ValueError('Not every available sample had a fresh forward')
        after=stage('full_input_inventory_after',lambda:inventory(path));write(out/'input_inventory_after.json',after)
        if before!=after:raise ValueError('Input file set or metadata changed')
        final_audit=stage('final_rules_output_audit',lambda:audit(out/'rules_report'))
        if final_audit['status']!='pass':raise ValueError('Final audit failed')
        elapsed=time.monotonic()-started;usage=resource.getrusage(resource.RUSAGE_SELF);io=io_counters()
        result=dict(status='complete_whole_package_execution',package=item,seconds=elapsed,under_3h=elapsed<=10800,stage_times=stages,
            rule_summary=rule_summary,rule_audit=final_audit,coverage=coverage,expert_completion=complete,
            resource=dict(peak_rss_kib=usage.ru_maxrss,cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,
                io_delta={k:io[k]-initial_io.get(k,0) for k in io},gpu_peak_allocated_bytes=complete['gpu_peak_allocated_bytes']),
            cold_application_outputs=True,os_page_cache_flushed=False,hardware='A100 40GB shared GPU0 / host CPU+storage; not A10',
            portable_output_root_override=str(outroot),rules_and_fresh_candidates_integrated=False,official_deployment_acceptance=False,
            limitations=plan['limitations'],optional_llm_calls=0,external_model_tokens=0)
        write(out/'complete.json',result);write(out/'state.json',dict(status='complete',seconds=elapsed,under_3h=elapsed<=10800))
        print(json.dumps(dict(package=package,status=result['status'],seconds=elapsed,under_3h=elapsed<=10800)),flush=True)
    except BaseException as e:
        write(out/'failure.json',dict(status='failed',error=repr(e),seconds=time.monotonic()-started,stage_times=stages));raise
    finally:
        stop.set();thread.join(timeout=12);write(out/'gpu0_monitor.json',hardware);fcntl.flock(lockfile,fcntl.LOCK_UN)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','run'))
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--plan',type=Path);p.add_argument('--package');p.add_argument('--device',choices=('cpu','cuda:0'))
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.out)
    else:
        if not all((a.plan,a.package,a.device)):p.error('run requires --plan --package --device')
        run(a.plan,a.package,a.device)


if __name__=='__main__':main()
