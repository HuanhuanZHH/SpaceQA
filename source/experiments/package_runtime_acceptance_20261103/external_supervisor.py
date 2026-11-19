"""External per-package wall timer, physical-GPU guard and process-tree watchdog.

Adopts the two already-running benchmarks without restarting them. Subsequent jobs
use the exact frozen pipeline with only a second CPU lock slot added in a new file.
The old running source is never edited. No model probability cache is consumed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

LIMIT=10800


def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def proc(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
        return dict(state=fields[0],ppid=int(fields[1]),start_ticks=int(fields[19]))
    except (OSError,ValueError):return None
def process_environment(pid):
    wanted={'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'};out={}
    try:
        for entry in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0'):
            if b'=' in entry:
                key,value=entry.split(b'=',1);key=key.decode(errors='replace')
                if key in wanted:out[key]=value.decode(errors='replace')
    except OSError:pass
    return out
def elapsed_from_proc(state):return time.clock_gettime(time.CLOCK_BOOTTIME)-state['start_ticks']/os.sysconf('SC_CLK_TCK')


def kill_tree(pid,identity):
    state=proc(pid)
    if not state or state['start_ticks']!=identity:return []
    found={pid:identity};todo=[pid]
    while todo:
        current=todo.pop()
        try:children=Path(f'/proc/{current}/task/{current}/children').read_text().split()
        except OSError:continue
        for child in children:
            child=int(child);s=proc(child)
            if s and child not in found:found[child]=s['start_ticks'];todo.append(child)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for target,ticks in reversed(list(found.items())):
            s=proc(target)
            if s and s['start_ticks']==ticks:
                try:os.kill(target,sig)
                except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(.2)
    return list(found)


def resources():
    memory={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,v=line.split(':',1)
        if k in ('MemTotal','MemAvailable'):memory[k]=v.strip()
    return dict(cpu_count=os.cpu_count(),loadavg=os.getloadavg(),memory=memory)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();base=a.root.resolve()
    if (base/'external_supervisor_request_v2.json').exists():raise ValueError('Refuse another external supervisor')
    plan=read(base/'plan_v1.json');runner=base/'package_benchmark.py'
    mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    uuids={int(x.split(',')[0]):x.split(',')[1].strip() for x in mapping.splitlines()};physical0=uuids[0]
    # A narrow scheduling-only new source; frozen preprocessing/model code unchanged.
    old="('gpu0-benchmark.lock' if gpu_family else 'cpu-benchmark.lock')"
    new="('gpu0-benchmark.lock' if gpu_family else ('cpu2-benchmark.lock' if os.environ.get('PACKAGE_BENCHMARK_CPU_SLOT')=='2' else 'cpu-benchmark.lock'))"
    source=runner.read_text()
    if source.count(old)!=1:raise ValueError('Unexpected frozen lock-slot expression')
    target=base/'package_benchmark_cpu_slots_v2.py'
    with target.open('x') as f:f.write(source.replace(old,new))
    import hashlib
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    plan2=dict(plan,runner_sha256=sha(target),derived_from_runner_sha256=sha(runner),
        scheduling_change='Second CPU lock slot only; external supervisor pins physical GPU0 UUID')
    write(base/'plan_v2.json',plan2)
    by_name={x['package']:x for x in plan['packages']};active={};completed={};retired=[]
    # Adopt live children before retiring their queue parents. Children continue.
    for lane in ('gpu0','cpu'):
        oldstate=read(base/(lane+'_queue_state.json'));pid=oldstate['child_pid'];state=proc(pid)
        if not state or state['state']=='Z':raise ValueError('Initial benchmark already exited; inspect before adoption')
        started=time.monotonic()-elapsed_from_proc(state)
        active[oldstate['package']]=dict(item=by_name[oldstate['package']],pid=pid,ticks=state['start_ticks'],
            started_monotonic=started,last_alive_monotonic=time.monotonic(),adopted=True,process=None,
            cpu_slot=1 if lane=='cpu' else None,environment=process_environment(pid),resources_at_adoption=resources(),
            actual_gpu_uuids_seen=[],reason=None)
        parent=state['ppid'];cmd=Path(f'/proc/{parent}/cmdline').read_bytes()
        if str(base/'queue_runner.py').encode() not in cmd:raise ValueError('Refuse to retire an unrelated parent')
        retired.append(dict(pid=parent,role='old_queue_parent',worker_kept=pid));os.kill(parent,signal.SIGTERM)
    watcher=base/'audit_watcher.py'
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:cmd=(directory/'cmdline').read_bytes()
        except OSError:continue
        if str(watcher).encode() in cmd:
            pid=int(directory.name)
            if pid!=os.getpid():os.kill(pid,signal.SIGTERM);retired.append(dict(pid=pid,role='old_internal_timer_audit_watcher'))
    write(base/'external_supervisor_request_v2.json',dict(pid=os.getpid(),source_sha256=sha(__file__),
        retired_supervisors=retired,adopted_workers=[{k:v for k,v in x.items() if k not in ('process',)} for x in active.values()],
        physical0_uuid=physical0,gpu_uuid_map=uuids,limit_seconds=LIMIT,max_gpu_workers=1,max_cpu_workers=2,
        initial_resources=resources(),first_worker_mapping_verified=(read(base/'initial_worker_cuda_mapping_probe_v2.json')['actual_cuda_uuid']==physical0.removeprefix('GPU-')),
        initial_mapping_probe_sha256=sha(base/'initial_worker_cuda_mapping_probe_v2.json')))
    pending=[x for x in plan['packages'] if x['package'] not in active]
    spec=importlib.util.spec_from_file_location('independent_external_auditor',base/'audit_benchmark.py');auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    last_gpu_check=0.;last_state_write=0.
    def arm_watchdog(job):
        job['finished_event']=threading.Event()
        def deadline():
            remaining=max(0.,LIMIT-(time.monotonic()-job['started_monotonic']))
            if not job['finished_event'].wait(remaining):
                state=proc(job['pid'])
                if state and state['state']!='Z' and state['start_ticks']==job['ticks']:
                    job['reason']='whole_process_wall_timeout_10800s'
                    job['terminated_process_tree']=kill_tree(job['pid'],job['ticks'])
        threading.Thread(target=deadline,daemon=True).start()
    def wait_child(job):
        job['return_code']=job['process'].wait();job['exit_observed_monotonic']=time.monotonic();job['finished_event'].set()
    for job in active.values():arm_watchdog(job)
    while active or pending:
        now=time.monotonic()
        if now-last_gpu_check>=.5:
            try:
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],text=True,timeout=5)
                for line in raw.splitlines():
                    pieces=[x.strip() for x in line.split(',')]
                    for job in active.values():
                        if pieces[0]==str(job['pid']):
                            if pieces[1] not in job['actual_gpu_uuids_seen']:job['actual_gpu_uuids_seen'].append(pieces[1])
                            if pieces[1]!=physical0:job['reason']='physical_gpu_mapping_violation'
                            try:
                                if int(pieces[2])>4096:job['reason']='process_gpu_memory_exceeds_4GiB'
                            except ValueError:pass
            except Exception as e:write(base/'gpu_guard_observation_error.json',dict(error=repr(e)))
            last_gpu_check=now
        for name,job in list(active.items()):
            state=proc(job['pid']);alive=state is not None and state['state']!='Z' and state['start_ticks']==job['ticks']
            if alive and now-job['started_monotonic']>=LIMIT:job['reason']='whole_process_wall_timeout_10800s'
            if job['reason'] and alive:
                job['terminated_process_tree']=kill_tree(job['pid'],job['ticks']);alive=False
            if alive:job['last_alive_monotonic']=now;continue
            job['finished_event'].set()
            end=job.get('exit_observed_monotonic',time.monotonic());upper=end-job['started_monotonic'];lower=job['last_alive_monotonic']-job['started_monotonic']
            if physical0 in job['actual_gpu_uuids_seen']:mapping_method='actual_worker_nvml_uuid'
            elif job['environment'].get('CUDA_VISIBLE_DEVICES')==physical0:mapping_method='explicit_single_physical_uuid_binding'
            elif job['adopted'] and job['cpu_slot'] is None:mapping_method='same_environment_cuda_probe_before_first_forward'
            else:mapping_method='not_applicable_cpu'
            receipt=dict(package=name,pid=job['pid'],adopted_existing_worker=job['adopted'],
                measurement='proc start ticks to observed exit interval' if job['adopted'] else 'external Popen launch to wait return',
                wall_seconds_upper_bound=upper,wall_seconds_lower_bound=lower if job['adopted'] else upper,
                under_3h_by_external_wall=upper<=LIMIT and job['reason'] is None,timeout_or_guard_failure=job['reason'],
                exit_code=job.get('return_code'),actual_gpu_uuids_seen=job['actual_gpu_uuids_seen'],
                expected_physical_gpu0_uuid=physical0,environment=job['environment'],cpu_slot=job['cpu_slot'],
                physical_mapping_evidence=mapping_method,
                resources_at_start=job.get('resources_at_start',job.get('resources_at_adoption')),resources_at_exit=resources(),
                terminated_process_tree=job.get('terminated_process_tree',[]),
                source_sha256=sha(__file__),A10_acceptance_established=False)
            folder=base/'runs'/name;write(folder/'external_runtime_receipt_v2.json',receipt)
            if (folder/'complete.json').exists() and not job['reason']:
                audited=auditor.audit(folder,plan);audited['internal_root_seconds']=audited.pop('seconds')
                audited['external_wall_seconds_upper_bound']=upper;audited['under_3h_on_measured_host']=receipt['under_3h_by_external_wall'];audited['external_timer_receipt_sha256']=sha(folder/'external_runtime_receipt_v2.json')
                if job['cpu_slot'] is None and mapping_method=='not_applicable_cpu':
                    audited['errors'].append('actual_physical_gpu_mapping_not_observed');audited['status']='fail'
                with (folder/'independent_audit_external_v2.json').open('x') as f:json.dump(audited,f,indent=2)
                receipt['independent_audit_status']=audited['status']
            else:receipt['independent_audit_status']='failed_or_incomplete_pipeline'
            completed[name]=receipt;del active[name]
        # Start only available slots. Pin the sole visible CUDA device by UUID.
        for item in pending[:]:
            isgpu=item['family'] in ('AIA_AR_sample','EPB_seg_sample','DMSP_SSUSI')
            gpu_jobs=[j for j in active.values() if j['cpu_slot'] is None];slots={j['cpu_slot'] for j in active.values() if j['cpu_slot'] is not None}
            if (isgpu and gpu_jobs) or (not isgpu and len(slots)>=2):continue
            slot=None if isgpu else next(s for s in (1,2) if s not in slots)
            command=['/staff/tangjiarui/OPC/.venv-image-models/bin/python','-u',str(target),'run','--plan',str(base/'plan_v2.json'),
                '--package',item['package'],'--device','cuda:0' if isgpu else 'cpu']
            env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',
                CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=physical0 if isgpu else '-1',PACKAGE_BENCHMARK_CPU_SLOT=str(slot or 0))
            log=(base/(item['package']+'.log')).open('x');begin=time.monotonic()
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            state=proc(process.pid)
            if state is None:raise RuntimeError('New benchmark disappeared during launch')
            job=dict(item=item,pid=process.pid,ticks=state['start_ticks'],started_monotonic=begin,last_alive_monotonic=time.monotonic(),
                adopted=False,process=process,cpu_slot=slot,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','PACKAGE_BENCHMARK_CPU_SLOT')},
                resources_at_start=resources(),actual_gpu_uuids_seen=[],reason=None)
            active[item['package']]=job;arm_watchdog(job);threading.Thread(target=wait_child,args=(job,),daemon=True).start();pending.remove(item)
        if now-last_state_write>=2:
            snapshot=dict(status='running' if active else 'complete',active=[dict(package=n,pid=j['pid'],cpu_slot=j['cpu_slot'],
                elapsed_seconds=time.monotonic()-j['started_monotonic'],actual_gpu_uuids_seen=j['actual_gpu_uuids_seen']) for n,j in active.items()],
                pending=[p['package'] for p in pending],completed=list(completed.values()),updated_unix=time.time())
            write(base/'external_supervisor_state_v2.json',snapshot);last_state_write=now
        time.sleep(.2)
    result=dict(status='complete',results=list(completed.values()),limit_scope='Each individual complete worker process, not sum of package times')
    write(base/'external_supervisor_complete_v2.json',result)


if __name__=='__main__':main()
