"""Wait without a CUDA context; train on at most two additional unoccupied GPUs."""
import argparse
import os
import socket
import subprocess
import time
from common import *
from scheduling import available_gpu,complete,identity,lock,unlock,verify_code

PRIORITY=['unetpp','dino_finetune','resnet18_unet','attention_unet','unet',
          'ijepa_finetune','deeplab_mobilenet','yolo11_detect_backbone',
          'yolo11_segment_backbone','yolov8_segment_backbone']


def main(gpu):
    host=socket.gethostname();folder=OUT/'parallel';folder.mkdir(parents=True,exist_ok=True)
    state=folder/f'{host}_gpu{gpu}.json';single=lock(f'worker_{host}_gpu{gpu}')
    if single is None:return
    uuid=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=15).strip()
    deadline=time.time()+72*3600;free_samples=0
    def status(value,**details):write(state,dict(status=value,gpu=gpu,uuid=uuid,**identity(),**details))
    status('waiting_for_inputs_and_idle_gpu')
    while time.time()<deadline:
        if (folder/'STOP').exists():status('stopped_by_control');return
        if all(complete(OUT/'training'/(r+'.json')) for r in ARCHS):status('all_architectures_complete');return
        if not (complete(OUT/'preparation.json') and complete(OUT/'prepared_validation.json')):
            status('waiting_for_validated_inputs');time.sleep(30);continue
        verify_code()
        if not available_gpu(uuid):
            free_samples=0;status('waiting_for_idle_gpu');time.sleep(30);continue
        free_samples+=1
        if free_samples<3:
            status('confirming_idle_gpu',consecutive_free_samples=free_samples);time.sleep(20);continue
        gpu_lock=lock('gpu_'+uuid)
        if gpu_lock is None:status('gpu_locked_by_project');time.sleep(30);continue
        slot=None;job=None
        try:
            for i in range(2):
                slot=lock(f'helper_slot_{i}')
                if slot is not None:break
            if slot is None:status('waiting_for_one_of_two_helper_slots');time.sleep(30);continue
            route=None
            for candidate in PRIORITY:
                if complete(OUT/'training'/(candidate+'.json')):continue
                try:previous=json.loads((folder/(candidate+'.json')).read_text())
                except (OSError,ValueError):previous={}
                if previous.get('status')=='failed':continue # main queue will resume and record the failure
                job=lock('stage_'+candidate)
                if job is not None:
                    if complete(OUT/'training'/(candidate+'.json')):unlock(job);job=None;continue
                    route=candidate;break
            if route is None:status('waiting_for_unclaimed_architecture');time.sleep(30);continue
            verify_code()
            if not available_gpu(uuid):free_samples=0;status('gpu_became_busy');continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
            log=OUT/'logs'/f'helper_{host}_gpu{gpu}_{route}.log'
            started=time.time()
            with log.open('a') as stream:
                proc=subprocess.Popen([str(ROOT/'.venv-image-models/bin/python'),'-u',str(CODE/'train.py'),route],
                    stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,
                    pass_fds=(single.fileno(),gpu_lock.fileno(),slot.fileno(),job.fileno()))
                details=dict(route=route,child_pid=proc.pid,started=started,log=str(log),test_read=False)
                status('training',**details);write(folder/(route+'.json'),dict(status='training',**identity(),gpu=gpu,uuid=uuid,**details))
                try:code=proc.wait(timeout=max(1,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);raise
            passed=code==0 and complete(OUT/'training'/(route+'.json'))
            if passed:
                status('evaluating_30_packages',route=route)
                with log.open('a') as stream:
                    evaluation=subprocess.Popen([str(ROOT/'.venv-image-models/bin/python'),'-u',str(CODE/'route_evaluation.py'),route],
                        stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True,
                        pass_fds=(single.fileno(),gpu_lock.fileno(),slot.fileno(),job.fileno()))
                    try:code=evaluation.wait(timeout=max(1,deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(evaluation.pid,signal.SIGTERM);evaluation.wait(timeout=60);raise
                passed=code==0 and complete(OUT/'evaluations'/route/'complete.json')
            result=dict(status='complete' if passed else 'failed',route=route,exit_code=code,seconds=time.time()-started,
                **identity(),gpu=gpu,uuid=uuid,log=str(log),test_read=False)
            write(folder/(route+'.json'),result);status(result['status'],route=route,exit_code=code)
            free_samples=0
        finally:
            unlock(job);unlock(slot);unlock(gpu_lock)
        time.sleep(20)
    status('wait_budget_exhausted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();main(a.gpu)
