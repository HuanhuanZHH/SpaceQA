"""Bounded, durable all-family training queue with a global test barrier."""
import json
import os
import subprocess
import sys
import time

from common import *
from scheduling import complete as is_complete,lock as stage_lock,unlock,verify_code,process_alive

IMAGE=ROOT/'.venv-image-models/bin/python'
QWEN=ROOT/'.venv-qwen3vl/bin/python'


def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True);start=time.monotonic();stages=[]
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']='1';environment['PYTHONUNBUFFERED']='1'
    environment['TOKENIZERS_PARALLELISM']='false';environment['OMP_NUM_THREADS']='4'
    # A single-run lock prevents two queues from concurrently updating the same checkpoints.
    import fcntl
    lock=(OUT/'runner.lock').open('a+')
    previous_queue=json.loads((OUT/'queue_state.json').read_text()) if (OUT/'queue_state.json').exists() else {}
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if not os.environ.get('OPC_ADOPT_ACTIVE_TRAINING'):raise
            write(OUT/'queue_state.json',dict(previous_queue,pid=os.getpid(),handoff_waiting_for_active_training=True,
                evaluation_policy='per_model_after_six_family_training',test_locked=False))
            if time.monotonic()-start>72*3600:raise TimeoutError('Training handoff exceeded queue budget')
            time.sleep(10)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    current_code={p.name:sha(p) for p in CODE.iterdir() if p.suffix in ['.py','.md']}
    existing=OUT/'code_manifest.json'
    has_training=bool(list((OUT/'training').glob('*.resume.pt'))) or bool(list((OUT/'training').glob('*.updated_ids.jsonl'))) or (OUT/'training/qlora_resume').exists() or (OUT/'pretraining/resume.pt').exists()
    if existing.exists() and has_training:assert json.loads(existing.read_text())==current_code,'Cannot resume trained weights under changed code'
    write(existing,current_code)
    def run(name,python,script,args=None,env=None,complete=None,adopt_pid=None):
        # Helpers use the same route lock. A child retains it if its controller exits.
        route_lock=stage_lock('stage_'+name,blocking=True)
        try:return run_locked(name,python,script,args,env,complete,adopt_pid,route_lock)
        finally:unlock(route_lock)
    def run_locked(name,python,script,args,env,complete,adopt_pid,route_lock):
        verify_code()
        if complete and is_complete(Path(complete)):
            stages.append(dict(name=name,status='previously_complete',path=str(complete)));return
        duration=72*3600-(time.monotonic()-start)
        if duration<=0:raise TimeoutError('72-hour queue budget reached; checkpoints retained')
        command=[str(python),'-u',str(CODE/script)]+(args or []);log=OUT/'logs'/(name+'.log')
        state=dict(status='running',stage=name,completed_stages=stages,pid=os.getpid(),started=time.time(),
            test_locked=False,evaluation_policy='per_model_after_six_family_training',gpu=1)
        write(OUT/'queue_state.json',state);emit('queue_stage',stage_name=name)
        if adopt_pid and process_alive(adopt_pid,script):
            state['child_pid']=adopt_pid;state['adopted_existing_process']=True;write(OUT/'queue_state.json',state)
            until=time.monotonic()+duration
            while process_alive(adopt_pid,script):
                if time.monotonic()>until:raise TimeoutError('Adopted process exceeded queue budget')
                time.sleep(10)
            if not complete or not is_complete(Path(complete)):raise RuntimeError('Adopted stage failed: '+name)
            stages.append(dict(name=name,status='adopted_complete',child_pid=adopt_pid));return
        with log.open('a') as f:
            proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env or environment,start_new_session=True,
                                  pass_fds=(lock.fileno(),route_lock.fileno()))
            state['child_pid']=proc.pid;write(OUT/'queue_state.json',state)
            try:code=proc.wait(timeout=duration)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);raise
        stages.append(dict(name=name,exit_code=code,log=str(log)))
        if code!=0:raise RuntimeError('Stage failed: '+name+'; inspect '+str(log))
    def evaluate_route(route):
        run('evaluate_'+route,QWEN if route=='qlora' else IMAGE,'route_evaluation.py',[route],
            complete=OUT/'evaluations'/route/'complete.json')
    def fit_route(route):
        run(route,QWEN if route=='qlora' else IMAGE,'qwen.py' if route=='qlora' else 'train.py',
            ['train'] if route=='qlora' else [route],complete=OUT/'training'/(route+'.json'))
        evaluate_route(route)
    try:
        # Preparation and registration can continue on CPU while a short isolated gradient check runs.
        previous=json.loads((OUT/'prepare_process.json').read_text()) if (OUT/'prepare_process.json').exists() else {}
        prep_pid=previous.get('pid')
        if not is_complete(OUT/'preparation.json') and not process_alive(prep_pid,'prepare.py'):
            prep_log=(OUT/'logs/prepare.log').open('a')
            prep=subprocess.Popen([str(IMAGE),'-u',str(CODE/'prepare.py')],stdout=prep_log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            prep_pid=prep.pid;write(OUT/'prepare_process.json',dict(pid=prep_pid,log=str(OUT/'logs/prepare.log')))
        pretrain=json.loads((OUT/'pretraining/state.json').read_text()) if (OUT/'pretraining/state.json').exists() else {}
        run('raw_encoder_pretraining',IMAGE,'warmup.py',complete=OUT/'pretraining/complete.json',adopt_pid=pretrain.get('pid'))
        smokeenv=dict(environment,OPC_ALL_FAMILY_OUT=str(REPO/'runs/all-family-smoke-20261031'))
        run('smoke_inputs',IMAGE,'smoke_inputs.py',env=smokeenv)
        for route in ['supervised_seed20261031','pruned_seed20261031']+ARCHS:
            run('smoke_'+route,IMAGE,'train.py',[route,'--smoke'],env=smokeenv)
        run('smoke_qlora',QWEN,'qwen.py',['train','--smoke'],env=smokeenv)
        state=dict(status='waiting_data_preparation',pid=os.getpid(),child_pid=prep_pid,completed_stages=stages,test_locked=True)
        write(OUT/'queue_state.json',state)
        while not is_complete(OUT/'preparation.json'):
            if not process_alive(prep_pid,'prepare.py'):raise RuntimeError('Preparation failed; inspect logs/prepare.log')
            if time.monotonic()-start>72*3600:raise TimeoutError('Preparation exceeded queue budget')
            time.sleep(10)
        run('validate_inputs',IMAGE,'validate.py',complete=OUT/'prepared_validation.json')
        # Each model completes all six training families before its own evaluation.
        fit_route('supervised_seed20261031')
        for route in ['dino_finetune','resnet18_unet','unetpp']:
            fit_route(route)
        for seed in SEEDS:
            for kind in ['supervised','pruned']:
                route=f'{kind}_seed{seed}'
                fit_route(route)
        fit_route('qlora')
        run('teacher_train_only',QWEN,'qwen.py',['teacher'],complete=OUT/'teacher_train.json')
        for seed in SEEDS:
            route=f'distilled_seed{seed}';fit_route(route)
        for route in ARCHS:fit_route(route)
        run('global_training_barrier',IMAGE,'barrier.py')
        for route in ROUTES:
            evaluate_route(route)
        run('score',IMAGE,'evaluate.py',['score'],complete=OUT/'evaluation_complete.json')
        run('validate_results',IMAGE,'validate.py',['--results'],complete=OUT/'result_validation.json')
        write(OUT/'queue_state.json',dict(status='complete',pid=os.getpid(),completed_stages=stages,seconds=time.monotonic()-start))
    except BaseException as exc:
        write(OUT/'queue_state.json',dict(status='failed',error=repr(exc),completed_stages=stages,pid=os.getpid(),
            test_locked=not (OUT/'training_barrier.json').exists(),seconds=time.monotonic()-start))
        raise


if __name__=='__main__':main()
