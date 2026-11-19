"""Continue the already-started GPU experiment through evaluation and reports."""
from __future__ import annotations
import datetime as dt,json,os,subprocess,time
from pathlib import Path

ROOT=Path('/staff/tangjiarui/OPC/eight_class_20261104')
PY='/staff/tangjiarui/OPC/.venv-image-models/bin/python'


def write(name,value):
    path=ROOT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    from status import snapshot
    started=time.monotonic();deadline=started+22*3600
    state=dict(start=dt.datetime.now().isoformat(),deadline=(dt.datetime.now().astimezone()+dt.timedelta(hours=22)).isoformat(),stage='waiting_localizer')
    write('completion_state.json',state)
    launch=json.loads((ROOT/'localizer_v2_launch.json').read_text());pid=launch['pid']
    def alive():
        try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
        except FileNotFoundError:return False
    while alive() or not (ROOT/'roadshow_v2/complete.json').exists():
        write('report/status.json',snapshot())
        if (ROOT/'localizer_v2/failure.json').exists():raise RuntimeError('Localizer failed; see localizer_v2/failure.json')
        if (ROOT/'pipeline_v2_state_failure.json').exists():raise RuntimeError('Classifier continuation failed')
        if not alive() and not (ROOT/'localizer_v2/comparison.json').exists():raise RuntimeError('Localizer exited without complete results')
        if time.monotonic()>deadline:raise TimeoutError('22 hour continuation budget reached')
        time.sleep(20)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',
        PYTHONPATH=str(ROOT/'runtime/src'))
    commands=[('spatial_comparison_v1','legacy_evaluation.py',['--localizer',str(ROOT/'localizer_v2'),'--output',str(ROOT/'spatial_comparison_v1')]),
        ('candidate_comparison_v1','candidate_localizer.py',['--localizer',str(ROOT/'localizer_v2'),'--output',str(ROOT/'candidate_comparison_v1')]),
        ('grounded_roadshow_v1','ground_roadshow.py',['--localizer',str(ROOT/'localizer_v2'),'--comparison',str(ROOT/'candidate_comparison_v1'),
            '--roadshow',str(ROOT/'roadshow_v2'),'--output',str(ROOT/'grounded_roadshow_v1')]),
        ('report_final','report.py',['--output',str(ROOT/'report')]),
        ('delivery_audit','finalize.py',[])]
    for name,script,args in commands:
        state.update(stage=name,elapsed_seconds=time.monotonic()-started);write('completion_state.json',state)
        write('report/status.json',snapshot())
        with (ROOT/(name+'.log')).open('ab') as log:
            child=subprocess.Popen([PY,'-u',str(ROOT/'code'/script)]+args,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:child.wait(timeout=max(1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                child.terminate();raise TimeoutError('Overall continuation deadline exceeded')
            if child.returncode:raise RuntimeError(name+' exited with '+str(child.returncode))
    state.update(stage='complete',elapsed_seconds=time.monotonic()-started,
        independent_official_localized_metrics=False,original_services_unchanged=True)
    write('completion_state.json',state)
    write('report/status.json',snapshot())


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('completion_failure.json',dict(type=type(exc).__name__,message=str(exc)))
        from status import snapshot
        write('report/status.json',snapshot());raise
