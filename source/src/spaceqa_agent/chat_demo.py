"""Isolated live package recomputation, with existing conversations preserved."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

class DemoMixin:
    def live_recompute(self, identity):
        runner=Path(__file__).with_name('demo_run_package.py')
        if not runner.is_file():
            raise self.insight_error(503,'服务器尚未安装整包现场重评程序。')
        with self.lock:
            if any(s.get('liveRun',{}).get('status') in {'queued','running'} for s in self.sessions.values()):
                raise self.insight_error(409,'已有整包重评任务正在执行，请等待完成。')
            ds, report=self._report_for_session(self.sessions[identity])
            if not report or ds.get('family') not in {'EPB_seg_sample','AIA_AR_sample'}:
                raise self.insight_error(422,'当前现场重评支持已登记且具有专家缓存的EPB、AIA路演包。其他数据仍可载入报告或执行常规分析。')
            run_id='LIVE-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            out=self.state_dir/'live-runs'/run_id
            out.mkdir(parents=True)
            (out/'baseline.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
            physical_gpu=None
            try:
                probe=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4)
                used,utilization=[int(x.strip()) for x in probe.strip().split(',')]
                if used<512 and utilization<5:physical_gpu='1'
            except (OSError,ValueError,subprocess.SubprocessError):pass
            device='cuda:'+physical_gpu if physical_gpu else 'cpu'
            live=dict(runId=run_id,status='queued',phase='准备整包重评',percent=0,elapsedSeconds=0,
                      scope='本次重新读取文件、核验哈希、执行规则、八类融合和全部可读图像定位；21个上游专家使用已校验预测缓存。',
                      device=device)
            self.sessions[identity]['liveRun']=live
            self._save(self.sessions[identity])
        def execute():
            started=time.monotonic()
            try:
                env=dict(os.environ)
                for key in list(env):
                    if 'API_KEY' in key or key.startswith('SPACEQA_CHAT_LLM_'):env.pop(key,None)
                env.update(PYTHONPATH='/staff/tangjiarui/OPC/eight_class_20261104/runtime/src',
                           OMP_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES=physical_gpu or '')
                with (out/'execution.log').open('wb') as log:
                    child=subprocess.Popen([sys.executable,str(runner),'--package',ds['name'],'--output',str(out),'--baseline',str(out/'baseline.json'),'--device','cuda' if physical_gpu else 'cpu','--physical-gpu',physical_gpu or ''],
                                           env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                    while child.poll() is None:
                        progress=out/'progress.json'
                        try: update=json.loads(progress.read_text(encoding='utf-8')) if progress.is_file() else {}
                        except (OSError,ValueError):update={}
                        with self._session_lock(identity):
                            # The session switches only after complete artifacts pass validation.
                            live.update(update,status='running',elapsedSeconds=round(time.monotonic()-started,2))
                            self._save(self.sessions[identity])
                        time.sleep(.8)
                if child.returncode:raise RuntimeError('整包重评失败，已保留上次完整报告；请查看服务器本次execution.log。')
                fresh=json.loads((out/'report.json').read_text(encoding='utf-8'))
                assert fresh['name']==ds['name'] and fresh['totalSamples']==report['totalSamples']
                assert fresh['inferenceCoverage']['actualFreshImageForwards']==fresh['inferenceCoverage']['expectedReadableImageSamples']
                with self._session_lock(identity):
                    session=self.sessions[identity]
                    session['_liveReport']=fresh
                    facts=self._saved_facts(ds,fresh)
                    facts.update(source='live_recompute',actualMode='fresh_rules_fusion_rgb_with_verified_expert_cache',elapsedSeconds=fresh['elapsedSeconds'])
                    session['analysis'].update(source='live_recompute',taskId=run_id,status='completed',phase='completed',
                        result=facts,evaluatedAt=fresh['evaluatedAt'],reportLabel=fresh['reportLabel'],elapsedSeconds=fresh['elapsedSeconds'])
                    session['status']='completed'
                    live.update(status='completed',phase='整包重评完成',percent=100,elapsedSeconds=fresh['elapsedSeconds'],audit=fresh['liveAudit'])
                    session['messages'].append(self._summary_message(facts))
                    self._save(session)
            except Exception as exc:
                with self._session_lock(identity):
                    live.update(status='failed',phase='重评未完成',error=str(exc),elapsedSeconds=round(time.monotonic()-started,2))
                    self._save(self.sessions[identity])
        threading.Thread(target=execute,daemon=True,name=run_id).start()
        return self.get(identity,refresh=False)
