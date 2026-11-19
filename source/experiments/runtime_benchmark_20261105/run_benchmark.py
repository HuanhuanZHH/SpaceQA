"""One isolated, sequential complete refresh pass with measured package timings."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time


BASE = Path('/staff/tangjiarui/OPC/eight_class_20261104')
PYTHON = '/staff/tangjiarui/OPC/.venv-image-models/bin/python'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).parent
    reference_path = BASE / 'refresh_20261105/complete.json'
    reference = json.loads(reference_path.read_text())
    dump(root / 'reference_complete_v2.json', reference)
    production_index = Path('/staff/tangjiarui/OPC/chat_console_20261105/report_index.json')
    production_index_sha = sha(production_index)
    source_hashes = {p.name: sha(p) for p in code.glob('*.py')}
    dump(root / 'source_hashes.json', source_hashes)
    env = dict(os.environ)
    for key in list(env):
        lower = key.lower()
        if 'api_key' in lower or 'access_token' in lower or key.startswith(('OPENAI_', 'DEEPSEEK_', 'DASHSCOPE_', 'QWEN_', 'SPACEQA_LLM_', 'SPACEQA_PLANNER_')):
            env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6', MKL_NUM_THREADS='6',
               PYTHONPATH=str(BASE/'code') + ':' + str(BASE/'runtime/src'), PYTHONUNBUFFERED='1', SPACEQA_BENCH_ROOT=str(root))
    hardware = dict(host=platform.node(), platform=platform.platform(), python=sys.version,
                    physicalGpu=1, visibleDevice='1', batch=16, cpuThreads=6)
    for key, cmd in [('nvidia_smi', ['nvidia-smi']), ('cpu', ['lscpu']), ('ram', ['free', '-b'])]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        hardware[key] = result.stdout.strip()
    dump(root/'hardware.json', hardware)
    stopped = threading.Event()
    gpu_rows = []

    def sample_gpu():
        with (root/'gpu_samples.jsonl').open('w') as stream:
            while not stopped.is_set():
                now = time.perf_counter()
                try:
                    result = subprocess.run(['nvidia-smi', '--id=1', '--query-gpu=memory.used,utilization.gpu,power.draw',
                                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                    memory, utilization, power = [float(value.strip()) for value in result.stdout.strip().split(',')]
                    row = dict(perf_counter=now, memoryMiB=memory, utilizationPercent=utilization, powerW=power)
                    gpu_rows.append(row)
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                except Exception as exc:
                    stream.write(json.dumps(dict(perf_counter=now, error=str(exc)))+'\n')
                stopped.wait(1.)

    monitor = threading.Thread(target=sample_gpu, daemon=True)
    monitor.start()
    commands = [
        ('classification', [PYTHON, str(code/'classification_bench.py'), '--fusion', str(BASE/'fusion_v3_missing4'), '--output', str(root/'classification')]),
        ('grounding', [PYTHON, str(code/'grounding_bench.py'), '--roadshow', str(root/'classification'), '--localizer', str(BASE/'localizer_v2'),
                       '--comparison', str(BASE/'candidate_comparison_v1'), '--output', str(root/'grounded'), '--batch', '16']),
        ('finalize', [PYTHON, str(code/'finalize_bench.py')]),
    ]
    started = time.perf_counter()
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    process_rows = []
    try:
        for phase, cmd in commands:
            phase_start = time.perf_counter()
            dump(root/'benchmark_state.json', dict(status='running', phase=phase, startedAt=started_at,
                elapsedSeconds=phase_start-started, processes=process_rows))
            with (root/(phase+'.log')).open('w') as stream:
                process = subprocess.Popen(cmd, cwd=str(BASE/'code'), env=env, stdout=stream, stderr=subprocess.STDOUT)
                code_value = process.wait()
            phase_end = time.perf_counter()
            process_rows.append(dict(phase=phase, pid=process.pid, startPerf=phase_start, endPerf=phase_end,
                                     seconds=phase_end-phase_start, returncode=code_value))
            print(json.dumps(dict(phase=phase, seconds=phase_end-phase_start, returncode=code_value)), flush=True)
            if code_value:
                raise RuntimeError(phase+' failed; see '+str(root/(phase+'.log')))
        ended = time.perf_counter()
    except Exception as exc:
        dump(root/'benchmark_state.json', dict(status='failed', error=str(exc), processes=process_rows))
        raise
    finally:
        stopped.set()
        monitor.join(timeout=8)

    events = []
    for path in (root/'timing').glob('*.jsonl'):
        events.extend(json.loads(line) for line in path.read_text().splitlines())
    events.sort(key=lambda row: row['perf_counter'])
    package_phases = {(row['package'], row['phase']): row for row in events if row['event']=='package_end'}
    shared_loads = collections.defaultdict(float)
    for event in events:
        if event['event']=='shared' and event['stage']=='load_pixel_model':
            shared_loads[event['package']] += event['seconds']
    current = json.loads((root/'complete.json').read_text())
    by_name = {row['package']: row for row in current['packages']}
    ref_by_name = {row['package']: row for row in reference['packages']}
    comparisons = []
    compared_keys = ['family','totalSamples','confirmedSamples','candidateOnlySamples','foundSamples','freshImageForwards','missingDataSamples','localizedByClass','addedLocatedCandidateInstances']
    package_rows = []
    for name in sorted(by_name):
        row = by_name[name]
        phases = {phase: package_phases[(name, phase)] for phase in ['classification','grounding','finalize']}
        seconds = sum(value['seconds'] for value in phases.values())
        shared = shared_loads[name]
        observed = dict(package=name, family=row['family'], samples=row['totalSamples'], imageForwards=row['freshImageForwards'],
                        classificationSeconds=phases['classification']['seconds'], groundingSeconds=phases['grounding']['seconds'],
                        contextScorecardAndIndexSeconds=phases['finalize']['seconds'],
                        measuredPackageTotalSeconds=seconds, sharedModelLoadIncludedSeconds=shared,
                        measuredPackageExcludingSharedModelLoadSeconds=seconds-shared,
                        millisecondsPerSample=1000*seconds/row['totalSamples'],
                        gpuForwardSeconds=phases['grounding']['forward_seconds'],
                        gpuForwardBatches=phases['grounding']['forward_batches'],
                        torchPeakAllocatedMiB=phases['grounding']['torch_peak_allocated_bytes']/1024**2,
                        torchPeakReservedMiB=phases['grounding']['torch_peak_reserved_bytes']/1024**2,
                        csvNumericAxesRead=phases['finalize']['csv_numeric_axes_read'],
                        csvAbsoluteDomainsRead=phases['finalize']['csv_absolute_domains_read'],
                        confirmedSamples=row['confirmedSamples'], candidateOnlySamples=row['candidateOnlySamples'], foundSamples=row['foundSamples'])
        package_rows.append(observed)
        for key in compared_keys:
            if row.get(key) != ref_by_name[name].get(key):
                comparisons.append(dict(package=name, key=key, reference=ref_by_name[name].get(key), benchmark=row.get(key)))
    stage_rows = [dict(package=row.get('package'), phase=row['phase'], stage=row['stage'], seconds=row['seconds'], event=row['event'])
                  for row in events if row['event'] in ['stage','shared']]
    for process in process_rows:
        entry = next(row for row in events if row['event']=='script_enter' and row['pid']==process['pid'])
        done = next(row for row in events if row['event']=='script_complete' and row['pid']==process['pid'])
        process['launchToScriptEnterSeconds'] = entry['perf_counter']-process['startPerf']
        process['scriptDoneToExitSeconds'] = process['endPerf']-done['perf_counter']
        process['packageSecondsSum'] = sum(row['seconds'] for row in events if row['event']=='package_end' and row['phase']==process['phase'])
        process['outsidePackageSeconds'] = process['seconds']-process['packageSecondsSum']
    total_package = sum(row['measuredPackageTotalSeconds'] for row in package_rows)
    summary = dict(status='complete', startedAt=started_at, completedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   benchmarkRoot=str(root), packageCount=len(package_rows), samples=sum(row['samples'] for row in package_rows),
                   imageForwards=sum(row['imageForwards'] for row in package_rows),
                   endToEndSeconds=ended-started, measuredPackageSecondsSum=total_package,
                   outsidePackageStartupAndGlobalSeconds=ended-started-total_package,
                   sharedModelLoadSeconds=sum(shared_loads.values()),
                   fullPassMillisecondsPerSample=(ended-started)*1000/24782,
                   maxMeasuredPackageSeconds=max(row['measuredPackageTotalSeconds'] for row in package_rows),
                   gpuPeakSampledMiB=max((row['memoryMiB'] for row in gpu_rows),default=None),
                   gpuMeanUtilizationPercent=sum(row['utilizationPercent'] for row in gpu_rows)/len(gpu_rows) if gpu_rows else None,
                   physicalGpu=1,batch=16,providerApiCalls=0,trainingRuns=0,
                   reusedUnderlyingExperts=21, freshUnderlyingExpertForwards=0,
                   stageMeasurement='Actual perf_counter spans per package; sequential-stage spans summed, not sample-count apportionment; shared first model load also reported separately.',
                   scope='Fresh current input checks and hashes, frozen expert cache verification, fresh eight-head fusion, every readable RGB image independently forwarded, matching/localization, actual ULF CSV context, scorecards/ranking/index.',
                   endToEndScope='Three subprocess launches through final exit, including imports/model loading/global ranking and writes. Excludes benchmark hardware discovery and summary comparison/report generation.',
                   caveats=['21 upstream experts use verified frozen prediction caches; this is not latency of fresh 21-expert production inference.',
                            'ULF package timings include CSV/rules/fusion/context/postprocessing; ULF has no RGB image forward.',
                            'Host GPU is A10040GB, not official target-hardware certification; model serving and API explanation not measured.',
                            'Single pass; live operating-system file caches not purged.'],
                   referenceCountsMatch=not comparisons, comparisonDifferences=comparisons,
                   productionReportIndexUnchanged=sha(production_index)==production_index_sha,
                   processes=process_rows, packages=package_rows)
    dump(root/'benchmark_summary.json', summary)
    write_csv(root/'package_timings.csv', package_rows)
    write_csv(root/'stage_timings.csv', stage_rows)
    write_csv(root/'process_timings.csv', process_rows)
    lines = ['# 12包完整刷新实测（隔离单次pass）', '',
             '- 范围：规则与输入哈希核验＋21专家缓存核验＋八头融合＋全部可读图像新一轮GPU前向与匹配＋ULF实际CSV上下文＋评分卡/排序/索引。',
             '- 21个上游专家复用已校验的预测缓存，没有重新运行这21个专家；不含API解释。',
             f"- 物理GPU1，A100 40GB，batch16；12包 {summary['samples']:,} 样本，{summary['imageForwards']:,} 次新图像前向。",
             f"- 全流程墙钟：{summary['endToEndSeconds']:.3f} 秒；逐包实际阶段耗时之和：{total_package:.3f} 秒；进程启动/导入及全局收尾另计 {summary['outsidePackageStartupAndGlobalSeconds']:.3f} 秒。",
             f"- 首次模型加载 {summary['sharedModelLoadSeconds']:.3f} 秒已包含在首次使用包内，另列以便区分共享开销。",
             f"- 最慢单包 {summary['maxMeasuredPackageSeconds']:.3f} 秒；采样GPU显存峰值 {summary['gpuPeakSampledMiB']} MiB。",
             f"- 与refresh v2的逐包计数/定位统计一致：{summary['referenceCountsMatch']}；生产索引未变：{summary['productionReportIndexUnchanged']}。", '',
             '每包“合计”是真实三个阶段计时相加，未按样本数分摊总耗时。ULF包含规则/融合与CSV时间上下文，不以其空图像阶段代表整包时间。', '',
             '| 数据包 | 样本 | 新图像前向 | 规则/缓存核验/融合(s) | 图像/匹配/聚合(s) | 上下文/最终评分卡(s) | 实测合计(s) | ms/样本 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in package_rows:
        lines.append('| '+row['package']+' | '+str(row['samples'])+' | '+str(row['imageForwards'])+' | '+
                     ' | '.join(f"{row[key]:.3f}" for key in ['classificationSeconds','groundingSeconds','contextScorecardAndIndexSeconds','measuredPackageTotalSeconds','millisecondsPerSample'])+' |')
    lines += ['', '本次仅在A100与操作系统当前缓存状态下测量一次；不能替代官方目标硬件的3小时验收，也不能代表21个上游专家全部新运行的耗时。完整明细见 package_timings.csv、stage_timings.csv、process_timings.csv 和 benchmark_summary.json。']
    (root/'benchmark_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(root/'benchmark_state.json', dict(status='complete', summary=str(root/'benchmark_summary.json'), endToEndSeconds=ended-started, referenceCountsMatch=not comparisons))
    print(json.dumps({k:summary[k] for k in ['status','endToEndSeconds','maxMeasuredPackageSeconds','referenceCountsMatch','productionReportIndexUnchanged']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    main(parser.parse_args().root)
