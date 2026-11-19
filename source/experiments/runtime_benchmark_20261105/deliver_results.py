"""Summarize actual timings without conflating cached and first-run costs."""
from pathlib import Path
import collections
import json
import zipfile

ROOT = Path('D:/OPC/outputs/runtime_benchmark_20261105')


def main():
    current = json.loads((ROOT/'benchmark_summary.json').read_text(encoding='utf-8-sig'))
    history = json.loads((ROOT/'historical_21expert_components.json').read_text(encoding='utf-8-sig'))
    comparison = json.loads((ROOT/'per_sample_output_comparison.json').read_text(encoding='utf-8-sig'))
    assert comparison['referenceSelection'] == 'published_report_index_sourcefiles'
    assert comparison['status'] == 'prediction_and_location_pass_with_documented_context_metadata_migration'
    assert comparison['samples'] == 24782 and comparison['packageCount'] == 12
    for key in ['sampleIdsEqual', 'predictionRecordsExactlyEqual', 'probabilitiesExactlyEqual',
                'semanticFindingLocationSetsEqual', 'allFindingContentEqualExceptNumericTimeContextEnrichment']:
        assert comparison[key], key
    rows = sorted(current['packages'], key=lambda row: row['package'])
    assert current['status']=='complete' and current['referenceCountsMatch'] and current['productionReportIndexUnchanged']
    assert len(rows)==12 and sum(row['samples'] for row in rows)==24782
    actual_sum = sum(row['measuredPackageTotalSeconds'] for row in rows)
    assert abs(actual_sum-current['measuredPackageSecondsSum'])<1e-5
    assert abs(current['endToEndSeconds']-actual_sum-current['outsidePackageStartupAndGlobalSeconds'])<1e-5
    totals = history['totals']
    minutes, seconds = divmod(current['endToEndSeconds'], 60)
    text = ['# 路演12包评价耗时实测', '',
        f"本次在icarus0的A100 40GB物理GPU1、图像batch16、数值库6线程配置下，重新跑完12包、24,782条登记样本。整轮墙钟 **{int(minutes)}分{seconds:.2f}秒**，含共享启动与汇总的加权平均 **{current['fullPassMillisecondsPerSample']:.2f}毫秒/条**。",
        '',
        '本轮测量的是当前复评流程：重新核验原始文件与哈希、验证21专家预测缓存、重新执行八类融合、10,332张现存图片的新RGB定位与匹配、ULF真实CSV上下文、评分卡、排序和索引写出。21个基础专家预测复用已校验缓存，未再次执行其神经前向；不包含按需大模型API解释。',
        '',
        '## 本轮逐包实际耗时', '',
        '包级合计为同次批处理内该包三阶段真实计时之和；包内读取与结果写出已计入。首次图像模型加载包含在首包，后续包共享模型。整批额外启动、导入和全局收尾单列，不按样本数分摊进每包。', '',
        '| 数据包 | 登记样本 | 核验与融合(s) | 图像与匹配(s) | 上下文与报告(s) | 包内合计(s) | 平均(ms/条) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append('| '+row['package']+' | '+str(row['samples'])+' | '+' | '.join(f"{row[key]:.3f}" for key in ['classificationSeconds','groundingSeconds','contextScorecardAndIndexSeconds','measuredPackageTotalSeconds','millisecondsPerSample'])+' |')
    text += ['', f"12包包内合计 **{actual_sum:.3f}秒**；额外共享启动/导入/全局收尾 **{current['outsidePackageStartupAndGlobalSeconds']:.3f}秒**；整轮 **{current['endToEndSeconds']:.3f}秒**。首次图像模型加载 **{current['sharedModelLoadSeconds']:.3f}秒**已经计入首包，不重复相加。",
        '',
        '平均值按登记样本数加权，包含272条缺数据记录；这是批处理摊销耗时。ULF的整包时间包含规则、融合和CSV上下文处理，不使用其跳过RGB任务的时间作为整包时间。',
        '', '## 按数据类型汇总', '',
        '| 类型 | 登记样本 | 包内实际秒数合计 | 加权平均(ms/条) |', '|---|---:|---:|---:|']
    grouped = collections.defaultdict(list)
    for row in rows:grouped[row['family'].split('_')[0]].append(row)
    for family, items in sorted(grouped.items()):
        count=sum(item['samples'] for item in items); elapsed=sum(item['measuredPackageTotalSeconds'] for item in items)
        text.append(f'| {family} | {count:,} | {elapsed:.3f} | {1000*elapsed/count:.3f} |')
    text += ['', '## 首次运行21个基础专家的成本', '',
        '另外核对了历史全部12包的264份原始计时回执，覆盖原始输入准备、20个神经专家及GBDT。以下为这些实际阶段的相加；不能将其当作一次完整冷启动墙钟。当前八类融合、最新定位与报告等后续费用另计。', '',
        '| 历史实测阶段 | 12包累计分钟 | 加权平均(ms/登记样本) |', '|---|---:|---:|',
        f"| 21个基础专家 | {totals['expert21Seconds']/60:.2f} | {1000*totals['expert21WeightedSecondsPerRegisteredSample']:.2f} |",
        f"| 原始输入准备＋21个基础专家 | {totals['sumRecordedPrepareAnd21Seconds']/60:.2f} | {1000*totals['prepareAnd21WeightedSecondsPerRegisteredSample']:.2f} |", '',
        '| 数据包 | 输入准备(s) | 21专家(s) | 已记录阶段合计(s) |', '|---|---:|---:|---:|']
    for row in history['packages']:
        text.append(f"| {row['package']} | {row['rawPrepareSeconds']:.2f} | {row['expert21Seconds']:.2f} | {row['sumRecordedPrepareAnd21Seconds']:.2f} |")
    text += ['',
        '历史神经专家计时包含逐包模型加载、数据加载、前向和CUDA同步；预测JSON序列化在计时后进行。GBDT全局模型加载缺独立计时。硬件记录为icarus1 A100，输入准备与GBDT的精确CPU型号未记。完整回执、SHA与范围见 historical_21expert_components.json/.md。',
        '', '## 复核与适用范围', '',
        f"本轮全部可读图片重新前向，GPU采样显存峰值 **{current['gpuPeakSampledMiB']:.0f}MiB**。与上一版各包检出、确认、候选和定位统计一致：**{current['referenceCountsMatch']}**；线上报告索引保持原样：**{current['productionReportIndexUnchanged']}**。",
        '',
        f"按已发布报告索引指定的真实结果文件，逐一复核全部{comparison['samples']:,}条样本：八类预测、规则、候选集合和精确概率一致，指向的样本、标注、图像区域和被质疑事件区间一致。存在{comparison['changedFindingRecords']:,}条ULF时段候选的说明文字或CSV时间上下文字段差异，已独立读取CSV验证数值一致；其中部分派生问题ID改变。结果不是逐字节相同，分类与定位语义未变。详见 per_sample_output_comparison.json。",
        '',
        '本轮是单次完整批处理，系统文件缓存未清空。整轮墙钟包含三阶段子进程启动至退出；硬件探测、测试结果比较和本说明生成不计入评价时间。每包数字适用于本次批内执行，12次分别启动的模型冷加载与全局费用需要另测。当前A100上的结果不等同于目标硬件认证。',
        '', '## 文件', '',
        '- benchmark_summary.json：完整结构化结果及计时范围。',
        '- package_timings.csv / stage_timings.csv / process_timings.csv：逐包、细阶段和子进程明细。',
        '- roadshow_runtime_current.png / .svg：可放入PPT的本轮实测图。',
        '- historical_21expert_components.json / .csv / .md：全部基础专家的历史成本证据。',
        '- historical_051_end_to_end.json / .md：旧版0.5.1独立任务计时，单独保留用于追溯，未混入本轮平均。',
        '- benchmark_script_readonly_review.md：三份计时副本的算法AST比对结论。',
        '- per_sample_output_comparison.json：按已发布报告索引进行的全量分类、概率及定位一致性审计。',
    ]
    (ROOT/'路演12包耗时实测.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    snapshot = dict(passed=True, samples=current['samples'], packageCount=12,
                    measuredPackageSecondsSum=actual_sum, overallWallSeconds=current['endToEndSeconds'],
                    weightedMillisecondsPerSample=current['fullPassMillisecondsPerSample'],
                    predictionAndLocationAuditPassed=True,
                    excludesFreshUpstream21=True, includesApiExplanation=False)
    (ROOT/'计时汇总独立复算.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
    names = ['路演12包耗时实测.md','benchmark_summary.json','benchmark_report.md','hardware.json',
             'package_timings.csv','stage_timings.csv','process_timings.csv','计时汇总独立复算.json',
             'historical_21expert_components.json','historical_21expert_components.csv','historical_21expert_components.md',
             'historical_051_end_to_end.json','historical_051_end_to_end.md','benchmark_script_readonly_review.md',
             'per_sample_output_comparison.json','exact_finding_published_v2_field_diff_audit.json',
             'roadshow_runtime_current.png','roadshow_runtime_current.svg']
    with zipfile.ZipFile(ROOT/'SpaceQA_路演耗时实测_20261105.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,name)
    print(json.dumps(snapshot,ensure_ascii=False))


if __name__=='__main__':main()
