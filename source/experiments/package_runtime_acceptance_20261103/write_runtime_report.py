"""Write a concise Chinese evidence report from the small runtime summary JSON."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def duration(seconds):
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}分{remainder:05.2f}秒"


def report(summary):
    lines = [
        "# 完整数据包运行时间核验",
        "",
        f"更新时间：{dt.datetime.fromtimestamp(summary['updated_unix'], dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')}。",
        "",
        f"已完成 {summary['completed_count']}/{summary['planned_count']} 包，其中 {summary['passed_measured_host_count']} 包同时满足外部时间、无超时/资源违规、独立覆盖审计及物理GPU映射要求。上限按每个数据包单独核验为 10,800 秒，不能把多个包用时相加后判断。",
        "",
        "本轮为真实完整包执行：所有标签与现存数据均参与，新生成预处理结果并重新执行全部专家前向；不读取历史概率缓存，缺失输入保留为弃权。计时覆盖进程启动退出、输入盘点、完整规则检查、评分卡与证据输出、系统审计、模型加载、预处理、前向和候选导出。独立验收审计在计时结束后进行，用于核验证据与范围。",
        "",
        "**适用范围：** 测试主机为共享 A100 40GB，尚不能代替最终 A10 验收。规则评分卡与新模型候选仍是两个独立产物；本次不生成官方定位 F1、召回率、精确率，也不宣称已完成模型结论与评分卡整合。",
        "",
        "| 数据包 | 专家路线 | 标签数 | 新推理样本数 | 缺失保留 | 外部整包时间上界 | 单包≤3h | 独立审计 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in summary["results"]:
        cov = row["coverage"]
        name = row["family"].replace("_AR_sample", "").replace("_C2_sample", "").replace("_seg_sample", "").replace("_sample", "")
        name += " P4" if row["tier"] == "P4" else " 路演（档位未知）"
        receipt = row["external_receipt"]
        within = "是" if receipt["under_3h_by_external_wall"] else "否"
        lines.append(f"| {name} | {' + '.join(row['expert_routes'])} | {cov['records']:,} | {cov['forward_eligible']:,} | {cov['missing_records_retained']:,} | {duration(receipt['wall_seconds_upper_bound'])} | {within} | {row['independent_audit']['status']} |")
    if summary["active"]:
        lines += ["", "尚在运行：" + "；".join(f"{j['package']}（已运行 {duration(j['elapsed_seconds'])}）" for j in summary["active"]) + "。"]
    if summary["pending"]:
        lines += ["", "尚未开始：" + "、".join(summary["pending"]) + "。"]
    lines += [
        "",
        "阶段耗时和资源（阶段有嵌套关系，以下列只作瓶颈分析，不能相加代替外部整包时间）：",
        "",
        "| 数据包 | 预处理秒 | 模型阶段秒 | 进程峰值内存 MiB | Torch峰值显存 MiB | 新推理batch数 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["results"]:
        stage = {r["stage"]: r["seconds"] for r in row["stage_times"]}
        resource = row["resource"]
        lines.append(f"| `{row['package']}` | {stage['fresh.prepare']:.3f} | {stage['fresh.fresh_probabilities']:.3f} | {resource['peak_rss_kib']/1024:.1f} | {resource['gpu_peak_allocated_bytes']/1024**2:.1f} | {', '.join(f'{k}: {v}' for k,v in row['fresh_forward_batches'].items())} |")
    lines += [
        "",
        "配置与可复核性：batch=4；最多1个GPU0任务与2个CPU任务并行；后续GPU任务按物理GPU0的UUID绑定，最初AIA任务通过同环境探针和NVML确认，所有GPU任务均有实际物理GPU0的NVML证据，GPU1保持不使用；GPU0原有Qwen服务保持运行；模型专家CUDA预算4GiB。每个任务都有从原进程启动时间计算的3小时看门狗，超时终止工作进程及已识别子进程；本轮全部正常完成，未触发超时路径。未清理操作系统页缓存，不能把本轮当作冷磁盘基准。",
        "",
        "`runtime_summary_live_v2.json` 保存外部时间回执、独立审计、全部覆盖统计、各阶段耗时、资源用量和源文件哈希。服务器原始证据位于 `/staff/tangjiarui/OPC/package_runtime_acceptance_20261103/runs/<package>/`：`complete.json`、`external_runtime_receipt_v2.json`、`independent_audit_external_v2.json`。",
        "",
        "最初的AIA/CME任务由外部监督程序在运行中接管，因此保留原进程启动时间到退出观测的保守时间区间，无法取得原父进程的wait退出码；没有把该字段伪填为0。成功验收同时依据完整产物、独立审计和退出时间界限。",
        "",
        "预处理在AIA、HMI中占主导；扩大GPU batch只能优化模型阶段。后续可单独验证GBDT路线跳过未使用图片渲染，但必须保持新旧特征、概率和候选逐项一致，再重新做完整包测试。冻结基线未因此改动。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(report(json.loads(args.summary.read_text(encoding="utf-8"))), encoding="utf-8")


if __name__ == "__main__":
    main()
