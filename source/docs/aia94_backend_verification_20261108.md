# AIA94 后端迁移实测核验（2026-11-08）

## 核验目的

把 AIA94 的后端思路以数据类型适配器形式接入 SpaceQA，对 AIA、HMI、CME、EPB、DMSP、ULF 各抽取真实发布包中的 3 个完整 data–label 配对，分别运行 `legacy` 与 `enhanced` 配置。核验只比较后端迁移是否改变原有规则结论；不把抽样结果写成全年准确率，也不把策略观测写成官方缺陷真值。

## 实测结果

| 数据类型 | 样本数 | 旧模式确认样本 | 增强模式确认样本 | 旧模式候选样本 | 增强模式候选样本 | 核心结果差异 | 增强策略回退/适用情况 |
|---|---:|---:|---:|---:|---:|---:|---|
| AIA | 3 | 0 | 0 | 0 | 0 | 0 | 3 个普通图像，无科学数组或 `.zattrs`，按显示图像降级 |
| HMI | 3 | 0 | 0 | 0 | 0 | 0 | 3 个普通图像，无科学数组或 `.zattrs`，按显示图像降级 |
| CME | 3 | 0 | 0 | 0 | 0 | 0 | 3 个事件记录上下文正常 |
| EPB | 3 | 0 | 0 | 0 | 0 | 0 | 3 个几何记录上下文正常 |
| DMSP | 3 | 1 | 1 | 0 | 0 | 0 | 2 个几何记录正常；1 个样本没有显式几何记录，保留回退说明 |
| ULF | 3 | 0 | 0 | 0 | 0 | 0 | 3 个样本需要完整 CSV 时间轴，当前子集按 `full_csv_axis_required` 记录 |
| **合计** | **18** | **1** | **1** | **0** | **0** | **0** | — |

“核心结果差异”比较每个样本的 `findings`、`confirmed_bits` 和 `candidate_bits`，18 个样本均为 0。原因是这 18 个样本没有触发独立侧车曝光异常或无效 CSV 时间轴这两类正式门槛；因此本次增强结果只增加 `observations.backend_strategy`，没有新的正式 finding。

对应图表由 `tools/plot_backend_comparison.py` 生成：

- `outputs/backend_comparison_20261108/backend_mode_comparison.png`：旧/增强模式的确认样本和候选样本对比；
- `outputs/backend_comparison_20261108/backend_observation_coverage.png`：增强策略实际可用上下文与安全降级数量；
- `outputs/backend_comparison_20261108/backend_mode_comparison.csv`：绘图用的审计数据。

第一张图几乎没有柱子是该抽样数据的实际结果，不代表正式融合永远不改变结果：增强扫描遇到独立侧车曝光异常或无效 CSV 时间轴时会生成标准正式 finding。第二张图体现了当前新增的上下文能力：CME/EPB/DMSP 能记录相应结构信息，AIA/HMI/ULF 在缺少科学源数据时明确标记需要源数据，而不是产生未经验证的物理结论。

## 各类型迁移内容

- **AIA**：保留曝光、稳健去噪、双阈值亮区和跨帧上下文的接口；只有科学数组与独立元数据齐备时才可继续做科学强度处理。普通 PNG/JPG 不推断 DN/s。
- **HMI**：复用区域候选、元数据和时间上下文，但不直接套用 AIA94 的阈值或太阳几何。
- **EPB/DMSP**：使用原生图像坐标，统计矩形框、多边形和掩膜的几何与覆盖关系；不伪造多边形，也不使用太阳自转假设。
- **CME**：围绕 JSON Pointer、事件时间和参数来源建立事件记录上下文；不做像素亮区分割。
- **ULF**：围绕完整 CSV 时间轴、采样率、时间段和单位来源建立时间序列上下文；不做图像分割。

## 可复现命令

本地回归：

```bash
PYTHONPATH=src python -m unittest discover -s tests -q
python scripts/reproduce.py verify --manifest configs/source_manifest_20261108_backend_final.json
```

实际包核验使用 icarus0 的只读临时目录 `/tmp/spaceqa_backend_verify_20261108`，原始包没有修改。完整发布包很大（AIA/HMI/CME 每包约 13–17 GB），因此采用每类 3 个配对做链路核验；全包重新运行属于性能测量任务，不能由本报告的 18 个样本外推。

## 结论与边界

迁移已经覆盖六类数据类型，并可通过 `backendProfile=legacy` 回退到旧对话模式。当前发布数据中的 AIA/HMI 是显示图像，缺少 AIA94 所需的科学数组和元数据，因此本次只能确认安全降级行为，不能宣称 AIA94 全年亮区或事件召回率提升。要让 AIA94 的科学处理真正参与候选生成，还需要提供全年 Zarr/FITS/HDF5 数组、对应元数据和运行环境；这属于输入条件补全，不应通过猜测完成。
