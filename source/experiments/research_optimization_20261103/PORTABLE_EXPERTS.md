# 冻结默认专家：独立 fresh 推理研究入口

`portable_experts.py` 读取一个明确的数据包路径和类型，在新目录重新生成当前样本输入，再实际运行冻结默认专家。没有训练、阈值调整、B 集拟合、clean 检索或默认 Web 服务改动。

| 数据类型 | 默认专家 | 冻结阈值 G / C / D |
| --- | --- | --- |
| AIA_AR_sample | U-Net++ | 0.85 / 0.80 / 禁用 |
| HMI_AR_sample | GBDT | 0.95 / 1.01 / 禁用 |
| CME_C2_sample | GBDT | 禁用 / 禁用 / 0.65 |
| EPB_seg_sample | YOLOv8 segmentation backbone adapted classifier | 0.60 / 0.95 / 禁用 |
| DMSP_SSUSI | pruned_seed20261101 | 0.65 / 0.90 / 禁用 |
| ULF_sample | GBDT | 禁用 / 禁用 / 0.95 |

1.01 表示冻结策略禁用该头。G/C/D 分别是几何、数量、元数据/类别/时段等技术分组。模型输出只做候选；它们不等于官方九类已定位缺陷，不能据此宣称官方召回率或 F1。

## 依赖和边界

- 服务器 Python：`/staff/tangjiarui/OPC/.venv-image-models/bin/python`。
- 验证环境：PyTorch 2.9.0+cu128、NumPy 2.2.6、Pillow 12.3.0、scikit-learn 1.7.2、Ultralytics 8.3.228。
- 冻结训练代码/权重依赖原 `/staff/tangjiarui/OPC/opc-agent` 树。入口校验其 SHA，不修改原文件；这还不是独立交付镜像。
- YOLO 构造额外依赖 `models/image-model-pilot/yolov8n-seg.pt`，已纳入 SHA 锁。不得只计算微调 checkpoint 大小就认定完整镜像合规。
- 研究输出必须位于 `/staff/tangjiarui/OPC/research_optimization_20261103/` 的全新子目录，拒绝覆盖已有结果或写入输入包。
- 显式 `--device cpu` 或 `--device cuda:0`。GPU 研究 batch 限制 1～4、PyTorch 分配预算最多 4GiB；GPU1 不可用。
- 缺数据、缺标注均保留记录并 `abstain`，无模型概率；准备失败单列、整次 smoke 失败，不能当正常缺图。
- 冻结规则适配器按文件名建立字典。递归包中出现重复标签文件名或重复数据文件名时，入口明确拒绝这类歧义输入，避免规则项被覆盖。
- 模型预处理直接使用冻结的 `feature_view`、`current_input`、Pillow 双线性尺寸变换。公开包支持 `data/label`；B 包支持原平铺结构。
- `public-v1` 沿用公开训练的 ULF dateObs 和 CME 当前 image_time 特征；`roadshow-b-v1` 精确重放历史 B 的 ULF 显示起点与 GBDT 特征约定。此差别被明确记录，不能静默混用。

## 建立一次锁

```bash
PY=/staff/tangjiarui/OPC/.venv-image-models/bin/python
ENTRY=/staff/tangjiarui/OPC/research_optimization_20261103/portable_experts.py
$PY "$ENTRY" freeze \
  --code-root /staff/tangjiarui/OPC/opc-agent \
  --training-root /staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031 \
  --policy-root /staff/tangjiarui/OPC/inference_prior_20261102 \
  --prior-code-root /staff/tangjiarui/OPC/spaceqa-agent/experiments/prior_optimization_20261102 \
  --output /staff/tangjiarui/OPC/research_optimization_20261103/portable_lock_NEW
```

已有验证锁为 `portable_lock_v1/lock.json`，包含 40 项源代码、策略、checkpoint、初始化文件、GBDT/vectorizer SHA。锁同时检查原训练覆盖 30 包、先冻结训练/验证阈值再打开测试的凭证。

## 显式运行一个新包

```bash
PY=/staff/tangjiarui/OPC/.venv-image-models/bin/python
ROOT=/staff/tangjiarui/OPC/research_optimization_20261103
$PY "$ROOT/portable_experts.py" run \
  --input /absolute/path/to/ONE_PACKAGE \
  --family AIA_AR_sample \
  --output "$ROOT/new_package_NEW" \
  --lock "$ROOT/portable_lock_v1/lock.json" \
  --device cuda:0 --batch-size 2 \
  --release-profile public-v1
```

`--family` 必须指定六种支持类型之一；B 包使用 `--release-profile roadshow-b-v1`。可用 `--limit N` 或 `--selection-json labels.json` 做固定样本验证；这时不会声称全包覆盖。正常全包模式还会保留无标注的孤立原始数据。

产物包括 `request.json`、`inventory.jsonl`、`legacy_check_events.json`、`selected_candidates.jsonl`、`reference_comparison.json` 和 `complete.json`。`fresh_forward_batches`/`fresh_forward_samples` 只统计实际当前样本模型前向；YOLO 构造探测和 GBDT 三个树头调用另列；缓存概率复用数固定为零。原始输入结束时复核 SHA。

## 已实施验证

- `test_portable_experts.py`：7 项保护测试，覆盖模型篡改、选择越界、缺文件路径保留、HMI/CME/AIA 目录配对、阈值冻结、输出覆盖/原始目录保护、缺标注数据 SHA 复核及重复文件名拒绝。
- `run_portable_smoke.py`：六类各固定两个公开 P1 样本，按排序与文件存在性选择，不看目标或预测。分别以 batch=1 和 batch=2 fresh 重放；对照历史缓存核验图像 SHA、256 特征、prompt、input SHA、概率和冻结候选位。
- `validate_portable_edges.py`：生成微型合成输入，分别实测缺数据与缺标注，二者均 abstain、零前向。
- 原始失败尝试被保留；只以成功目录的 `complete.json` 为准。不能用 smoke 的耗时推算全包或 A10 上的 3 小时验收，也未重新计算 F1。

成功验证目录及逐类结果见研究根目录的汇总文件。此入口暂未接网页，线上任务仍保留现有执行能力标识。

### 2026-11-03 实测

| 验证 | 覆盖 | 状态 | 与历史概率最大绝对差 |
| --- | --- | --- | --- |
| 公开数据 batch=2 | 六类×2固定P1样本 | 全部通过 | 0.000496641 |
| 公开数据 batch=1 | 同一批12样本 | 全部通过 | 0.000496641 |
| 路演 B batch=2 | AIA/EPB/ULF各2样本 | 全部通过 | 0.002386630 |
| 缺数据、缺标注 | 2种合成输入 | 均abstain、零前向 | 不适用 |

以上所有对照的当前视图SHA、256特征、prompt和input SHA均完全一致；冻结模型候选位也均一致。B 包中没有 HMI/CME/DMSP，明确记为未覆盖，不把另外三类结果代替它们。模型概率的微小差异可能来自 CUDA/bfloat16 与 batch 数值路径，全部低于预先设定的0.02 smoke容差；它不是精度指标。

batch2 的最高 PyTorch 分配峰值为 619,357,696 字节（AIA）；GBDT未使用GPU。这个小样本的加载和推理总耗时每类约1.8～4.7秒，样本过少，不能用于推算全量时间，也不能据此声称batch2比batch1更快。原GPU0常驻Qwen服务保持运行，GPU1没有使用。

研究根的主要成功目录：`portable_public_smoke_b2_v3`、`portable_public_smoke_b1_v1`、`portable_roadshow_smoke_b2_v2`、`portable_missing_edges_v1`。增加库缓存落入研究目录的约束后，用 `portable_env_epb_v1` 再次验证 EPB 两样本通过。

最终回执为 `portable_validation_summary_v2.json`，旧 `v1` 原样保留。v2计入公开12样本×两种batch的24次、B的6次、环境复测的2次，共32次认可的样本推理；对应18条不同原始标注记录。18次使用GPU图像模型，14次为CPU树模型。两个缺输入合成案例没有前向，不计入32次。另有保留失败尝试中的10次前向（例如汇总程序失败前已完成AIA），所以所有保留尝试共42次；它们不加入认可对照数。

每次运行的`request.json`记录实际执行入口SHA。v2同时记录当前代码/文档SHA、各历史运行实际SHA及最新7项测试输出。模型重放之后增加的重复文件名保护已通过最新单元测试，但不会把旧模型重放改称为当前源码执行。当前源码SHA与某次历史运行不同，会明确记录`executed_entry_matches_current=false`；没有覆盖原始成功或失败结果。
