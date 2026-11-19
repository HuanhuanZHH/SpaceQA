# 3 小时 / 20GB 约束下的指标优化

原始数据和所有模型产物保存在 icarus0。本地只有代码与小体积指标。完整报告见 `D:/OPC/OPC_官方限制下指标优化实测_2026-10-31.md`。

服务器结果目录：

`/staff/tangjiarui/OPC/opc-agent/runs/constraint-optimization-20261031/`

正式规则程序为 v0.1.4 / 2026.10.31-7；没有将模型分类自动并入正式评分。CLI 和现有 demo 不变。更新规则需要重新启动常驻程序才能生效。

## 结果文件

- `full-rule-release_runtime.json`：最终 30 包全流程计时与输出审计。
- `fixed_field_regression.json`：六族固定字段修正前后公开技术参照的一致率，匹配单位为样本+类型。
- `expanded_dataset_summary.json` / `student_optimization.json`：真实多档训练、三种子、阈值敏感性。
- `selection_and_ablation.json`：仅使用结构特征的对照、图像置换和消零实验。
- `all_family_vlm_runtime.json` / `startup.json`：原权重 NF4 的五族、两种分辨率、并发32测速。抽样，不是全包模型运行。
- `feature_export_validation.json` / `selected_candidate/feature_forest.json.gz`：仅依赖 Python 标准库的 9 特征候选分类器。
- `selected_candidate/student.pt` / `config.json`：通过验证集选择的小图像分类器，2.48MB；依赖 torch。
- `localization_review_queue.jsonl` / `.csv`：127 条待专家确认的定位/困难负例建议，独立专家标注数为 0。
- `optimization_validation.json`：独立数值与产物一致性复核。

## 正式规则复现

```bash
cd /staff/tangjiarui/OPC/opc-agent
python3 opc_agent.py scan \
  --input /staff/tangjiarui/OPC/opc1018/newpublish \
  --output /staff/tangjiarui/OPC/opc-agent/runs/meeting-v014 \
  --seed 20261021
python3 tools/audit_run.py /staff/tangjiarui/OPC/opc-agent/runs/meeting-v014
```

## 可选候选分类器演示

下列输入是本轮预先抽取了 9 个结构特征的 JSONL。它是研究用候选筛查入口，不是正式数据集评分接口，不接受自然语言，也不输出官方细分类定位结论。

```bash
cd /staff/tangjiarui/OPC/opc-agent
python3 experiments/constraint_optimization_20261031/portable_candidates.py \
  --model runs/constraint-optimization-20261031/selected_candidate/feature_forest.json.gz \
  --input runs/constraint-optimization-20261031/expanded_dataset.jsonl \
  --output runs/constraint-optimization-20261031/candidate-demo.jsonl
```

输出文件必须是新路径。每条候选保留 `may_enter_formal_score=false`；仍需定位与可信证据。

## 重训和复核

`optimize_students.py prepare/train` 会写本轮产物，重训前应换用新的独立结果目录。训练扩展只使用原训练组的同源 P2～P4，所有公开测试仍冻结。`all_family_runtime.py` 需要实验端口8002；原8001服务保持原状。`followup.py` 会在自己使用的 GPU1 上启动、停止实验服务。

`validate_results.py` 只复算已有结果，无需启动模型。原版和中间版全量输出只保留工程迭代记录，最终结论采用 `full-rule-release`。

Docker image 的 20GB 限制还需实际构建、导出和离线复现验收；服务器 Docker 权限不足，A10 也尚未实测。
