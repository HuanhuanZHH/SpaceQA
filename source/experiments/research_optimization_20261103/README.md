# 2026-11-03：独立目标定位与新包专家推理

参考材料为 `D:/download/deep-research-report (1).md`。本目录实现并实际测试其中的独立对象预测、实例核对和独立复核方向，保留冻结默认组合及历史结果。

## 已完成与保留决策

| 工作 | 实测范围 | 结论 |
| --- | --- | --- |
| 冻结专家重新推理 | 六类公开样本，路演实际具有的AIA/EPB/ULF；认可对照32次，18条不同标注 | 输入与特征一致、冻结候选位一致；具备研究用新包入口，未接网页 |
| 图像独立定位 | AIA/HMI/EPB/DMSP四类，两个微型U-Net分别联合训练三轮 | 加权BCE+Dice改善弱掩膜IoU，但空掩膜预测率高；不替换默认 |
| ROI固定网格消融 | 相同权重、相同阈值，对照全图 | 未证明ROI稳定提升；不可声称解决P1轻污染 |
| 实例核对 | 594条原生参考导出，80例对接核心 | 19条待复核候选；未知类别不判类别错误，模型漏检不判虚标 |
| 双人盲审队列 | 六类各20例，共120例公开验证样本 | 空表合并仍为120例未知；尚无独立科学真值或新官方F1 |

本轮未改变模型、规则的正式评分路径、默认阈值、旧指标和原始数据。线上网页仍使用原能力标识。新模型只输出待复核区域，不能扣科学质量分。

## 文件

- `portable_experts.py`、`PORTABLE_EXPERTS.md`：明确类型与包路径的冻结专家真实推理入口。
- `portable_validation_summary_v2.json`：成功对照、失败尝试、前向计数及源码版本分别记录。
- `spatial_localizer.py`、`README_localizer.md`：图像独立预测的研究入口及边界。
- `review_protocol.py`：按同源组抽样、双人盲审、共识及未知范围保护。
- `../../src/spaceqa_agent/instance_audit.py`、`../../docs/instance_audit.md`：一对一匈牙利匹配、空间/时间定位及独立缺陷评价。
- `../../refine-logs/EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`：假设、门槛与实际决策。

小型实验产生的是P0标注掩膜复现指标，不是官方九类缺陷定位P/R/F1。原始G/C/D分类指标也必须另外报告。CME具有图像，但没有本路线已验证的像素监督；ULF是时序，两者对此像素模型明确记为N/A。

## 服务器产物

根目录：`/staff/tangjiarui/OPC/research_optimization_20261103/`。

- `portable_public_smoke_b2_v3/`、`portable_public_smoke_b1_v1/`：公开六类小样本对照。
- `portable_roadshow_smoke_b2_v2/`：路演三类小样本对照。
- `portable_env_epb_v1/`：最后的环境改动复测。
- `portable_missing_edges_v1/`：缺数据/缺标签均弃权、零前向。
- `localizer/pilot_v1/`：实际三轮训练、验证阈值、测试、原生参考与候选。
- `review/queue_v2/`：双人复核队列；私有抽样映射不得发给盲审员。

源样本、权重和预览保留在服务器，不提交仓库。各历史运行保留其实际启动代码SHA；后来增加的保护不伪装成旧运行已执行的逻辑。

## 使用

新包专家运行命令见 `PORTABLE_EXPERTS.md`，定位训练见 `README_localizer.md`。每次必须选择全新输出目录，不能覆盖历史、输入或训练数据。

实例核对及独立评价：

```bash
python -m spaceqa_agent.instance_audit candidates --input objects.json --output candidates.json
python -m spaceqa_agent.instance_audit evaluate --input independently_reviewed.json --output metrics.json
```

`evaluate`需要独立、已审阅的真值范围与显式定位容差。未审样本/类别、未知真值不当负例，关键标记不全时关键召回为空；`official_metrics`始终为空，不冒充正式评测。

已由两人分别填写队列后，在服务器使用新输出目录合并：

```bash
/staff/tangjiarui/OPC/.venv-image-models/bin/python \
  /staff/tangjiarui/OPC/research_optimization_20261103/code/review_protocol.py reconcile \
  --queue-dir /staff/tangjiarui/OPC/research_optimization_20261103/review/queue_v2 \
  --output /staff/tangjiarui/OPC/research_optimization_20261103/review/reconciliation_NEW
```

空缺陷列表只在双方完成独立复核且范围可观察时有意义。缺图不能声明视觉缺陷已全部检查，损坏图像不能形成视觉负例；图像位置必须匹配实际尺寸和数据SHA。人员身份与独立性仍需团队组织核实，软件中的声明不能认证真实人工工作。

合并结果使用盲审 `review_id` 作为 `sample_id`。正式内部复算前，用私有 `private_sampling_map.jsonl` 将其映射到原始 `source_id`，并将每例 `reviewed_classes` 显式转为核心评价的 `reviewed_sample_classes`；不能把其余样本或类别当正常。不能拿这批已用于开发的验证样本声称独立未触碰测试。

## 报告建议中的修正

SAM不保证识别空间科学目标；HMI是太阳磁图；CME有观测图像；ULF与CME不能套用同一时间基准；路演四包对象不重叠，不能取一包做配对clean。模型共识、无掩膜或更多掩膜均不能充当独立科学真值。本轮不从头训练大型MAE/JEPA，也不以缺少证据的SAM结果替代判定。
