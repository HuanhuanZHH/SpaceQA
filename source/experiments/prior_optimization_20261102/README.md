# 先验规则、轻量树模型与分组专家优化 / Prior and expert optimization

本目录归档 2026-11-02 的探索性研究代码；旧训练目录保持原样。聚合证据、
冻结策略、逐包指标与复核结果位于
[results/reference/prior_optimization_20261102](../../results/reference/prior_optimization_20261102/)。
真实数据、逐样本 JSONL、权重、图片与缓存均不在 Git 中。

原始 30 包采用相同分组划分：train 86,505 条、val 18,610 条、test 18,450 条。
每个新 GBDT 头完成全部六类训练后，才打开 test；新增阈值与专家选择仅用 val，
先冻结策略后复评 test。未知 `truth_valid` 位不计分。历史测试结果此前已被查看，
所以本轮是**探索性历史复测**，不是全新的独立测试，也不是官方隐藏比赛成绩。
G/C/D 标签来自公开 P0 对照形成的技术 proxy，不能代替科学定位真值。

| 冻结配置 | P1–P4 F1 | 精确率 | 召回率 | P0 样本误报率 | C 类召回率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认 `val_selected_gbdt_extension` | **91.37%** | 92.06% | 90.69% | 0.51% | **24.37%** |
| 可选 `high_recall_review` | 89.84% | 86.04% | **93.98%** | 0.38% | **82.77%** |

主指标是 24 个 P1–P4 包已知 G/C/D 位的微平均，30 包全部单列在参考表中；
P0 没有正例，单列误报。计入 P0 后，默认 F1 为 91.12%，review 为 89.67%。
默认的漏标/虚标 C 类召回仅 24.37%，仍是明确短板；整体 F1 不能代替 C 类验收。
review 是供复核的可选取舍，未替换默认配置，也未用 test 重新选策略。

默认分工：AIA 使用 UNet++ 的 G/C；HMI 使用 GBDT 的 G/C；CME、ULF 使用
GBDT 的 D；EPB 使用 YOLOv8 分割骨干适配的 G/C；DMSP 使用剪枝学生的 G/C。
新规则负责可确认的 D，旧 G/C 硬结构检查保留。确切路线、种子和阈值以
[默认冻结策略](../../results/reference/prior_optimization_20261102/gbdt_extension_v1/frozen_validation_extension.json)
及其引用的
[基础策略](../../results/reference/prior_optimization_20261102/comparison_v1/frozen_validation_policy.json)为准。
[按头策略](../../results/reference/prior_optimization_20261102/headwise_v1/frozen_validation_policy.json)
保留 `f1` 和 `high_recall_review` 两个独立配置。

默认模型产物约 10.63 MB；review 检查点加已知初始化依赖约 109.30 MB。
这些都不等于完整容器体积。**完整 Docker 镜像≤20GB 和 A10 上单数据集完整评价≤3h
尚未完成验收**；缓存概率融合耗时、CPU 树模型前向或 A100 计时不能替代该验收。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `prior_rules.py` | 仅当前输入的字段/尺寸/CSV 检查；先验、硬确认与待核验分开 |
| `compare_versions.py` | 校验原 20 路缓存、在 val 校准、冻结后复评全 30 包 |
| `compare_gbdt_extension.py` | 验证 GBDT 六类训练屏障，再做 val 冻结的分数据类型组合 |
| `compare_headwise.py` | family × G/C/D 选专家与阈值，记录 P0 上限及不可达召回目标 |
| `gbdt/train_gbdt.py` | 三个 HGB 头，全六类训练；当前字段白名单、有效位 mask |
| `gbdt/predict_roadshow.py` | 冻结 GBDT 对路演 12 包作 CPU 候选预测，无 B 拟合 |
| `run_roadshow_prior.py` | fresh 规则的正式独立入口，只运行至 `prior_complete.json` |
| `fuse_roadshow_extension.py` | 默认配置的最终路演融合，支持缺失输入状态 |
| `fuse_headwise_roadshow.py` | 可选按头 review 配置的最终路演融合 |
| `stage_replay.py` | 仅路径字符串的 AST 重定位，创建新副本及哈希回执，不启动实验 |
| `verify_archive.py` | 不用私有数据核验归档哈希和聚合混淆公式 |

`evaluate_roadshow_prior.py` 保留历史执行源码以便溯源。其 fresh 规则阶段成功，
但后半段旧融合不兼容 `data_missing`，不是最终入口。`run_roadshow_prior.py`
在独立副本的 AST 中显式截止到 fresh 完成检查点；不修改归档源码、不执行旧融合尾部。
最终融合使用上表两个 `fuse_*` 脚本。

CME 的目录事件时刻、组锚点和观测帧时刻不能直接强制相等；“约12分钟”不作为
绝对判错依据。ULF 原生 summary/list 矛盾保留待核验，裸 CSV 不证明物理单位。
原始 comparison 的 `trust_ulf_dateobs=False`、时长约定 `either` 已冻结；复现不能
悄悄换成更强的时间锚点假设。GBDT 的额外时间特征只是学习输入，模型发现仍为候选。

路演 12 包共 24,782 条记录、272 个缺失输入均保留。默认与 review 已分别融合并导出，
集成审计均通过；**B 无独立缺陷真值，F1/精确率/召回率保持 null**，模型候选不得
自动变为科学错误。集成汇总/审计在参考目录 `integrated_default/`、`integrated_review/`。

复现命令、外部依赖和路径副本回执见 [protocol.md](protocol.md)。归档与入口自检：

```bash
python experiments/prior_optimization_20261102/verify_archive.py
cd experiments/prior_optimization_20261102
python -m unittest -v test_replay_entries
python stage_replay.py --help
python run_roadshow_prior.py --help
```

## English scope note

This archive contains exploratory re-evaluation code and aggregate evidence,
not data or trained weights. Train/validation/test groups are disjoint; all six
families train before each new model reaches test. Thresholds and routing freeze
on validation, but historical held-out outcomes had already been examined.
Scores are masked P0-derived technical proxies, not official scientific truth.
The default reaches 91.37% P1–P4 micro F1 while C recall remains 24.37%; the
optional review profile trades F1 for 93.98% overall recall and 82.77% C recall.
Roadshow has no independent truth. Complete container size and A10 runtime remain
unverified. Replay copies record path-only AST edits; executed snapshots stay intact.
