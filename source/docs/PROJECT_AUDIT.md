# SpaceQA 项目交接审计（系统架构与文件索引）

- 审计日期：2026-11-08（本地工作区）
- 仓库：`D:\OPC\spaceqa-agent`
- 当前仓库版本：`0.6.0`（见 `pyproject.toml` / `README.md`）
- 审计范围：源码入口、模块依赖、研究代码、配置、交付文档、可运行回归；不读取或修改真实数据集。
- 结论性质：这是仓库级交接审计。外部服务器上的私有数据、模型权重、缓存、API Key 和完整运行目录必须由部署负责人另行提供收据。

## 1. 实际系统架构

```text
数据包目录 / publish 根目录
        |
        v
spaceqa CLI: scan -> engine.py
        |
        +--> 清单、配对、字段/尺寸/时间/来源规则
        +--> 评分卡 04_评分卡
        +--> 排序表 05_排序表
        +--> 问题清单 06_问题清单
        +--> 证据索引 07_证据索引
        +--> 运行日志 08_运行日志
        |
        +--> audit.py：独立读取产物并复算
        +--> priors.py：当前输入先验观测（不静默修改正式分数）
        +--> chat_service.py/chat_insights.py：数据集会话、案例库、九档样本、追问
        +--> web_service.py：浏览器控制台、任务、历史运行、样本和证据接口
        +--> refinement.py/review_mcp.py：有独立参考时的细分复核协议
```

### 主入口与运行命令

| 能力 | 主入口 | 可独立执行命令 | 当前状态 |
|---|---|---|---|
| 规则评价 | `src/spaceqa_agent/cli.py`、`engine.py` | `spaceqa scan --input ... --output ...` | 可运行；不依赖大模型 |
| 结果审计 | `audit.py` | `spaceqa audit <run_dir>` | 可运行；复算证据与排序 |
| 当前输入先验 | `priors.py` | `spaceqa check-priors --input ... --family ... --output ...` | 可运行；结果独立保存 |
| 单案例解释 | `explain.py` | `spaceqa explain --case ... --output ...` | 可运行；API 为可选项 |
| 浏览器控制台 | `web_service.py` | `spaceqa serve --host 127.0.0.1 ...` | 可运行；回环监听，需 SSH 转发 |
| 对话工作台 | `chat_service.py`、`chat_insights.py` | 服务器 `manage.py start` | 已有独立部署脚本；案例库与九档档案集成在对话页 |
| 质量评分/审计 | `chat_scoring.py`、`quality_ledger.py` | 由报告/复核流程调用 | 源码存在；正式分数需完整规范材料和运行台账 |
| 细分复核 MCP | `review_mcp.py`、`refinement.py` | `python -m spaceqa_agent.review_mcp ...` | 源码与协议存在；需独立复核样本才能形成正式召回 |
| 研究专家/融合 | `experiments/`、`results/reference/` | 见各实验 `protocol.md`/`README` | 研究代码与聚合指标在仓库；权重和逐样本缓存在外部 |

## 2. 文件与目录索引

### 交接必须先看

- `README.md`：边界、版本、规则扫描、发布包与硬件声明。
- `docs/architecture.md`：数据流和组件职责。
- `docs/data_contract.md`：样本配对、`base_sample_id`、`split_group_id` 和防泄漏划分。
- `docs/quality_evaluation_spec_v2.md` / `.yaml`：八类缺陷、六维规范、容差、四状态和计分边界。
- `docs/delivery_audit_20261108.md`：官方交付清单逐项核对。
- `docs/delivery/`：交付说明源文件。
- `DEMO_GUIDE.md`、`docs/chat_console.md`、`docs/web_service.md`：演示和服务启动说明。

### 程序模块

- `src/spaceqa_agent/engine.py`：规则检查、清单、产物写出。
- `src/spaceqa_agent/audit.py`：运行结果独立审计。
- `src/spaceqa_agent/instance_audit.py`：有独立参考时的类别/位置/时间指标。
- `src/spaceqa_agent/quality_ledger.py`：四状态和复核台账。
- `src/spaceqa_agent/chat_scoring.py`、`pollution_metrics.py`：会话画像、污染指数和评分展示。
- `src/spaceqa_agent/chat_service.py`、`chat_insights.py`、`chat_reporting.py`：对话式工作台、案例库、报告。
- `src/spaceqa_agent/web_service.py`：服务端 API、只读数据预览、任务队列和历史运行。
- `src/spaceqa_agent/review_mcp.py`、`review_inputs.py`、`refinement.py`：细分复核输入、MCP 工具和状态约束。
- `src/spaceqa_agent/web/`：`index.html` 主控制台，`chat.html` 对话工作台，`evaluation.html` 兼容案例核查入口。

### 研究模型与实验

`experiments/` 当前包含：

- `all_family_training_20261031`：跨六类训练协议。
- `image_models_20261031`：U-Net、YOLO、DINO、I-JEPA 等图像模型实验。
- `expert_routing_20261101`：专家路由实验。
- `eight_class_20261104`：八类融合与定位实验。
- `jwvl_optimization_20261104`：视觉语言模型优化实验。
- `planner_pilot_20261105`：调查规划器与工具调用实验。
- `prior_optimization_20261102`：先验/GBDT 结构特征实验。
- `runtime_benchmark_20261105`、`package_runtime_acceptance_20261103`：运行时间和打包验收。

仓库内保存协议、脚本和聚合结果；`README.md` 明确说明完整 21 专家权重、预测缓存和数据集不在 Git 仓库。

## 3. 已知可复核结果

- README 记录的正式压缩发布：`spaceqa-agent-0.6.0-all21-nf4-20261107`，镜像和 `docker save` 均低于 20 GB；但这是外部发布收据，当前工作区不含完整镜像 tar。
- README 记录的 A100 GPU1 全六类 30 包专家前向：18,450 条分组测试记录，约 52.1 分钟；压缩前后代理指标 Precision 95.15%、Recall 94.95%、F1 95.05%。这些标签由公开 P0 对照推导，不能替代官方独立真值。
- A10 24 GB 的完整端到端验收在 README 中明确标记为尚未完成，不能在交接时写成已验收。
- `results/reference/` 保存轻量聚合结果；不得把它们误认成当前服务器的全量逐样本运行结果。

## 4. 本地回归测试记录

执行命令：

```powershell
python -m unittest discover -s tests -q
```

本次工作区结果：`232 tests`，`4 failures`、`1 error`、`4 skipped`。失败项应进入待修清单：

1. `test_refinement.ContractTest.test_mcp_protocol_and_path`：MCP 子进程返回非零，需单独复现协议输出。
2. `test_instance_audit.EvaluatorTests.test_json_cli_round_trip`：子进程没有安装包路径，出现 `ModuleNotFoundError: spaceqa_agent`；先用 `pip install -e .` 或显式设置 `PYTHONPATH=src` 重跑，区分环境失败和代码失败。
3. `test_quality_ledger_runner.NewPackageRunnerTest.test_new_package_four_states_replay_and_source_immutability`：实验脚本子进程同样没有找到 `spaceqa_agent`，优先修复测试环境入口。
4. `test_review_protocol.ReviewProtocolTests.test_locator_dimensions_must_match_actual_image`：定位尺寸不匹配的原因字段与测试期望不一致，需要核对协议实现。
5. `test_review_protocol.ReviewProtocolTests.test_only_complete_two_person_agreement_yields_declared_consensus`：两人一致性聚合结果与测试期望不一致，需要核对复核聚合规则。

本轮通过了浏览器服务、案例接口和静态 Demo 入口相关测试；不能据此宣称所有研究实验和指定硬件已验收。

### AIA94 后端迁移后的复核

随后新增了 `backend_strategy.py`、六类策略配置和 `backendProfile=enhanced/legacy` 回退入口。正式增强扫描在规则之后融合通过证据门槛的后端信号，写入标准 finding、证据、评分和排序；没有独立证据的观测保持降级或待判定，legacy 仍可完整回退。当前全仓库回归为 **242/242 通过，4 skipped**。

## 5. 交接时明确的边界

- 规则确认、模型候选、独立复核三种状态必须分开；候选概率不等于官方 Precision/Recall/F1。
- 路演包没有独立九类缺陷真值时，只能交付可追溯的问题记录、污染指数和证据覆盖，不能声称官方召回率。
- 真实数据、标签、权重、缓存、API Key 和运行目录不应加入 Git 或客户端 ZIP。
- 端到端模型能力、A10 运行时间、完整 21 专家原始输入、隐藏集指标和现场运行仍需以外部收据或现场实测补齐。

## 6. 接手人第一天建议顺序

1. `pip install -e .`，运行 `spaceqa --version`。
2. 运行 `python -m unittest discover -s tests -q`，先修复上述环境路径和复核协议失败。
3. 用小型 fixture 执行 `scan -> audit`，确认 04～08 产物链。
4. 阅读 `docs/quality_evaluation_spec_v2.md` 和 `docs/delivery_audit_20261108.md`，锁定正式分数边界。
5. 连接服务器后，用已有运行目录验证浏览器控制台和对话式案例库。
6. 最后再接入外部模型权重/缓存，重新记录指定硬件的完整运行时间和显存。
