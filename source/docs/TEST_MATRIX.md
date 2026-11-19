# 交接测试矩阵（2026-11-08）

这份矩阵记录交接时已经实际执行的检查、证据文件和剩余边界。`PASS` 只表示对应命令和范围通过；它不等于官方隐藏集的九类缺陷召回率，也不等于指定 A10 硬件验收。

## 本机可复核测试

| 范围 | 命令 | 结果 | 证据 |
|---|---|---|---|
| 当前仓库完整单元测试（推荐入口） | `PYTHONPATH=src python -m unittest discover -s tests -q` | **242 tests：242 passed，4 skipped**；正式后端融合后复测通过 | 本轮终端回执（2026-11-08） |
| 未设置源码路径的裸命令 | `python -m unittest discover -s tests -v` | 裸命令依赖本机安装状态；交接时使用上表推荐入口或先执行 `python -m pip install -e .` | 历史终端回执 |
| 复现源文件清单（原清单） | `python scripts/reproduce.py verify --manifest configs/source_manifest.json` | FAIL：旧清单包含 453 个文件，当前工作区有 28 个哈希不一致；不能作为当前版本收据 | 终端回执（2026-11-08） |
| 复现源文件清单（最终快照） | `python scripts/reproduce.py verify --manifest configs/source_manifest_20261108_final.json` | **PASS**：488 文件、17 项冻结科研源码、3960 项混淆矩阵字段复算 | 本轮终端回执、`configs/source_manifest_20261108_final.json` |
| 旧版规则原型回归 | `python -m unittest discover -s tests -v`（`D:\OPC\opc-agent`） | **35/35 PASS** | `D:\OPC\opc-agent\tests` |
| 候选规则基准 | `python benchmark/benchmark_candidates.py`（`D:\OPC\opc-agent`） | A_schema_pairing F1=0.2000；B_single_package_rules F1=0.6154；C_rules_plus_sibling_diff F1=0.9412，耗时 0.2322 s | `D:\OPC\opc-agent\benchmark\results.json`（本次终端回执） |

### 测试边界

推荐命令已通过全部 242 个用例；4 个跳过项均有明确条件：Windows 无法创建真实目录软链接，或未安装可选的定位器 SciPy。裸命令依赖本机安装状态，交接时必须使用上表中的推荐入口，或先执行 `python -m pip install -e .`。

## 已有离线、体积和运行时证据

| 项目 | 实测结果 | 证据 |
|---|---|---|
| 0.6.0 压缩发布 | Docker 镜像 19,748,024,384 bytes；`docker save` tar 19,748,043,776 bytes，低于 20 GB；压缩前后代理 Precision=0.9515306、Recall=0.9495121、F1=0.9505203 | `docs/release_metrics_0.6.0.md`；服务器发布收据（不在 Git） |
| 旧版完整默认候选包 | Docker 兼容 tar 14,314,756,608 bytes；Podman 空 store 离线 load、规则 scan/audit、六类 tiny fresh forward、DMSP P4 全流程均通过 | `D:\OPC\outputs\package_runtime_acceptance_20261103\完整默认程序_打包与离线验收.md`、`acceptance.json` |
| 12 包当前刷新 | A100、batch=16、24,782 样本、10,332 次新图像前向，总墙钟 459.790 s；最慢单包 104.945 s，峰值 4473 MiB | `D:\OPC\outputs\runtime_benchmark_20261105\benchmark_report.md`、`benchmark_summary.json` |
| 公开包时间 | A100 上已记录的 9 个完整包均低于 3 h，最慢 CME P4 67 min 02 s | `D:\OPC\outputs\package_runtime_acceptance_20261103\20GB与3小时_核验结论.md` |

这些时间不是 A10 验收。0.6.0 的 A10 端到端时间、显存峰值和 GPU 容器验收仍需在指定硬件上重新记录。

## 交付物就绪度

| 交付物 | 当前状态 | 交接说明 |
|---|---|---|
| DEL-1 方案规划 | 源文件具备，PDF 待冻结 | `docs/delivery/DEL-1_方案规划.md` |
| DEL-2 产品/镜像 | Dockerfile 和离线规则运行时具备；完整 tar 在受控服务器 | 不把数据、权重缓存或密钥提交 Git |
| DEL-3 部署说明 | Markdown 具备，最终 PDF 待生成 | `docs/deployment_offline.md`、`docs/delivery/DEL-3_部署与使用说明.md` |
| DEL-4 评价规范 | YAML、Markdown、PDF 已生成；需与冻结运行版本绑定 | `docs/quality_evaluation_spec_v2.*` |
| DEL-5～DEL-10 | 生成逻辑和历史产物具备；最终提交包需来自同一次冻结 runId | 评分卡、排序、问题、证据、日志、版本/种子不可手工拼接 |
| DEL-11 未完成事项 | 源文件具备，PDF 待生成 | `docs/delivery/DEL-11_未完成事项说明.md` |
| DEL-12 演示视频 | **未在仓库核验到 MP4** | 需要 ≤5 分钟、H.264、时长和 SHA-256 收据 |
| DEL-13 现场运行 | 只能现场验收 | 需在指定硬件和现场数据执行 scan→audit |

## 建议的多人并行开发方式

在冻结规则版本和测试数据之后，建议按三个互不覆盖的工作包并行：

| 角色 | 负责范围 | 先交付什么 |
|---|---|---|
| 交付与复现 | 固化 `PYTHONPATH=src`/可编辑安装入口；冻结 manifest、版本、种子和 runId；把全量测试收据纳入提交包 | `unittest` 通过回执、`source_manifest_*.json`、`audit_delivery` 收据 |
| 指标与科学复核 | 准备独立九类实例/位置/时间参考，复算 P/R/F1、容差和严重程度；检查八类到九类映射 | `instance_audit` 指标、混淆矩阵、定位容差报告 |
| 部署与演示 | 在 A10 完成单包 ≤3 h 和显存记录；生成 Docker load/run 收据、PDF 和 ≤5 分钟视频 | `image/`、PDF、H.264 视频、现场启动说明 |

三组只通过冻结的 `docs/quality_evaluation_spec_v2.yaml`、`runId` 和证据索引接口交换结果。这样可以同时推进，而不会互相覆盖模型、规则或评分口径。

## 当前结论

本次已经把可在本机立即复核的交接测试跑通，并补了一份与当前代码一致的来源快照。当前可以交接源码、规则扫描、评分/排序、证据审计、聊天工作台和离线运行时；仍不能把缺少独立九类真值、未完成 A10 验收或缺少最终视频写成“全部交付达标”。
