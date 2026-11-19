# 交付要求核对报告（2026-11-08）

本报告把官方《S2-空间科学智能就绪数据集质量评价智能体需求说明》和《交付材料清单-v1.1》与当前仓库 `spaceqa-agent` 逐项核对。`PASS` 表示仓库中已有可复核实现或文件，`PARTIAL` 表示代码/源文件存在但最终提交格式或全量证据尚未放入仓库，`MISSING` 表示当前没有，`BLOCKED` 表示必须在指定评测环境或现场确认。

## 结论

当前仓库已经具备可运行的规则评价、八类问题账本、六维评分卡、全局排序、证据索引、审计日志、离线 Docker 规则运行时和浏览器工作台。仓库本身还不能直接宣称“完整交付合格”，原因是比赛要求的最终 PDF、H.264 视频、完整 `docker save` tar 和对应路演/隐藏集运行收据不在 Git 仓库内。仓库外的 0.6.0 完整研究发布收据显示镜像 19,748,024,384 bytes、save tar 19,748,043,776 bytes，低于 20 GB；A100 全量前向 3,124.28 s，但指定 A10 仍需复测。

## 逐项状态

| 交付物 | 状态 | 当前证据 | 缺失或风险 | 处理方式 |
| --- | --- | --- | --- | --- |
| DEL-1 方案规划 | PARTIAL | `docs/architecture.md`、`docs/quality_ledger.md`、`README.md` | 没有最终签字版 PDF，方案章节需集中成一份 | 使用 `docs/delivery/DEL-1_方案规划.md` 生成 PDF |
| DEL-2 产品/镜像 | PARTIAL | `Dockerfile`、`docker-compose.yml`、`docs/deployment_offline.md` | 完整研究镜像 tar 在仓库外，缺离线 load/run 收据 | 将 tar、inspect、SHA-256 收据放入提交包 `image/`；不要把数据和密钥放入 Git |
| DEL-3 部署说明 | PARTIAL | `docs/deployment_offline.md`、`docs/quickstart.md` | 官方要求 PDF，当前只有 Markdown | 由 `docs/delivery/DEL-3_部署与使用说明.md` 输出 PDF |
| DEL-4 评价规范 | PARTIAL | `spec/quality_spec.yaml`、`docs/quality_ledger.md` | 之前没有显式的类型对应主表、逐检查边界表和不适用性矩阵；没有最终 PDF | 本次已补进 YAML 的 `typeCorrespondence`、`checkBoundaries`、`notApplicableMatrix`、`scopeBoundary`、`consistencyAudit`；再生成 PDF |
| DEL-5 逐数据集评分卡 | PARTIAL | `src/spaceqa_agent/engine.py` 会生成 `04_评分卡/*.json` 和 HTML | 当前仓库没有本次最终全量评分卡 | 从冻结运行目录完整复制，不得手工改分 |
| DEL-6 全局排序 | PARTIAL | `engine.py` 生成 `05_排序表/ranking_global.csv` | 当前仓库没有与最终评分卡绑定的全量 CSV/PDF 说明 | 使用同一 `runId` 的排序和评分卡，并审计名次连续、无并列 |
| DEL-7 问题清单 | PASS（实现）/PARTIAL（交付） | `findings.jsonl` 生成逻辑及八类账本 | 最终全量 JSONL 未在仓库 | 打包全部记录；每条保留四状态、程度和定位字段 |
| DEL-8 证据索引 | PASS（实现）/PARTIAL（交付） | `audit.py`、`engine.py`、`evidence_index.csv` 生成逻辑 | 最终 evidence 文件夹未在仓库 | 全量复制并保存哈希；不得只放截图或抽样 |
| DEL-9 运行日志 | PASS（实现）/PARTIAL（交付） | `08_运行日志/run.log.jsonl`、`runtime_metrics.json` 生成逻辑 | 需要最终运行收据 | 从最终运行目录复制，确认每条含 UTC、checkId、状态、耗时、资源、token、重试 |
| DEL-10 版本与随机种子 | PASS（已修复）/PARTIAL（收据） | `agent/manifest.yaml`、`agent/tools.yaml`、`spec/quality_spec.yaml` | 运行环境和 GPU 峰值仍需随提交包附上 | 版本统一为 0.6.0；每次运行附 `system_env.txt`、`seeds.yaml` |
| DEL-11 未完成事项 | PARTIAL | `engine.py` 生成 `09_未完成事项说明.md`，`docs/scope.md` | 官方要求 PDF | 使用 `docs/delivery/DEL-11_未完成事项说明.md` 输出 PDF，如实列明独立真值、A10 复测和视频等缺口 |
| DEL-12 演示视频 | MISSING | 当前仓库没有可核验 MP4 | 必须 ≤5 分钟、H.264 | 录制后检查时长、编码和 SHA-256；视频不能用静态截图替代 |
| DEL-13 现场运行与答辩 | BLOCKED | `DEMO_GUIDE.md`、`docs/demo_live.md` | 需要比赛现场和指定硬件 | 现场执行 `manifest → scan → audit`，保留终端和报告证据 |

## 规范与实现一致性

- 四类状态保留：`通过`、`不通过`、`待判定`、`不适用`。适用但未实现的检查仍留在完成度分母中。
- 八类问题和六个评价维度分开：八类用于内容缺陷账本，六维用于官方评价框架。
- 规则确认的问题才进入正式技术分；模型候选保留概率和定位，不能冒充官方真值或官方 F1。
- 模型/LLM 只负责候选、定位和解释；比例、程度、评分和排序由确定性程序计算。
- EPB 源标签若只有矩形，系统保留原始矩形；不会把矩形伪装为独立预测的多边形。

## 提交前必须补齐

1. 对最终冻结运行目录执行 `python tools/audit_delivery.py <submission-root>`，修复所有 `MISSING` 和 `ERROR`。
2. 将四份 Markdown 源转换为可选中文字的 PDF：方案规划、部署说明、评价规范、未完成事项。
3. 放入完整研究镜像的 Docker save tar、`docker image inspect`、SHA-256，并在无网络环境执行 load/run/audit。
4. 从同一次冻结运行复制 DEL-5～DEL-10 全量文件，确认 `runId`、`ruleVersion`、`specVersion` 一致。
5. 完成 ≤5 分钟 MP4，并在指定 A10 上复测单数据集 ≤3 小时、峰值显存和资源日志。


## 2026-11-08 补充核对

本轮新增 `docs/quality_evaluation_spec_v2.md`、`docs/quality_evaluation_spec_v2.yaml` 和 `docs/quality_evaluation_spec_v2.pdf`，并将同一规则写入 `spec/quality_spec.yaml` 及安装包资产。新增内容包括：

- 矩形、 多边形、时间区间和尺寸字段的项目默认容差；
- 无独立参考时保留候选、有独立参考才允许命中的定位状态；
- 八类按概率阈值累计的污染指数和升序排序规则；
- 专家原始概率存在时的 median/MAD 异常审计；
- 六维正式分条件、适用分母、未决处理和不能宣称官方 P/R/F1 的边界；
- 浏览器总览中的污染指数表、定位容差、专家审计和正式分阻断原因。

这些改动证明了系统有相应的计算和展示功能，但不会替代外部独立真值、字段字典、用途材料或指定硬件验收。因此“全部达标”仍需 DEL-5～DEL-13 的最终冻结运行、独立缺陷真值和现场材料共同满足。
