# 当前 complete 发布版本核对（2026-11-03）

本记录来自服务器既有文件和验收回执的只读检查，不包含新构建或新测试。用于补充早先 0.4 候选包报告；不修改历史报告及其 SHA。

## 当前目录与版本

服务器完整交付目录为 `/staff/tangjiarui/OPC/complete_delivery_20261103`。

| 对象 | 已核实状态 | 路径或版本 |
| --- | --- | --- |
| 最新已完成的完整离线镜像 | 0.5.1，构建回执为 complete | `complete_delivery_20261103/release051/` |
| 当前在线 Web 源码 | 0.5.2 | `/staff/tangjiarui/OPC/spaceqa-agent-release-052-20261103` |
| 当前 Web 服务 | 回执显示端口 8767、PID 1889964 | `complete_delivery_20261103/final_web_process.json` |
| 0.5.2 离线镜像 | 检查时尚未发现最终 complete_image_build.json 或最终 tar，不能称已交付 | `complete_delivery_20261103/release052/` |

## 0.5.1 已成包回执

- 回执：`/staff/tangjiarui/OPC/complete_delivery_20261103/release051/complete_image_build.json`
- 镜像标签：`localhost/spaceqa-agent:0.5.1-complete-offline-20261103`
- 源码：`/staff/tangjiarui/OPC/spaceqa-agent-release-051-20261103`
- sourceCommit：`2e28a1dbb81c4a091d34d5367e1647e9d053ce8f`
- image ID：`d4a263f9d66986fde39ff362ca6228a5e366da11084bce8040972ab264ef882e`
- 镜像展开大小：19,366,245,622 字节。
- docker tar 大小：19,366,273,536 字节，即 19.366273536 GB，低于 20,000,000,000 字节。
- tar：`/staff/tangjiarui/OPC/complete_delivery_20261103/release051/spaceqa-agent-complete-offline-0.5.1-20261103.docker.tar`
- SHA-256：`20c53e69363563484d22f982b0e18782be796bdbf55a7246627b8308d792a198`
- 回执记录：冻结模型资源未变，科学算法文件与已验证基线一致；构建未使用 GPU、未训练、网络为 none。

该完整版本加入了冻结的小专家、空间定位器和 Q4_K_M 文本规划器。规划器来自 Qwen3-VL-8B-Instruct 的文本部分，以 GGUF 形式进行工具规划，图像感知仍由小专家承担。GGUF 为 5,027,785,184 字节。不能将其描述成完整 8B 视觉模型离线视觉推理。

## 时间与验收的适用范围

主机运行快照 `compact_run_snapshot.json` 和 `package_acceptance_051/run_1_summary.json` 显示第一轮 12 个目录均已形成审计结果，审计通过，最慢目录计时 4,549.639 秒（约 75.83 分钟）。其中 3 项沿用了早先 fresh 结果，计时口径包含部分共享前置开销；这不是在统一 0.5.1 代码下重新完成的独立全量复测。

第一轮总体仍为 partial，officialScoreApproved=false。第二轮当时仅完成 1/12，最终汇总尚未生成。因此“0.5.1 全量双轮复现及正式验收完成”尚不成立。

容器离线证据为 CPU、断网、只读根文件系统的真实样本 smoke：根目录 `offline_acceptance/acceptance.json` 对应 0.5.0 的 EPB、ULF，每类实际推理 2 条；0.5.1 的 `extra_family_smokes.json` 覆盖 AIA、HMI、CME、DMSP，每类实际推理 1 条。完整规则检查与输出审计通过；模型未执行部分有明确记录。这些不能替代 A10 上完整单数据集三小时验收，也不能用旧规则基线分数代替比赛正式科学指标。

## 与历史 0.4 候选包的关系

早先 `/staff/tangjiarui/OPC/package_runtime_acceptance_20261103/full_candidate_v2/` 的 14,314,756,608 字节 tar 是 0.4 历史候选包，也是后续完整版本的基础；当前可核实的最新已成包版本为上面的 0.5.1。早先未量化 8B 原权重连同环境超过 31GB 的结论，不适用于现已加入约 5GB Q4 文本规划器的完整包。
