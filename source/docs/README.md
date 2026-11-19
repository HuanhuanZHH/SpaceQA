# 文档索引

| 文档 | 用途 |
| --- | --- |
| [quickstart.md](quickstart.md) | 从安装到第一次扫描、审计和查看结果 |
| [deployment_offline.md](deployment_offline.md) | Docker 离线加载、运行、停止和验收 |
| [hardware_runtime.md](hardware_runtime.md) | 官方限制、实测体积、A100 结果与 A10 评估 |
| [architecture.md](architecture.md) | 数据流、规则、专家融合和证据边界 |
| [data_contract.md](data_contract.md) | 六类数据、标签布局和输入约定 |
| [web_service.md](web_service.md) | 浏览器服务、SSH 转发和 API |
| [llm_explanations.md](llm_explanations.md) | DeepSeek/Qwen 等 OpenAI 兼容接口的可选解释 |
| [reproducibility.md](reproducibility.md) | 外部依赖、复现命令和指标口径 |
| [release_metrics_0.6.0.md](release_metrics_0.6.0.md) | 0.6.0 压缩发布验收和压缩前后对比 |
| [quality_evaluation_spec_v2.md](quality_evaluation_spec_v2.md) | 六维规则、八类缺陷、定位容差、污染指数和不能宣称达标的边界 |
| [quality_evaluation_spec_v2.yaml](quality_evaluation_spec_v2.yaml) | 上述规则的机器可读版本 |
| [评价规范pro.txt](评价规范pro.txt) | 按当前实现修订后的中文评价规范文本 |
| [项目结构与阶段交接总结_20261108.md](项目结构与阶段交接总结_20261108.md) | 系统结构图、后端结构图、模型结构、调研结果和问题清单 |
| [项目结构与阶段交接讲稿_20261108.md](项目结构与阶段交接讲稿_20261108.md) | 三张结构图及项目原理的汇报讲稿 |
| [aia94_backend_migration_20261108.md](aia94_backend_migration_20261108.md) | AIA94 算法分析、六类数据迁移和后端回退说明 |

## 运行时原则

- 数据集、权重、缓存、密钥和逐样本记录不进 Git。
- 规则确认、模型候选和人工/独立真值在结果中分开保存。
- 路演包没有独立缺陷真值时，只报告污染指数、证据和排序，不伪造官方 P/R/F1。
- 发布验收必须同时记录镜像大小、Docker save 大小、输入哈希、规则版本、随机种子和真实耗时。
