# 根目录运行记录

本目录现汇集修复后的 2 个完整 AIA 包、3 类小样成功运行，以及共享内存修复前的 2 次真实失败。不是根目录 12 包历史评分卡的重跑记录。

原始 6 条旧导出事件及环境、种子、回执完整保存在 `history/pre_runtime_sync/`。新日志每条保留 sourceLog/sourceLine，可回查未经修改的原始日志；ts 是 timestamp 的别名。

两条失败事件原 elapsedMs 由异常处理器错误继承起始事件，已保留为 sourceElapsedMs，归一化 elapsedMs 置空并写明原因；真实容器外耗时见 run_summary.json 中 executionReceipt。没有补造单调时钟耗时或重试。修复后重新启动使用新 runId，原生 retryCount 保持其真实值。

环境与种子来自实际容器 system.json。tokenUsage 仅指外部 API，本地模型 token 未计量。历史 12 包结果与当前运行验证的对应边界见主清单。
