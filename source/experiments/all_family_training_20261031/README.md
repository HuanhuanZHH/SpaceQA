# 全六类训练队列

这是新一轮全六类训练，禁止把旧的 EPB/DMSP 权重的迁移结果写成正式全量成绩。

服务器源码：`/staff/tangjiarui/OPC/opc-agent/experiments/all_family_training_20261031`

服务器输出：`/staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031`

所有原始数据、模型权重、派生图像留在服务器。

查看当前阶段：

```bash
cat /staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031/queue_state.json
```

查看当前训练日志（阶段名以 queue_state.json 的 stage 为准）：

```bash
tail -n 20 /staff/tangjiarui/OPC/opc-agent/runs/all-family-training-20261031/runner.log
```

正常队列持续运行时不重复启动。重启可恢复时，用原代码、原输出目录执行：

```bash
CUDA_VISIBLE_DEVICES=1 /staff/tangjiarui/OPC/.venv-image-models/bin/python -u /staff/tangjiarui/OPC/opc-agent/experiments/all_family_training_20261031/runner.py
```

独占锁会拒绝同时启动两个队列。任意路线失败会停止测试阶段，查看对应
`logs/<阶段>.log`；不能删除未完成路线或用历史检查点填补通过屏障。
权重更新按轮次记录，QLoRA每250次更新额外保存恢复点。代码发生改变时，
已有训练不能混用新代码静默恢复，需明确开新实验目录。

`training_barrier.json` 的 `status=pass` 表示所有路线的30个训练分组覆盖、
逐训练ID、权重哈希均通过。它不表示测试完成。测试及独立复核分别查看
`evaluation_complete.json` 和 `result_validation.json`。

最终主表 `heldout_test_expanded.csv` 对30包逐包展开指标；
`full_inventory_replay_expanded.csv` 另列含训练数据的全清单回放。
P0无注入正例时F1留空，报告误报率；unknown真值不当成正常负例。
本轮是公开P0技术参考、多标签缺陷分类试验，不能宣称官方隐藏科学定位成绩。
