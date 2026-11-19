# SpaceQA Agent 演示与操作指南

本仓库提供两种调用方式：规则扫描使用固定 CLI 参数；浏览器服务提供数据集浏览、任务提交、证据查看和报告下载。自然语言只用于可选的解释层，不能直接改变缺陷、分数或排名。

## 固定请求格式

```bash
spaceqa scan \
  --input /data/publish \
  --output /runs/demo-001 \
  --seed 20261021
spaceqa audit /runs/demo-001
```

输入可以是单个数据包或包含多个包的根目录，输出必须是新的外部目录。每次运行都会保留输入哈希、检查编号、规则版本、种子和日志。

## 浏览器演示

服务器上启动：

```bash
spaceqa serve --host 127.0.0.1 --port 8765 \
  --data-root /data/publish \
  --run-root /data/historical-runs \
  --state-dir /runs/spaceqa-web
```

本机建立 SSH 转发：

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18765:127.0.0.1:8765 user@server
```

打开 `http://127.0.0.1:18765`。演示顺序建议为：

1. **Overview**：展示数据包数量、服务版本和运行状态。
2. **Dataset explorer**：打开图像或 ULF 曲线，切换原始标签叠加并查看 JSON 字段。
3. **Run evaluation**：选择数据包和种子，启动一次规则任务，查看阶段日志。
4. **Evidence**：打开问题、文件哈希、样本、标注记录或区域定位。
5. **Scorecard and ranking**：展示八类画像、六维技术观测、污染指数和排序。
6. **Audit**：执行独立审计，说明报告可以复算，但候选不等于官方真值。

## 结果解读

- **已确认问题**：有文件、字段或内部一致性证据，可进入确定性技术分。
- **模型候选**：有专家概率和定位，但缺少独立参考时只展示证据，不直接扣确认分。
- **污染指数**：按固定阈值累加问题概率并按样本数归一；指数越低表示当前可见信号越少，不等同于真实污染率或官方 F1。
- **路演 F1**：路演包没有独立缺陷真值时必须留空，不能用候选数计算官方召回率。

## 现场答疑命令

```bash
# 查看一条运行是否能独立复算
spaceqa audit /runs/demo-001

# 只运行文件/配对相关检查
spaceqa scan --input /data/package --output /runs/structure-only \
  --checks C01,C02,C03 --seed 20261021
```

模型解释接口通过 `SPACEQA_CHAT_LLM_BASE_URL`、`SPACEQA_CHAT_LLM_MODEL` 和 `SPACEQA_CHAT_LLM_API_KEY` 配置。密钥不放进脚本、报告或 Git。
