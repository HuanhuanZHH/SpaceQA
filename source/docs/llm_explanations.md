# 可选的大模型结果解释

模型解释是独立的后处理层：`检测 → 证据/分数 → 大模型组织语言`。原有离线扫描无需网络或 API Key，评分、污染指数、排序和正式缺陷状态继续由确定性程序给出。模型返回的文字不能覆盖原始结果。

## 两种接口

- `spaceqa explain` 读取一个案例 JSON 或已完成的运行目录，使用 `SPACEQA_LLM_BASE_URL`、`SPACEQA_LLM_MODEL`、`SPACEQA_LLM_API_KEY` 调用 OpenAI 兼容的 `/v1/chat/completions`。
- 浏览器对话服务使用 `SPACEQA_CHAT_LLM_BASE_URL`、`SPACEQA_CHAT_LLM_MODEL`、`SPACEQA_CHAT_LLM_API_KEY`。未配置时页面仍能展示已有证据，但不会访问网络模型。

附图必须使用支持视觉输入的模型。DeepSeek 文本模型可以解释字段和规则证据；Qwen-VL 等视觉模型可以在明确给出原图、标注和坐标系时补充图像说明。接口是否支持图片以服务商当前文档为准。

## CLI 示例

```bash
export SPACEQA_LLM_BASE_URL='https://provider.example/v1'
export SPACEQA_LLM_MODEL='provider-supported-model'
read -rsp 'API Key: ' SPACEQA_LLM_API_KEY; export SPACEQA_LLM_API_KEY; echo

spaceqa explain --case /external/case.json \
  --image /external/current-case.png \
  --output /external/explanation.json
```

PowerShell 使用 `$env:SPACEQA_LLM_BASE_URL`、`$env:SPACEQA_LLM_MODEL`、`$env:SPACEQA_LLM_API_KEY`。也可以用 `--base-url` 和 `--model` 覆盖地址与模型，密钥仍只从环境变量读取。

输入案例只描述已有事实，例如：

```json
{
  "case_id": "example-001",
  "dataset_type": "HMI_AR_sample",
  "question": "这张图的标注为什么需要复核？",
  "model_predictions": {
    "route": "resnet18_unet",
    "boundary_candidate_probability": 0.72,
    "threshold": 0.25
  },
  "findings": [
    {
      "finding_id": "F-EXAMPLE-001",
      "status": "候选",
      "defect_type": "boundary_candidate",
      "locator": "/annotation/events/0/bbox_pixel",
      "evidence": "视觉模型提出边界候选；尚无独立位置真值"
    }
  ]
}
```

解释结果记录请求哈希、模型名称、调用用量、尝试次数、耗时和图像哈希。最多读取命令明确选择的结论，未知或重复的 `finding_id` 会被拒绝。接口失败时原有检测报告保留。

## 不能做的事

解释层不生成官方 F1、精确率或召回率，不把 G/C/D 候选概率转换成九类独立真值，也不自动修复原始文件。没有独立真值的路演数据只能生成证据范围内的文字说明；正式无网络评测只运行本地检测和报告功能。

密钥不要写入代码、仓库、截图、运行 JSON 或 shell 历史；优先从受控密钥管理器注入环境变量。
