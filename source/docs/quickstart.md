# 快速开始

## 1. 安装规则运行时

规则评价只需要 Python 3.10+（推荐 3.11），没有第三方运行依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
spaceqa --version
```

Windows PowerShell 将激活命令替换为 `.venv\\Scripts\\Activate.ps1`。

## 2. 扫描一个包或一批包

```bash
spaceqa scan \
  --input /absolute/path/to/publish \
  --output /absolute/path/to/runs/scan-20261108 \
  --seed 20261021
```

输入目录只读，输出目录必须尚不存在并且不能位于输入目录中。输入可以是 `data/` 与 `label/` 分目录，也可以是 PNG/CSV 与 JSON 平铺布局。扫描会建立文件清单、哈希和数据—标签配对关系，随后运行适用的确定性检查。

## 3. 审计与查看

```bash
spaceqa audit /absolute/path/to/runs/scan-20261108
python -m json.tool /absolute/path/to/runs/scan-20261108/04_评分卡/<package>.json
head /absolute/path/to/runs/scan-20261108/05_排序表/ranking_global.csv
```

审计通过表示证据引用、计分和排序内部一致；它不等同于所有科学缺陷均已被正确识别。

## 4. 可选先验和解释

```bash
spaceqa check-priors \
  --input /absolute/path/to/package \
  --family AIA_AR_sample \
  --output /absolute/path/to/prior-results.jsonl
```

模型解释必须显式配置环境变量，且只组织已有证据：

```bash
export SPACEQA_CHAT_LLM_BASE_URL=https://provider.example/v1
export SPACEQA_CHAT_LLM_MODEL=provider-model
export SPACEQA_CHAT_LLM_API_KEY=read-from-a-secret-manager
spaceqa explain --run-dir /absolute/path/to/run \
  --finding-id finding-001 --output /absolute/path/to/explanation.json
```

不要把 API Key 写入命令历史、配置文件或提交记录。
