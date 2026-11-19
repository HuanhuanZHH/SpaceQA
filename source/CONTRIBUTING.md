# Contributing / 贡献说明

Keep changes small, explain the evidence, and run the dataset-free tests:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/reproduce.py verify --manifest configs/source_manifest.json
```

Do not commit datasets, sample-level manifests/labels/predictions, derived
images, weights, tokens, private `.local.json` settings, or runtime directories.
Keep raw observations and P0–P4 duplicates in the same split group. Never use
held-out outcomes to select a route or threshold. P0 recall/F1 is undefined
when there are no positive truth bits. Label unknown bits explicitly.

The archived scientific files are immutable reference material: implement new
scientific variants in a new experiment directory, register their configuration
and provenance, then evaluate them separately. New training is not silently
resumed under changed code. A new source manifest is a new reviewable snapshot;
do not rewrite historical provenance to conceal a mismatch.

修改时请附依据及相应测试。不要提交原始数据、逐样本标签/预测、权重或密钥；
不要把全清单回放当成独立测试，也不要把技术参考指标称为官方隐藏分数。
新增算法开新实验版本，不覆盖冻结实验。仓库尚未指定开源许可证，发布前需由
代码所有者确定许可，并逐项核对模型、库和数据的使用条款。
