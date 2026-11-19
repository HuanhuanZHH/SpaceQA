# Frozen checkpoint replay / 冻结权重推理复核

This optional Linux/CUDA check requires the private historical run, its prepared
image/feature cache, local base weights, and an isolated workspace created by
`scripts/reproduce.py stage`. It never trains or overwrites the historical run.

此可选检查需要 Linux/CUDA、历史私有运行目录及其派生图像/特征、本地基础权重，
以及 `scripts/reproduce.py stage` 创建的独立工作区。检查不训练模型，也不覆盖历史运行。

```bash
python scripts/check_inference.py \
  --workspace /absolute/path/to/isolated-workspace \
  --run-dir /absolute/path/to/original/all-family-training-20261031 \
  --output /absolute/path/to/new-inference-check.json \
  --gpu 1
```

- Experts come from the frozen validation-selected `configs/dispatch_20_models.json`.
  专家由已冻结验证集配置确定，不使用测试结果重新选模型。
- Each expert replays the first historical batch and complete batches surrounding
  at most two held-out examples per assigned family, using the original batch size
  16, image size 384×768, saved feature normalization and BF16 autocast.
  每个专家复核历史首批及每类最多两个独立测试样本所在的完整批次，沿用历史推理参数。
- The new JSON contains aggregate counts, probability differences and threshold
  decision mismatches only. Absolute probability tolerance is 0.005; a passing
  result also requires matching threshold decisions. It contains no sample IDs.
  输出只含聚合计数、概率差与阈值决策差；不包含样本编号。
- Batch positions are reconstructed from nonmissing inventory order. A historical
  interrupted/resumed evaluation may have used different batch boundaries. Any
  difference must be investigated; this script does not modify predictions to
  manufacture agreement.
  若历史任务中断续跑导致批次边界不同，须解释差异，不能据此修改旧预测。
- This verifies a small checkpoint replay. It does not establish new F1/recall,
  complete training reproducibility, final Docker size, or A10 runtime compliance.
  本检查只验证小规模推理重放，不代表重新获得全量准确率或完成官方部署验收。
