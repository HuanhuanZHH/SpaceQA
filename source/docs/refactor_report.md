# Repository preparation and verification / 仓库整理与验证

Prepared on 2026-11-02. The original project directory, experiment source,
datasets, checkpoints and historical run outputs remain the provenance basis.
The new repository is `spaceqa-agent`, product name **SpaceQA Agent**.

## Engineering changes / 工程改动

- Installable `src/spaceqa_agent` package, separate CLI, rule engine, independent
  audit module, and bundled versioned assets. The minimal runtime uses Python's
  standard library. The original script interfaces remain compatible.
- Preserve scientific source snapshots and their recorded hashes instead of
  rewriting algorithms while claiming that old training ran under new code.
- Config-driven preflight and isolated workspace staging. Generated scientific
  source changes only root-path string constants; an AST check rejects other
  changes. Explicit receipts hash the generated import closure.
- External large inputs remain outside Git. Metadata links/remappings handle
  prepared views, data paths and rule-evidence locators consistently.
- Original package metrics and actual model-OR-rule combination metrics have
  separate artifacts. Unknown truth stays masked. Clean P0 has no positive
  recall/F1 denominator; its false positives still affect the all-tier aggregate.
- Git and Docker allow/ignore rules exclude datasets, weights, per-sample JSONL,
  credentials, images and runtime caches. The CPU CI workflow covers installation,
  dataset-free tests, frozen hashes and confusion-matrix recomputation.

中文说明：把可安装运行包、历史科学实验、外部路径配置、聚合结果和文档分开。
源码未被“整理”为一个与旧结果无法对应的新训练实现。复现工作区有独立回执，
拒绝覆盖旧运行，也不会通过修改输入或阈值来制造一致结果。

## Verification evidence / 验证依据

1. Regression and installation tests run on Windows and the actual Linux
   Python-3.10 research environment. Installed-wheel scanning/auditing works from
   an arbitrary working directory, and input hashes remain unchanged.
2. The frozen historical rule engine and refactored engine produce identical
   finding JSONL, ranking CSV and scorecards on a generated six-family,
   five-tier fixture (30 packages).
3. The actual frozen combination is recomputed from hash-verified private
   splits, model probabilities, training thresholds, checkpoints and rule
   evidence. Only the 30-package aggregates are exported to
   `results/reference/dispatch_metrics.json`.
4. Four experts are loaded offline in the relocated workspace. Across 128
   model-sample batch contexts, maximum probability difference from the saved
   predictions is 0.0016480088 (tolerance 0.005); all threshold decisions match.
   This includes two held-out examples for each of five model-served families.
   CME uses rules. See `results/reference/inference_check.json`.

## Boundaries / 边界

The existing predictions and aggregate statistics can be independently
recomputed. This refactor does not retrain all 20 models or assert bit-identical
new GPU training. Historical encoder warmup depends on readiness/time, and CUDA
training was not strictly deterministic. Complete source-cache regeneration is
not silently claimed: the frozen prepared cache/registrations are explicit
external inputs. Complete Docker size and end-to-end A10 runtime remain separate
acceptance tasks.

本次已验证安装、规则等价、实际组合指标复算和小规模冻结权重推理重放。
没有把新一轮完整重训、全量模型重新推理、最终镜像大小或A10耗时写成已经完成。
