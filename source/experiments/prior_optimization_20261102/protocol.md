# 复现协议 / Replay protocol

本协议复算本轮研究，不重训旧 20 路深度模型。克隆仓库只提供代码和聚合证据；
完整复算必须能访问原始当前输入、已冻结划分、旧概率、规则证据和权重哈希。
聚合 JSON/CSV 可以在无数据环境验证公式，不能从聚合表反推逐样本预测。

## 外部输入

历史服务器根目录为 `/staff/tangjiarui/OPC`，下表为只读来源。

| 名称 | 原始位置及内容 |
| --- | --- |
| A | `opc-agent/runs/all-family-training-20261031/`；train/val/test、barrier、training、全部 20 路 predictions 与 checkpoints |
| R | `opc-agent/runs/full-matrix-20261031/blind-rules/`；finding 和证据 locator |
| B | `opc-agent/runs/roadshow-evaluation-20261101/`；dispatch_current、12 包 inventory、模型概率与 SHA marker |
| 当前标签/数据 | 上述 inventory 的 label_path/data_path 指向的原始文件；六类 30 包和路演解压目录 |
| 构造依赖 | `models/` 下的 DINO、YOLO、I-JEPA 初始化文件；按头 footprint 检查需要真实文件大小 |

A 的 split SHA：

- train：`3b4cf96425cb393a4c27e1f6c59a26f0f32682d6342d432aded41a99771a7a5c`
- val：`b1ec2c53b73e2da04595ede4849344302fca55c25ed0925aca17c6d640d5f270`
- test：`4872995ab5df92f80e4fc7386c0beeda327f4328427034839b68fa3c908286b6`

研究环境为原 `.venv-image-models`，包括 sklearn 1.7.2 和其记录的 NumPy 版本。
新 GBDT/字段预测使用 CPU；缓存复评和 fresh 规则只需标准库。仓库中的 GBDT
`reference/.../training.json` 记录版本、150 迭代/15 叶/seed 20261102、样本与 family 平衡、
当前字段白名单和阈值来源。不是任意版本下逐字节同模型的承诺。

## 先创建隔离副本

归档脚本保留原始绝对路径和拒绝覆盖行为。不要在原 `inference_prior_20261102/gbdt`
直接重训。以下命令在可访问上述外部输入的 Linux 主机执行；`WORK` 必须尚不存在，
应位于克隆仓库之外。`stage_replay.py` 只改变已知路径字符串，通过 AST 还原检查，
并记录原始/生成源码 SHA 和每项路径修订。它不读训练或测试数据、不启动训练。

```bash
export OPC_ROOT=/staff/tangjiarui/OPC
export WORK=/staff/tangjiarui/OPC/replays/prior_20261102_001
export HIST_REPO="$OPC_ROOT/opc-agent"
export A="$HIST_REPO/runs/all-family-training-20261031"
export R="$HIST_REPO/runs/full-matrix-20261031/blind-rules"
export B="$HIST_REPO/runs/roadshow-evaluation-20261101"
export PY="$OPC_ROOT/.venv-image-models/bin/python"

python experiments/prior_optimization_20261102/stage_replay.py \
  --workspace "$WORK" --opc-root "$OPC_ROOT" --repo "$HIST_REPO"
cd "$WORK/code"
export RUN="$WORK/runs"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=''
```

`stage_receipt.json` 是复现来源证据。生成副本的源码哈希不同于历史执行源码，
不能冒称逐字节未改；科学 AST 只允许列明的路径字符串变化。所有输出写入新 `RUN`。
修改映射后，A/B inventory 内部路径仍须实际可访问；代码重定位不会悄悄改写私有输入。

## 全六类 CPU 训练、冻结与30包复评

```bash
"$PY" gbdt/train_gbdt.py --source "$A" --output "$RUN/gbdt" --read-cme-anchor

"$PY" compare_versions.py --base "$A" --dispatch "$B/dispatch_current.json" \
  --old-rules "$R" --output "$RUN/comparison_v1" --workers 8 \
  --ulf-duration-convention either

"$PY" compare_gbdt_extension.py --comparison "$RUN/comparison_v1" \
  --gbdt "$RUN/gbdt" --base "$A" --old-rules "$R" \
  --output "$RUN/gbdt_extension_v1"

"$PY" compare_headwise.py --comparison "$RUN/comparison_v1" \
  --gbdt "$RUN/gbdt" --base "$A" --old-rules "$R" \
  --output "$RUN/headwise_v1" --p0-head-cap .05
```

第一步三个头使用六类×五档 train 中各自已知位，全部训练结束后冻结向量器和 val
阈值，写 `training_barrier.json` 后才首次打开 test。当前 CME image_time 只从当前
标签读取并校验标签 SHA；不读 P0 对照。既有 256 维特征与 Current fields 均有明确
白名单，target/tiers/IDs/source paths 不进入 X。验证/测试群组需不相交。

后续脚本使用原 20 模型缓存，不重新 GPU 前向。阈值、专家、按头策略仅使用 val，
冻结 JSON 落盘后才解析 test。保留全部30包、缺失输入、未知位和每种方案的计数。
GBDT 路由选择与重训练仍是探索性历史复测，不成为新独立测试。

若仅复算历史概率和策略，可使用外部已经完成且哈希一致的 GBDT 目录，并对
`compare_gbdt_extension.py` / `compare_headwise.py` 显式传 `--gbdt /existing/gbdt`。
不要把参考聚合目录误当该私有概率/权重目录。

## 路演：先 fresh 规则，再冻结模型候选，最后融合

```bash
"$PY" run_roadshow_prior.py --roadshow "$B" \
  --frozen-policy "$RUN/comparison_v1/frozen_validation_policy.json" \
  --output "$RUN/roadshow_v2" --workers 8

"$PY" gbdt/predict_roadshow.py

"$PY" fuse_roadshow_extension.py --roadshow "$B" --prior-output "$RUN/roadshow_v2" \
  --base-policy "$RUN/comparison_v1/frozen_validation_policy.json" \
  --extension-policy "$RUN/gbdt_extension_v1/frozen_validation_extension.json" \
  --gbdt "$RUN/gbdt" --output "$RUN/roadshow_final_v1"

"$PY" fuse_headwise_roadshow.py --roadshow "$B" --prior-output "$RUN/roadshow_v2" \
  --frozen-policy "$RUN/headwise_v1/frozen_validation_policy.json" \
  --profile high_recall_review --gbdt "$RUN/gbdt" \
  --output "$RUN/headwise_review_roadshow_v1"
```

fresh 入口保留历史 runner，但只执行至 `roadshow_prior_complete`，写独立入口回执；
不执行历史中失败的旧融合尾部。该阶段对全部 ULF CSV 重新哈希并完整扫描，
当前标签读取/哈希；PNG只读头部，不宣称已全量重新哈希原图。
三种时钟不强等、单位不凭裸 CSV 证明、依赖当前 startTime 的起点不自证正确。

GBDT B 预测和两种最终融合都不训练、不调阈值；B 没有真值，P/R/F1 为 null。
模型发现必须保持候选。原始 B 272 个缺失输入保留，`data_missing` 不能被替代图片掩盖。
标准评估目录的集成导出属于仓库运行时功能，参见根 CLI 文档；本研究入口输出
`prior_observations.jsonl` / `selected_candidates.jsonl` 等私有中间工件。

## 比较与可复现边界

无数据可运行 `verify_archive.py` 校验归档 SHA、策略链接与聚合 TP/FP/FN/TN 公式。
完整独立复算还需私有逐样本输入/概率和历史外部权重；归档汇总不替代它们。
比较语义、混淆计数和阈值，而非 runId、绝对路径或计时逐字节一致。

默认 91.37% F1 的 C 类召回仅 24.37%；review 的 F1 89.84%、总召回 93.98%、
C 类召回82.77%，是另一配置，不能把两者优点拼成不存在的结果。所有数字均为
P1–P4 已知代理位，P0误报单列。ULF100%不等于真实地磁事件识别100%。
本目录没有执行最终完整容器体积和A10端到端3小时时限验收。
