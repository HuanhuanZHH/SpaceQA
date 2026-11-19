# 复现说明 / Reproducibility

复现分为规则运行、历史指标复算、冻结权重推理与完整训练。仓库不含真实观测、逐样本标签、权重、派生视图或完整预测；克隆代码不能替代外部文件准备。来源记录及保存预测的指标可精确核验/复算；冻结权重已完成小规模容差复核；完整重训不承诺逐位一致。

## 规则运行与离线验证

```bash
python -m pip install -e .
python scripts/reproduce.py verify --manifest configs/source_manifest.json
python -m unittest discover -s tests -v
spaceqa scan --input /absolute/path/to/publish --output /absolute/path/to/new-run --seed 20261021
spaceqa audit /absolute/path/to/new-run
```

测试在临时目录生成小型夹具，输入只读，输出使用尚不存在且位于输入目录之外的新路径。规则包支持Python≥3.10，推荐3.11且无第三方运行依赖；研究训练另需完整环境。跨路径或机器时，时间戳、绝对路径和运行 ID 可能不同，应比较检查语义、问题状态、规范和分数，而非整个输出目录逐字节一致。

`verify` 检查仓库来源清单，不能证明外部数据完整或模型正确。`results/reference/` 保留聚合指标、完成与复核记录、代码清单、环境、组合复算和来源说明。它们是可核验快照，不是运行时读取的缺陷答案。

## 外部环境与显式重定位

复制 `configs/reproduce.example.json` 到本机配置，填写数据、清单、基础权重、历史中间工件和 Python 环境路径。明确的外部起点包含已完成的 `full-matrix-20261031` 预处理缓存及配准/注册记录，不能仅凭原始数据重造这条入口的全部前置工件。配置中的清单/源缓存SHA-256采用真实历史值；不同版本必须显式记录。不要把本机配置或凭据提交 Git。

```bash
python scripts/reproduce.py preflight --config /path/to/reproduce.local.json
python scripts/reproduce.py stage --config /path/to/reproduce.local.json --workspace /path/to/new-workspace
python scripts/reproduce.py run --config /path/to/reproduce.local.json --workspace /path/to/new-workspace --stage prepare
python scripts/reproduce.py run --config /path/to/reproduce.local.json --workspace /path/to/new-workspace --stage train
python scripts/reproduce.py run --config /path/to/reproduce.local.json --workspace /path/to/new-workspace --stage evaluate
```

预检报告缺失工件和环境差异，并核对研究环境的Python 3.10.20及PyTorch CUDA 12.8构建等历史要求；这与只用标准库、支持Python≥3.10的规则包不同。`stage` 在新工作区生成显式路径重定位副本，保留冻结源码，不能静默复用旧运行完成标记。

`run --stage train` 运行原始完整队列，每模型六类训练后分别评价；`run --stage evaluate` 只汇总已有prediction并执行复核，不补跑缺失推理。`run --stage roadshow` 在本地按序回放12包，需额外路演数据及校验清单，不自动SSH占用其他机器，也不参与训练或阈值更新。

历史训练依据为 `experiments/all_family_training_20261031/` 的冻结源码及记录；它还依赖早期预处理与外部标注解析代码。必须检查完整导入闭包、文件哈希与输出来源，不能只复制训练脚本。重定位运行属于记录了路径修订的复现运行，不应冒称原始源码逐字节未改的历史执行。

## 冻结训练配置

| 路线 | 轮数 | batch | 关键参数 |
| --- | --- | --- | --- |
| Qwen3-VL-8B QLoRA | 2 | micro 4，有效16 | NF4、r=8、alpha=16、q/v、dropout=0.05、AdamW lr=1.5e-4 |
| 监督 CNN 学生 | 12 | 64 | 三种子、当前双面板＋256维特征 |
| 蒸馏 CNN 学生 | 12 | 64 | 仅 train 软目标，T=2、监督与蒸馏等权 |
| 结构剪枝学生 | 6 | 64 | 继承同种子监督学生，通道/隐藏单元减半后恢复 |
| U-Net 系列 / DeepLab | 12 | 16 | 384×768双面板，分类＋可信区域辅助头 |
| YOLO 骨干适配 | 12 | 16 | 自定义多任务头，不报告原生检测mAP |
| DINOv2-small | 12 | 16 | 最后两层与新头，lr=1e-5，编码器输入224×224 |
| I-JEPA ViT-H/14 | 12 | 8 | 最后两层与新头，lr=1e-5，编码器输入224×224 |

种子为 `20261031`、`20261101`、`20261102`；单种子架构使用首个种子。学生双面板128×256。QLoRA可为显存减小microbatch并保持有效batch，实际配置以训练报告为准。监督与蒸馏学生还依赖只在train原始观测上训练的编码器。历史编码器warmup的停止取决于监督输入准备时机和时间预算，种子相同也不保证重训经历相同轮数；需保留历史预训练检查点及完整记录才能使用相同初始化。

同源/重复组合并后按固定组哈希约70%/15%/15%划分。每模型每轮访问全部可用且至少一位已知的train记录，并核验六类×五档覆盖。模型完成训练、val选权重与阈值并通过独立屏障后，立即测试30包；全模型结束再做总屏障与混淆矩阵独立复核。

## 指标复算

```bash
python scripts/recompute_dispatch.py --run-dir /external/all-family-training-20261031 --dispatch configs/dispatch_20_models.json --output /new/dispatch_metrics.json
```

主表为 `heldout_test`；含训练与验证的 `full_inventory_replay` 单列。种子均值是独立种子指标统计，不是预测集成。

已知位上计算 `recall=TP/(TP+FN)`、`precision=TP/(TP+FP)`、`F1=2TP/(2TP+FP+FN)`。无正例时F1/recall留空，其他零分母也不伪造值。微平均先加TP/FP/FN；宏平均须注明平均对象。

组合用原val选定专家和family阈值，逐位执行 `model >= threshold OR blind_rules`。复算需要对应预测、阈值、规则与有效位，不能直接拼接纯模型指标。独立测试P1–P4的24包组合为微召回94.81%、精确率53.10%、F1 68.07%；加入P0后，全30包精确率为46.33%、F1为62.25%，召回仍为94.81%。CME的P0样本误报为754/1570（48.03%），ULF为141/950（14.84%）。此前78.94%是纯模型行拼接，口径不同。原始逐包计数以 `results/reference/dispatch_metrics.json` 为准。

这些是公开P0技术proxy指标，未知科学归因被mask，不是官方隐藏科学定位成绩。路演B没有独立真值，禁止生成其F1/precision/recall。

## 数值和资源边界

图像与Qwen环境分别记录Python、PyTorch、CUDA、transformers、PEFT、bitsandbytes、torchvision、Ultralytics等版本及权重哈希；图像环境锁也包含路演入口使用的PEFT、bitsandbytes和accelerate依赖并集。宽版本范围安装不足以重放历史环境。固定种子和数据不保证不同GPU/驱动/CUDA算子上梯度逐位相同，完整重训应报告环境、指标差异和容差。

四专家在服务器通过真实推理重放：合计128条模型批次上下文记录，含五类各2个heldout示例对应的完整批次，沿用历史batch、归一化和BF16设置；概率最大绝对差0.001648低于0.005容差，所有阈值决策一致。CME仅用规则不需要模型推理。见[检查方法](inference_check.md)及[聚合记录](../results/reference/inference_check.json)。这是小规模推理核验，不是全量新准确率，也不是完整重训验证。

部分模型在 `pretrained=False` 构造时仍读取基础权重或YOLO初始文件。四个训练检查点约102MiB；加上DINO基础权重和两个YOLO原.pt，当前所需模型文件合计约207.8MB，不含环境、依赖和镜像层。完整离线包需统计这些构造依赖，不能只统计检查点；运行时4bit加载也不意味着磁盘BF16文件已量化。

比赛约束单独验收：完整Docker镜像≤20GB；指定A10硬件上每数据集完整评价≤3h。A100单模型耗时不能替代A10端到端测量，后者包含扫描、预处理、加载、输出和审计。当前不宣称这两项已经验收。

Git只提交源码、配置模板、测试、小型聚合指标及哈希。数据、逐样本真值、观测CSV、图像、权重、缓存、运行目录、虚拟环境和凭据均在外部保存。
