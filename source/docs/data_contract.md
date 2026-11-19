# 数据约定 / Data contract

真实观测与标签不随仓库分发。输入由命令行或复现配置提供；缺失、解析失败、引用失配和无法判定必须保留，不能删除记录来提高覆盖率。

## 数据类型与目录

原公开数据为六类 × P0–P4，共 30 包。类型前缀后的脱敏后缀没有质量含义。

| 类型 | family ID | 数据体 |
| --- | --- | --- |
| AIA | `AIA_AR_sample` | PNG 图像 |
| HMI | `HMI_AR_sample` | PNG 图像 |
| CME | `CME_C2_sample` | PNG 与事件参数 |
| EPB | `EPB_seg_sample` | PNG 与 mask shapes |
| DMSP | `DMSP_SSUSI` | PNG 与区域标注 |
| ULF | `ULF_sample` | 数值 CSV 与事件时间窗口 |

支持分目录和平铺布局：

```text
publish/
  AIA_AR_sample_<suffix>/
    data/X.png
    label/X.png.json
  EPB_seg_sample_<suffix>/
    X.png
    X.json
  ULF_sample_<suffix>/
    X.csv
    X.csv.json
```

此示例对应路演 AIA、EPB、ULF 的配对规则。原发布版其他变体由清单和标签引用解析，不能假定任何 JSON 都适用相同后缀规则。

## 关键标注字段

| 类型 | 事件或区域入口 | 注意事项 |
| --- | --- | --- |
| AIA / HMI | `hek_event_annotation.matched_events` | 路演 AIA 类别为 `concept`；空几何不能直接判科学错误 |
| CME | 事件列表、`cme_time`、角域与物理参数 | 单帧可观察范围与真实事件时间不同 |
| EPB | `science_annotation.mask.shapes` | 类别字段 `label`；`shape_type=polygon` 时必须保留原始顶点，展示可以计算外接框作索引但不能用外接框替代掩膜；比较避免依赖数组顺序 |
| DMSP | 区域形状，含 `specialShapes` | 保留形状类型和声明尺寸 |
| ULF | `ulf_event_annotation.matched_events` | 事件声明含 `kind`、`start_dt`、`end_dt` |

通用参数可能位于 `basic_annotation`、`science_annotation` 或 `label_header`。尺寸与真实 PNG 头对照，采样率和时长从 CSV 实际时间轴推算。脱敏文件名不作为台站等语义证据。

路演 ULF 的 CSV `t` 是相对 `basic_annotation.startTime` 的秒偏移。旧公开版视图曾优先使用 `science_annotation.dateObs`；复现需保留发布版本差异，不能混用时间锚点。

## 六类 × 八类展示子集

使用 `scripts/export_demo_subset.py` 可以从服务器上的私有 30 包数据和全量八类测试 CSV 导出一个小型审阅包。它按 AIA、HMI、CME、EPB、DMSP、ULF 各选一个样本，生成八类统计表、证据形式、哈希和缩略图；不会把原始数据、权重或逐样本台账写入仓库。

```bash
python scripts/export_demo_subset.py \
  --data-root /path/to/private/newpublish \
  --metrics /path/to/all30_classes_test.csv \
  --variant selected_eight \
  --copy-media \
  --output /path/to/demo_subset \
  --zip /path/to/demo_subset.zip
```

导出的 `index.html` 是给评审查看的入口。EPB 缩略图使用原始 `mask.shapes[*].points` 绘制多边形；若页面出现矩形，只能是模型候选区域的索引框，不能称为 EPB 的原始多边形标注。

## 清单、防泄漏与目标

历史清单覆盖 123,565 个逐档记录、24,713 个 `base_sample_id`。五档标签同源不代表像素相同，AIA/HMI 存在图像变化。缺失引用、预期路径和路径别名异常保留。

划分合并 `base_sample_id`、清单 `split_group_id` 与相同原始内容的哈希关系，再用固定组哈希划分 train/val/test。同源版本及重复内容必须整体划分，test 不用于梯度、选轮次、阈值或蒸馏。

P0 仅用于离线构造技术目标或可信参照检查，研究推理只读当前观测、标注与参数。`target` 为 `G/C/D` 三位，`truth_valid` 表示每位是否已知；未知位不进入对应损失或混淆矩阵，不能当负例。P0 无正例时 F1/recall 留空，报告误报率。每包同时报告 n、已知/未知、缺输入、弃权及 TP/FP/FN。

P0无正例不等于无需计入误报：比较总体方案时须同时报告P1–P4的24包汇总和P0–P4的全30包汇总。当前组合的CME P0有754/1570个样本误报，ULF有141/950个；不能以P1–P4的较高F1代替全包表现。

复现入口依赖历史外部清单及已完成的full-matrix预处理缓存、配准/注册记录；这些是明确的私有起点，不由代码仓库或仅有原始数据自动补齐。

## 路演边界

路演为三类 × 四目录，共 12 包；每类四目录对象互不重叠，不是 P0–P4 配对。tar 须先通过清单 SHA-256 校验再解包，拒绝越界路径和链接。

待评价 JSON 是研究对象，不是缺陷真值。无额外独立裁定时，只输出候选、覆盖率、完整性和耗时，不能计算路演 F1、精确率或召回率，也不能用自己的预测生成真值。
