# 多数据族合成消融基准

该基准构造 6 个公开数据族、每族 `clean/P1/P2/P3/P4` 共 30 个小包，比较：

- A：仅格式、配对、引用一致性；
- B：增加单包结构、几何、尺寸、时间、重复和类别规则；
- C：增加同源 clean/P1-P4 差分、证据链和待判定输出；
- D：只知道每族 clean 包的弱监督异常分数，不输出缺陷类别或位置。

运行：

```bash
python benchmark/benchmark_multifamily.py
python -m unittest discover -s tests -v
```

默认结果写入 `benchmark/multifamily_results.json`。所有输入均在临时目录生成，不读取或写入比赛原始数据。

指标边界：

- precision/recall/F1 的单位是“包 × 缺陷类别”，不是隐藏位置级官方 D；
- `L0-01` 与 `L0-03` 统一为 `L0-MISSING`，避免对公开文字无法区分的同一观测双重计分；
- 等级排序比较每族 5 档的全部 10 个二元对，共 60 对；
- 严重度单调性只验证受控注入，不代表官方隐藏集 `S_Q`；
- EPB/DMSP 保留官方参考 JSON 中 header 指向 JSON、basic 指向 PNG 的形式，用于监测 schema 误报。
