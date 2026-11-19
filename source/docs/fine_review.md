# 九类缺陷的多模态结构化复核

本模块在现有规则与HGB7之后追加实例复核，无需微调任何专家。旧版检测、分数与服务保持原入口，调用本入口才执行新的复核。细分复核不是把G/C/D直接重命名为九类。

## 输入与输出

每个case包含当前原始标注JSON、真实解码尺寸/文件状态、图像或ULF波形、当前标注视图、21个专家的63个G/C/D概率、HGB7三概率及结构特征。图像区域使用原图像素，字段使用原JSON Pointer，时间区间使用明确的秒数轴。病例ID和数据路径在请求中采用无档位信息的别名，不发送P0、参考位置、训练标签、污染档位或评估答案。

输出九类检查的全部四状态、逐处问题类别、定位、解释、模型自报置信度和独立严重程度。计数由程序去重，不能由模型随意填写。JSON Schema约束格式；程序额外检查实际字段、越界、时间轴、类别与定位兼容、fail是否有实例、重复问题、缺文件不能判虚标等。失败至多修复两次，最终保留undetermined。

L2-F尺寸矛盾可由程序重算像素差；其他未经独立核验的模型严重程度保留`llm_estimate_unverified`。复核结果存为candidate，不能因为通过格式校验就成为已确认科学错误。

机械后处理`mechanical-reconcile-1`不依赖模型是否答对：缺文件和尺寸冲突单独生成`program_findings`，合并为`combined_defects`；大模型报告的缺文件若与当前文件存在状态矛盾，或尺寸错误的字段不在真实冲突清单，会被隔离到`blocked_mechanical_claims`。同一规则事实与大模型复述只计一次，保留别名链。最终状态在`final_checks`，原始模型说法仍可查。API失败也不删除已核实的机械问题。

256维结构特征的后240维是带符号哈希桶，不存在“一桶一个物理字段”的解释。短上下文请求发送前16维及哈希向量范数，完整256维通过MCP读取；多边形长坐标列表可投影为外接框摘要，程序仍用完整原JSON核验字段。不能将摘要当成精细分割真值。

## API调用

使用OpenAI兼容Chat Completions。DeepSeek文本模型可组织解释；视觉复核要选实际通过图像测试的Qwen或其他视觉模型。仅模型名称出现在列表中不代表支持图片。

先用`python -m spaceqa_agent.refinement prepare --data /input/sample.png --label /input/sample.json --family EPB_seg_sample --signals /external/current_expert_signals.json --output /external/new-case`准备新样本。signals文件提供`fusion_probabilities`、`experts`和可选`structural_features`，来自该样本实际专家推理。该入口读取原始图像绘制当前标注，不需要训练集或P0。CSV只在存在已知`t`秒数列时给出时间轴，不按行数假设1Hz。然后使用返回的case和images调用下面的review命令。

```bash
export PYTHONPATH=/absolute/path/to/spaceqa-agent/src
export SPACEQA_LLM_BASE_URL=https://api.llm.ustc.edu.cn/v1
export SPACEQA_LLM_MODEL=qwen3.8-chat
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}api.llm.ustc.edu.cn"
# SPACEQA_LLM_API_KEY从既有私有配置注入，不写入代码或报告。
python -m spaceqa_agent.refinement review \
  --case /external/cases/CASE-example.json \
  --image /external/cases/media/CASE-example-observation.jpg \
  --image /external/cases/media/CASE-example-annotation.jpg \
  --output /external/new-review.json
```

供应商响应仅支持JSON对象时，在提示中提供Schema并进行本地校验；支持原生JSON Schema的服务可用Python接口`native_schema=True`。这两个接口都必须通过相同的本地定位校验。响应、请求哈希、图像哈希、模型名、耗时、用量和全部尝试保存在结果中。温度为0不保证服务商逐字确定；复算应使用冻结响应。

## Skill与MCP

技能入口：[SKILL.md](../.agents/skills/spaceqa-fine-review/SKILL.md)。仓库内技能支持项目发现，不修改全局Codex配置。

```bash
python -m spaceqa_agent.review_mcp --case-root /external/cases
```

这是MCP标准stdio传输、JSON-RPC协议的实际服务，提供7个工具：list_cases、review_schema、get_case、get_case_image、validate_review、aggregate_reviews、evaluate_locations。仅开放指定病例根目录下的证据；没有任意命令执行接口。服务器客户端用[服务器配置](../deploy/review_mcp.example.json)；本机客户端用[SSH配置](../deploy/review_mcp.ssh.example.json)，SSH 主机、路径和源地址均由用户环境配置，不写入示例。此配置是可供接入的示例，本轮未替用户修改桌面应用的全局MCP注册。

Skill指导调查流程，MCP暴露证据及校验工具，JSON Schema限制格式，定位验证器限制引用。四者都不能保证语言模型判断科学正确。`evaluate_locations`复用现有独立定位评估器，只有具备独立审查范围与真值时才给出P/R/F1。

## 用户指定的负担统计

令样本i的G/C/D疑点概率为p(i,g)，大模型定位到的去重缺陷数为n(i,g)：

`B = sum_i sum_g n(i,g) * p(i,g) / N`

这就是“各有问题样本的问题数×混合专家概率，求和除以总样本数”。N为原始清单记录数，缺文件、执行失败和没有完成检查的样本仍保留。B越高表示模型估计的缺陷负担越大，可超过1；它不是污染样本比例，不是独立物理严重程度，也不是官方q。G/C/D概率没有做逐实例校准，故B不能严格解释为已校准的缺陷期望。

示例：100个样本中，某样本2处G问题且pG=.8，另一条1处C问题且pC=.6，其余完整检查未见问题，则B=(2×.8+1×.6)/100=.022。若只检查了这两条，其余98条未知，完整B返回null，另列已观察部分贡献.022，不能把其余98条当作正常。

正式评分、候选负担、已确认缺陷密度分别输出；不把模型自报概率二次相乘，不用无依据的1-B强行转成0–100质量分。排序只在可比口径和完成范围明确后生成，无法区分的同分不可用目录名字冒充真实优劣。

全部样本收到响应也不代表全部检查完成；任何适用检查仍为undetermined时，`response_coverage_complete`可以为true，但`complete`为false，完整负担分仍返回null。已观察的贡献和程序已确认的问题继续单列，失败样本不从分母删除。

## 评估与复现

`experiments/jwvl_optimization_20261104/fine_review_eval.py`对冻结验证分组中的全部30包分层取样，进行图像/无图像、含/不含专家信息的对照，参考答案仅在评估进程中使用。规则与HGB7使用相同病例。专家消融在原3730条选型验证集运行，冻结权重、冻结阈值，将一个专家概率置为中性.5，以测量当前模型对该信号的依赖。此敏感性实验不等于重新训练后删掉该专家的因果增益。

九类缺陷的类别与位置联合指标需要独立参考；不能拿P0派生的G/C/D标签或大模型自己的答案充当官方九类真值。耗时报告区分完整前向、缓存复算和API复核，不能沿用旧19.37GB镜像与4022秒实测为新21专家+API方案背书。
