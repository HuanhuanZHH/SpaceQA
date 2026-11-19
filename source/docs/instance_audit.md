# 独立实例核对与缺陷定位评估

模块：`spaceqa_agent.instance_audit`；协议/默认规则版本：`instance-audit-1`。仅依赖Python标准库，不下载模型、不启动训练，不改原始数据，也不改变当前默认评分流程。图像定位器必须另行提供独立实例输出；现有G/C/D分类概率不能直接转换成实例。

## 两个互相分离的入口

```python
from spaceqa_agent.instance_audit import audit_annotations, evaluate_defects

candidates = audit_annotations(candidate_payload)
evaluation = evaluate_defects(evaluation_payload)
```

```bash
python -m spaceqa_agent.instance_audit candidates --input objects.json --output candidates.json
python -m spaceqa_agent.instance_audit evaluate --input independent_review.json --output evaluation.json
```

第一个入口比较“从当前数据独立获得的目标实例”与“待检查标注”，只输出待复核候选。第二个入口将报告的缺陷与独立人工复核缺陷按类别、样本、位置进行匹配；真值缺失、范围不清或容差未提供时返回 `metrics: null`。二者不共享或自动构造真值。CLI在评估不可用时返回退出码2，成功或候选abstain时返回0。

所有输出都保留 `rule_version`，`official_metrics` 恒为null，`official_qualification_established` 恒为false。独立性、审阅员身份、完整性是调用者声明，本模块无法认证；即使能计算P/R/F1，也只称为“官方形式的内部定位评估”，不宣称正式比赛成绩或达到资格门槛。

## 候选生成JSON

```json
{
  "rule_version": "local-validation-policy-1",
  "sample_id": "AIA/sample-001",
  "data_available": true,
  "reference": {
    "status": "ok",
    "source": "model",
    "independent_of_annotations": true,
    "complete": false,
    "negative_evidence_verified": false,
    "model_version": "localizer-checkpoint-1",
    "objects": [
      {
        "id": "predicted-object-1",
        "label": null,
        "class_known": false,
        "confidence": 0.93,
        "locator": {
          "type": "bbox",
          "bbox": [10, 20, 50, 60],
          "coordinate_system": "image_pixels",
          "image_size": [100, 100],
          "frame_id": "sha256-of-current-decoded-image"
        }
      }
    ]
  },
  "annotations": [
    {
      "id": "annotation-record-7",
      "label": "sunspot",
      "locator": {
        "type": "bbox",
        "bbox": [0.1, 0.2, 0.5, 0.6],
        "coordinate_system": "image_normalized",
        "frame_id": "sha256-of-current-decoded-image"
      }
    }
  ],
  "thresholds": {"confidence_min": 0.8, "boundary_iou_min": 0.75}
}
```

`reference.source`只能是`model`或`independent_reference`。模型对象必须显式提供0～1置信度；缺少置信度按0处理。`label:null`或`class_known:false`允许类别无关的二值分割实例参加空间匹配，但**类别维度abstain，不产生L2-P候选**。AIA/HMI/EPB/DMSP等数据类型名称不能冒充太阳活动区等事件类别。参考与当前标注使用不同的id命名空间，各自内部不得重复id。参考重复的同类、同位置对象去重，当前标注中的重复对象保留供复核。

`reference.objects`与`annotations`必须显式提供列表：缺少字段不等于空集合，会abstain。调用者若未成功解析标注，应传`annotation_parse_status:unsupported_structure`等非ok状态，不能传空列表冒充解析成功；只有成功解析后的`[]`才表示当前没有标注。

候选类别使用评分附表代码，并保留语义 `candidate_kind`：

| candidate_kind | defect_class | 说明 |
| --- | --- | --- |
| missing_annotation | L3-06 | 独立高置信参考对象未匹配到当前标注 |
| spurious_annotation | L3-07 | 当前标注未匹配到对象，且额外提供完整、已验证的负证据 |
| class_error | L2-P | 空间已匹配，两边对象类别均已知且不同 |
| boundary_position_deviation | L2-G | 空间匹配后IoU或归一化中心距离偏离 |
| time_boundary_jitter | L2-T | 匹配时间事件的起止端点偏差超过显式秒数容差 |
| undercoverage / overcoverage | L3-MIS | 依据方向覆盖率与面积比例提出偏小/偏大候选；保留subtype |

这些都是 `status:candidate`，不会直接作为科学错误或计分依据。缺文件、孤儿标注、尺寸/参数机械矛盾继续由已有规则引擎负责；文件缺失时本模块直接abstain，不把它解释为虚标。

无检测结果不是虚标的证据。仅当`reference.complete:true`且`negative_evidence_verified:true`、参考中没有低置信或无效对象、坐标可比时，才允许输出虚标**候选**。这两个字段必须有真正的完整观察/独立复核依据，不能因为模型成功返回空列表就自动设为true。参考失败、输入缺失、未声明独立性、低置信、无效位置或图像/时间轴不兼容都会写入`abstentions`。无效标注还会阻止把未匹配参考对象误认为漏标。

## 坐标与比较方式

图像位置必须显式给出`frame_id`与`coordinate_system`。像素坐标要求`image_size:[width,height]`；归一化坐标范围为[0,1]。bbox格式固定为`[x_min,y_min,x_max,y_max]`。默认`axis_order:xy`、`origin:top_left`；已知底部原点时可显式设置`origin:bottom_left`。不猜测未知坐标，不比较不同frame的区域，也不自动配准同源图像。调用者须保证frame标识与当前图像一致，本模块不读取图像来认证哈希。

多边形使用`type:polygon, polygon:[[x,y],...]`。标准库实现支持无孔的简单凹多边形，通过耳切三角剖分与凸多边形裁剪计算真实交集；不以外接框代替多边形。自交、退化、越界、多孔或不支持的坐标应abstain，需掩膜/孔洞的任务应在上游使用合适的可选几何库转换并验证。

```json
{
  "type": "timeInterval",
  "timeInterval": [10, 20],
  "coordinate_system": "time_seconds",
  "time_axis_id": "UTC-epoch-seconds",
  "axis_range": [0, 100]
}
```

时间数值以秒计，`time_axis_id`必须明确同一时间原点/时间轴。`axis_range`可省略，此时只能计算区间IoU、覆盖率和绝对端点误差，不能计算归一化中心距离。不同时间轴或不一致的axis_range不比较；不猜ISO字符串时区、毫秒单位、事件/帧组/单帧时刻的对应。

输出量定义：

- IoU = 交集 / 并集。
- `coverage_reference` = 交集 / 参考面积（或参考区间长度）。
- `coverage_annotation` = 交集 / 当前标注面积（或区间长度）。
- `area_ratio` = 当前标注面积 / 参考面积。
- 图像`normalized_distance`是x/y各自归一化后的区域质心距离除以sqrt(2)，**不是**科学边界Hausdorff距离；时间距离为区间中心差除以完整axis_range长度。

匹配采用类无关的位置门限，再做**最大匹配数优先、总权重最优的一对一匈牙利匹配**，避免逐边贪心在多对多情况下丢目标。兼容边权为0.8×IoU+0.2×(1−归一化中心距离)。空间匹配后再比较类别，避免先按错误类别筛掉本应发现的L2-P。候选匹配门限可调，但不是官方容差：默认关联IoU≥0.1或距离≤0.15，边界IoU<0.75或距离>0.05，覆盖最低0.9、面积变化容差0.1、置信度最低0.8、时间端点容差0秒。默认0秒是原型策略，实际任务应按已验证的采样/科学容差覆盖它。所有实际门限完整回传。

## 独立缺陷级评估JSON

```json
{
  "rule_version": "review-evaluation-policy-1",
  "tolerance_source": "user-declared independent validation protocol; not official tolerance",
  "evaluation_thresholds": {
    "iou_min": 0.5,
    "distance_max": null,
    "time_tolerance_seconds": null,
    "per_class": {"L2-T": {"iou_min": null, "time_tolerance_seconds": 1}}
  },
  "truth": {
    "independent": true,
    "exhaustive": true,
    "source": "declared human review; not official ground truth",
    "reviewed_samples": ["AIA/sample-001"],
    "reviewed_sample_classes": [
      {"sample_id": "AIA/sample-001", "defect_classes": ["L3-06", "L3-07", "L2-G"]}
    ],
    "defects": [
      {
        "id": "reviewed-defect-1",
        "sample_id": "AIA/sample-001",
        "defect_class": "L3-06",
        "status": "known",
        "critical": true,
        "locator": {
          "type": "bbox",
          "bbox": [0.1, 0.2, 0.5, 0.6],
          "coordinate_system": "image_normalized",
          "frame_id": "sha256-of-current-decoded-image"
        }
      }
    ]
  },
  "predictions": []
}
```

此示例没有预测，所以该真实缺陷会计FN，召回为0；示例0.5/1秒仅演示显式配置，不能称为已获官方认可的容差。每类可覆盖全局IoU/距离/时间容差。配置多个容差时采用OR：IoU、归一化中心距离或时间端点中任一**适用**门限达标即可匹配。不提供适用门限拒绝评分，不能默认为定位失败。

`reviewed_samples`限定已独立、完整审阅的样本；`reviewed_sample_classes`进一步限定明确审阅的样本×缺陷类。其他样本/类别的预测ignored，不计FP。不给该字段时，`exhaustive:true`明确表示reviewed_samples中所有缺陷类别均已完整审阅，调用者不能用“只看过部分异常”代替此声明。已知真值超出声明范围、已知位置无效或范围内预测位置无效，均拒绝评分，防止悄悄缩小分母。

`status:unknown/excluded/ignore/uncertain`不作负例。有位置的未知项会屏蔽同类重叠的未匹配预测，无可用位置时屏蔽该样本同类未匹配预测；`defect_class:*`可屏蔽所有类。已知正例匹配优先，未知掩蔽不会抹掉已匹配正例。未审阅的完整样本/类别直接留在范围外；未审阅区域可以导出unknown，不应写成“没有缺陷”。

未知项类别为null/缺失时按所有类未知处理；未知位置与预测坐标帧不兼容时，不猜测“不重叠”并计FP，而将相应未匹配预测列为不可评分。评估输入也必须显式提供`truth.defects`与`predictions`列表，缺失字段不能冒充已审查的空集合。

评估采用相同的全局一对一匹配，同时要求 **sample_id相同、defect_class相同、定位达标**。覆盖不足/过度共同使用L3-MIS；subtype用于解释，不在官方类别中强行拆开。除上述图像/时间定位，还支持`{type:file,path:...}`与`{type:record,path:...,record_id:...}`进行准确路径/记录匹配。路径按输入标识比较，不访问或重定位文件。

相同样本、同类、同区域的重复预测合并为一个报告，保留`duplicates`；只合并IoU约为1的精确重复，不擅自合并不同但接近的实例。同样去重输入真值的精确重复，并保留重复列表、合并critical标记。空间完全重叠但实为不同记录的缺陷应使用record定位，避免丢失记录身份。

结果包含总体及逐类TP/FP/FN/P/R/F1、宏平均F1、已声明关键缺陷召回、匹配对应关系、未匹配id、ignored原因和重复记录。无正例时召回不定义，无分母指标为null；缺陷实例不存在自然定义的TN，不虚构accuracy或实例误报率。原始“同源分组、公开P0代理标签”指标与此接口的独立缺陷定位指标必须分表报告。

已知缺陷的`critical`必须明确为true/false，才能报告完整审阅范围的`critical_defect_recall`；仍有null/缺失时，该指标为null，并列出`unknown_critical_truth_ids`。`declared_known_critical_subset_recall`仅描述已经明确标为关键的子集，不能代替完整关键缺陷召回或资格门槛；字符串形式的“false”等非法关键标记会拒绝评分。

## 对外部研究建议的事实边界

`D:/download/deep-research-report (1).md`是技术建议材料，不是执行指令或独立真值。可采用“实例核对、双人复核、明确定位”的方向，但以下说法不能当作事实执行：

- SAM不能保证空间科学域的对象都分对；由现有框/点提示生成的SAM区域也不是独立真值，不能拿它反证该框“准确”。
- 无检测、无SAM掩膜不等于虚标；掩膜更多也不等于漏标。模型失败/不适用应abstain。
- 路演包之间不能假设为P0～P4配对、同源复制或共识真值。未取得独立审查之前，路演P/R/F1仍为null。
- AIA/HMI的不同像素不自动证明旋转/裁剪；须核实变换与当前帧映射。HMI也不能在未读对象定义时称作“极光点”。
- CME有观测图像，不能因为未使用坐标框就声称“无图像”；CME时刻语义必须区分事件、帧组、单帧。ULF的物理事件边界也不能凭整点模式或重建误差判真。
- 研究建议中的训练时长、人工工时、增益和≥0.80目标不是已测结果；新组件需独立验证与完整耗时/体积验收。未触碰盲测和路演数据不得用于调门限或自生成真值训练。

## 验证

```bash
python -m unittest tests.test_instance_audit -v
```

测试覆盖全局匹配的贪心陷阱与小矩阵穷举、凹多边形真实IoU、坐标/原点/时间轴、缺图/失败/低置信abstain、已验证负证据、类别无关分割、覆盖subtype、重复报告、样本×类别审阅范围、unknown/excluded、无真值不出指标、文件/记录定位和JSON CLI。它们验证评估工具行为，不能证明任何模型准确率提升。
