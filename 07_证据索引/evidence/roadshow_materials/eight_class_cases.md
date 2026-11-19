# 八类真实输出案例与讲解

案例来自本轮真实输出。每例均可通过包名与sampleId在前端检索；模型候选与规则确认分开说明。

## 数据文件缺失（L0-01）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:0kJ5J1no1gcOJ7b6a1eS.png.json`。

状态：规则确认、可计技术分。

程序读取标签的文件引用并在实际文件系统核验，对应文件不存在时保留预期路径。这是直接可核验的文件缺失，未凭模型猜测补图。

本例输出：标注存在，对应数据文件缺失

定位：`{"type": "file", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/label/0kJ5J1no1gcOJ7b6a1eS.png.json", "annotation_pointer": null, "expected_data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0kJ5J1no1gcOJ7b6a1eS.png"}`

测量：`{}`

## 边界与位置偏差（L2-G）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:0N_IIOqw3WTgnPVTKjez.png.json`。

状态：模型/上下文候选、不可计确认分。

RGB模型独立看原图预测目标区域，再与输入标注进行匹配，测量IoU和中心位移。当前框标出被比较的输入对象；真实科学边界仍需独立参考确认。

本例输出：boundary_position_deviation；原始图像预测与当前标注对比，尚未获独立科学确认

定位：`{"type": "imageRegion", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0N_IIOqw3WTgnPVTKjez.png", "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0N_IIOqw3WTgnPVTKjez.png", "bbox": [3005.9, 1570.5, 3453.7, 1962.1], "coordinate_system": "image_pixels", "origin": "top_left", "axis_order": "xy", "image_size": [4096, 4096], "frame_id": "fc8fabb38c3b50148f0bea91c30590d3c2e504d0f2609259c04a0a0d1e2f0513", "annotation_pointer": "/hek_event_annotation/matched_events/4/bbox_polygon_pixel"}`

测量：`{"mask_iou": 0.45364536453645365, "coverage_reference": 0.5299684542586751, "coverage_annotation": 0.7590361445783133, "area_ratio": 0.6982124079915878, "normalized_distance": 0.026749448244949132}`

## 类别标注错误/类别契约冲突（L2-P）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:0JStaZ6ZlwMRSTTqqYQY.png.json`。

状态：规则确认、可计技术分。

程序比较具体对象类别与本任务声明的允许类别集合。冲突可在输入JSON字段核验；它证明类别契约不一致，不自动证明图片在科学上属于哪一类。

本例输出：类别'CH'不属于声明的任务类别集合['AR']；依据：Participant task-family vocabulary, not independent semantic truth

定位：`{"type": "imageRegion", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/label/0JStaZ6ZlwMRSTTqqYQY.png.json", "annotation_pointer": "/hek_event_annotation/matched_events/3/event_type", "bbox": [1988.0, 2830.5, 2265.1, 3043.4], "coordinate_system": "image_pixels", "origin": "top_left", "axis_order": "xy", "image_size": [4096, 4096], "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0JStaZ6ZlwMRSTTqqYQY.png"}`

测量：`{"actual": "CH", "allowed": ["AR"], "contract_source": "Participant task-family vocabulary, not independent semantic truth", "affected_object_fraction": 0.2727272727272727, "invalid_object_count": 6, "classified_object_count": 22}`

## 时段起止时刻疑点（L2-T）

包：`ULF_sample_0yu0mc7d5a`；样本：`ULF_sample_0yu0mc7d5a:02SqBLVBdGSVk_kp7uo9.csv.json`。

状态：模型/上下文候选、不可计确认分。

冻结时段分类头筛出此样本后，程序将疑点关联到输入事件的起止记录。数值t轴缺独立UTC原点，因此区间是复核入口，不是已经证实的真实错误端点。

本例输出：八类时段候选已关联到被质疑的输入事件起止记录。CSV仅含数值t轴，未提供独立UTC原点；列出其首末值供复核，不把输入区间认作真实错误时段或独立端点偏差。

定位：`{"type": "timeInterval", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/091319f73a6cdfadaadf9e68525bf0d1adcb1067938215b2be40b6ac7e60fc3d/ULF_sample_0yu0mc7d5a/02SqBLVBdGSVk_kp7uo9.csv.json", "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/091319f73a6cdfadaadf9e68525bf0d1adcb1067938215b2be40b6ac7e60fc3d/ULF_sample_0yu0mc7d5a/02SqBLVBdGSVk_kp7uo9.csv", "annotation_pointer": "/ulf_event_annotation/matched_events/0", "timeInterval": ["2016-08-03T04:20:00+00:00", "2016-08-03T11:00:00+00:00"], "time_unit": "ISO8601 UTC", "interval_source": "questioned_input_event_record_not_independent_true_endpoint", "time_axis_domain": {"kind": "numeric_axis", "column": "t", "columnIndex": 0, "firstValue": 0.0, "lastValue": 3599.0, "unit": "not declared by CSV header", "source": "actual CSV first and last rows", "independent_absolute_time_mapping_established": false, "interpretation": "Raw numeric axis only; no independent UTC origin is supplied by this CSV."}, "scientific_endpoint_reference_available": false}`

测量：`{"declared_event_interval": ["2016-08-03T04:20:00+00:00", "2016-08-03T11:00:00+00:00"], "interval_role": "questioned_input_event_interval", "independent_endpoint_reference_available": false, "actual_csv_numeric_axis": {"kind": "numeric_axis", "column": "t", "columnIndex": 0, "firstValue": 0.0, "lastValue": 3599.0, "unit": "not declared by CSV header", "source": "actual CSV first and last rows", "independent_absolute_time_mapping_established": false, "interpretation": "Raw numeric axis only; no independent UTC origin is supplied by this CSV."}}`

## 元数据尺寸矛盾（L2-F）

包：`EPB_seg_sample_fbbqy66u2d`；样本：`EPB_seg_sample_fbbqy66u2d:2KKzxROcF9GHYym46zRB.json`。

状态：规则确认、可计技术分。

程序读取PNG头中的真实宽高，与JSON整图尺寸字段直接比较。这里只检查整图字段，避免将目标框宽高误当图像宽高。

本例输出：声明尺寸[1024, 512]，PNG实测尺寸[1024, 1024]

定位：`{"type": "annotationRecord", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/5f75097e1ed2fc55cd45bc2dc98e53f0b0265458fc14a4281b281c19fd65bf50/EPB_seg_sample_fbbqy66u2d/2KKzxROcF9GHYym46zRB.json", "annotation_pointer": "/basic_annotation/imgSize"}`

测量：`{"declared": [1024, 512], "actual": [1024, 1024], "max_pixel_error": 512}`

## 漏标（L3-06）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:0aETy9ibah5RnT7sMUvm.png.json`。

状态：模型/上下文候选、不可计确认分。

RGB模型在图像中预测出一个与输入对象没有匹配的区域，并满足候选阈值。该框来自独立图像预测，是漏标待核区域；模型检出不等于已证实真实目标。

本例输出：unmatched_independent_target；原始图像预测与当前标注对比，尚未获独立科学确认

定位：`{"type": "imageRegion", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0aETy9ibah5RnT7sMUvm.png", "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0aETy9ibah5RnT7sMUvm.png", "bbox": [2986.6666666666665, 1376.0, 3264.0, 1685.3333333333333], "coordinate_system": "image_pixels", "origin": "top_left", "axis_order": "xy", "image_size": [4096, 4096], "frame_id": "6b0de3e789cd1165d77fee25c0c1d2c215f4f03c28fe405629d18fcfa2275d3f", "annotation_pointer": null}`

测量：`{"predicted_component_pixels": 563}`

## 虚标（L3-07）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:4lSCczKt5zTYXMU2Bij4.png.json`。

状态：模型/上下文候选、不可计确认分。

八类分类器提出虚标疑点，新RGB组件与输入对象匹配后仍无支持。本框来自被质疑的输入标注，不是模型预测的独立缺陷框；目标未检出也可能是模型漏检，所以不能自动扣分。

本例输出：八类虚标候选阳性且该输入标注对象未匹配新一轮独立RGB组件；框标明被质疑的输入对象位置，不是模型独立预测的缺陷框。模型未检出不能确认虚标。

定位：`{"type": "imageRegion", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/label/4lSCczKt5zTYXMU2Bij4.png.json", "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/4lSCczKt5zTYXMU2Bij4.png", "annotation_pointer": "/hek_event_annotation/matched_events/0/bbox_polygon_pixel", "bbox": [1891.7, 2567.8, 2009.9, 2739.9], "coordinate_system": "image_pixels", "origin": "top_left", "axis_order": "xy", "image_size": [4096, 4096], "region_source": "questioned_input_annotation_not_independent_predicted_defect_region", "independent_model_evidence": "fresh RGB components failed to support this input object at recorded matching tolerance"}`

测量：`{"max_predicted_component_bbox_iou": 0.0, "independent_component_count": 3, "matching_min_mask_iou": 0.1, "negative_evidence_verified": false}`

## 覆盖不足/覆盖过度（L3-MIS）

包：`AIA_AR_sample_3mogtggl50`；样本：`AIA_AR_sample_3mogtggl50:0N_IIOqw3WTgnPVTKjez.png.json`。

状态：模型/上下文候选、不可计确认分。

把匹配对象的输入标注面积与独立RGB预测面积比较，报告相对覆盖率、面积比等。覆盖不足或过度的方向来自相对预测的测量，科学正确边界仍待核。

本例输出：undercoverage；原始图像预测与当前标注对比，尚未获独立科学确认

定位：`{"type": "imageRegion", "path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0N_IIOqw3WTgnPVTKjez.png", "data_path": "/staff/tangjiarui/OPC/opc-agent/runs/roadshow-evaluation-20261101/extracted/86e990ec307fc61a688715b3ae267346cea18e770792426cc00fb77966c36049/AIA_AR_sample_3mogtggl50/data/0N_IIOqw3WTgnPVTKjez.png", "bbox": [1010.2, 2344.4, 1140.2, 2442.4], "coordinate_system": "image_pixels", "origin": "top_left", "axis_order": "xy", "image_size": [4096, 4096], "frame_id": "fc8fabb38c3b50148f0bea91c30590d3c2e504d0f2609259c04a0a0d1e2f0513", "annotation_pointer": "/hek_event_annotation/matched_events/2/bbox_polygon_pixel"}`

测量：`{"mask_iou": 0.1, "coverage_reference": 0.10015649452269171, "coverage_annotation": 0.9846153846153847, "area_ratio": 0.10172143974960876, "normalized_distance": 0.047418478236705724}`

