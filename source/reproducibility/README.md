# Frozen scientific code / 冻结科学代码

`legacy_runtime/opc_agent.py` is the byte-identical historical rule engine read by
the archived experiments. `external_preprocessing/annotation_diff.py` is the
previously external phase-2 annotation parser, vendored to close an import
dependency. Neither directory contains observations, annotations, model weights,
or generated previews. Author/license provenance has not been independently
established; no permissive license is asserted for this snapshot.

这里保留实验实际使用的旧规则引擎，以及原先位于阶段二目录中的标注解析代码，
避免新工程包装改变历史实验依赖。它们由来源哈希校验。新命令行入口位于
`src/spaceqa_agent`；历史源码在 `experiments`。路径重定位只写入单独的新工作区，
不修改冻结源码、原始数据、原权重或既有实验目录。

完整重训依赖私有 manifest、准备完成的 full-matrix 缓存/配准工件和基础权重。
这些都必须通过预检提供，不能声称仅凭本仓库就能无数据重训。由于历史编码器
预训练的停止条件依赖准备进度/耗时，GPU完整重训不保证逐比特相同；已保存的
预测和混淆矩阵可独立精确复算，新的完整训练会有独立来源回执。
