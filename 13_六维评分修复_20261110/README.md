# 六维复算入口

本目录是已测试的独立 CPU 六维复算组件。R4 负责已有模型推理，随后可用本入口补充协议并重算六维；不需要重跑 GPU。`pipeline.py` 和 `publish_run.py` 是供后续镜像集成的补丁，尚未装入当前 R4，也未修改另一条 0.7.0 候选线。不要把 `--target-protocol` 传给当前 R4 镜像。

实际运行示例（服务器 icarus0）：

```bash
ROOT='/staff/tangjiarui/OPC/LTH团队-SpaceQA智能体'
SIX="$ROOT/13_六维评分修复_20261110"
RUN="$ROOT/repair_20261110/runtime_candidate"
OUT="/staff/tangjiarui/OPC/sixdim_results/AIA_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT")"
PYTHONDONTWRITEBYTECODE=1 python3 "$SIX/rescore_six_dimensions.py"   --input "$RUN/inputs/AIA_AR_sample_p86y2cz08y"   --source-run "$RUN/acceptance/output/clean_final_AIA_AR_sample"   --protocol "$SIX/approved_protocols/AIA_AR_sample/target_protocol.json"   --output "$OUT"
```

以上协议的六维等权已获用户批准；其他目标与参考证据仍不完整，因此输出局部观测和缺证原因，完整六维分仍为空。批准权重不能替代独立标注真值、目标覆盖/分布、来源或许可文件。协议 records 指向旁边的 JSON 证据表；C45/C50 的 path 相对输入包，审核证据 evidencePath 相对协议目录。

输出为 `six_dimension_report.json`、`formal_checks/` 和 `run.log.jsonl`。报告含新评分 runId、原模型 sourceRunId 与实际源文件哈希，明确标识为既有推理的 CPU 复算。`approved_weight_replays/` 是批准权重后的三类真实小样实测；`actual_replays/` 保留批准前记录，`tests.log` 中的完整六维 75 分仅为合成测试夹具。

尚需补充的字段和证据见 `六维计算口径.md`、`待补协议清单.csv`。不要填写虚构批准人，不把模型候选或自动标注当成独立真值。PPT 和根目录历史 12 包评分结果未改变。

## 最终提交所用验收入口

交付包中的十二包正式验收使用 review_archive.py，仅复核随包附带的历史推理、账本与检查证据。请从 12包正式验收/README.md 开始。上面的独立 CPU 小样复算命令所列 RUN/inputs 是服务器保留的原始验证输入。完整计分需要正式协议和逐维证据。
