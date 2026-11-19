# 12个路演包结果交接材料（2026-11-09）

这份材料用于组内查看当前12个路演包的评价结果、排序、典型案例、八类缺陷定位展示和前端验收证据。包内不含原始图像、JSON标签或服务器数据，只包含结果、摘要、截图、校验记录和复现索引。

## 12个路演包

- AIA：`AIA_AR_sample_3mogtggl50`、`AIA_AR_sample_7qn2pdeube`、`AIA_AR_sample_p86y2cz08y`、`AIA_AR_sample_z4h764sf2d`
- EPB：`EPB_seg_sample_fbbqy66u2d`、`EPB_seg_sample_mc4t7fou91`、`EPB_seg_sample_qmn05l7r91`、`EPB_seg_sample_z443g9dgh7`
- ULF：`ULF_sample_09b7y3z4hn`、`ULF_sample_0yu0mc7d5a`、`ULF_sample_ta382376ar`、`ULF_sample_xfnogkdey9`

## 建议先看

1. `roadshow_report.md`：整体范围、逐包统计、八类缺陷和限制。
2. `12_package_ranking.csv`：12包排序及发现比例。当前排序按“确认或候选发现样本比例”升序，比例越小排在前面；它不是官方真实污染率。
3. `eight_class_cases.md`：典型漏标、虚标、边界、覆盖、尺寸、缺图和时段案例。
4. `eight_class_localization_summary.csv`：确认问题、模型候选、位置指认和仅样本级结果的区分。
5. `01_fresh_overview.png through 13_axes_and_coverage.png`：可直接用于前端和汇报查看的截图证据。
6. `frontend_acceptance.md`：前端功能、后端调用和已知边界。

## 本轮结果说明

- 12包分类结果已完成，记录数 24,782；分类回放使用冻结策略，结果可由 JSON、CSV 和报告互相核对。
- 本轮对现存图片重新做了 RGB 几何前向；21个基础专家使用经过校验的预测缓存，没有重新进行21个专家的冷启动前向。
- 规则确认问题和模型候选分开保存；候选不能直接当作真实问题，也不能在没有独立真值时计算官方 P/R/F1。
- ULF 的时间区间是输入事件区间核验，当前 CSV 缺少独立 UTC 原点，不能宣称完成真实起止时刻准确率。

## 已知缺陷和需要向查看者说明的地方

1. **路演包没有独立缺陷真值。** 当前确认数来自文件、字段、尺寸、配对等直接证据；模型候选来自冻结模型和P0/结构差异，不能用它们计算官方隐藏集九类缺陷 P/R/F1。
2. **区域定位覆盖不完整。** 漏标、虚标、覆盖等问题有一部分已经指到图像区域，但仍有样本只有样本级结论；详细计数见 `eight_class_localization_summary.csv`。
3. **EPB区域表达仍受标注格式限制。** 当前结果可以展示矩形/候选区域，但不能把矩形自动宣称为真实多边形；需要独立多边形参考才能做正式区域命中评估。
4. **ULF时间真值不完整。** 已能检查时间轴、采样率和输入事件区间，但缺少独立时间基准时，起止时刻只能作为疑点或结构核验。
5. **排序是技术展示排序。** “发现比例越小越好”便于展示相对差异，但不能替代官方质量等级或隐藏集评分。
6. **当前材料不包含原数据。** 要查看原图、原始JSON和服务器绝对路径，需要访问icarus0上的结果目录。

## 服务器查看位置

服务器结果索引：

```text
/staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105/report_index.json
```

登录后可查看：

```bash
ssh tangjiarui@icarus0
cd /staff/tangjiarui/OPC/eight_class_20261104/refresh_20261105
less report_index.json
```

前端查看应使用现有对话式工作台；材料中的 `report_index.json` 保存了各包结果、样本路径和证据索引。不要把原始数据目录复制到代码仓库或公开群聊。

## 校验

`SHA256SUMS.txt` 是本交接目录内文件的校验清单。压缩包解压后可在当前目录执行：

```bash
sha256sum -c SHA256SUMS.txt
```

Windows PowerShell 可使用：

```powershell
Get-FileHash * -Algorithm SHA256
```

