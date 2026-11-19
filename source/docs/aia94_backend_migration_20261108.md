# AIA94 后端迁移说明（2026-11-08）

## 结论

AIA94 包不是需要重新训练的分类模型，也没有需要替换的权重。它是一套确定性科学图像处理流程：异常曝光上下文、WATROO 去噪、双阈值亮区候选、连通区域、跨帧几何匹配，以及可选的时间残差候选。因此本次迁移只改后端策略层，保留原对话式前端、接口路径、评分和历史运行读取。

原始包的分析报告和验证材料保存在：

- `D:/OPC/outputs/aia94_inspect_20261108/`
- `D:/OPC/outputs/aia94_inspect_20261108/AIA94分析与跨数据集迁移建议.md`

## AIA94 实际做了什么

1. 对 `EXPTIME ≤ 0.02s` 的异常曝光帧，用前后有效帧逐像素均值生成工作缓存；不修改原始数据。
2. 使用 WATROO 三层 B3 à trous 去噪，尺度权重为 3、2、1。
3. 对去噪强度做平滑、开平方和 Otsu 阈值估计，`TH=1.5O`、`TL=0.8TH`。
4. 对低阈值区域进行闭运算、填孔、8 连通域和最小面积过滤，并保留包含高阈值种子的区域。
5. 对相邻帧做太阳自转投影，以重叠系数 `≥0.25`、IoU `≥0.08` 和匈牙利匹配继承亮区编号。
6. 可选分支计算时间归一化残差并生成三维时空候选，但该实现不是官方 AQuA2 完整复现。

包内验证只覆盖缓存空间分割：2024-01-11 的三帧亮区数为 13、13、12，分区一致且前景像素差为 0。它没有证明全年事件召回率，也没有提供训练权重。

## 当前后端迁移实现

新增模块：`src/spaceqa_agent/backend_strategy.py`。

正式 `scan` 现在在基础规则和同源参照检查之后调用 `engine.apply_backend_checks`。增强模式会把满足独立证据门槛的后端信号转成标准 finding，进入同一套证据索引、六维分数和排序；没有科学源数据或证据不足的情况仍只记录观测/回退，不会强行扣分。`check-priors` 仍保留独立先验接口，便于对照。

| 数据类型 | 后端策略 | 迁移内容 | 禁止直接复用 |
|---|---|---|---|
| AIA | `aia94_bright_region` | 曝光上下文、元数据检查、亮区候选、区域和帧证据 | 把 R 编号当作物理耀斑；把 PNG 亮度当 DN/s |
| HMI | `solar_region_adapted` | 元数据感知预处理、区域候选和时间上下文 | 直接套用 AIA94 阈值，跳过单位/几何校准 |
| EPB | `segmentation_geometry` | 框/多边形/掩膜统计、区域匹配和覆盖率 | 太阳自转投影；把矩形输入伪装成多边形 |
| DMSP | `segmentation_geometry` | 几何归一化、区域匹配和面积统计 | AIA94 强度阈值和太阳几何 |
| CME | `event_parameter_context` | 事件记录、时间上下文和参数来源 | 像素亮区分割和太阳自转跟踪 |
| ULF | `timeseries_context` | 完整 CSV 轴扫描、采样率、时间段和单位来源 | 图像分割和太阳亮区阈值 |

当 AIA/HMI 只有普通 PNG/JPG，策略会明确返回 `display_image_without_scientific_array_or_metadata`，不会猜测科学强度。存在 `.zattrs`、`.metadata.json` 或 `.meta.json` 侧车时，才记录曝光和几何元数据上下文。EPB、DMSP 的候选使用原生图像坐标；CME 使用 JSON Pointer/事件记录；ULF 使用 CSV 时间轴。

## 服务接口与回退

原有调用不变：

```bash
spaceqa check-priors --input <package> --family <family> \
  --output <new-output.jsonl> --backend-profile enhanced
```

正式评价入口：

```bash
spaceqa scan --input <package-or-publish> --output <new-run> \
  --backend-profile enhanced
```

`enhanced` 是正式扫描的默认配置。它目前只将两类具有直接证据的信号纳入正式判定：独立科学侧车中的 AIA/HMI 异常曝光，以及完整 ULF CSV 时间轴中的无效时间行。亮区候选、缺失科学元数据和缺少显式几何记录不会自动判为缺陷。需要回到迁移前正式判定时，给 `scan` 或网页任务传入 `backendProfile=legacy`。

服务器侧受控验收已验证这条正式路径：在 icarus0 上用真实 AIA PNG 加独立 `.zattrs` 侧车构造单样本，`legacy` 输出 1 个确认问题、技术分 87.5；`enhanced` 输出 2 个确认问题、技术分 75.0，新增的 `C06/L2-M` 同时出现在 `findings.jsonl`、证据索引和评分卡中。该侧车是链路验收输入，不是路演集缺陷统计。

网页任务可选传入：

```json
{
  "datasetId": "...",
  "mode": "rules_priors",
  "backendProfile": "enhanced"
}
```

需要回到提交前对话模式时，使用：

```bash
spaceqa check-priors --input <package> --family <family> \
  --output <new-output.jsonl> --backend-profile legacy
```

网页任务将 `backendProfile` 改为 `legacy` 即可。未传该字段时默认使用 `enhanced`；增强策略只有在满足独立证据门槛时才增加正式 finding，其余只保留观测。服务健康接口会报告策略版本、默认 profile 和 rollback profile。

## 安全备份

对话式后端已在迁移前保存：

- `D:/OPC/outputs/backend_backup_20261108/spaceqa-agent-dialog-backup-75b6bff.zip`
- 基线提交：`75b6bff6bc665b2a31f5b4d8db12359cecc8e225`
- SHA-256：`8e698bdf511550da487b95c8f7446f199113ea9cf5105e41c1b7f5c10f0c4cbd`

备份不含数据集、权重、缓存和 API Key。迁移没有改动 `chat_service.py`、聊天页面、案例核查入口和历史报告格式；出现新策略异常时可以直接切回 `legacy`，原提交版本仍可交付。

## 当前验证

- AIA94 策略与所有六类数据类型均有显式适配器；
- 增强策略和 legacy 策略在未触发高证据门槛的抽测样本上保持原有 findings、confirmed bits 一致；触发门槛的信号会作为标准正式 finding 进入评分；
- 普通图像缺少科学侧车时会明确降级，不产生伪科学数值；
- Web 服务继续支持原有 `rules`、`rules_priors` 模式和 `evaluation.html`；
- 新策略输出不参与污染指数、六维分数和全局排序。

因此，本次是后端策略增强和可回退的正式融合，不是替换模型权重，也不是把 AIA94 的太阳物理假设强行套用到所有数据类型。
