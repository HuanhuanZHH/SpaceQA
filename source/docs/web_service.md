# Browser console / 浏览器控制台

SpaceQA 0.4.0 adds a browser frontend and a Python backend. The backend runs beside
the datasets on the server; the browser connects through an SSH local forward.
No dataset archive or model weights need to be copied to the laptop.

## 已部署实例

- 服务：远端 Linux 主机，`127.0.0.1:8765`。
- 本机浏览器：<http://127.0.0.1:18765>，需要保持SSH转发连接。
- 代码：远端仓库中的 `spaceqa-agent`。
- 新任务与结果：由 `--state-dir` 指定的外部目录。
- 数据源：由 `--data-root` 指定的数据包目录。
- 历史结果：由 `--run-root` 指定的只读运行目录。

本机PowerShell执行：

```powershell
cd /path/to/spaceqa-agent
.\deploy\web_service\connect.ps1
```

打开 <http://127.0.0.1:18765>。保持终端开启；`Ctrl+C`关闭转发，服务器仍运行。
指定本机端口可使用 `-LocalPort 18767`。脚本在本机存在既有的<local-bind-address>地址时
沿用该SSH来源地址；也可显式传入 `-BindAddress`。从其他电脑连接时应满足已有SSH
认证条件，脚本不保存密码或API Key。

案例核查 Demo 兼容原来的 `evaluation.html` 地址，但现在直接读取当前服务的审计结果。
当前服务器服务监听回环地址 `8765`，本机建立 `18867 -> 8765` 转发。推荐直接执行：

```powershell
cd D:\OPC\spaceqa-agent
powershell -ExecutionPolicy Bypass -File .\deploy\web_service\connect_evaluation.ps1
```

连接脚本默认使用 `tangjiarui@222.195.72.220`，不要求本机配置 `icarus0`。打包客户端会先执行服务器上的
`deploy/web_service/start_server.sh start`（已有服务时不会重复启动），再建立 SSH 转发。

然后打开 <http://127.0.0.1:18867/evaluation.html>，或 <http://127.0.0.1:18867/demo.html>，并保持这个 PowerShell 窗口开启。
如果使用了没有 `icarus0` SSH 别名的电脑，可指定服务器参数：

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\web_service\connect_evaluation.ps1 `
  -Server tangjiarui@222.195.72.220
```

如果服务器代码路径不同，同时传入远端仓库路径：

```powershell
.\deploy\web_service\launch_evaluation_console.cmd 用户名@服务器IP /服务器上的spaceqa-agent路径
```

`evaluation.html` 是只读案例核查入口，使用 `/api/runs`、`/api/runs/{run}/findings` 和证据接口，
与数据控制台共用同一份运行结果。`127.0.0.1` 只代表当前浏览器所在电脑；其他人的电脑也必须各自建立自己的 SSH 转发，
不能直接把你的 `127.0.0.1` 地址发给对方。

等价手动命令：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:18765:127.0.0.1:8765 user@server
```

服务器管理：

```bash
bash /path/to/spaceqa-agent/deploy/web_service/start_server.sh start
bash /path/to/spaceqa-agent/deploy/web_service/start_server.sh status
bash /path/to/spaceqa-agent/deploy/web_service/start_server.sh stop
```

服务日志位于`<state-dir>/service.log`。该脚本启动普通用户进程，
不是系统开机自启服务；服务器重启后再次运行`start`即可。使用`stop`时活动任务可能被
中断，重启后会明确标记`interrupted`，不会把它伪装成成功。

## 界面操作

分析工作台和对话式工作台的右上角均有 `中文 / EN` 切换。选择会保存在浏览器中，页面导航、表格、任务状态、筛选器和弹窗都会同步切换；源文件名、原始 JSON、证据内容和模型回答保留原文。

1. **Overview**：查看服务器连接、数据包数量、已有结果、队列和CPU／内存／存储快照。
2. **Dataset explorer**：选择数据包，搜索样本，打开原图与原始标注叠加；ULF显示抽样曲线
   和原始JSON。可关闭叠加层。缺图引用保留为缺失状态。
3. **Evaluation results**：选择默认或高召回复核的历史运行，查看六维评分卡、排序、问题
   筛选、证据、定位范围与SHA核验。点击证据中的样本入口可回到原始样本。
4. **Run an evaluation**：选择服务器数据包、执行模式和种子，提交后台任务。查看实际
   扫描检查数、当前阶段及日志；可取消。只有扫描和独立审计成功，任务才完成。

可用模式：

| 模式 | 实际执行 | 评分关系 |
| --- | --- | --- |
| `rules` | 当前输入规则扫描 → 输出 → 独立审计 | 当前项目规范下的规则分数 |
| `rules_priors` | 上述流程 → 独立当前输入先验检查 | 先验单独保存；不静默修改既有分数 |

界面由`/api/health`真实能力字段决定可提交的模式。**当前没有启用任意新数据包的完整
视觉专家在线推理**。历史默认／review整合可浏览，但其中已保存的视觉概率不等于本次
新GPU推理。界面不会用历史候选替代新任务，也不会显示未经测量的官方F1。

## 记录、复算与统计范围

新任务的输出位于`state-dir/runs/<job-id>`，任务状态、进程日志及独立先验位于
`state-dir/jobs/`。任务串行执行，最多8个活动或排队任务；排队时间不计入执行上限。
运行超过3小时会失败并保留原因，不能据此声称已完成正式3小时验收。

0.4.0的新扫描会实时写入检查开始／结束事件、检查编号、规则／系统版本、随机种子、
实际耗时与进程CPU时间，并在`08_运行日志/runtime_metrics.json`保存运行摘要。
RSS为扫描进程生命周期高水位，不是每检查独占内存或多进程总和；GPU未测时保持null。
它不能代替完整专家容器、指定A10及全链路资源验收。

```bash
spaceqa audit /absolute/path/to/state/runs/job-id
```

新日志采用`loggingVersion=2`，审计要求真实开始／结束覆盖及字段一致。旧运行保留
兼容审计与限制提示，不补写或伪造历史开始日志。证据SHA通过表示产物完整、引用一致，
不证明每项科学判断正确。

## 安全边界与预览语义

- 服务仅绑定loopback，通过SSH访问；不提供公网账号系统。
- 客户端只能提交登记的数据包ID，不能提交shell命令、任意文件路径或模型权重。
- 路径经规范化并限制在配置根目录内；原始数据与历史运行不作为输出目录。
- 图片叠加来自输入标签，不是自动修复或模型新预测的缺陷区域。样本级候选没有区域时
  明确保留样本级定位。
- AIA／HMI、EPB、DMSP按原标签像素坐标展示。CME角宽参数保留JSON，不凭未验证的
  朝向假设画成确证区域。ULF曲线为有上限的抽样预览，不能替代全CSV数值检查。
- 主机资源是机器快照，可能包含其他用户的任务，不能当作本任务资源消耗。
- 当前控制台不调用外部大模型；已有`spaceqa explain`仍为单独的显式可选流程。

## Generic setup / 自定义数据根

```bash
python -m pip install -e .
spaceqa serve --host 127.0.0.1 --port 8765 \
  --data-root /external/publish \
  --data-root /external/extracted-roadshow \
  --run-root /external/historical-integrated-run \
  --state-dir /external/spaceqa-web-state
```

`data-root`与`state-dir`不能互相包含。支持分目录`data/label`和平铺PNG／CSV＋JSON。
新数据放到已配置数据根后刷新即可发现；陌生科学类型或不兼容schema仍需适配和验证。
前端无外部CDN或构建依赖，HTML／CSS／JS随wheel打包，后端仅用Python标准库。

## API examples

```text
GET  /api/health
GET  /api/datasets
GET  /api/datasets/{dataset-id}/samples?offset=0&limit=24&q=
GET  /api/datasets/{dataset-id}/samples/{sample-id}
GET  /api/datasets/{dataset-id}/samples/{sample-id}/image
GET  /api/runs
GET  /api/runs/{run-id}
GET  /api/runs/{run-id}/findings?offset=0&limit=40&status=待判定
GET  /api/runs/{run-id}/evidence/{finding-id}
GET  /api/jobs
POST /api/jobs
POST /api/jobs/{job-id}/cancel
```

POST example, using an ID returned by `/api/datasets`:

```json
{"datasetId":"ds-registered-id","mode":"rules_priors","seed":20261021}
```

Browser POST requests must come from the same origin. Use the browser service URL consistently;
switching `localhost` and `127.0.0.1` between origins is deliberately rejected.

## Remaining work / 尚待完成

- 为任意新包建立与训练契约一致的完整专家预处理、权重校验和冻结分流入口，并通过
  六类重放验证，再对外启用该模式。
- 按类别与位置的独立真值测量官方关键缺陷召回；当前代理F1不能代填资格门槛。
- 改进漏标／虚标定位及可靠区域真值，验证其召回与误报取舍。
- 对完整专家镜像进行≤20GB和目标A10单数据集≤3小时验收，纳入模型调用及资源消耗。
- 获得官方跨类型质量参考后，验证全局排序与不并列条件。

These limitations are capability boundaries, not hidden benchmark successes. This release improves
the operator workflow and auditability without claiming a new scientific accuracy measurement.
