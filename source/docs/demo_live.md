# 真实整包重评与报告

浏览器入口：`http://127.0.0.1:18780/chat.html?new=1`。

已登记并有专家缓存的EPB/AIA路演包支持“现场重评整包”。当前演示使用EPB_seg_sample_z443g9dgh7：787条记录、721张可读图片；两次CPU实测分别152.867和150.406秒。上游21专家预测从校验后的缓存读取，当前规则、八类融合与全部可读图像定位重新执行。图像定位不复用旧区域缓存。

接口：

- `POST /api/chat/sessions/{id}/live-run`，JSON正文`{}`，启动新任务。已有任务运行时返回409。
- `GET /api/chat/sessions/{id}`的`liveRun`提供任务编号、阶段、进度、已前向图片、耗时与完成后的审计摘要。
- `GET /api/chat/sessions/{id}/report.html`，完整可打印HTML报告。
- `GET /api/chat/sessions/{id}/report`，分析摘要JSON。
- `GET /api/chat/sessions/{id}/evidence.jsonl`，完整问题记录与定位台账。

任务成功后从新目录读取`samples.jsonl`、`predictions.jsonl`和`coverage.json`，案例库和总览切换为本次结果。失败时保留上一份完整报告及本次错误信息。页面重载不影响后台任务；升级服务应等待任务完成。当前现场入口依赖已登记目录和专家缓存，不代表任意新包都具有同样的全流程推理接入。

大模型请求使用固定事实快照；不得在网络请求期间持有会话锁，避免阻塞任务进度更新。相关回归测试位于`tests/test_chat_insights.py::test_language_model_does_not_block_progress_reads`。

演示材料与字幕视频：`D:/OPC/outputs/demo_polish_20261106/`。执行流程、解释用语与统计口径详见该目录说明文件。
