# 离线部署

## 规则容器

仓库根目录的 `Dockerfile` 是轻量规则运行时，不含数据、研究权重和 GPU 依赖。构建和运行：

```bash
docker build -t spaceqa-agent:0.6.0 .
docker run --rm \
  -v /absolute/path/to/publish:/input:ro \
  -v /absolute/path/to/runs:/output \
  spaceqa-agent:0.6.0 scan --input /input --output /output/run-001
```

## 完整研究发布

完整 21 专家发布包由比赛环境外部提供，不在 Git 仓库中。离线验收顺序如下：

```bash
docker load --input spaceqa-agent-0.6.0-all21-nf4-20261107.docker.tar
docker image inspect spaceqa-agent:0.6.0-all21-nf4-20261107
# 在无网络、只读输入挂载下执行一次 manifest/help/scan/audit
```

运行时必须挂载数据目录和新的输出目录；容器不得依赖在线下载或 API Key。提交前保存 `docker image inspect`、`docker save` 字节数和 SHA-256。压缩后的完整包不应重新 gzip 后冒充 Docker save 大小。

## 浏览器服务

远端启动：

```bash
spaceqa serve --host 127.0.0.1 --port 8765 \
  --data-root /absolute/path/to/publish \
  --run-root /absolute/path/to/historical-runs \
  --state-dir /absolute/path/to/web-state
```

本机转发：

```bash
ssh -N -T -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18765:127.0.0.1:8765 user@server
```

浏览器打开 `http://127.0.0.1:18765`。停止服务时先结束远端 `spaceqa serve` 进程，再关闭 SSH 转发；运行中的任务会在日志中标记为中断，不会伪装成成功。

## 生产前检查

- 输入目录为只读，输出目录与输入不重叠。
- 服务仅监听 loopback，外部访问通过受控 SSH 转发。
- 数据、权重、缓存和凭据均通过挂载或环境变量提供。
- 运行日志含版本、检查 ID、规则版本、随机种子、实际耗时和错误原因。
