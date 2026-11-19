# SpaceQA 部署与使用说明（交付源文件）

## 规则容器

```bash
docker load --input spaceqa-agent-0.6.0.docker.tar
docker run --rm -v /data/publish:/input:ro -v /data/runs:/output spaceqa-agent:0.6.0 scan --input /input --output /output/run-001
docker run --rm -v /data/runs/run-001:/run:ro spaceqa-agent:0.6.0 audit /run
```

运行环境无网络，输入只读，输出必须是新的目录。单次运行必须产生评分卡、排序、问题清单、证据索引和日志。

## 浏览器工作台

```bash
spaceqa serve --host 127.0.0.1 --port 8765 --data-root /data/publish --run-root /data/historical-runs --state-dir /data/web-state
ssh -N -T -L 127.0.0.1:18765:127.0.0.1:8765 user@server
```

浏览器打开 `http://127.0.0.1:18765`。关闭时先停止远端服务，再关闭 SSH 转发。工作台只读展示源文件，并把问题候选与已确认问题分开展示。

## 典型输出

`04_评分卡/` 是逐包维度分数，`05_排序表/` 是全局排序，`06_问题清单/` 是逐条问题，`07_证据索引/` 是回查入口，`08_运行日志/` 是检查编号、规则版本、种子和资源记录。若证据不足，维度为 null 或范围值，绝不填 0 或 100。

