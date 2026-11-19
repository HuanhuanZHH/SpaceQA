# SpaceQA 评价控制台连接包

这个小包从使用者电脑通过服务器 IP 建立 SSH 转发并打开浏览器页面。数据集、模型权重、推理环境和 API Key 都留在服务器上，不会复制到使用者电脑。

## 使用前提

1. Windows 已安装 OpenSSH Client。PowerShell 中执行 `ssh -V` 能看到版本号即可。
2. 使用者拥有服务器 `tangjiarui@222.195.72.220` 的 SSH 登录权限。
3. 服务器代码默认位于 `/staff/tangjiarui/OPC/spaceqa-agent`。启动脚本会检查并启动已安装的 SpaceQA 服务，不会复制数据、模型或密钥。

## 启动

双击 `launch_evaluation_console.cmd`，默认直接使用 `tangjiarui@222.195.72.220`，不依赖 `icarus0` SSH 别名。

页面地址：<http://127.0.0.1:18867/evaluation.html>（当前案例核查 Demo；`/demo.html` 同样可用）

如果服务器用户名或路径不同，在 PowerShell 中进入本目录并执行：

```powershell
.\launch_evaluation_console.cmd 用户名@服务器IP /服务器上的spaceqa-agent路径
```

连接窗口必须保持打开。关闭窗口只会关闭本机转发，不会删除服务器数据。

## 常见问题

- “连接被拒绝”：通常是没有运行启动脚本，或者启动窗口已被关闭。
- “SSH 认证失败”：需要先配置服务器账号、密码或 SSH Key，并确认该账号可以通过 IP 登录服务器。
- “本机端口被占用”：关闭占用 18867 的旧连接，或联系项目负责人调整本地端口。
- 如果服务器仍只部署旧版 8775 服务，可在脚本参数中追加 `-RemotePort 8775`；旧服务没有当前新版案例目录时，建议改用 8765。
- 对方只能看到自己的 `127.0.0.1`，不能直接把你电脑上的 `127.0.0.1:18867` 发给对方；每台电脑都要各自建立转发。
