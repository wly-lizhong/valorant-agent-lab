# 一键启动云端试玩

适用：朋友在 Ubuntu 22.04/24.04 x86_64 + NVIDIA 云服务器运行项目，再从他自己的电脑通过 SSH 隧道试玩。两台机器无需处于同一局域网。服务器应能联网安装依赖；普通账号需要 sudo。

## 1. 服务器只运行一条启动命令

在已经克隆的完整仓库根目录执行：

```bash
bash scripts/start_cloud.sh
```

如果已有 SSH 配置别名，可以让脚本直接生成可复制的连接命令：

```bash
bash scripts/start_cloud.sh --ssh-target my-server
```

`my-server` 替换成朋友电脑中实际配置的别名。也支持 `--ssh-target user@host --ssh-port 23456`，端口填平时 SSH 登录使用的实际端口；不根据容器里的 22 猜测平台外部端口。

首次安装、以后重启都使用同一条命令。脚本依次完成：

1. 检查 Ubuntu、架构、NVIDIA GPU 和网页端口。
2. 安装项目虚拟环境、系统图形工具及网页依赖；依赖指纹和导入检查通过时复用已有安装。
3. 解包并校验地图；已有地图只校验，校验失败时保留现场并退出。
4. 自动生成本机路径配置、构建引擎并导入资源。
5. 复用已验证的 NVIDIA 显示，或新建私有 GPU 显示。普通账号只为显示服务调用 sudo，网页保持原账号。
6. 检查实际 GPU 渲染，完成真实地图移动、开火、重置自检。
7. 启动网页，确认 HTTP 页面就绪后显示连接步骤和登录口令。

安装和显示阶段可能需要输入 sudo 密码。脚本不安装/替换 NVIDIA 驱动，不修改系统显示配置；平台须提供图形驱动与设备权限。已有显示未通过检查时尝试新建独立显示，不接管或停止原显示。复用显示时 GPU 以该显示的实际设备为准。

本入口用于第一轮试玩和采集准备，不会开始长时间模型训练。模型学习步骤见[第一阶段手册](stage1-guide.md)。真实云端手感和性能仍需要人工验收。

## 2. 朋友电脑运行连接命令

服务器出现“网页已就绪”后，在朋友自己的 PowerShell/终端粘贴脚本打印的 SSH 命令。未填写 SSH 参数时，输出会带有 `SSH_PORT`、`USER@HOST` 占位符，换成平时的登录信息即可。例如：

```text
ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 -p SSH_PORT USER@HOST
```

保持连接窗口运行，浏览器打开 [http://127.0.0.1:8765](http://127.0.0.1:8765)，输入服务器终端显示的网页口令。若用了 `--port`，浏览器和隧道端口都使用脚本打印的值。

网页继续只绑定服务器 loopback，无需开放公网网页端口；SSH 服务需要允许端口转发。使用 VPN 或跳板机时沿用朋友现有的 SSH 配置别名。[OpenSSH 转发说明](https://man.openbsd.org/ssh.1)

连接说明另存至 `runs/quickstart/connection.txt`，不含口令。口令保存在 `runs/web/access-token`；仅直接连接终端时显示口令，输出被重定向到文件时不打印口令。

## 3. 首次验收与停止

先完成“路线行走”，再试射击和换弹；成功后确认保存一段示范，按[录制回放步骤](stage1-guide.md#4-数据内容和验收)验证。当前为 PNG 步进采集，网络延迟可能使操作减速，页面能打开不等于原速性能已经达标。

服务器启动窗口保持运行。结束时按 `Ctrl+C` 并等待退出；脚本关闭本次网页和自己新建的显示，已有显示、地图、示范、题库与模型保留。SSH 断开通常也会终止前台服务，长时间保留服务可在 tmux 中运行同一条命令。

常用选项：

```bash
bash scripts/start_cloud.sh --check-only
bash scripts/start_cloud.sh --reinstall
bash scripts/start_cloud.sh --gpu 1 --port 8767
bash scripts/start_cloud.sh --help
```

`--check-only` 完成安装与 GPU/地图自检后退出；`--reinstall` 重新检查并安装依赖，不删除数据。端口占用时停止并提示，不会关闭原服务。不能靠重新安装项目依赖补齐平台缺失的 GPU 图形权限。

失败时保留第一处报错，并查看：

- `runs/quickstart/setup.log`：安装、地图校验、构建和自检输出。
- `runs/quickstart/cloud-check.json`、`runs/quickstart/smoke/`：图形与地图检查结果。
- `runs/quickstart/web.log`：网页进程输出。
- `runs/quickstart/display-worker.log`：普通账号的 sudo 显示子进程错误。
- `runs/gpu-display-*/`：本次私有 Xorg 日志。文件是否存在取决于执行到了哪一步。

本脚本不会拉取、提交或推送 Git，也不会自动改动服务器防火墙。平台目录与持久化要求见[矩池云指南](matpool.md)。
