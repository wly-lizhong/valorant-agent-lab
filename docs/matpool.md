# 矩池云首次启动与验收

更新：2026-11-26。训练平台已确定为矩池云，GPU 为 NVIDIA，可使用 sudo；具体型号、镜像和连接信息尚未提供。本页是待执行的实例操作指南，**尚未在矩池云实机验证 GPU 渲染、采集速度或训练性能**。

本地低配电脑只运行编辑器、SSH 和浏览器；游戏画面、示范数据与训练均在云端。当前代码使用 X11/OpenGL 渲染和 PNG 步进示范台，不要求完整桌面或 NVENC。

首次试玩可在确认持久化目录并克隆仓库后执行 `bash scripts/start_cloud.sh`，自动完成本页的安装、自检和网页启动，详见[一键启动说明](one-click.md)。下面的分步命令用于手动操作和排错。

## 1. 先核实实例类型和图形能力

使用 Ubuntu 22.04/24.04 x86_64、可用 SSH、可安装系统依赖的环境。平台预装的 PyTorch/CUDA 不一定等于项目虚拟环境里的版本，后续仍按仓库依赖安装。

矩池云的概念文档把普通镜像环境描述为 Docker 容器；其保存环境文档也提到虚拟机等其他主机类型，所以应以实际所选产品为准。容器内的 root/sudo 不能替代宿主机的设备和驱动配置。[平台概念](https://www.matpool.com/supports/doc-overview-concepts)、[环境类型说明](https://www.matpool.com/supports/snapshot)

本项目需要同时满足：

| 能力 | 验收方式 |
| --- | --- |
| CUDA 计算 | `nvidia-smi` 可见 GPU，安装训练依赖后 `doctor` 的 `cuda_available` 为 true |
| NVIDIA 图形驱动 | GPU 显示启动后，`cloud check --require-gpu` 通过 |
| 实际游戏渲染 | `run_cloud.sh smoke` 通过引擎设备检查，并生成有效地图画面 |
| 远程人工操作 | 浏览器能录制、成功提交并重放少量示范，检查网络减速和输入对齐 |

NVIDIA 容器的 `compute` / `utility` 与 `graphics` / `display` 是不同能力。仅有 `nvidia-smi` 或 CUDA 通过，不足以验收图形；在已经运行的容器里设置 `NVIDIA_DRIVER_CAPABILITIES`，也不能补挂缺失的宿主机设备和驱动库。[NVIDIA 官方说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)

矩池云部分镜像提供 VNC，但官方教程没有承诺该桌面的 OpenGL 一定使用 NVIDIA。现有 VNC 桌面也应检查实际 renderer；本项目浏览器示范台本身不依赖 VNC。[平台 VNC 教程](https://www.matpool.com/supports/doc-vnc-connect-matpool)

如需向平台确认，可说明：需要在无物理显示器的 Linux 实例中运行 Godot 4 的 X11/OpenGL 3D 程序，能否提供 NVIDIA 图形驱动、匹配的 Xorg NVIDIA 模块及启动所需设备权限，或已配置好的 NVIDIA X11 显示；CUDA 可用之外，还需 `glxinfo -B` 和实际游戏渲染验收。此处是本项目的需求清单，不是平台已承诺的能力。

## 2. 把代码和数据放到持久化网盘

矩池云默认把网盘挂载到 `/mnt`，可自定义挂载位置；不同区域的网盘隔离。先在控制台确认实际挂载路径、读写属性和剩余容量，再执行以下命令。`findmnt` 和 `df` 只辅助核对，不能单凭目录名字判断数据会持久保存。[入门教程](https://www.matpool.com/supports/doc-quick-start)、[区域与挂载概念](https://www.matpool.com/supports/doc-overview-concepts)

```bash
cat /etc/os-release
uname -m
nvidia-smi
findmnt -T /mnt
df -h /mnt
```

确认 `/mnt` 是本实例的可写持久化网盘后，首次克隆：

```bash
cd /mnt
git clone https://github.com/wly-lizhong/valorant-agent-lab.git
cd valorant-agent-lab
```

如果已经有该目录，先检查已有内容；不要重复克隆或覆盖。整个项目放在网盘上便于首版使用：地图、`runs/demonstrations/`、纠错题库、模型和日志都留在同一持久化目录。后续若实测网盘小文件 I/O 成为瓶颈，再设计本地临时盘加数据同步；当前不增加这层复杂度。

随后从 [README 的 Ubuntu 准备步骤](../README.md#ubuntu-服务器准备)中 `bash scripts/setup_ubuntu.sh` 开始执行，完成地图安装、引擎构建和 `setup_cloud.sh`。训练依赖按[服务器说明](server.md#ubuntu)安装，PyTorch CUDA 构建须与实际驱动兼容。

在服务器重新生成本机配置，不复制 Windows 的 `local.json` 或虚拟环境。租用页面的“入口命令”先保留所选镜像默认值：平台文档说明自定义入口命令结束可能触发实例自动释放，因此安装脚本和自检应在 SSH 终端里运行。[入口命令说明](https://www.matpool.com/supports/doc-overview-concepts)

## 3. 启动并验收 GPU 显示

安装完成后，在仓库目录采集初始诊断：

```bash
source .venv/bin/activate
python -m valorant_lab.cloud check --output runs/cloud-check.json
```

尚未启动图形显示时，`graphics.ok=false` 可以是正常结果，不能据此判定平台不兼容。下一步在**云端终端 A**启动独立 GPU 显示，并保持该终端运行。平台默认 root 账号使用：

```bash
cd /mnt/valorant-agent-lab
.venv/bin/python -m valorant_lab.cloud display --display 99
```

普通账号在第二条命令前加 `sudo`。脚本打印实际 `DISPLAY` 和 `XAUTHORITY`；显示号占用时改选空闲编号，不删除其他进程的锁。已有经检查的 NVIDIA 显示也可直接复用，详见[通用显示指南](cloud-workflow.md#3-启动-gpu-图形会话)。

**云端终端 B**进入相同目录，填入终端 A 打印的实际值：

```bash
cd /mnt/valorant-agent-lab
source .venv/bin/activate
export DISPLAY=:99
export XAUTHORITY=/实际路径/runs/gpu-display-实际编号/Xauthority
python -m valorant_lab.cloud check --require-gpu
bash scripts/run_cloud.sh smoke
```

只有显示检查和 smoke 都通过后，再进行浏览器试采与正式训练。`llvmpipe` / `softpipe` 是软件渲染，不能视为 NVIDIA 图形验收通过。

若 Xorg 启动失败，保留 `runs/cloud-check.json` 及本次 `runs/gpu-display-*/Xorg.log`、`launcher.log`，根据具体错误检查驱动模块、设备或权限。项目脚本不安装 NVIDIA 宿主机驱动、不修改系统显示配置。若实例只开放计算能力，由平台提供合适的图形环境；离线 BC 仍可在仅支持 CUDA 的实例中运行，但当前 PPO/交互采集需要能渲染游戏的实例，尚未实现跨主机采样服务。

## 4. 用本地浏览器操作

保持终端 A 的显示服务运行，在已经配置 `DISPLAY` / `XAUTHORITY` 的云端终端 B 启动：

```bash
bash scripts/run_cloud.sh web --queue runs/corrections
```

另开一个云端终端，在仓库目录读取登录口令：

```bash
cat runs/web/access-token
```

本地 Windows PowerShell 执行下列命令，把 `SSH_PORT`、`HOST` 和账号替换为矩池云控制台显示的实际值：

```text
ssh -p SSH_PORT -N -L 8765:127.0.0.1:8765 root@HOST
```

这里 `SSH_PORT` 是平台映射后的 SSH 端口；`8765` 是本地浏览器端口和云端示范服务端口。保持隧道窗口打开，浏览器访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)，输入刚才的口令。矩池云官方支持这种本地 SSH 转发，因此该路径无需额外导出公网 8765。[平台端口访问说明](https://www.matpool.com/supports/reference/faqs)

首次先试“路线行走”等简单课程：加载 → 开始操作并录制 → 成功后确认提交，再按[第一阶段手册](stage1-guide.md)重放检查。按 Esc、切走窗口或断线会中止示范，不应作为合格训练数据。当前 PNG 通道等待画面确认，网络延迟会降低操作速度；先检查手感和数据质量，再扩大采集，不把“网页能打开”当作“原速流畅”验收。

如本地 8765 已占用，按[通用连接指南](cloud-workflow.md#4-浏览器连接先使用-ssh-隧道)修改端口及 Origin。新租实例后重新核对 SSH 主机与端口。

## 5. 学习、停止和换机恢复

首批示范通过检查后，按 [README 的学习命令](../README.md#浏览器录制与云端学习)做 BC、独立评估和纠错，再比较 PPO 微调。首次分时运行人工采集与训练，减少同卡争用。长期进程可放入 tmux；它维持 SSH 断开后的进程，但不能防止实例到期或释放。

释放实例前，先让训练正常保存并退出，确认模型 `.zip` 与同名 `.json`、示范和纠错题库均在持久化网盘，并另存重要结果。平台“保存环境”主要保留网盘以外的环境改动，`.snap` 本身占用网盘；保存期间机器不可用。环境快照不能代替训练检查点，也不能理解为保存了正在运行的训练进程。[保存环境说明](https://www.matpool.com/supports/snapshot)

换机后先确认区域、网盘挂载和镜像，再进入仓库检查：

```bash
cd /mnt/valorant-agent-lab
git status --short --branch
```

工作区干净、当前为 `main` 时，使用 `git pull --ff-only origin main` 更新代码；有本地改动或分叉时先保留处理。镜像、系统 Python 或项目路径变动后重新准备依赖和本机配置，不假定网盘上的旧 `.venv` 一定可用。GPU 显示和 SSH 隧道每次实例启动都需重新建立并检查；模型恢复按[服务器说明](server.md#正式训练恢复和日志)使用 `--resume`。

本页验收完成后，下一项交付是少量真实人工示范及首次 BC 的任务成功率、失败分类和实时表现报告，而非仅检查训练损失下降。
