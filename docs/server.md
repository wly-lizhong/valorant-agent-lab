# 服务器与本地启动

目标平台：Ubuntu 22.04/24.04 x86_64，系统 Python 3.10/3.12。脚本安装系统运行库，在项目内建立 Python 虚拟环境，并下载校验固定的 Godot 4.7.2 .NET 和 .NET SDK 8.0.428。首次需要网络，建议预留至少 10 GB 磁盘；GPU 库、训练日志和模型会继续占用空间。

## Ubuntu

```bash
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

代码克隆后已包含地图包，按 [地图说明](assets.md) 校验解包并配置 `maps_dir`，无需另外上传地图。然后安装训练依赖。先检查 `nvidia-smi`，选择驱动支持的 PyTorch 构建。下面给出固定 2.8.0 的 CUDA 12.8 示例：

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
pip check
bash scripts/run_linux.sh doctor
bash scripts/run_linux.sh smoke
```

无 GPU 的工作电脑可先使用 CPU 构建：

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
bash scripts/run_linux.sh train --device cpu --steps 256
```

以上 CPU/CUDA 安装源来自 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。其他显卡或驱动组合按官方说明选择对应构建；不要仅根据 Ubuntu 版本猜 CUDA 兼容性。`doctor` 的 `cuda_available` 必须为 true 才使用 `--device cuda`。

## 画面渲染与 GPU 训练是两件事

Godot 的 `--headless` 使用无画面的渲染器，不能生成训练需要的图像。本项目只在导入资源时使用它，运行阶段明确拒绝它。

`run_linux.sh` 在没有 `DISPLAY` 时使用 Xvfb + Mesa 软件渲染。这样无桌面服务器也能验收、训练；模型仍可以由 CUDA 加速，但地图渲染占 CPU。需要更高吞吐时，为 NVIDIA 配置能实际创建 OpenGL 上下文的显示服务，再设置 `DISPLAY`，检查 `runs/engine-*/engine.log` 中的渲染设备。当前未自动配置 GPU 显示服务。

WSLg 自带显示时可以直接运行。要测试无桌面路径：

```bash
DISPLAY= bash scripts/run_linux.sh smoke
```

## 正式训练、恢复和日志

```bash
bash scripts/run_linux.sh train --device cuda --steps 100000 \
  --output runs/lotus-v1 --checkpoint-every 10000
tensorboard --logdir runs/lotus-v1/tensorboard --host 127.0.0.1
```

```bash
bash scripts/run_linux.sh train --device cuda --steps 100000 \
  --output runs/lotus-v1 --resume runs/lotus-v1/latest.zip
```

首次长任务建议保留 `--envs 1`，确认内存、CPU 和采样速度后再增加。每个环境有独立进程、随机端口和数据目录。每 10,000 个采样步保存 checkpoint；正常退出或 Ctrl+C 保存 `latest.zip`。进程被强杀/断电时使用此前 checkpoint。

训练保存优化器与采样步数；恢复不会恢复旧场景和全部随机数状态，所以不是逐比特继续。改场景、观察尺寸或动作空间时建立新实验。当前训练任务容易，长期奖励增长不代表学会完整比赛。

## 更新代码

在干净的 `main` 分支：

```bash
git pull --ff-only origin main
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/bootstrap.py
bash scripts/run_linux.sh smoke
```

保留 `local.json`、地图和 `runs/`。保存当前 Git SHA 以及 `pip freeze`，便于回到对应代码。不要在训练进程运行时切换代码。

## Windows 调试

需要 Python 3.10–3.12、[.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) 和可用的桌面显卡驱动。PowerShell 在仓库根目录运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.lock.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python scripts/bootstrap.py --maps-dir 'TRAINER_ROOT'
.\.venv\Scripts\python -m valorant_lab.cli smoke
```

训练依赖同上，用 `.\.venv\Scripts\python -m pip` 安装 CPU/CUDA wheel。Windows 和 WSL 使用各自 checkout 或重新生成 `local.json`；该文件的可执行文件和地图路径不能跨系统共用。当前真实引擎验收在 WSL Ubuntu 24.04 完成，原生 Windows 启动路径仍需单独实测。

## 常见问题

- `Missing real map assets`：检查 maps_dir 层级，运行 `assets.py verify` 校验解包目录。
- 画面空白或启动超时：看错误给出的 `engine.log`；先用 `DISPLAY=` + Xvfb 确认软件渲染。
- 首次导入较慢：bootstrap 需要完成 Godot 资源导入后再启动。
- 下载停在约一分钟：检查 HTTP/HTTPS 代理；可在当前终端取消代理后重试。下载文件必须通过锁定校验值，校验失败不应直接解压使用。
- 下载未完成：缓存 `.part` 可续传；校验失败时保留坏包备查，移开该文件后重新下载。
