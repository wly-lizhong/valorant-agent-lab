# VALORANT Agent Lab

基于 [CA-VAL-AIM](https://github.com/Pyromesis/CA-VAL-AIM) 的独立 FPS 强化学习环境。首个场景使用莲华古城（Lotus）地图资源，训练智能体根据第一人称画面走到目标点。

这是训练环境和 PPO 基线代码，尚未提供训练好的游戏智能体。运行时不需要安装 VALORANT。

## 当前能力

- Gymnasium `reset / step` 接口；160×90 RGB 图像，PPO 使用连续 4 帧。
- 前后左右移动、走路/蹲伏、转视角、跳跃、开枪、换弹动作。
- 固定 60 Hz 模拟；默认每次动作推进 6 个模拟子步。
- 真实地图缺失时明确报错，避免意外在简化地图上训练。
- PPO 训练、TensorBoard、定期存档、断点继续和评估入口。
- Ubuntu 22.04/24.04 x86_64 服务器脚本；Windows 本地启动入口。

首个任务是约 4 米的简单导航，目标用青色立柱标记。还没有敌人奖励、5v5 回合、爆能器、技能、队友跟随、声音输入、HUD 识别或外接键鼠控制。移动和枪械沿用上游模拟，尚未按官方客户端逐项校准。

## 服务器开始使用

仓库已包含约 74 MB 的莲华古城地图包，`git clone` 会一起下载，无需另外传文件或配置 Git LFS。地图资源归属见 [资源说明](assets/NOTICE.md)，安装方法见 [地图说明](docs/assets.md)。引擎和模型权重单独准备。

```bash
git clone https://github.com/wly-lizhong/valorant-agent-lab.git
cd valorant-agent-lab
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

克隆完成后，直接校验并解包仓库自带的地图：

```bash
python scripts/assets.py install --archive assets/lotus-assets.zip \
  --sha256 309ae815598e3180e626eac0fbb693555abce65740e877fb3e40e0530d3746a2
python scripts/bootstrap.py --maps-dir assets/maps-data --skip-build
bash scripts/run_linux.sh smoke
```

安装 PyTorch 2.8.0 对应的 CPU 或 CUDA 版本后安装训练依赖，具体命令见 [服务器说明](docs/server.md)：

```bash
pip install -r requirements-train.txt
bash scripts/run_linux.sh doctor
bash scripts/run_linux.sh train --device cuda --steps 100000
```

先用 `smoke` 检查真实画面、移动、模拟时间及开枪，再启动训练。结果在 `runs/smoke/`，模型和日志默认在 `runs/ppo-navigation/`。没有 GPU 时把 `--device cuda` 改成 `--device cpu`。

## 如何继续训练

```bash
bash scripts/run_linux.sh train --device cuda --steps 100000 \
  --resume runs/ppo-navigation/latest.zip
bash scripts/run_linux.sh evaluate --device cuda \
  --model runs/ppo-navigation/latest.zip --episodes 10
```

`--steps` 是本次追加的采样步数，PPO 会按完整 rollout 向上取整。`--envs 2` 可创建独立引擎进程；先单环境验收再增加并发。断点保存策略与优化器，不保存当时的场景状态，继续时重新开始回合。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `configs/lotus_navigation.json` | 地图、起点、终点和动作频率 |
| `engine/overlay/` | Godot C# 训练入口和本机通信桥 |
| `src/valorant_lab/` | Gymnasium 环境、PPO 与命令行 |
| `scripts/` | 固定版本下载、构建和地图迁移 |
| `dependencies.lock.json` | 上游源代码、Godot、.NET 版本及校验值 |
| `requirements*.txt` | Python 基础依赖及训练依赖版本 |
| `docs/` | 服务器、资源和接口说明 |

`assets/` 仅跟踪地图 ZIP、SHA-256 文件和资源说明；解包后的地图保存在 `assets/maps-data/`，不提交 Git。`external/`、`.tools/`、`local.json` 和 `runs/` 也只保存在本机。每次训练记录配置及主要库版本。Python 平台相关间接依赖由 pip 解析，若要严格复现实验，还应保存本次 `pip freeze`、驱动和硬件信息。

## 开发与验证

```bash
pip install -e '.[test]'
python -m pytest -q
python scripts/bootstrap.py
bash scripts/run_linux.sh smoke
```

`ci/checks.yml` 提供 Python 协议测试和 C# 编译的 GitHub Actions 模板，当前未启用。具备 workflow 写入权限后可将它放入 `.github/workflows/`。模板未执行地图解包和渲染，因此 CI 通过也不代表真实地图渲染验收。实际运行记录见 [验证说明](docs/validation.md)。观察和动作定义见 [环境接口](docs/environment.md)。

修改 overlay 后重跑 `bootstrap.py`；它只更新 `external/` 内生成的副本，保留原始下载和已安装的训练器。更换上游版本需要一起更新哈希、补丁锚点和实际地图检查。

## 来源与许可

代码按 GPL-3.0-or-later 提供，参见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。本项目修改并连接 CA-VAL-AIM v2.11.0，保留其对 ValTrainer 的归属。地图资源不在代码许可证范围内；VALORANT 及相关商标、素材属于其权利人。本项目与 Riot Games 无隶属或背书关系。
