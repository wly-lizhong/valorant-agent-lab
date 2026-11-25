# VALORANT Agent Lab

基于 [CA-VAL-AIM](https://github.com/Pyromesis/CA-VAL-AIM) 的视觉 FPS 智能体项目，使用莲华古城（Lotus）地图资源。目标是让智能体依据第一人称画面、可见 HUD 和事先公开的任务，连续完成移动、观察、射击与基本攻防职责。

当前已接通环境控制、图像观察和 PPO 导航基线，尚未提供训练好的游戏智能体。运行这个独立环境不需要安装 VALORANT。

## 首个正式训练目标

**在 Lotus 的一段真实地图区域，完成“沿路线移动 → 遇到可见敌人 → 停稳瞄准并射击 → 必要时换弹 → 恢复行进”的连续闭环。** 固定角色和一把步枪，先不使用技能；场景包含拐角、掩体、无敌人路段以及走偏后的恢复。具体区域和武器在实施前锁定。

第一阶段按以下顺序推进，表中能力均为**待实现和验收的计划**：

| 里程碑 | 智能体要完成的行为 | 交付与验收重点 |
| --- | --- | --- |
| 1A 移动与交战 | 转弯、观察、停稳射击、换弹、交战后继续原任务 | 可训练场景、同步示范、模型和原速录像；分别检查单项、组合及失败恢复 |
| 1B 跟队与守点 | 攻方跟随可见队友；守方到固定位置观察入口，交战后恢复职责 | 跟丢能找回、不过度贴人堵路、不会打完就停住 |
| 1C 基础爆破回合 | 携包、安装、守包、回防、拆除及死亡重置 | 先少量角色跑通回合，再加入脚本队友和对手形成简化 5v5 |

1A 的正式训练与验收使用地图本身的地标和公开的固定目的地，不依赖额外的青色路标或透视提示；基础弹药、血量等信息通过画面中的 HUD 提供。具体数据、奖励和验收条件见[路线图](docs/research/project-status-and-roadmap.md)及[数据与评估规范](docs/research/data-training-evaluation.md)。

移动、停稳精度和枪械行为的基线检查放在正式示范采集之前；完整的官方客户端对照校准在后续逐项推进。环境音、技能、更多武器与外接键鼠后端按后续阶段扩展，CPU/低配显卡优化放在能力基线之后。

## 研究与后续路线

[研究文档目录](docs/research/README.md)集中说明项目目标、参考依据、设计取舍和计划。当前方向是先在 Lotus 上实现移动与交战，再扩展跟队、守点和基础爆破回合；这些仍是后续计划，不属于下方已实现能力。

| 文档 | 内容 |
| --- | --- |
| [目标、现状与路线图](docs/research/project-status-and-roadmap.md) | 第一阶段 1A/1B/1C、工作包、依赖和拟定验收门槛 |
| [CS:GO 与具身智能参考](docs/research/reference-projects.md) | 视觉模仿学习、MLMove、ACT、Diffusion Policy、RTC、OpenVLA、GR00T 的用途和限制 |
| [系统与训练架构](docs/research/system-design.md) | 玩家信息约束、视觉策略、实时控制、动作语义和迁移 |
| [数据、训练与评估规范](docs/research/data-training-evaluation.md) | 示范录制、纠错、奖励、留出评估及服务器复现 |
| [决策与待验证事项](docs/research/decisions.md) | 为什么选择当前路线、暂缓什么、何时调整 |
| [教学手册](docs/research/beginner-guide.md) / [来源与评论台账](docs/research/teaching-source-ledger.md) / [20 张训练卡](docs/research/training-task-cards.md) | 基础游戏行为、公开教学与社区证据、可训练场景设计 |

参考项目的论文成绩与本项目成绩分开记录；未完整观看的视频和未取得的评论均注明读取范围。文档不会把短程 smoke 测试写成模型已经学会游戏。

## 当前实现与环境自检

- Gymnasium `reset / step` 接口；160×90 RGB 图像，PPO 使用连续 4 帧。
- 前后左右移动、走路/蹲伏、转视角、跳跃、开枪、换弹动作。
- 固定 60 Hz 模拟；默认每次动作推进 6 个模拟子步。
- 真实地图缺失时明确报错，避免意外在简化地图上训练。
- PPO 训练、TensorBoard、定期存档、断点继续和评估入口。
- Ubuntu 22.04/24.04 x86_64 服务器脚本；Windows 本地启动入口。

`configs/lotus_navigation.json` 保留约 4 米、带青色立柱的短路线，作为**环境连通性自检场景**：检查地图加载、画面、动作、奖励、重置及训练存档链路。通过它只代表基础管线可运行，不能作为 1A 的训练成果或基础参战能力验收。

**当前默认 `train` 仍运行这份导航配置。** 1A 所需的交战任务与奖励、基础 HUD、细粒度瞄准和独立实时运行尚未接入。移动和枪械继续使用上游模拟；地图外观接近并不代表移速、停止和后坐力已经与官方客户端一致。后续实施将保留自检配置，并另建正式任务配置和评估集。

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

先用 `smoke` 检查真实画面、移动、模拟时间及开枪，再用上面的 `train` 验证现有导航训练链路。这些命令还不训练 1A 移动交战任务。结果在 `runs/smoke/`，模型和日志默认在 `runs/ppo-navigation/`。没有 GPU 时把 `--device cuda` 改成 `--device cpu`。

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
| `configs/lotus_navigation.json` | 导航自检配置：地图、起点、终点和动作频率 |
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
