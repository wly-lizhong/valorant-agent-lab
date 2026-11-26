# VALORANT Agent Lab

基于 [CA-VAL-AIM](https://github.com/Pyromesis/CA-VAL-AIM) 的视觉 FPS 智能体训练项目，使用莲华古城（Lotus）真实地图资源。运行独立训练器不需要安装官方 VALORANT。

**当前版本提供第一阶段的环境和学习工具：沿路线移动 → 处理可见敌人 → 必要时换弹 → 继续到达目的地。尚未提供训练成熟、能流畅参战的模型。** 工程验证、学习效果和实时性能分别报告，见[验证记录](docs/validation.md)。

**0.3 版已加入云端 GPU 启动检查和中文浏览器示范台。** 本地低配、无 GPU 电脑用于写代码、SSH 和浏览器操作；地图渲染、示范保存、训练与评估放在无桌面 Linux 云端。浏览器采用逐帧确认的 PNG 步进通道，网络慢时会减速；实际云服务器的 GPU 渲染与远程手感仍待验收。见[云端安装与浏览器操作指南](docs/cloud-workflow.md)。

## 第一阶段现在能做什么

| 能力 | 当前实现 |
| --- | --- |
| 正式任务 | Lotus A Root 到 A 入口的一段约 20 米路线；多个无可见标记的路线检查点；固定 Vandal |
| 交战 | 复用上游 Bot 的模型、命中、伤害、遮挡与回击；固定位置的弱对手；混入无敌人回合 |
| 完成判定 | 经过路线检查点、击杀本回合生成的敌人，并回到/到达终点；击杀不会单独结束任务 |
| 观察 | 256×144 第一人称 RGB；基础 HP、护盾、弹药及换弹 HUD；最近 4 帧 |
| 操作 | 移动、静步/蹲伏、跳跃、开火、换弹；水平/垂直最小非零视角增量 0.25° |
| 人工数据 | 中文浏览器示范台与可选本地窗口；画面/动作绑定、场景/种子、时间和诊断记录；确认保存与重放 |
| 云端运行 | 无桌面 NVIDIA Xorg 启动、OpenGL 与实际引擎设备检查；SSH 隧道连接浏览器 |
| 学习 | 离线行为克隆（BC）；BC 权重初始化 PPO；PPO 存档与恢复；模型与数据的控制契约检查 |
| 纠错 | 开发评估自动生成去重题目；按原种子从回合起点重做；示范通过与模型复测通过分别记录 |
| 时钟 | 训练使用精确步进；独立实时评估让世界在推理期间继续运行，报告超时、丢时与按键释放 |

策略只读取画面。坐标、敌人数量、路线进度等后台字段只用于评分、诊断与测试，不进入学习模型。首版任务固定，模型尚未接入图像目标导航、长期路线记忆或战术状态机。

`configs/lotus_navigation.json` 中约 4 米的青色立柱任务继续保留为环境自检。**`smoke` 默认用它；`train`、`record`、`bc-train`、`evaluate` 默认用正式移动交战配置。** 旧版导航模型不能直接加载到新动作协议。

## 推荐使用顺序

1. 在云端准备地图和引擎，核验 GPU 图形渲染，完成自检、机制检查。
2. 通过浏览器示范台试采少量路线/射击示范，检查手感、画面/动作对齐与重放。
3. 在云端训练 BC，独立评估整段任务；本地电脑不承担训练器渲染或模型训练。
4. 按失败队列补课，混合正常示范重新训练；再比较 PPO 微调收益。
5. 在实际目标硬件上做实时评估，随后扩展跟队、守点和爆破回合。

详细操作见[第一阶段使用手册](docs/stage1-guide.md)。

## Ubuntu 服务器准备

目标为 Ubuntu 22.04/24.04 x86_64 + NVIDIA。仓库包含约 74 MB 的 Lotus 地图包，无需 Git LFS；地图归属见[资源说明](assets/NOTICE.md)。

当前训练平台已确定为矩池云，先按[矩池云首次启动与验收](docs/matpool.md)确认镜像、网盘挂载和 GPU 图形能力。标准 Linux 镜像通常运行在容器中；能使用 CUDA 尚不能证明能渲染游戏，具体实例仍待实机检查。

```bash
git clone https://github.com/wly-lizhong/valorant-agent-lab.git
cd valorant-agent-lab
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
python scripts/assets.py install --archive assets/lotus-assets.zip \
  --sha256 309ae815598e3180e626eac0fbb693555abce65740e877fb3e40e0530d3746a2
python scripts/bootstrap.py --maps-dir assets/maps-data --skip-build
bash scripts/setup_cloud.sh
```

接着按[云端指南第 3 节](docs/cloud-workflow.md#3-启动-gpu-图形会话)准备 GPU 显示，运行 `bash scripts/run_cloud.sh smoke`。启动器会检查真实渲染设备；普通账号可用 sudo 单独启动显示服务。

按[服务器说明](docs/server.md)安装 PyTorch 2.8.0 的 CPU/CUDA 版本，再安装训练依赖：

```bash
pip install -r requirements-train.txt
bash scripts/run_linux.sh doctor
```

无桌面服务器可以训练。**离线 BC 只读取示范数据，无需启动游戏；PPO 和游戏评估仍需渲染进程。** `run_linux.sh` 在没有 `DISPLAY` 时使用 Xvfb + 软件渲染，这是兼容回退，不代表已启用 NVIDIA 图形加速。不要用 Godot `--headless` 生成视觉观察。

## 浏览器录制与云端学习

GPU 显示通过检查后，在云端启动：

```bash
bash scripts/run_cloud.sh web --queue runs/corrections
```

本地建立 SSH 隧道后打开 `http://127.0.0.1:8765`，输入云端 `runs/web/access-token` 文件中的口令。选择课程或纠错题，点击“加载场景”→“开始操作并录制”，成功后确认提交。切走窗口、断线或超时会停止并排除本段。完整命令见[连接与操作步骤](docs/cloud-workflow.md#4-浏览器连接先使用-ssh-隧道)。

可选桌面录制器供已有显示器和图形环境的机器使用：

```bash
pip install -e '.[collect]'
valorant-lab record --episodes 5 --split train
valorant-lab record --episodes 2 --split validation
```

窗口内按 Space 开始；WASD 移动、鼠标瞄准、左键开火、R 换弹、Shift 静步、Ctrl 蹲、Space 跳跃。回合成功后按 S 确认保存，R 重试；Esc 放弃当前示范。失败、失焦或未完成的记录不会自动成为正确示范。

云端录制的数据可直接交给学习器；若在另一台合适机器采集，需复制整个 `runs/demonstrations/` 到训练实例，再执行：

```bash
valorant-lab bc-train --data runs/demonstrations --device cuda
bash scripts/run_cloud.sh evaluate --model runs/bc/best.zip \
  --queue runs/corrections --output runs/bc-evaluation
valorant-lab queue --queue runs/corrections
```

示范数量仅是命令示例，不是达到能力目标所需的数据量承诺。首批先确认采集和学习有效，再扩大规模。CPU 训练将 `--device cuda` 改为 `--device cpu`。

## 强化学习与实时评估

```bash
bash scripts/run_cloud.sh train --device cuda --init-bc runs/bc/best.zip
bash scripts/run_cloud.sh train --device cuda \
  --resume runs/ppo-movement-combat/latest.zip --steps 100000
bash scripts/run_cloud.sh evaluate --model runs/ppo-movement-combat/latest.zip \
  --split test --realtime --output runs/realtime-test
```

PPO 默认每 25,000 步做 3 个开发回合评估，并把代表性失败放入本次训练目录的 `corrections/`；评估会暂时占用训练时间和一个额外渲染进程。可用 `--eval-every` 调整，设为 0 关闭。正式测试集不生成纠错题。

`--steps` 为追加采样步数，PPO 会按完整 rollout 向上取整。保存模型旁边的同名 `.json`，其中包含动作语义、观察、任务和来源信息；迁移服务器时需要与 `.zip` 一起复制。低离线损失不等于游戏通过，步进模式成功也不等于实时通过。

## 后续计划与研究

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| 1A 基础移动交战 | 走路线、观察、停稳射击、换弹、恢复行进 | 环境、采集、训练与评估工具已接入；真人数据及模型能力验收待进行 |
| 1B 跟队与守点 | 可见队友跟随、跟丢恢复、固定岗位守点 | 待实现 |
| 1C 基础爆破回合 | 携包、安装、守包、回防、拆除、简化多人回合 | 待实现 |

浏览器步进示范台已实现，原速视频串流与实际云端手感仍需后续验证。任意时刻场景快照、声音输入、图像目标导航、长期记忆、技能和外接键鼠仍未实现。完整官方客户端机制校准、模型量化和低配部署优化继续按后续阶段推进。

- [第一阶段使用手册](docs/stage1-guide.md)：课程、录制、重放、BC/PPO、纠错、原速验收。
- [云端工作流](docs/cloud-workflow.md)：低配开发端、无桌面 Linux、GPU 渲染与远程人工入口的分工及缺口。
- [矩池云操作指南](docs/matpool.md)：持久化目录、实例图形验收、带平台端口的 SSH 隧道和换机恢复。
- [目标、状态与路线图](docs/research/project-status-and-roadmap.md)：工作包、能力门槛与后续安排。
- [人工纠错与训练路线](docs/research/human-correction-plan.md)：DAgger/DART、规则协作、云端分工、选型理由。
- [CS:GO 与具身智能参考](docs/research/reference-projects.md)、[系统设计](docs/research/system-design.md)、[数据与评估规范](docs/research/data-training-evaluation.md)、[决策记录](docs/research/decisions.md)。
- [教学手册](docs/research/beginner-guide.md)、[来源及评论台账](docs/research/teaching-source-ledger.md)、[训练任务卡](docs/research/training-task-cards.md)。

## 开发与验证

```bash
pip install -e '.[test,web]'
python -m pytest -q
python scripts/bootstrap.py
bash scripts/run_linux.sh smoke
```

地图机制的完整集成检查使用 `scripts/validate_stage1.py`，见[验证记录](docs/validation.md)。其中的测试控制器读取坐标来检查环境，数据明确标为 `script_test`，普通 BC 命令会拒绝将它冒充真人示范。

源码在 `engine/overlay/` 与 `src/valorant_lab/`；上游版本由 `dependencies.lock.json` 固定。`external/`、`.tools/`、本机配置、示范、日志和权重不提交 Git。GitHub Actions 模板在 `ci/checks.yml`，尚未启用；功能验证结果以实际记录为准。
