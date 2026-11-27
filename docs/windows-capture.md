# Windows 本机示范与 Linux 训练

这条流程让游戏在 Windows 本机运行，直接接收本机键鼠。操作期间不经过云服务器、SSH、浏览器画面传输；网络只用于下载代码、接收题目和上传完成的示范。Windows 不需要安装 VALORANT、PyTorch 或 CUDA，但需要能运行 CA-VAL-AIM 的图形硬件。

当前开发分支为 `windows-recorder`，两端共用同一套代码、场景、动作和数据格式。引擎锁定 **CA-VAL-AIM v2.15.0**，Lotus 使用该安装版本的地图和碰撞数据。原 `main` 与云端正在运行的服务不会因下载本分支而自动更新。

## 1. Windows：第一次安装

准备 Windows 10/11 x64、Python 3.10–3.13（建议 python.org 的 3.12，包含 Tcl/Tk）、Git。预留至少 5 GB 安装空间，录制数据另计。以下在 PowerShell 执行：

```powershell
git clone -b windows-recorder https://github.com/wly-lizhong/valorant-agent-lab.git
cd valorant-agent-lab
.\start_windows.cmd
```

已有本分支的目录，关闭录制器后先运行 `git status`；工作区干净时执行 `git pull --ff-only origin windows-recorder`，再双击 `start_windows.cmd`。不要在正在修改的仓库中强制切分支或覆盖文件。

首次启动自动创建 `.venv-windows`，下载已固定校验值的 Godot/.NET 工具，校验地图并构建。以后源码未变时直接打开中文选关窗口。工具保存在项目目录，不需要全局安装 .NET SDK，也不需要管理员权限。下载阶段可能较慢；它不是实际游戏延迟。

启动失败会保留命令窗口。若提示找不到 Python，请先在新 PowerShell 中确认 `python --version`；不要通过双击旧版 CA-VAL-AIM 安装程序进入录制。

## 2. Windows：选关并录制

1. 选择“路线行走”“停稳射击”等课程，或下载来的“纠错”题目。
2. 选择“训练示范”或“独立验证”。纠错题自动使用原题的种子和训练分组，不能改成验证题。
3. 点击“开始本机录制”。原生游戏窗口显示中文说明，按 **Enter** 开始。
4. **WASD** 移动，鼠标转向，左键射击，**R** 换弹，**Shift** 静步，**Ctrl** 蹲伏，**Space** 跳跃。
5. 完成任务后会显示“任务完成”，按 **Enter** 确认保存并返回选关；**Esc** 放弃返回。

失焦、未完成、长时间卡顿或写盘跟不上会使本段记录不进入训练。拒绝的记录保留在本地供诊断，上传器不会把它当作成功示范。独立验证的数据不会用于 BC 参数学习；应另外录不同种子的验证示范。

数据位于 `runs/demonstrations-v215/episode-…/`。保存内容是 **每次动作之前的游戏 RGB 画面 + 实际执行的动作 + 场景/种子 + 诊断**，可直接供 BC 读取。它不是普通 MP4 录像，也不录桌面、麦克风或其他应用。原始采集暂存在 `runs/native/`，可用于重新转换和排错；首次小批量验收后再规划清理与大规模存储。

也可以跳过选关器，使用 CLI：

```powershell
.venv-windows\Scripts\valorant-lab.exe record-native --scenario configs/lotus_route.json
```

## 3. GitHub 负责传什么

| 内容 | 载体 | 用处 |
| --- | --- | --- |
| 源码、课程配置、固定 Lotus 地图包 | Git 分支 | 两端安装同版本环境 |
| 纠错题包 | GitHub Release 附件 | 场景、随机种子、失败类型、失败时画面与诊断 |
| 真人示范包 | GitHub Release 附件 | 经确认的 RGB、动作和元数据，Linux 校验后并入数据集 |
| 模型权重 | 目前仍在 Linux 训练目录 | 做题不需要把模型下载到 Windows；本工具不交换权重 |

题包是“重做这次回合的题目”，**尚不是从模型卡住的那一帧恢复的存档**。Windows 从原种子、回合起点开始，人在相同任务中提供完整正确示范。要让人直接接管失败瞬间，还需实现完整世界状态快照，包括敌人、随机状态、枪械、门和物理状态。

附件不写入 Git 历史；`git pull` 本身只更新代码，不会下载 Release 数据。每个 ZIP 附带 SHA-256 和文件清单；导入检查版本、哈希、动作/画面维度、训练/验证分组。相同记录跳过，冲突保留本地数据并报错，不用覆盖旧文件解决冲突。

**本仓库公开，上传的游戏画面、动作和题目附件也公开。** 打包器只收集规定的数据文件，不会遍历上传整个项目、个人设置或凭据。训练或采集仍在写数据时，请等本回合结束再上传。

## 4. 两端的 GitHub 登录

两台机器分别安装 [GitHub CLI](https://cli.github.com)，然后运行：

```text
gh auth login
gh auth status
```

上传需有目标仓库的写权限；下载使用自己的已有 GitHub 登录。Linux 没有桌面时可选择网页登录/设备码授权，在另一台电脑的浏览器完成。不要把令牌放进题包、仓库或发到聊天里。两端默认仓库都是 `wly-lizhong/valorant-agent-lab`；若换自己的仓库，两边都使用同一 `--repo 用户/仓库`。

## 5. Linux：发题、收示范、继续训练

### 代码准备

首次验证建议另建目录克隆 `windows-recorder`，按 README 的 Ubuntu 安装流程配置；现有训练实例的数据和服务保留。在新目录中确认 `git branch --show-current` 输出 `windows-recorder`。此分支兼容 Linux，并非只能 Windows 运行。

已有实例如果只是想接入导入命令，也必须先准备同版本的 Python 包；不能用旧 v2.11 引擎训练新 v2.15 示范。不要把“导入数据”误做成直接覆盖服务器源码。

### 发纠错题

PPO 的开发评估会向对应训练目录的 `corrections/` 写失败题；也可显式运行开发评估生成：

```bash
source .venv/bin/activate
bash scripts/run_cloud.sh evaluate --model runs/bc/best.zip \
  --split dev --queue runs/corrections-v215 --output runs/dev-evaluation
python -m valorant_lab.exchange push --kind tasks \
  --directory runs/corrections-v215
```

模型必须已在当前版本训练；以上 `best.zip` 不是仓库自带的成熟模型。若用 PPO 自动失败队列，将目录替换为 `runs/ppo-movement-combat/corrections`。没有失败题时不上传空包。

Windows 点击 **“下载云端纠错题”**。会读取最近最多 1000 个 Release 中尚未接收的所有对应批次，不只取最后一包；收到的题目加入下拉列表。完成后点击 **“上传成功示范”**。

### 收示范并训练

```bash
python -m valorant_lab.exchange pull --kind demos \
  --directory runs/demonstrations-v215
valorant-lab bc-train --data runs/demonstrations-v215 --device cuda
bash scripts/run_cloud.sh evaluate --model runs/bc/best.zip \
  --split dev --queue runs/corrections-v215 --output runs/bc-evaluation
```

BC 同时需要适量训练与独立验证示范；读取合同兼容的课程变体，遇到其他版本的数据会拒绝。先小批量检查示范质量与模型能力，再扩大。新数据导入后不会自动开始训练、覆盖已有模型或重启云端服务；选择输出目录和训练命令仍由训练操作者决定。

重复下载通过接收记录避免重复传输；重复示范按 ID/哈希去重。当前上传会打包所选目录中全部成功示范，接收端去重；数据变大后按批次使用不同的目录。单包上限 1800 MiB、解压后上限 4 GiB。更大规模的数据集应改用对象存储，不应把大量录像塞进 Git 提交。

## 6. 无法连接 GitHub 时

同一格式也支持手动传包，例如 SCP、网盘或移动硬盘。必须把 ZIP 和旁边的 `.sha256` 一起复制：

```bash
python -m valorant_lab.exchange pack --kind demos \
  --directory runs/demonstrations-v215 --archive runs/batch.valab.zip
python -m valorant_lab.exchange unpack --kind demos \
  --directory runs/demonstrations-v215 --archive /path/to/batch.valab.zip
```

第一条在采集端运行，第二条在接收端运行。题包把 `demos` 换成 `tasks` 并使用纠错队列目录。可用 `pull --tag 批次名` 重取指定旧批次，绕过本机的“已经接收”记录。

## 7. 手感与数据质量边界

- 步行、枪械与碰撞沿用 v2.15.0，不额外调低人物速度。你对上游 v2.15.0 的良好试玩反馈，是选型依据，不等于已完成官方客户端逐项校准。
- 游戏物理按 60 Hz 固定步进，正式课程每 3 个物理帧采样一次动作，约 20 Hz。鼠标转角映射到模型可执行的离散动作，并分摊至物理帧；它与上游无限精度、逐鼠标事件输入仍有区别，需要真人检查瞄准手感。
- 本机渲染上限 120 FPS，不承诺低配电脑实际达到这个数值。超过 250 ms 的累计卡顿会停止采集，避免长时间积压后突然快进移动；起始帧加载不当成游戏卡顿。
- 录制用 960×540 游戏窗口，学习画面缩小到 256×144。后台保存压缩帧，结束后转换为现有 NPZ 数据集。声音仍不作为学习输入。
- 正式训练画面没有仅为老师提供的路线指示或敌人坐标。先熟悉路线再录制；本窗口暂未提供练习小地图。
- 旧 v2.11 的题目、示范、模型和地图目录保留，但不能仅修改版本标签冒充新数据。新版数据合同明确使用 `ca-val-aim-v2.15.0-lab-v3`。

验证结果与尚未验收项见[验证记录](validation.md)。
