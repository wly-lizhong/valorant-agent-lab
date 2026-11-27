# 有限批次的自动训练与评估

`scripts/run_bc_experiments.py` 用普通 Python 子进程执行已经确定的实验。它不调用大语言模型，也不创建 Goal、定时唤醒或自动调参服务。训练过程中不需要 AI 反复读日志。

首轮配置在 `configs/bc-experiments-round01.json`：

| 候选 | 视觉特征 | 动作头 | 动作损失 |
|---|---|---|---|
| normalized | LayerNorm | Tanh | 原始分布 |
| balanced | LayerNorm | Tanh | 仅从训练集计算的有限平方根逆频率权重 |
| balanced-relu | LayerNorm | ReLU | 同上 |

三个候选使用同一数据、随机种子 42、batch 128、学习率 3e-4、60 epochs。每个候选依次运行：训练、best 的离线诊断、latest 的离线诊断、4 个 dev 路线场景、4 个 dev 射击场景。游戏评估使用 best；同时保留 latest 诊断，便于识别仅按验证 NLL 选出的模型是否忽略少数动作。

这是候选实验，不是证明已经学会游戏的正式验收。仅 dev 场景用于此流程，最终 test 场景不会打开；人类示范中的 train 和 validation 始终分开。

## 在 Linux 云端运行

从当前候选源码根目录执行。先加载既有 NVIDIA 图形环境；不要安装或更换系统驱动。源码与 `runs/gpu-display-*` 应在实例本地盘，输出目录可以放在持久盘；FUSE 持久盘不适合作为 Xauthority 运行目录。

```bash
python scripts/run_bc_experiments.py \
  --manifest configs/bc-experiments-round01.json \
  --data /path/to/demonstrations-v215 \
  --output /path/to/persistent/experiments-round01 \
  --device cuda --gpu-display
```

`--gpu-display` 为本次任务创建私有 NVIDIA X 显示，运行结束时只清理它自己的进程。已有桌面、VNC 和其他显示不动。指定 GPU 时使用 `--gpu N`；显存和利用率采样记录整张指定 GPU，不冒充本进程独占用量。

长任务可由现有 `tmux` 会话启动。启动后检查 `state.json`、阶段日志与第一个检查点即可；结束当前聊天不会让 tmux 中的 Python 训练停止。报告写完本身不会唤醒 AI，之后再让助手分析汇总。

低配 Windows 可阅读报告；要在 CPU 上运行整条流程，先复制 manifest 并把 `training.precision` 改为 `fp32`，命令使用 `--device cpu` 且不传 `--gpu-display`。这不保证低配机器训练很快，Windows 仍主要用于本地采集。

## 上限与恢复

- 每个 manifest 最多 3 个不同候选；参数不会由脚本自行增加或搜索。
- 首轮累计子进程预算 1800 秒；每次训练最多 600 秒、离线诊断 180 秒、每门游戏评估 300 秒。
- 每个阶段最多 2 次尝试。失败或超时即停止整批，没有自动无限重试；检查原因后显式传 `--resume`。
- 超时或正常中断会关闭该阶段自己的进程组/进程树。Linux `SIGTERM` 会转成可清理的中断；`SIGKILL` 无法执行清理，如恢复时检测到原子进程仍存在，脚本拒绝并发重跑，需先检查该 PID。
- 完成阶段先核对产物 SHA-256，再跳过。源码、manifest、实际数据、依赖版本或运行设备改变时拒绝恢复，避免混合实验。
- 训练恢复使用同一工作目录里的完整 `resume.pt`；恢复前复制输入检查点到新 attempt 目录。模型、优化器、随机状态的恢复由 BC 实现负责。若在第一个恢复点前失败，不猜测重置状态，应保留旧目录并新建实验目录。
- 诊断、评估的每次尝试都有独立目录，失败日志和轨迹不会被后一次覆盖。最终成功结果指向最近完成的 attempt。

恢复命令与首次运行相同，追加 `--resume`。修改 epoch 数量属于另一份 manifest，不能在已有结果目录里悄悄追加。脚本不会自动升级环境、安装依赖、改正式部署或推送 GitHub。

## 看哪些文件

| 文件 | 内容 |
|---|---|
| `state.json` | 当前阶段、状态、累计时间、每次尝试的命令、PID、错误、完整日志路径及产物哈希 |
| `identity.json` | 冻结的配置、源码哈希、数据清单和依赖版本 |
| `summary.json` | 每个候选的训练/离线诊断/真实游戏结果及 GPU 汇总；每阶段更新 |
| `gpu.jsonl` | 普通线程采样的 GPU 利用率、显存、功耗和所属阶段 |
| `<候选>/training/` | `best.zip`、`latest.zip`、恢复点、元数据和逐 epoch 指标 |
| `<候选>/attempts/<阶段>/attempt-NNN/` | 日志；诊断报告、游戏报告、逐步轨迹、起点与终点图片 |

`summary.json` 统计实际发射子弹数、命中、击杀、成功率、动作分布、连续相同动作长度、视角极限和疑似卡墙窗口。开火按钮按下的帧数与实际子弹数分开；疑似卡墙定义为连续 40 步中超过 70% 请求移动，但离窗口起点的最大位移不足 0.5 米，它是诊断指标，不是改写游戏成功条件。

游戏采用 lockstep 评估。推理 p95 会保留，但它不是实时端到端流畅性认证。原始 `trace.jsonl`、`start.png`、`final.png` 保留；默认不逐帧保存 PNG，避免小持久盘被图片撑满。loss、GPU 占用或离线准确率都不能替代实际射击、命中与路线完成。
