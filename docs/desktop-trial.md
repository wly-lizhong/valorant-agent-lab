# 中文选关实时试玩

入口流程：中文选择关卡 → 点击“开始这一关” → 自动进入场景。开始时锁定鼠标用于瞄准；Esc、离开窗口或返回选关会结束本次试玩并释放鼠标。重来会重新启动所选关卡，不需要操作英文首页。游戏内部 HUD 和结算文字仍沿用上游英文。

## 与训练示范台的区别

| 入口 | 用途 | 数据 |
| --- | --- | --- |
| `valorant-lab web`，默认 8765 | 五阶段课程、带路线引导练习、正式逐帧示范 | 正式模式可确认提交，带引导练习不保存示范 |
| `scripts/desktop_trial/server.py`，默认 6080 | 中文选关、连续游戏画面、相对鼠标控制 | 仅试玩，不生成 BC/PPO 训练示范 |

当前选关列表：Lotus 入门清点（上游 Site Clear tier 0，每轮最多 2 名守点敌人）、Lotus 进阶清点（tier 2，每轮最多 3 名）、独立训练场的急停点射（Counter-Strafe tier 0）。它们不是原有五阶段课程的实时版本，也不表示已实现队友跟随、5v5 或爆能器任务。

## 运行前提

这是从已验收云端运维工具整理出的独立入口，尚未集成进 `start_cloud.sh`。先按[云端指南](cloud-workflow.md)完成仓库安装、引擎编译、地图准备及实际 NVIDIA OpenGL 验证。需要：

- Ubuntu 22.04/24.04、已可用的 NVIDIA Xorg 图形会话；本机电脑只需浏览器和 SSH。
- 仓库根目录 `local.json` 中有效的 `godot`、`project` 路径；不上传该文件。
- 同一个图形会话的 `DISPLAY`、`XAUTHORITY`，以及需要时的 `DOTNET_ROOT`、NVIDIA 图形库环境变量。
- `aiohttp`（项目 `web` 可选依赖）、系统 `libX11` / `libXtst`、`x11vnc` 和发行版 `novnc`。
- 上游正常模式能找到真实地图：`external/CA-VAL-AIM-v2.15.0/CaValAim/maps` 应指向已解压资源中的 `maps` 目录；已有路径先检查，不覆盖。

以下命令在服务器执行，先进入仓库根目录，并使用**实际可用的图形会话环境**。不要盲目猜 `DISPLAY` 或替换正在运行的显示服务。

```bash
sudo apt-get install -y x11vnc novnc libxtst6 x11-xserver-utils
python -m pip install -e '.[web]'
export VALORANT_LAB_ROOT="$PWD"
```

在已激活项目虚拟环境、已设置图形会话环境的终端 A 启动画面服务：

```bash
x11vnc -display "$DISPLAY" -auth "$XAUTHORITY" \
  -localhost -rfbport 5901 -forever -shared -nopw \
  -noxdamage -repeat -xkb -wait 10 -defer 10
```

终端 B 使用相同环境启动选关服务：

```bash
python scripts/desktop_trial/server.py
```

工具根据仓库位置或 `VALORANT_LAB_ROOT` 读取 `local.json`，只从固定白名单选择模式；日志写到 `runs/desktop-trial/selected-game.log`。它只终止自己创建的游戏进程，不停止其他示范台、显示服务或训练任务。开始前停止旧的手工试玩实例，避免两个游戏窗口争抢焦点。

朋友电脑建立 SSH 隧道（替换占位值）：

```bash
ssh -N -L 16080:127.0.0.1:6080 -p SSH_PORT USER@HOST
```

浏览器打开 `http://127.0.0.1:16080/`。目前一次只允许一个操作页面；关闭其他试玩页面后再刷新。6080、5901 均只监听回环地址，依靠 SSH 身份认证；**不要把这两个无独立登录验证的端口公开到公网**。代理协议检查只接受文档中的本机 Host/Origin；端口变更需要同步修改允许列表。

## 显示与操控注意事项

- VNC 只传画面；游戏键鼠通过独立 WebSocket 发送相对移动。释放鼠标不会重新启用普通桌面绝对坐标输入。
- 加载时与释放控制后不发送游戏移动；过滤鼠标锁定瞬间和异常大的单次位移。2 秒无输入消息释放按键，15 秒无消息断开操作连接；断线结束本次试玩。
- 若黑屏，先检查 `xset q` 是否显示 `Monitor is Off`。专用试玩显示器可使用 `xset dpms force on`、`xset -dpms`、`xset s off`；桌面电源管理器可能重新应用节能设置，需要在同一图形会话中配置。不要关闭共享服务器的认证或屏幕锁定保护。
- 游戏鼠标灵敏度、路线引导仍来自独立的上游设置目录，可用 `XDG_CONFIG_HOME` 隔离。已验收环境设置 `UseSensOverride=true`、`SensOverride=0.4`、`GuideLine=2`、`FpsCapOverride=60`、`VSyncOverride=0`、`Quality=0`，关闭自动检查/下载更新。代码不修改用户已有设置。

## 已验证与限制

2026-11-26：Ubuntu 22.04 + V100 的实际 GPU 图形会话上，已从中文选关直接进入真实 Lotus，验证横向转向、开枪（弹药 25→24）、释放控制后恢复中文选关。该次转向检查没有出现视角突然朝地面。引擎日志为 60 FPS，**不是浏览器帧率或端到端延迟测试结果**。其他两项关卡的模式映射来自上游代码，尚未逐项完成真人手感验收。

本次发布把已运行工具迁入仓库，并去除固定服务器路径、临时鼠标诊断输出；此可迁移副本已通过本地测试，尚未在第二台新服务器安装验收。不是 WebRTC 视频通道，VNC/TCP 仍可能受带宽、网络抖动和浏览器性能影响。实时画面/动作时间对齐、中文结算反馈和正式训练示范录制需要后续实现。

本地可运行轻量回归测试，无需显卡：

```bash
python -m pytest scripts/desktop_trial/test_helper.py -q
node --test scripts/desktop_trial/test_input.cjs
```
