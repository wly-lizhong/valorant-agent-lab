"""Prepare the existing simulator, then supervise a GPU-backed browser demo.

No driver installation, git operations, training, or public listener is performed.
The optional sudo child owns only its private display and exits on parent pipe EOF.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
STATE = ROOT / "runs/quickstart"
READY_PREFIX = b"LAB_GPU_ENV "


class StartError(RuntimeError):
    pass


def options(argv=None):
    parser = argparse.ArgumentParser(description="一键安装、自检并启动云端浏览器试玩；重复运行可复用依赖。")
    parser.add_argument("--reinstall", action="store_true", help="重新检查并安装项目依赖；保留地图、示范和模型")
    parser.add_argument("--check-only", action="store_true", help="安装并完成真实 GPU/地图自检后退出")
    parser.add_argument("--gpu", type=int, default=0, help="新建显示使用的 NVIDIA GPU 编号，默认 0")
    parser.add_argument("--port", type=int, default=8765, help="网页和本地 SSH 转发端口，默认 8765")
    parser.add_argument("--ssh-target", help="生成连接命令时使用的 SSH 别名或 user@host")
    parser.add_argument("--ssh-port", type=int, help="实际 SSH 端口；使用别名时可沿用 SSH 配置")
    parser.add_argument("--runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--display-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu < 0 or not 1 <= args.port <= 65535:
        parser.error("GPU 编号不能为负数，网页端口须为 1–65535")
    if args.ssh_port is not None and not 1 <= args.ssh_port <= 65535:
        parser.error("SSH 端口须为 1–65535")
    if args.ssh_target and (args.ssh_target.startswith("-") or not re.fullmatch(r"[\w.@:\[\]-]+", args.ssh_target)):
        parser.error("SSH 目标只能是配置别名或 user@host，不要填写整条命令")
    return args


def stop_child(process, *, grace=10):
    """Signal only the process group created by this launcher."""
    if process.poll() is not None:
        return
    for sig, timeout in ((signal.SIGINT, grace), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
    raise StartError(f"本次子进程 {process.pid} 未退出，请检查对应日志。")


def run_stage(title, command, *, env=None, keep_tty=False):
    print(f"\n>>> {title}", flush=True)
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "setup.log").open("a", encoding="utf-8") as log:
        log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {title}\n")
        log.flush()
        process = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", start_new_session=not keep_tty,
            preexec_fn=os.setpgrp if keep_tty else None)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise StartError(f"{title}失败。完整输出：{STATE / 'setup.log'}")
        finally:
            stop_child(process)
            process.stdout.close()


def require_free_port(port):
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", port))
    except OSError as exc:
        raise StartError(f"本机端口 {port} 不可用；可加 --port 8767。不会停止已有服务。") from exc


def preflight(args):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise StartError("当前脚本支持 Ubuntu 22.04/24.04 x86_64。")
    release = platform.freedesktop_os_release()
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") not in ("22.04", "24.04"):
        raise StartError("请使用 Ubuntu 22.04 或 24.04 镜像。")
    if not shutil.which("nvidia-smi"):
        raise StartError("未找到 nvidia-smi，请先让平台提供可用的 NVIDIA 驱动和 GPU。")
    result = subprocess.run(["nvidia-smi", "-i", str(args.gpu),
        "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise StartError("无法访问所选 NVIDIA GPU：\n" + result.stdout + result.stderr)
    print(result.stdout.strip(), flush=True)
    if not args.check_only:
        require_free_port(args.port)


def setup_fingerprint():
    digest = hashlib.sha256()
    for value in (str(ROOT), platform.python_version(), Path("/etc/os-release").read_text()):
        digest.update(value.encode())
        digest.update(b"\0")
    for name in ("requirements.lock.txt", "pyproject.toml", "dependencies.lock.json",
                 "scripts/setup_ubuntu.sh", "scripts/setup_cloud.sh"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def dependencies_ready(fingerprint):
    marker = ROOT / ".venv/.cloud-setup-fingerprint"
    if not PYTHON.is_file() or not marker.is_file() or marker.read_text().strip() != fingerprint:
        return False
    # A marker alone is insufficient after moving or damaging a virtualenv.
    try:
        result = subprocess.run([str(PYTHON), "-c",
            "import gymnasium,numpy,PIL,aiohttp,valorant_lab.web_control"],
            capture_output=True, timeout=30)
        return result.returncode == 0 and all(shutil.which(x) for x in ("Xorg", "xauth", "glxinfo"))
    except (OSError, subprocess.TimeoutExpired):
        return False


def prepare(args):
    preflight(args)
    STATE.mkdir(parents=True, exist_ok=True)
    fingerprint = setup_fingerprint()
    if args.reinstall or not dependencies_ready(fingerprint):
        if os.geteuid() != 0:
            if not shutil.which("sudo"):
                raise StartError("安装系统依赖需要 root 或 sudo。")
            subprocess.run(["sudo", "-v"], check=True)
        # Keep the controlling terminal for sudo's tty-scoped credential cache,
        # but own a separate process group so cancellation cannot signal the shell.
        run_stage("准备 Ubuntu 依赖和引擎工具", ["bash", "scripts/setup_ubuntu.sh", "--skip-build"], keep_tty=True)
        if os.geteuid() != 0:
            subprocess.run(["sudo", "-v"], check=True)
        run_stage("安装云端显示工具与网页依赖", ["bash", "scripts/setup_cloud.sh"], keep_tty=True)
        if not dependencies_ready_after_install():
            raise StartError("项目依赖安装后仍无法导入，请检查 .venv 和 setup.log。")
        (ROOT / ".venv/.cloud-setup-fingerprint").write_text(fingerprint + "\n")
    else:
        print("依赖已准备，复用现有 .venv；需要重装时加 --reinstall。", flush=True)
    maps = ROOT / "assets/maps-data-v2.15.0"
    if maps.exists() and any(maps.iterdir()):
        run_stage("校验已有地图（保留原文件）", [PYTHON, "scripts/assets.py", "verify", "--root", maps])
    else:
        digest = (ROOT / "assets/lotus-assets.sha256").read_text().split()[0]
        run_stage("校验并安装 Lotus 地图", [PYTHON, "scripts/assets.py", "install",
            "--archive", ROOT / "assets/lotus-assets.zip", "--sha256", digest, "--destination", maps])
    run_stage("生成本机配置、构建和导入引擎", [PYTHON, "scripts/bootstrap.py", "--maps-dir", maps])


def dependencies_ready_after_install():
    return subprocess.run([str(PYTHON), "-c",
        "import gymnasium,numpy,PIL,aiohttp,valorant_lab.web_control"],
        capture_output=True, timeout=30).returncode == 0


def cloud_module():
    # Runtime runs with the prepared virtualenv; no graphics imports during --help.
    from valorant_lab import cloud
    return cloud


def display_worker(gpu):
    with cloud_module().gpu_display(gpu=gpu) as env:
        message = {key: env[key] for key in ("DISPLAY", "XAUTHORITY")}
        print(READY_PREFIX.decode() + json.dumps(message), flush=True)
        # Parent death closes the pipe too: no orphan root display after disconnect.
        sys.stdin.buffer.read(1)


def read_display_ready(process, timeout=45):
    deadline = time.monotonic() + timeout
    pending = b""
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                break
            for key, _ in selector.select(timeout=.2):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    raise StartError("GPU 显示进程提前退出，请查看 display-worker.log。")
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(READY_PREFIX):
                        result = json.loads(line[len(READY_PREFIX):])
                        if set(result) != {"DISPLAY", "XAUTHORITY"} or not all(isinstance(v, str) for v in result.values()):
                            raise StartError("GPU 显示就绪消息格式错误。")
                        return result
                if len(pending) > 16384:
                    raise StartError("GPU 显示就绪消息过长。")
    raise StartError("GPU 显示未就绪，请查看 display-worker.log 和 runs/gpu-display-* 中的日志。")


@contextmanager
def display_environment(gpu):
    cloud = cloud_module()
    if cloud.graphics_check()["ok"]:
        print("复用现有 NVIDIA 显示，GPU 以该显示的实际设备为准。", flush=True)
        yield os.environ.copy()
    elif os.geteuid() == 0:
        with cloud.gpu_display(gpu=gpu) as env:
            yield env
    else:
        print("通过 sudo 启动独立 GPU 显示；网页仍使用当前账号。", flush=True)
        subprocess.run(["sudo", "-v"], check=True)
        with (STATE / "display-worker.log").open("a") as log:
            process = subprocess.Popen(["sudo", "-n", str(PYTHON), "-u", str(Path(__file__).resolve()),
                "--display-worker", "--gpu", str(gpu)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=log, preexec_fn=os.setpgrp)
            try:
                env = {**os.environ, **read_display_ready(process)}
                env.pop("LIBGL_ALWAYS_SOFTWARE", None)
                yield env
            finally:
                process.stdin.close()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    stop_child(process)
                process.stdout.close()


def tunnel_command(args):
    parts = ["ssh", "-N", "-L", f"127.0.0.1:{args.port}:127.0.0.1:{args.port}"]
    if args.ssh_port is not None:
        parts.extend(["-p", str(args.ssh_port)])
    elif not args.ssh_target:
        parts.extend(["-p", "SSH_PORT"])
    parts.append(args.ssh_target or "USER@HOST")
    return " ".join(parts)


def wait_for_web(process, port, timeout=30):
    # Ignore HTTP_PROXY, including when probing localhost on a proxy-enabled host.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    expected = (ROOT / "src/valorant_lab/web_static/index.html").read_bytes()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise StartError("网页进程提前退出，请查看 runs/quickstart/web.log。")
        try:
            with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
                if response.status == 200 and response.read(len(expected) + 1) == expected:
                    if process.poll() is None:
                        return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.2)
    raise StartError("网页启动超时，请查看 runs/quickstart/web.log。")


def serve(args, env):
    from valorant_lab.web_control import token_file
    require_free_port(args.port)
    token_path = ROOT / "runs/web/access-token"
    token = token_file(token_path)
    with (STATE / "web.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(PYTHON), "-u", "-m", "valorant_lab.cli", "web",
            "--port", str(args.port), "--queue", "runs/corrections"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for_web(process, args.port)
            instructions = ("请在朋友自己的电脑上运行 SSH 命令（沿用原有 VPN/跳板配置）：\n"
                + tunnel_command(args) + f"\n浏览器：http://127.0.0.1:{args.port}\n"
                + f"口令文件：{token_path}\n保持本窗口运行；Ctrl+C 停止本次网页和自建显示。\n")
            (STATE / "connection.txt").write_text(instructions, encoding="utf-8")
            print("\n网页已就绪，GPU 与地图自检通过。\n" + instructions, flush=True)
            # Redirected stdout (nohup/tee) must not silently become a token log.
            if sys.stdout.isatty():
                print("网页登录口令：" + token, flush=True)
            else:
                print("输出已重定向，登录口令请从上面的口令文件读取。", flush=True)
            print("首次先完成路线/射击录制和回放；网页手感与模型原速成绩仍需验收。", flush=True)
            result = process.wait()
            if result:
                raise StartError(f"网页服务退出（{result}），请查看 {STATE / 'web.log'}。")
        finally:
            stop_child(process, grace=95)


def runtime(args):
    STATE.mkdir(parents=True, exist_ok=True)
    with display_environment(args.gpu) as env:
        env = {**env, "VALORANT_LAB_GPU_REQUIRED": "1", "PYTHONUNBUFFERED": "1"}
        run_stage("验收 NVIDIA 图形显示", [PYTHON, "-m", "valorant_lab.cloud", "check",
            "--require-gpu", "--output", STATE / "cloud-check.json"], env=env)
        run_stage("真实地图移动、开火和重置自检", [PYTHON, "-m", "valorant_lab.cli", "smoke",
            "--output", STATE / "smoke"], env=env)
        if args.check_only:
            print(f"自检通过。结果：{STATE}；去掉 --check-only 可启动网页。", flush=True)
        else:
            serve(args, env)


def main(argv=None):
    args = options(argv)
    if args.display_worker:
        display_worker(args.gpu)
        return 0
    if args.runtime:
        runtime(args)
        return 0
    prepare(args)
    # Re-enter with the prepared venv, rather than mixing its packages into system Python.
    forwarded = list(sys.argv[1:] if argv is None else argv)
    os.execv(str(PYTHON), [str(PYTHON), "-u", str(Path(__file__).resolve()), "--runtime", *forwarded])


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n启动器已停止。已保存的数据仍在 runs/ 中。", file=sys.stderr)
        raise SystemExit(130)
    except (StartError, RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"\n启动未完成：{exc}\n日志目录：{STATE}", file=sys.stderr)
        raise SystemExit(1)
