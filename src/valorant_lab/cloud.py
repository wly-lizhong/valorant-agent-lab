"""Cloud graphics diagnostics and an isolated NVIDIA X server; no driver installation."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = ("llvmpipe", "softpipe", "swrast", "software rasterizer")


def probe(command, *, env=None, timeout=12):
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
        return {"ok": result.returncode == 0, "code": result.returncode,
                "output": (result.stdout + result.stderr)[-12000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "output": str(exc)}


def nvidia_renderer(text):
    fields = dict(line.strip().split(":", 1) for line in text.splitlines()
                  if line.strip().startswith(("OpenGL vendor string:", "OpenGL renderer string:")))
    # Tesla drivers identify the vendor separately; their renderer can omit NVIDIA.
    return bool(fields.get("OpenGL vendor string", "").strip().lower() == "nvidia corporation"
                and fields.get("OpenGL renderer string", "").strip()
                and not any(word in text.lower() for word in SOFTWARE))


def graphics_check(env=None):
    env = os.environ.copy() if env is None else env
    if not env.get("DISPLAY"):
        return {"ok": False, "output": "DISPLAY is absent; prepare a GPU display first."}
    result = probe(["glxinfo", "-B"], env=env)
    result["ok"] = (result["ok"] and nvidia_renderer(result["output"])
                    and env.get("LIBGL_ALWAYS_SOFTWARE", "").lower() not in ("1", "true"))
    return result


def inventory():
    return {"system": platform.system(), "display": os.environ.get("DISPLAY"),
            "container_detected": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
            "nvidia": probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv"]),
            "graphics": graphics_check(),
            "tools": {name: bool(shutil.which(name)) for name in ("Xorg", "xauth", "glxinfo")},
            "note": "CUDA, graphics and browser timing need separate validation. This check does not run the game."}


def pci_bus_id(value):
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value.strip())
    if not match:
        raise ValueError("Unexpected NVIDIA PCI bus ID")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    return f"PCI:{bus}@{domain}:{device}:{function}" if domain else f"PCI:{bus}:{device}:{function}"


def xorg_config(bus):
    if not re.fullmatch(r"PCI:\d+(?:@\d+)?:\d+:\d+", bus):
        raise ValueError("Invalid Xorg bus ID")
    return '''Section "ServerFlags"
    Option "AutoAddDevices" "false"
    Option "AutoEnableDevices" "false"
EndSection
Section "Device"
    Identifier "AgentGPU"
    Driver "nvidia"
    BusID "__BUS__"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "ProbeAllGpus" "false"
EndSection
Section "Screen"
    Identifier "AgentScreen"
    Device "AgentGPU"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1280 720
    EndSubSection
EndSection
'''.replace("__BUS__", bus)


def display_free(number):
    return not Path(f"/tmp/.X11-unix/X{number}").exists() and not Path(f"/tmp/.X{number}-lock").exists()


@contextmanager
def gpu_display(*, gpu=0, display=None, timeout=25):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("Starting Xorg requires Linux host privileges. With sudo, run the documented display command in another terminal, then set DISPLAY/XAUTHORITY. Container sudo may still lack devices.")
    for name in ("Xorg", "xauth", "glxinfo", "nvidia-smi"):
        if not shutil.which(name):
            raise RuntimeError(f"Missing {name}; see docs/cloud-workflow.md. No driver is installed automatically.")
    number = display if display is not None else next((n for n in range(90, 120) if display_free(n)), None)
    if number is None or not 1 <= number <= 199 or not display_free(number):
        raise RuntimeError("Display is occupied or outside [1,199]; existing displays are never removed")
    card = probe(["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id", "--format=csv,noheader"])
    if not card["ok"]:
        raise RuntimeError("Cannot query the selected NVIDIA GPU: " + card["output"])
    bus = pci_bus_id(card["output"].strip())
    uid = int(os.environ.get("SUDO_UID", "0")); gid = int(os.environ.get("SUDO_GID", "0"))
    runs = ROOT / "runs"
    if not runs.exists():
        runs.mkdir()
        os.chown(runs, uid, gid)
    directory = runs / ("gpu-display-" + uuid.uuid4().hex[:12])
    directory.mkdir(mode=0o700)
    auth = directory / "Xauthority"
    auth.touch(mode=0o600)
    config = directory / "xorg.conf"
    config.write_text(xorg_config(bus), encoding="utf-8")
    # Root-created files must remain usable by the sudo caller, not world-readable.
    for path in (directory, auth, config):
        os.chown(path, uid, gid)
    env = os.environ.copy()
    env.update(DISPLAY=f":{number}", XAUTHORITY=str(auth))
    env.pop("LIBGL_ALWAYS_SOFTWARE", None)
    process = None
    try:
        result = subprocess.run(["xauth", "-f", str(auth)],
                                input=f"add :{number} . {secrets.token_hex(16)}\n", text=True,
                                capture_output=True, timeout=10)
        if result.returncode:
            raise RuntimeError("Could not create private X authority")
        # xauth may replace the file atomically, resetting its owner.
        os.chown(auth, uid, gid)
        auth.chmod(0o600)
        with (directory / "launcher.log").open("wb") as log:
            process = subprocess.Popen([shutil.which("Xorg"), f":{number}", "-config", str(config),
                "-auth", str(auth), "-logfile", str(directory / "Xorg.log"),
                "-nolisten", "tcp", "-noreset", "-novtswitch", "-sharevts"], stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"GPU display exited. Inspect {directory / 'Xorg.log'}; driver/container may not support graphics.")
                if Path(f"/tmp/.X11-unix/X{number}").exists() and graphics_check(env)["ok"]:
                    print(f"GPU display ready: DISPLAY=:{number}\nXAUTHORITY={auth}", flush=True)
                    yield env
                    return
                time.sleep(.25)
            raise RuntimeError(f"NVIDIA OpenGL check failed; inspect {directory}. No CPU fallback was used.")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=5)
        # Remove only our authority; preserve logs/config for troubleshooting.
        auth.unlink(missing_ok=True)


def run_cli(arguments, env):
    env = {**env, "VALORANT_LAB_GPU_REQUIRED": "1"}
    process = subprocess.Popen([sys.executable, "-m", "valorant_lab.cli", *arguments],
                               env=env, start_new_session=True)
    try:
        return process.wait()
    finally:
        if process.poll() is None:
            # Give Python time to reject unfinished recordings/save checkpoints and close its engine.
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=95)
            except subprocess.TimeoutExpired:
                if sys.platform == "linux":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if sys.platform == "linux":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="GPU-backed cloud launcher; no desktop environment required")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--require-gpu", action="store_true")
    check.add_argument("--output")
    display = sub.add_parser("display", help="Keep a private GPU X server alive; run via sudo when necessary")
    display.add_argument("--gpu", type=int, default=0)
    display.add_argument("--display", type=int, default=99)
    run = sub.add_parser("run")
    run.add_argument("--gpu", type=int, default=0)
    run.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "check":
        result = inventory()
        text = json.dumps(result, indent=2)
        if args.output:
            path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        print(text)
        return 1 if args.require_gpu and not result["graphics"]["ok"] else 0
    if args.gpu < 0:
        parser.error("GPU index must be nonnegative")
    if args.command == "display":
        # Exit through the context manager so only this process's X server is stopped.
        def terminate(_sig, _frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            with gpu_display(gpu=args.gpu, display=args.display):
                print("Keep this process alive. Use the DISPLAY/XAUTHORITY above in the training shell.", flush=True)
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        return 0
    arguments = args.args[1:] if args.args[:1] == ["--"] else args.args
    if not arguments:
        parser.error("Provide a valorant-lab command after --")
    if arguments[0] in ("bc-train", "queue", "doctor") or arguments == ["--help"]:
        return subprocess.call([sys.executable, "-m", "valorant_lab.cli", *arguments])
    def terminate(_sig, _frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        if os.environ.get("DISPLAY"):
            result = graphics_check()
            if not result["ok"]:
                raise RuntimeError("Existing DISPLAY is not verified NVIDIA rendering:\n" + result["output"])
            return run_cli(arguments, os.environ.copy())
        with gpu_display(gpu=args.gpu) as env:
            return run_cli(arguments, env)
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
