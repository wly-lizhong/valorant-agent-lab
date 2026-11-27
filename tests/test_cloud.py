import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from valorant_lab import cloud


@pytest.mark.parametrize("text,ok", [
    ("OpenGL vendor string: NVIDIA Corporation\nOpenGL renderer string: NVIDIA RTX 4090/PCIe/SSE2", True),
    ("OpenGL vendor string: NVIDIA Corporation\nOpenGL renderer string: Tesla V100-SXM2-16GB/PCIe/SSE2", True),
    ("OpenGL vendor string: NVIDIA Corporation\nOpenGL renderer string: llvmpipe", False),
    ("OpenGL vendor string: NVIDIA Corporation\nOpenGL renderer string: ", False),
    ("OpenGL renderer string: NVIDIA RTX 4090", False),
    ("OpenGL vendor string: Mesa\nOpenGL renderer string: NVIDIA RTX 4090", False),
    ("OpenGL renderer string: llvmpipe (LLVM 19)\nNVIDIA driver installed", False),
    ("NVIDIA-SMI 580.105.08", False),
    ("OpenGL renderer string: Intel UHD", False)])
def test_cuda_inventory_is_not_graphics_proof(text, ok):
    assert cloud.nvidia_renderer(text) is ok


def test_graphics_requires_display_and_does_not_hide_software_override(monkeypatch):
    monkeypatch.setattr(cloud, "probe", lambda *a, **kw: {"ok": True, "output": "OpenGL vendor string: NVIDIA Corporation\nOpenGL renderer string: NVIDIA Test"})
    assert not cloud.graphics_check({})["ok"]
    assert cloud.graphics_check({"DISPLAY": ":99"})["ok"]
    assert not cloud.graphics_check({"DISPLAY": ":99", "LIBGL_ALWAYS_SOFTWARE": "1"})["ok"]


def test_pci_decimal_conversion_and_config_injection():
    assert cloud.pci_bus_id("00000000:AF:01.0") == "PCI:175:1:0"
    assert cloud.pci_bus_id("0001:0A:00.0") == "PCI:10@1:0:0"
    # V100 virtual displays reject the obsolete NoScanout option.
    assert 'UseDisplayDevice' not in cloud.xorg_config("PCI:175:1:0")
    assert 'AllowEmptyInitialConfiguration' in cloud.xorg_config("PCI:175:1:0")
    with pytest.raises(ValueError):
        cloud.pci_bus_id("0; command")
    with pytest.raises(ValueError):
        cloud.xorg_config('PCI:1:0:0\nSection "Other"')


def test_probe_failure_is_reported(monkeypatch):
    def fail(*a, **kw):
        raise FileNotFoundError("missing tool")
    monkeypatch.setattr(cloud.subprocess, "run", fail)
    assert cloud.probe(["absent"]) == {"ok": False, "output": "missing tool"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only display lifecycle")
@pytest.mark.parametrize("failure", ["body", "xauth", "none"])
def test_private_display_cleanup_never_touches_other_displays(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(cloud, "ROOT", tmp_path)
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    monkeypatch.setattr(os, "chown", lambda *a: None)
    monkeypatch.setattr(cloud.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(cloud, "display_free", lambda n: n == 99)
    monkeypatch.setattr(cloud, "probe", lambda *a, **kw: {"ok": True, "output": "00000000:01:00.0"})
    monkeypatch.setattr(cloud, "graphics_check", lambda env: {"ok": True, "output": "NVIDIA"})
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: True if str(path) == "/tmp/.X11-unix/X99" else exists(path))
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=int(failure == "xauth")))
    calls = []
    class Process:
        def poll(self): return None
        def terminate(self): calls.append("terminate-own")
        def wait(self, timeout): calls.append("wait-own")
    def launch(command, **kwargs):
        assert "-nolisten" in command and "tcp" in command and "-auth" in command
        calls.append(command)
        return Process()
    monkeypatch.setattr(cloud.subprocess, "Popen", launch)
    try:
        with cloud.gpu_display(display=99) as env:
            auth = Path(env["XAUTHORITY"])
            assert auth.exists() and auth.stat().st_mode & 0o077 == 0
            assert env["DISPLAY"] == ":99"
            if failure == "body":
                raise RuntimeError("test work failure")
    except RuntimeError:
        assert failure != "none"
    assert not list(tmp_path.glob("runs/gpu-display-*/Xauthority"))
    if failure != "xauth":
        assert calls[-2:] == ["terminate-own", "wait-own"]
    else:
        assert calls == []


def test_interrupted_child_gets_cleanup_time(monkeypatch):
    events = []
    class Process:
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt
            events.append(("wait", timeout)); return 0
        def poll(self): return None
        def send_signal(self, sig): events.append(("signal", sig))
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *a, **kw: Process())
    with pytest.raises(KeyboardInterrupt):
        cloud.run_cli(["web"], {})
    assert events == [("signal", cloud.signal.SIGINT), ("wait", 95)]


@pytest.mark.parametrize("vendor,device,accepted", [
    ("NVIDIA Corporation", "NVIDIA Test", True),
    ("Mesa", "llvmpipe", False), ("NVIDIA", "llvmpipe", False), ("", "", False)])
def test_actual_engine_renderer_is_verified(tmp_path, monkeypatch, vendor, device, accepted):
    from valorant_lab import env as module
    from valorant_lab.scenario import load_scenario, action_sizes
    scenario = module.ROOT / "configs/lotus_route.json"
    config = load_scenario(scenario)
    (tmp_path / "local.json").write_text(json.dumps({"godot": "test-godot", "dotnet": "test-dotnet",
        "project": "test-project", "maps_dir": "test-maps",
        "upstream_sha256": json.loads((module.ROOT / "dependencies.lock.json").read_text())["upstream"]["sha256"]}))
    monkeypatch.setattr(module, "validate_assets", lambda *a: {})
    monkeypatch.setenv("VALORANT_LAB_GPU_REQUIRED", "1")
    class Process:
        def poll(self): return None
        def wait(self, **kw): return 0
    def launch(command, **kw):
        path = Path(command[command.index("--agent-ready") + 1])
        path.write_text(json.dumps({"protocol": 1, "revision": 2, "scenario": config["id"],
            "action_sizes": action_sizes(config).tolist(), "clock": "lockstep", "port": 12345,
            "render_vendor": vendor, "render_device": device}))
        return Process()
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    monkeypatch.setattr(module, "RpcClient", lambda *a: SimpleNamespace(
        socket=SimpleNamespace(settimeout=lambda x: None), call=lambda *a: None, close=lambda: None))
    if accepted:
        with module.ValorantEnv(scenario, root=tmp_path) as env:
            assert env.engine_metadata["render_device"] == device
    else:
        with pytest.raises(RuntimeError, match="did not report NVIDIA"):
            module.ValorantEnv(scenario, root=tmp_path)
