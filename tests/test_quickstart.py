"""Startup acceptance without installing packages, rendering maps or using sudo."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_cloud.py"
spec = importlib.util.spec_from_file_location("lab_quickstart", SCRIPT)
quick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    root.mkdir()
    state = root / "runs/quickstart"
    state.mkdir(parents=True)
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(quick, "ROOT", root)
    monkeypatch.setattr(quick, "STATE", state)
    monkeypatch.setattr(quick, "PYTHON", python)
    monkeypatch.setattr(quick, "preflight", lambda args: None)
    monkeypatch.setattr(quick, "setup_fingerprint", lambda: "test-fingerprint")
    return root


def test_help_needs_no_gpu_or_optional_packages():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--check-only" in result.stdout and "--ssh-target" in result.stdout
    assert "--runtime" not in result.stdout and "--display-worker" not in result.stdout


def test_no_gpu_fails_before_install_or_subprocess(monkeypatch):
    monkeypatch.setattr(quick.sys, "platform", "linux")
    monkeypatch.setattr(quick.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(quick.platform, "freedesktop_os_release", lambda: {"ID": "ubuntu", "VERSION_ID": "24.04"})
    monkeypatch.setattr(quick.shutil, "which", lambda name: None)
    monkeypatch.setattr(quick.subprocess, "run", lambda *a, **kw: pytest.fail("No-GPU preflight launched a command"))
    with pytest.raises(quick.StartError, match="nvidia-smi"):
        quick.preflight(quick.options([]))


def test_busy_port_keeps_existing_listener():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(quick.StartError, match="端口"):
            quick.require_free_port(port)
        assert listener.getsockname()[1] == port


@pytest.mark.parametrize("arguments", [
    ["--port", "0"], ["--port", "65536"], ["--gpu", "-1"],
    ["--ssh-port", "0"], ["--ssh-target", "host; echo bad"],
    ["--ssh-target=--proxy"], ["--ssh-target", "host\ncommand"],
])
def test_invalid_options_fail_before_installation(arguments):
    with pytest.raises(SystemExit):
        quick.options(arguments)


def test_tunnel_uses_actual_host_options_without_guessing_container_port():
    default = quick.tunnel_command(quick.options([]))
    assert "-p SSH_PORT USER@HOST" in default
    configured = quick.tunnel_command(quick.options(["--port", "8767", "--ssh-target", "my-server"]))
    assert configured == "ssh -N -L 127.0.0.1:8767:127.0.0.1:8767 my-server"
    explicit = quick.tunnel_command(quick.options(["--ssh-target", "friend@cloud", "--ssh-port", "23456"]))
    assert explicit.endswith("-p 23456 friend@cloud")


def test_repeated_start_preserves_data_and_skips_system_install(workspace, monkeypatch):
    maps = workspace / "assets/maps-data"
    maps.mkdir(parents=True)
    (maps / "keep.bin").write_bytes(b"map")
    demo = workspace / "runs/demonstrations/keep.json"
    demo.parent.mkdir()
    demo.write_text("human data")
    monkeypatch.setattr(quick, "dependencies_ready", lambda fp: True)
    calls = []
    monkeypatch.setattr(quick, "run_stage", lambda title, cmd, **kw: calls.append([str(x) for x in cmd]))
    quick.prepare(quick.options([]))
    assert len(calls) == 2
    assert calls[0][2] == "verify" and calls[1][1] == "scripts/bootstrap.py"
    assert (maps / "keep.bin").read_bytes() == b"map"
    assert demo.read_text() == "human data"


def test_corrupt_existing_map_is_not_overwritten(workspace, monkeypatch):
    maps = workspace / "assets/maps-data"
    maps.mkdir(parents=True)
    old = maps / "unrecognized.txt"
    old.write_text("keep")
    monkeypatch.setattr(quick, "dependencies_ready", lambda fp: True)
    calls = []
    def fail(title, command, **kwargs):
        calls.append(command)
        raise quick.StartError("invalid inventory")
    monkeypatch.setattr(quick, "run_stage", fail)
    with pytest.raises(quick.StartError, match="inventory"):
        quick.prepare(quick.options([]))
    assert len(calls) == 1 and calls[0][2] == "verify"
    assert old.read_text() == "keep"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux installation path")
@pytest.mark.parametrize("failed_stage", [None, "scripts/setup_cloud.sh"])
def test_first_install_stamps_only_completed_dependencies(workspace, monkeypatch, failed_stage):
    assets = workspace / "assets"
    assets.mkdir()
    (assets / "lotus-assets.sha256").write_text("a" * 64 + "  lotus-assets.zip\n")
    monkeypatch.setattr(quick.os, "geteuid", lambda: 0)
    monkeypatch.setattr(quick, "dependencies_ready", lambda fp: False)
    monkeypatch.setattr(quick, "dependencies_ready_after_install", lambda: True)
    calls = []
    def stage(title, cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        if failed_stage in cmd:
            raise quick.StartError("install failed")
    monkeypatch.setattr(quick, "run_stage", stage)
    marker = workspace / ".venv/.cloud-setup-fingerprint"
    if failed_stage:
        with pytest.raises(quick.StartError):
            quick.prepare(quick.options([]))
        assert not marker.exists() and len(calls) == 2
    else:
        quick.prepare(quick.options([]))
        assert marker.read_text().strip() == "test-fingerprint"
        assert calls[0] == ["bash", "scripts/setup_ubuntu.sh", "--skip-build"]
        assert calls[2][2] == "install"
        assert calls[3][1] == "scripts/bootstrap.py"


@pytest.mark.parametrize("failed_stage", [0, 1, None])
def test_gpu_and_map_checks_gate_web_and_release_display(workspace, monkeypatch, failed_stage):
    events = []
    @contextmanager
    def display(gpu):
        events.append("open display")
        try:
            yield {"DISPLAY": ":99"}
        finally:
            events.append("close display")
    calls = []
    def stage(title, cmd, *, env):
        assert env["VALORANT_LAB_GPU_REQUIRED"] == "1"
        current = len(calls)
        calls.append(cmd)
        if current == failed_stage:
            raise quick.StartError("check failed")
    monkeypatch.setattr(quick, "display_environment", display)
    monkeypatch.setattr(quick, "run_stage", stage)
    monkeypatch.setattr(quick, "serve", lambda *args: events.append("web"))
    if failed_stage is None:
        quick.runtime(quick.options([]))
        assert events == ["open display", "web", "close display"]
    else:
        with pytest.raises(quick.StartError):
            quick.runtime(quick.options([]))
        assert events == ["open display", "close display"]


def test_check_only_does_not_start_web(workspace, monkeypatch):
    @contextmanager
    def display(gpu):
        yield {}
    monkeypatch.setattr(quick, "display_environment", display)
    monkeypatch.setattr(quick, "run_stage", lambda *a, **kw: None)
    monkeypatch.setattr(quick, "serve", lambda *a: pytest.fail("check-only started web"))
    quick.runtime(quick.options(["--check-only"]))
    assert not (workspace / "runs/web/access-token").exists()


def test_existing_gpu_display_is_not_owned_or_stopped(monkeypatch):
    fake = SimpleNamespace(graphics_check=lambda: {"ok": True})
    monkeypatch.setattr(quick, "cloud_module", lambda: fake)
    monkeypatch.setenv("DISPLAY", ":77")
    with pytest.raises(ValueError):
        with quick.display_environment(0) as env:
            assert env["DISPLAY"] == ":77"
            raise ValueError("web failed")
    assert os.environ["DISPLAY"] == ":77"


@pytest.mark.skipif(sys.platform != "linux", reason="sudo-helper pipe lifecycle")
@pytest.mark.parametrize("worker_output", ["ready", "early_exit"])
def test_sudo_helper_stops_on_parent_pipe_close(workspace, monkeypatch, worker_output):
    flag = workspace / "worker-exited"
    fake_cloud = SimpleNamespace(graphics_check=lambda: {"ok": False})
    monkeypatch.setattr(quick, "cloud_module", lambda: fake_cloud)
    monkeypatch.setattr(quick.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(quick.subprocess, "run", lambda cmd, **kw: SimpleNamespace(returncode=0))
    real_popen = subprocess.Popen
    children = []
    def spawn(command, **kwargs):
        assert command[:2] == ["sudo", "-n"]
        assert "--display-worker" in command
        code = "import sys; from pathlib import Path; "
        if worker_output == "ready":
            code += "print('GPU display ready', flush=True); "
            code += "print('LAB_GPU_ENV {\"DISPLAY\":\":99\",\"XAUTHORITY\":\"/test/auth\"}', flush=True); "
            code += "sys.stdin.buffer.read(1); "
        code += f"Path({str(flag)!r}).write_text('closed')"
        child = real_popen([sys.executable, "-u", "-c", code], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(quick.subprocess, "Popen", spawn)
    if worker_output == "ready":
        with pytest.raises(ValueError, match="operation failed"):
            with quick.display_environment(0) as env:
                assert env["DISPLAY"] == ":99" and env["XAUTHORITY"] == "/test/auth"
                raise ValueError("operation failed")
    else:
        with pytest.raises(quick.StartError):
            with quick.display_environment(0):
                pytest.fail("dead worker was accepted")
    assert flag.read_text() == "closed"
    assert children[0].poll() == 0


@pytest.mark.skipif(sys.platform != "linux", reason="owned process group")
def test_failed_command_logs_and_stops_pipeline(workspace):
    with pytest.raises(quick.StartError, match="失败"):
        quick.run_stage("fixture failure", [sys.executable, "-c", "print('first error'); raise SystemExit(3)"])
    assert "first error" in (workspace / "runs/quickstart/setup.log").read_text()


def test_http_readiness_ignores_proxy_and_requires_expected_page(workspace, monkeypatch):
    import http.server
    import threading
    page = workspace / "src/valorant_lab/web_static/index.html"
    page.parent.mkdir(parents=True)
    page.write_bytes(b"test lab page")
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(page.read_bytes())
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    try:
        quick.wait_for_web(SimpleNamespace(poll=lambda: None), server.server_port, timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("terminal", [True, False])
def test_web_token_is_only_shown_after_readiness_not_written_to_logs(workspace, monkeypatch, capsys, terminal):
    from valorant_lab import web_control
    monkeypatch.setattr(web_control, "token_file", lambda path: "private-fixture-token")
    monkeypatch.setattr(quick, "require_free_port", lambda port: None)
    monkeypatch.setattr(quick.sys.stdout, "isatty", lambda: terminal)
    calls = []
    monkeypatch.setattr(quick, "wait_for_web", lambda *a: calls.append("ready"))
    class Process:
        result = None
        def wait(self):
            self.result = 0
            return 0
        def poll(self):
            return self.result
    monkeypatch.setattr(quick.subprocess, "Popen", lambda *a, **kw: Process())
    quick.serve(quick.options(["--ssh-target", "my-alias"]), {})
    assert calls == ["ready"]
    assert ("private-fixture-token" in capsys.readouterr().out) is terminal
    for name in ("web.log", "connection.txt"):
        assert "private-fixture-token" not in (workspace / "runs/quickstart" / name).read_text()
