import base64
import json
import socketserver
import threading
from pathlib import Path
import numpy as np
import pytest
from valorant_lab.env import ValorantNavigationEnv, NOOP, ROOT, validate_assets


@pytest.fixture
def fake_engine():
    calls = []
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while line := self.rfile.readline():
                request = json.loads(line)
                calls.append(request)
                image = np.zeros((90, 160, 3), dtype=np.uint8)
                image[:, :, 0] = 17
                response = {"id": request["id"], "ok": True, "protocol": 1,
                    "rgb": base64.b64encode(image.tobytes()).decode(), "width": 160, "height": 90,
                    "reward": 1.5, "terminated": request["command"] == "step", "truncated": False,
                    "info": {"position": [123, 456, 789], "is_success": True}}
                self.wfile.write(json.dumps(response).encode() + b"\n")
    server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], calls
    server.shutdown()
    server.server_close()
    thread.join(2)


def test_pixel_only_observation_and_episode_boundary(fake_engine):
    port, calls = fake_engine
    with ValorantNavigationEnv(port=port) as env:
        with pytest.raises(RuntimeError, match="reset"):
            env.step(NOOP)
        image, info = env.reset(seed=42)
        assert image.shape == (90, 160, 3)
        assert image.dtype == np.uint8
        assert np.all(image[:, :, 0] == 17)
        assert info["position"] == [123, 456, 789]
        assert env.observation_space.contains(image)
        _, reward, terminated, truncated, _ = env.step(NOOP)
        assert reward == 1.5 and terminated and not truncated
        with pytest.raises(RuntimeError, match="reset"):
            env.step(NOOP)
        env.reset(seed=42)
        assert calls[0]["seed"] == calls[2]["seed"]
        assert calls[1]["action"] == NOOP.tolist()


@pytest.mark.parametrize("bad", [[0] * 7, [0, 0, 0, 5, 1, 0, 0, 0], [0.5] * 8, [-1] * 8])
def test_invalid_actions_never_reach_engine(fake_engine, bad):
    port, calls = fake_engine
    with ValorantNavigationEnv(port=port) as env:
        env.reset(seed=1)
        with pytest.raises(ValueError):
            env.step(bad)
        assert len(calls) == 1


def test_missing_real_map_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="real map"):
        validate_assets(tmp_path, "lotus")


def test_runtime_and_map_cannot_silently_use_old_version(tmp_path, monkeypatch):
    from valorant_lab import env as module
    with pytest.raises(ValueError, match="Engine version"):
        module.validate_runtime({})
    directory = tmp_path / "maps/lotus"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({"levels": ["geometry"]}))
    (directory / "geometry.glb").write_bytes(bytes(4096))
    (directory / "collision.json").write_text("old collision")
    (tmp_path / "dependencies.lock.json").write_text(json.dumps({
        "assets": {"map": "lotus", "files": {"maps/lotus/collision.json": "0" * 64}}}))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="Map version"):
        module.validate_assets(tmp_path, "lotus")


def test_incomplete_and_unsafe_map_manifest(tmp_path):
    directory = tmp_path / "maps/lotus"
    directory.mkdir(parents=True)
    (directory / "collision.json").write_text("[]")
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"levels": ["missing"]}))
    with pytest.raises(FileNotFoundError):
        validate_assets(tmp_path, "lotus")
    manifest.write_text(json.dumps({"levels": ["../escape"]}))
    with pytest.raises(ValueError):
        validate_assets(tmp_path, "lotus")


def test_renderer_bytes_are_validated(fake_engine):
    port, _ = fake_engine
    with ValorantNavigationEnv(port=port) as env:
        with pytest.raises(ValueError, match="byte count"):
            env._observation({"protocol": 1, "width": 160, "height": 90, "rgb": "AA=="})


def test_render_returns_a_copy(fake_engine):
    port, _ = fake_engine
    with ValorantNavigationEnv(port=port) as env:
        env.reset()
        frame = env.render()
        frame[:] = 255
        assert env.render()[0, 0, 0] == 17
