import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
import numpy as np
import pytest

pytest.importorskip("aiohttp")
from aiohttp import CookieJar, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer
from valorant_lab.web_control import Settings, HumanSession, Catalog, create_app, token_file, shutdown_controller
from valorant_lab.scenario import load_scenario, noop
from valorant_lab.dataset import load_episodes
from valorant_lab.corrections import enqueue_failure, load_case


class FakeEnv:
    """Finite synthetic renderer; tests never launch Godot or generate production data."""
    instances = []

    def __init__(self, path):
        self.config = load_scenario(path)
        self.noop = noop(self.config)
        self.engine_metadata = {"render_device": "synthetic test"}
        self.closed = False
        self.actions = []
        self.instances.append(self)

    def frame(self, value):
        return np.full((self.config["height"], self.config["width"], 3), value, np.uint8)

    def reset(self, *, options):
        self.seed = options["engine_seed"]
        return self.frame(13), {"clock": "lockstep", "step": 0, "position": [999, 999, 999]}

    def step(self, action):
        self.actions.append(action.copy())
        return self.frame(99), 1., True, False, {"is_success": True, "step": 1, "action_applied": action.tolist()}

    def close(self):
        self.closed = True


@pytest.fixture
def settings(tmp_path):
    FakeEnv.instances.clear()
    return Settings(output=tmp_path / "data", queue=tmp_path / "queue", state_dir=tmp_path / "web",
                    source="script_test")


def begin(settings, **request):
    session = HumanSession(settings, FakeEnv)
    packet = session.prepare({"task": "lotus_movement_combat", **request})
    assert packet["state"] == "ready" and "position" not in packet
    assert session.writer is None
    packet = session.start()
    return session, packet


def step(session, packet, *, age=.1, **changes):
    received = session.sent_at + age
    session.acknowledge({"frame_id": packet["frame_id"], "displayed_ms": 1000}, received - .05)
    request = {"frame_id": packet["frame_id"], "sampled_ms": 1050, "keys": ["KeyW", "ControlLeft"],
               "fire": True, "reload": True, "focused": True, "dx": 12.5, "dy": -12.5, **changes}
    return session.step(request, received)


def test_browser_labels_pre_action_pixels_and_requires_confirmation(settings):
    session, packet = begin(settings, split="validation", seed=42)
    env = session.env
    result = step(session, packet)
    assert env.seed == 100042 and env.closed
    assert env.actions[0].tolist() == [1, 0, 2, 6, 6, 0, 1, 1]
    assert result["state"] == "review"
    assert load_episodes(settings.output, session.config) == []
    saved = session.save()
    assert saved["accepted"]
    # Synthetic tests can never be silently treated as human training data.
    with pytest.raises(ValueError, match="Unusable"):
        load_episodes(settings.output, session.config)
    [(directory, metadata)] = load_episodes(settings.output, session.config, allow_test_data=True)
    chunk = np.load(directory / metadata["chunks"][0]["file"])
    assert np.all(chunk["frames"] == 13)  # never the post-action frame (99)
    assert metadata["collection"]["transport"] == "browser_png_lockstep_v1"
    timing = json.loads((directory / "diagnostics.jsonl").read_text())["timing"]
    assert timing["browser_input_ms"] - timing["browser_displayed_ms"] == 50
    assert timing["frame_age_ms"] == pytest.approx(100)
    assert "position" not in metadata["contract"]
    session.abort()


@pytest.mark.parametrize("reason", ["late", "unconfirmed", "disconnect"])
def test_unusable_episodes_never_enter_training(settings, reason):
    session, packet = begin(settings)
    if reason != "disconnect":
        step(session, packet, age=.8 if reason == "late" else .1)
    if reason == "late":
        result = session.save()
        assert not result["accepted"] and result["quality_issues"] == ["browser_frame_age_exceeded"]
    else:
        session.abort(reason)
    assert load_episodes(settings.output, session.config, allow_test_data=True) == []
    assert FakeEnv.instances[-1].closed


@pytest.mark.parametrize("changes", [{"frame_id": -1}, {"keys": ["KeyQ"]}, {"focused": False},
                                     {"dx": float("nan")}, {"fire": 1}, {"sampled_ms": 999}])
def test_invalid_or_stale_input_never_steps_engine(settings, changes):
    session, packet = begin(settings)
    try:
        with pytest.raises(ValueError):
            step(session, packet, **changes)
        assert session.env.actions == []
    finally:
        session.abort()


def test_missing_or_duplicate_ack_is_rejected(settings):
    session, packet = begin(settings)
    try:
        with pytest.raises(ValueError, match="acknowledged"):
            session.step({"frame_id": packet["frame_id"]}, time.monotonic())
        ack = {"frame_id": packet["frame_id"], "displayed_ms": 1000}
        session.acknowledge(ack, time.monotonic())
        with pytest.raises(ValueError, match="Duplicate"):
            session.acknowledge(ack, time.monotonic())
        assert session.env.actions == []
    finally:
        session.abort()


def test_correction_preserves_seed_and_marks_only_demonstration(settings):
    config, _, _ = Catalog(settings).resolve("lotus_movement_combat")
    path = enqueue_failure(settings.queue, config, 200042, [{"step": 40}],
                           np.zeros((144, 256, 3), np.uint8), split="dev", model_hash="test")
    task = "case:" + path.parent.name
    assert Catalog(settings).items()[-1]["id"] == task
    with pytest.raises(ValueError, match="never validation"):
        begin(settings, task=task, split="validation")
    session, packet = begin(settings, task=task)
    assert session.env.seed == 200042
    step(session, packet)
    session.save()
    assert load_case(path)["status"] == "demonstrated"
    assert session.writer.manifest["case_id"] == path.parent.name
    session.abort()


@pytest.mark.parametrize("task", ["../../private.json", "case:../secret", [], None])
def test_browser_catalog_accepts_only_named_tasks(settings, task):
    with pytest.raises(ValueError):
        Catalog(settings).resolve(task)


def test_token_persists_and_is_private(tmp_path):
    path = tmp_path / "state" / "token"
    token = token_file(path)
    assert len(token) >= 32 and token == token_file(path)
    import os
    if os.name != "nt":
        assert path.stat().st_mode & 0o077 == 0
        path.chmod(0o644)
        with pytest.raises(ValueError, match="permissions"):
            token_file(path)


def test_web_auth_origin_lease_and_disconnect_cleanup(settings):
    async def check():
        origin = "http://127.0.0.1:8765"
        headers = {"Host": "127.0.0.1:8765", "Origin": origin}
        app = create_app(settings, token="t" * 40, origins=[origin], env_factory=FakeEnv)
        async with TestClient(TestServer(app), cookie_jar=CookieJar(unsafe=True)) as client:
            response = await client.get("/", headers=headers)
            assert response.status == 200 and "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            assert "人工示范台" in await response.text()
            response = await client.get("/static/app.js", headers=headers)
            assert response.status == 200
            assert (await client.get("/api/catalog", headers=headers)).status == 401
            assert (await client.post("/api/login", headers={**headers, "Origin": "http://evil.test"}, json={"token": "t" * 40})).status == 403
            assert (await client.post("/api/login", headers=headers, json={"token": "wrong"})).status == 401
            response = await client.post("/api/login", headers=headers, json={"token": "t" * 40})
            assert response.status == 200
            assert response.cookies["agent_session"]["httponly"]
            assert (await client.get("/api/catalog", headers={**headers, "Host": "evil.test"})).status == 403
            assert (await client.get("/api/catalog", headers=headers)).status == 200
            ws = await client.ws_connect("/ws", headers=headers)
            assert (await ws.receive_json())["type"] == "connected"
            with pytest.raises(WSServerHandshakeError) as exc:
                await client.ws_connect("/ws", headers=headers)
            assert exc.value.status == 409
            await ws.send_json({"type": "open", "task": "lotus_route"})
            assert (await ws.receive_json())["type"] == "loading"
            assert (await ws.receive_json())["state"] == "ready"
            await ws.send_json({"type": "start"})
            assert (await ws.receive_json())["state"] == "active"
            await ws.close()
            for _ in range(100):
                if FakeEnv.instances[-1].closed:
                    break
                await asyncio.sleep(.01)
            assert FakeEnv.instances[-1].closed
            assert not json.loads(next(settings.output.glob("*/episode.json")).read_text())["accepted"]
            await asyncio.sleep(.02)
            ws2 = await client.ws_connect("/ws", headers=headers)
            await ws2.close()
    asyncio.run(check())


def test_web_control_timeout_closes_own_engine(settings):
    settings.control_timeout = .05
    async def check():
        headers = {"Host": "localhost:8765", "Origin": "http://localhost:8765"}
        app = create_app(settings, token="t" * 40, origins=[headers["Origin"]], env_factory=FakeEnv)
        async with TestClient(TestServer(app), cookie_jar=CookieJar(unsafe=True)) as client:
            await client.post("/api/login", headers=headers, json={"token": "t" * 40})
            ws = await client.ws_connect("/ws", headers=headers)
            await ws.receive_json()
            await ws.send_json({"type": "open", "task": "lotus_route"})
            await ws.receive_json(); await ws.receive_json()
            await ws.send_json({"type": "start"})
            await ws.receive_json()
            assert (await ws.receive_json())["type"] == "error"
            await ws.receive()
            assert FakeEnv.instances[-1].closed and FakeEnv.instances[-1].actions == []
    asyncio.run(check())


def test_cancelled_request_still_finalizes_episode_and_closes_engine(settings):
    session, _ = begin(settings)
    engine = session.env
    release = threading.Event()
    entered = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    def busy():
        entered.set()
        release.wait(3)
    executor.submit(busy)
    assert entered.wait(1)
    async def check():
        task = asyncio.create_task(shutdown_controller(session, executor))
        await asyncio.sleep(0)
        task.cancel()
        # Give cancellation a turn while cleanup is still queued behind the worker.
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        asyncio.run(check())
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert engine.closed
    metadata = json.loads(next(settings.output.glob("*/episode.json")).read_text())
    assert not metadata["accepted"] and not metadata["complete"]
    assert "connection_closed" in metadata["quality_issues"]
