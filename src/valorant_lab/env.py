from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .protocol import RpcClient
from .scenario import load_scenario, action_sizes, noop

ROOT = Path(__file__).resolve().parents[2]
ACTION_SIZES = np.array([3, 3, 3, 5, 3, 2, 2, 2], dtype=np.int64)
NOOP = np.array([0, 0, 0, 2, 1, 0, 0, 0], dtype=np.int64)


def validate_assets(asset_root: Path, map_key: str) -> dict:
    directory = asset_root / "maps" / map_key
    manifest = directory / "manifest.json"
    collision = directory / "collision.json"
    if not manifest.is_file() or not collision.is_file():
        raise FileNotFoundError(f"Missing real map assets in {directory}. See docs/assets.md")
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    levels = data.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Map manifest has no levels")
    for level in levels:
        if not isinstance(level, str) or Path(level).name != level or "/" in level or "\\" in level:
            raise ValueError("Invalid map level name")
        path = directory / (level + ".glb")
        if not path.is_file() or path.stat().st_size < 4096:
            raise FileNotFoundError(f"Missing/empty map geometry: {path}")
    lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
    pinned = lock.get("assets", {})
    if pinned.get("map") == map_key:
        for relative, expected in pinned["files"].items():
            target = (asset_root / relative).resolve()
            if not target.is_relative_to(asset_root.resolve()) or not target.is_file():
                raise ValueError("Missing pinned map file: " + relative)
            h = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Map version mismatch; install the current bundled assets: " + relative)
    return {"map": map_key, "levels": len(levels), "directory": str(directory)}


def validate_runtime(local: dict):
    expected = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))["upstream"]["sha256"]
    if local.get("upstream_sha256") != expected:
        raise ValueError("Engine version changed; rerun bootstrap/start_windows with the current source and map bundle")


class ValorantNavigationEnv(gym.Env):
    """Gymnasium RGB observations. Ground-truth diagnostics are info-only, never policy inputs."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, scenario: str | Path | None = None, *, root: str | Path = ROOT,
                 render_mode: str | None = None, port: int | None = None,
                 startup_timeout: float = 180, timeout: float = 60, realtime: bool = False):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array is supported; the engine window is controlled by its renderer")
        self.root = Path(root).resolve()
        self.scenario_path = Path(scenario).resolve() if scenario else self.root / "configs/lotus_navigation.json"
        self.config = load_scenario(self.scenario_path)
        self.realtime = realtime
        self.last_engine_seed = None
        self.engine_metadata = {}
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete(action_sizes(self.config))
        self.noop = noop(self.config)
        self.observation_space = spaces.Box(0, 255,
            (self.config["height"], self.config["width"], 3), dtype=np.uint8)
        self.metadata = {**type(self).metadata,
            "render_fps": self.config["physics_hz"] / self.config["action_repeat"]}
        self.process = None
        self.client = None
        self.log = None
        self.run_dir = None
        self.last_frame = None
        self.needs_reset = True
        try:
            if port is None:
                port = self._launch(startup_timeout)
            self.client = RpcClient(port, timeout)
        except BaseException:
            self.close()
            raise

    def _launch(self, startup_timeout: float) -> int:
        local_path = self.root / "local.json"
        platform_config = self.root / ("local.windows.json" if os.name == "nt" else "local.linux.json")
        if platform_config.is_file():
            local_path = platform_config
        if not local_path.is_file():
            raise FileNotFoundError("Run python scripts/bootstrap.py first; local.json is missing")
        local = json.loads(local_path.read_text(encoding="utf-8"))
        validate_runtime(local)
        map_key = self.config["map"]
        if self.config.get("require_real_map", True):
            if not local.get("maps_dir"):
                raise ValueError("No maps_dir configured. Real-map training never falls back to a blockout.")
            validate_assets(Path(local["maps_dir"]), map_key)
        self.run_dir = self.root / "runs" / ("engine-" + uuid.uuid4().hex[:12])
        self.run_dir.mkdir(parents=True)
        ready = self.run_dir / "ready.json"
        self.log = (self.run_dir / "engine.log").open("wb")
        env = os.environ.copy()
        dotnet = Path(local["dotnet"]).parent
        env.update(DOTNET_ROOT=str(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1",
                   PATH=str(dotnet) + os.pathsep + env.get("PATH", ""))
        command = [local["godot"], "--path", local["project"], "--rendering-method", "gl_compatibility",
            "--audio-driver", "Dummy", "--resolution", f'{self.config["render_width"]}x{self.config["render_height"]}',
            "--", "--dev", "--data-dir", str(self.run_dir / "data"),
            "--agent-config", str(self.scenario_path), "--agent-port", "0", "--agent-ready", str(ready)]
        if local.get("maps_dir"):
            command.extend(["--maps-dir", local["maps_dir"]])
        if self.realtime:
            command.append("--agent-realtime")
        # A rendering display is required. Server entry scripts provide Xvfb when DISPLAY is absent.
        self.process = subprocess.Popen(command, env=env, cwd=self.root,
            stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise RuntimeError(f"Engine exited ({code}); see {self.run_dir / 'engine.log'}")
            if ready.is_file():
                try:
                    data = json.loads(ready.read_text())
                    if data["protocol"] != 1 or data["scenario"] != self.config["id"]:
                        raise RuntimeError("Engine protocol/scenario mismatch")
                    if data.get("revision", 0) < 2 or data.get("action_sizes") != self.action_space.nvec.tolist():
                        raise RuntimeError("Engine overlay is outdated; rerun scripts/bootstrap.py")
                    if data.get("clock") != ("realtime" if self.realtime else "lockstep"):
                        raise RuntimeError("Engine clock mismatch")
                    if data.get("completion_rule", "ordered_waypoints") != self.config.get("completion_rule", "ordered_waypoints"):
                        raise RuntimeError("Engine completion rule is outdated; rerun scripts/bootstrap.py")
                    self.engine_metadata = {k: data.get(k) for k in (
                        "revision", "render_device", "render_vendor", "clock", "visual_revision")}
                    if os.environ.get("VALORANT_LAB_GPU_REQUIRED") == "1":
                        from .cloud import SOFTWARE
                        renderer = (str(data.get("render_vendor", "")) + " " + str(data.get("render_device", ""))).lower()
                        if "nvidia" not in renderer or any(word in renderer for word in SOFTWARE):
                            raise RuntimeError("The game did not report NVIDIA rendering; rebuild the overlay and check GPU graphics configuration")
                    return int(data["port"])
                except json.JSONDecodeError:
                    pass  # Atomic readiness is not assumed while file is being written.
            time.sleep(0.1)
        raise TimeoutError(f"Engine startup timed out; see {self.run_dir / 'engine.log'}")

    def _observation(self, response: dict) -> np.ndarray:
        if response.get("protocol") != 1:
            raise ValueError("Unsupported engine protocol")
        shape = (response["height"], response["width"], 3)
        if shape != self.observation_space.shape:
            raise ValueError("Engine image dimensions differ from observation space")
        pixels = base64.b64decode(response["rgb"], validate=True)
        if len(pixels) != int(np.prod(shape)):
            raise ValueError("Engine image byte count mismatch")
        self.last_frame = np.frombuffer(pixels, dtype=np.uint8).reshape(shape).copy()
        return self.last_frame.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"engine_seed"}:
            raise ValueError("Only engine_seed reset option is supported")
        if self.client is None:
            raise RuntimeError("Environment is closed")
        self.needs_reset = True
        # PPO samples only the training seed bank. Evaluation banks are selected explicitly.
        engine_seed = options.get("engine_seed", int(self.np_random.integers(0, 100000)))
        if type(engine_seed) is not int or not 0 <= engine_seed < 2**31:
            raise ValueError("engine_seed must be an integer in [0, 2**31)")
        self.last_engine_seed = engine_seed
        response = self.client.call("reset", seed=engine_seed)
        observation = self._observation(response)
        self.needs_reset = False
        return observation, response["info"]

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset before step / after episode end")
        array = np.asarray(action)
        if array.shape != (8,) or array.dtype.kind not in "iu" or not self.action_space.contains(array):
            raise ValueError("Action must be eight integers in the documented MultiDiscrete ranges")
        try:
            response = self.client.call("step", action=array.tolist())
            observation = self._observation(response)
        except BaseException:
            self.needs_reset = True
            raise
        terminated, truncated = bool(response["terminated"]), bool(response["truncated"])
        self.needs_reset = terminated or truncated
        return observation, float(response["reward"]), terminated, truncated, response["info"]

    def render(self):
        return None if self.last_frame is None else self.last_frame.copy()

    def close(self):
        if self.client is not None:
            try:
                # Only stop engines launched by this env; attached clients don't own the process.
                if self.process is not None and self.process.poll() is None:
                    self.client.socket.settimeout(2)
                    self.client.call("close")
            except (OSError, RuntimeError, ValueError):
                pass
            finally:
                self.client.close()
                self.client = None
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.needs_reset = True


# The original name is kept for existing scripts; the class now supports both task modes.
ValorantEnv = ValorantNavigationEnv
