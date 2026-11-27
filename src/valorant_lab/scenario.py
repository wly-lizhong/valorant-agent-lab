"""Versioned observation/action contract shared by collection, BC, PPO and evaluation."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

FRAME_STACK = 4
LEGACY_YAW = [-15, -5, 0, 5, 15]
LEGACY_PITCH = [-5, 0, 5]
FINE_LOOK = [-12, -4, -1, -0.25, 0, 0.25, 1, 4, 12]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_scenario(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_scenario(config)
    return config


def validate_scenario(c: dict) -> None:
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    def point(p):
        return isinstance(p, list) and len(p) == 3 and all(finite(x) for x in p)
    for key in ("id", "map"):
        if not isinstance(c.get(key), str) or not c[key]:
            raise ValueError(f"Missing scenario {key}")
    if c.get("mode", "navigation") not in ("navigation", "move_combat"):
        raise ValueError("Unknown scenario mode")
    if c.get("completion_rule", "ordered_waypoints") not in ("ordered_waypoints", "goal_and_targets"):
        raise ValueError("Unknown completion_rule")
    if not all(point(c.get(k)) for k in ("start", "goal")):
        raise ValueError("start and goal must be finite 3D coordinates")
    for key, default in (("yaw_bins", LEGACY_YAW), ("pitch_bins", LEGACY_PITCH)):
        bins = c.get(key, default)
        if (not isinstance(bins, list) or not 3 <= len(bins) <= 31 or bins.count(0) != 1
                or not all(finite(v) and abs(v) <= 45 for v in bins)
                or any(a >= b for a, b in zip(bins, bins[1:]))):
            raise ValueError(f"Invalid {key}: sorted, unique, finite bins including zero required")
    limits = {"physics_hz": (60, 60), "action_repeat": (1, 30), "max_steps": (1, 100000),
              "width": (64, 1024), "height": (64, 1024), "render_width": (64, 1920), "render_height": (64, 1080)}
    for key, (lo, hi) in limits.items():
        if type(c.get(key)) is not int or not lo <= c[key] <= hi:
            raise ValueError(f"Invalid {key}")
    if c["width"] > c["render_width"] or c["height"] > c["render_height"]:
        raise ValueError("Render dimensions must cover the observation")
    for key, default, lo, hi in (("goal_radius", .8, .01, 10), ("yaw_jitter", 5, 0, 180),
                                ("start_jitter", 0, 0, 2), ("start_pitch", 0, -85, 85),
                                ("enemy_probability", 1, 0, 1)):
        value = c.get(key, default)
        if not finite(value) or not lo <= value <= hi:
            raise ValueError(f"Invalid {key}")
    if not finite(c.get("start_yaw")):
        raise ValueError("Invalid start_yaw")
    for key, hi in (("waypoints", 32), ("enemies", 8)):
        if not isinstance(c.get(key, []), list) or len(c.get(key, [])) > hi or not all(point(p) for p in c.get(key, [])):
            raise ValueError(f"Invalid {key}")
    for key, default, hi in (("enemy_tier", 0, 4), ("initial_ammo", 25, 25)):
        if type(c.get(key, default)) is not int or not 0 <= c.get(key, default) <= hi:
            raise ValueError(f"Invalid {key}")
    if c.get("mode") == "move_combat":
        if c.get("goal_marker", True) or not c.get("player_hud") or not c.get("waypoints") or not c.get("family"):
            raise ValueError("move_combat needs unmarked waypoints, family and player_hud")


def action_sizes(c: dict) -> np.ndarray:
    return np.array([3, 3, 3, len(c.get("yaw_bins", LEGACY_YAW)),
                     len(c.get("pitch_bins", LEGACY_PITCH)), 2, 2, 2], dtype=np.int64)


def noop(c: dict) -> np.ndarray:
    return np.array([0, 0, 0, c.get("yaw_bins", LEGACY_YAW).index(0),
                     c.get("pitch_bins", LEGACY_PITCH).index(0), 0, 0, 0], dtype=np.int64)


def contract(c: dict) -> dict:
    # Enemy/start variants may differ within an explicitly shared task family.
    # Route, timing, pixels, HUD, map, weapon and action meanings must match.
    result = {"version": 3, "engine": "ca-val-aim-v2.15.0-lab-v3", "map": c["map"],
            "mode": c.get("mode", "navigation"), "family": c.get("family", "navigation-selftest"),
            "task": c.get("task", "Reach the fixed destination."), "waypoints": c.get("waypoints", []), "goal": c["goal"],
            "shape": [c["height"], c["width"], 3], "stack": FRAME_STACK,
            "physics_hz": c["physics_hz"], "action_repeat": c["action_repeat"],
            "yaw_bins": c.get("yaw_bins", LEGACY_YAW), "pitch_bins": c.get("pitch_bins", LEGACY_PITCH),
            "action_sizes": action_sizes(c).tolist(), "player_hud": c.get("player_hud", False),
            "goal_marker": c.get("goal_marker", True), "weapon": "Vandal"}
    # Keep legacy contracts readable, but never silently mix the changed success rule.
    if c.get("completion_rule", "ordered_waypoints") != "ordered_waypoints":
        result.update(completion_rule=c["completion_rule"], goal_radius=c.get("goal_radius", .8))
    return result


def model_metadata_path(model: str | Path) -> Path:
    p = Path(model)
    return p.with_suffix(".json")


def require_contract(model: str | Path, config: dict) -> dict:
    path = model_metadata_path(model)
    if not path.is_file():
        raise ValueError(f"Missing model contract: {path}. Legacy checkpoints must be retrained for v2.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("contract") != contract(config):
        raise ValueError("Model/scenario contract mismatch (map, family, pixels, controls or timing)")
    return metadata
