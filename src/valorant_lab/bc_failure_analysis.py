"""Read-only failure evidence. Diagnostic game state never becomes a policy input."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from .dataset import load_episodes, read_demonstration_chunk, sha256, write_json
from .scenario import action_sizes, load_scenario, noop


def _first(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else None


def _bounds(points):
    if not len(points):
        return None
    points = np.asarray(points, dtype=float)
    return {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}


def trajectory_evidence(rows, config):
    """Use actual positions/counters, not the bridge's current-waypoint distance."""
    if not rows:
        raise ValueError("Empty trajectory")
    actions = np.asarray([r["requested_action"] for r in rows])
    sizes = action_sizes(config)
    if (actions.shape != (len(rows), 8) or actions.dtype.kind not in "iu"
            or np.any(actions < 0) or np.any(actions >= sizes)):
        raise ValueError("Invalid trajectory actions")
    infos = [r["info"] for r in rows]
    if any(i.get("action_applied") != a.tolist() for i, a in zip(infos, actions)):
        raise ValueError("Requested/applied actions disagree")
    positions = np.asarray([i["position"] for i in infos], dtype=float)
    if positions.shape != (len(rows), 3) or not np.isfinite(positions).all():
        raise ValueError("Invalid trajectory positions")
    for key in ("shots", "hits", "kills", "enemies_spawned"):
        values = [i.get(key) for i in infos]
        if (any(type(x) is not int or x < 0 for x in values)
                or any(b < a for a, b in zip(values, values[1:]))):
            raise ValueError("Invalid cumulative counter: " + key)
    movement = np.any(actions[:, :2] != 0, axis=1)
    active = np.any(actions != noop(config), axis=1)
    firing = actions[:, 6] != 0
    goal_distances = np.linalg.norm(positions - np.asarray(config["goal"]), axis=1)
    in_goal = goal_distances <= config.get("goal_radius", .8)
    pending = np.asarray([i["kills"] < i["enemies_spawned"] for i in infos])
    trailing_idle = _first(active[::-1])
    if trailing_idle is None:
        trailing_idle = len(rows)
    flags = []
    if not active.any():
        flags.append("all_noop")
    elif not movement.any():
        flags.append("no_translation_requested")
    step_seconds = config["action_repeat"] / config["physics_hz"]
    if movement.any() and trailing_idle * step_seconds >= 2:
        flags.append("moved_then_idle_at_least_2s")
    if pending.any() and not firing.any():
        flags.append("targets_remained_without_fire_request")
    if np.any(in_goal & pending):
        flags.append("goal_entered_with_live_targets")
    if in_goal.any() and not in_goal[-1]:
        flags.append("left_goal_region")
    return {"steps": len(rows), "success": bool(infos[-1]["is_success"]),
            "first_action_index": _first(active), "first_move_index": _first(movement),
            "first_fire_index": _first(firing), "move_requests": int(movement.sum()),
            "fire_requests": int(firing.sum()), "trailing_noop_steps": trailing_idle,
            "actual_shots": infos[-1]["shots"], "actual_hits": infos[-1]["hits"],
            "actual_kills": infos[-1]["kills"], "pending_target_steps": int(pending.sum()),
            "minimum_final_goal_distance_m": float(goal_distances.min()),
            "first_goal_entry_index": _first(in_goal), "final_position": positions[-1].tolist(),
            "flags": flags, "step_seconds": step_seconds,
            "note": "Positions are post-action. No visibility, aim correctness or collision claim is inferred."}


def demonstration_evidence(path, meta):
    actions = np.concatenate([read_demonstration_chunk(path, meta, i)[1]
                              for i in range(len(meta["chunks"]))])
    rows = [json.loads(line) for line in (path / "diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(actions) or any(row["index"] != i for i, row in enumerate(rows)):
        raise ValueError("Demonstration diagnostics length/order mismatch")
    if any(row["after"].get("action_applied") != a.tolist() for row, a in zip(rows, actions)):
        raise ValueError("Demonstration action/diagnostic mismatch")
    before = [row["before"] for row in rows]
    active = np.any(actions != noop(meta["scenario"]), axis=1)
    firing = actions[:, 6] != 0
    pending = np.asarray([i.get("kills", 0) < i.get("enemies_spawned", 0) for i in before])
    fire_positions = [i["position"] for i, fire in zip(before, firing) if fire]
    first_action = _first(active)
    return {"episode": meta["episode_id"], "split": meta["split"], "engine_seed": meta["engine_seed"],
            "scenario": meta["scenario"]["id"], "case_key": meta["case_key"], "steps": len(actions),
            "leading_noop_steps": len(actions) if first_action is None else first_action,
            "first_action_index": first_action, "first_forward_index": _first(actions[:, 0] == 1),
            "first_fire_index": _first(firing), "fire_requests": int(firing.sum()),
            "forward_while_target_alive": int(((actions[:, 0] == 1) & pending).sum()),
            "strafe_while_target_alive": int(((actions[:, 1] != 0) & pending).sum()),
            "fire_position_bounds": _bounds(fire_positions),
            "actual_shots": meta["final"].get("shots"), "actual_kills": meta["final"].get("kills"),
            "manifest_sha256": sha256(path / "episode.json"),
            "diagnostics_sha256": sha256(path / "diagnostics.jsonl")}


def analyze_failures(data, scenario, report_paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh analysis output; preserve previous evidence")
    config = load_scenario(scenario)
    demos = [demonstration_evidence(path, meta) for path, meta in load_episodes(data, config)]
    games = []
    for report_path in report_paths:
        report_path = Path(report_path).resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["split"] != "dev" or report.get("model_test_only") is not False:
            raise ValueError("Failure analysis only consumes development games from non-diagnostic models")
        for index, episode in enumerate(report["episodes"]):
            if episode.get("training_overlap") is not False:
                raise ValueError("Game overlaps model training cases")
            directory = (report_path.parent / episode.get("artifact_directory", f"episode-{index:04d}")).resolve()
            if not directory.is_relative_to(report_path.parent):
                raise ValueError("Unsafe trajectory path")
            trace = directory / "trace.jsonl"
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            evidence = trajectory_evidence(rows, report["scenario"])
            if rows[-1]["info"] != episode["final"] or evidence["success"] != episode["success"]:
                raise ValueError("Trace/report final result mismatch")
            games.append({"report": str(report_path), "model_sha256": report["model_sha256"],
                          "scenario": report["scenario"]["id"], "engine_seed": episode["engine_seed"],
                          "trace_sha256": sha256(trace), **evidence})
    result = {"schema": "bc-failure-analysis-v1", "demonstrations": demos, "games": games,
              "game_flag_counts": dict(Counter(flag for row in games for flag in row["flags"])),
              "diagnostic_state_is_policy_input": False,
              "notes": ["Development analysis, not final acceptance.",
                        "Idle/stopped behavior is not automatically wall collision.",
                        "Initial human waiting is a hypothesis, not a proven cause of game failure."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
