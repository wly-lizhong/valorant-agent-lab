"""Rendered integration checks. Uses privileged geometry ONLY as a test controller.

Produced demonstrations are tagged script_test and rejected by ordinary BC training.
This validates mechanisms/data flow, never learned gameplay or human demonstration quality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from valorant_lab.dataset import EpisodeWriter, write_json
from valorant_lab.env import ROOT, ValorantEnv
from valorant_lab.scenario import load_scenario


def steer(env, info, target, *, fire=False, move=False):
    position = np.array(info["position"], dtype=float)
    if fire:
        position[1] += 1.62
    delta = np.array(target) - position
    yaw = np.degrees(np.arctan2(delta[0], -delta[2]))
    yaw_error = (yaw - info.get("view_yaw", info["yaw"]) + 180) % 360 - 180
    pitch = np.degrees(np.arctan2(delta[1], np.linalg.norm(delta[[0, 2]]))) if fire else 0
    pitch_error = pitch - info.get("view_pitch", info["pitch"])
    action = env.noop.copy()
    action[3] = np.argmin(abs(np.array(env.config["yaw_bins"]) - yaw_error))
    action[4] = np.argmin(abs(np.array(env.config["pitch_bins"]) - pitch_error))
    action[0] = int(move and abs(yaw_error) < 25)
    action[6] = int(fire and abs(yaw_error) < 1.5 and abs(pitch_error) < 1.5 and info.get("speed", 0) < .1)
    action[7] = int(info["ammo"] == 0)
    return action


def check_enemy_damage(output, config):
    """An exposed, stationary player must take damage, die, then reset cleanly."""
    damage_config = {**config, "start": [-17.26, 0, 26.57], "start_jitter": 0,
                     "yaw_jitter": 0, "enemy_probability": 1, "enemy_fire": True,
                     "enemy_tier": 4, "initial_ammo": 25, "max_steps": 400}
    path = output / "damage-scenario.json"
    write_json(path, damage_config)
    with ValorantEnv(path) as env:
        _, initial = env.reset(options={"engine_seed": 42})
        for _ in range(damage_config["max_steps"]):
            frame, _, done, truncated, final = env.step(env.noop)
            if done or truncated:
                break
        assert done and final["termination_reason"] == "death", final
        assert final["hp"] <= 0 and not final["is_success"]
        Image.fromarray(frame).save(output / "damage-death.png")
        _, reset = env.reset(options={"engine_seed": 42})
        assert reset["hp"] == initial["hp"] == 100 and reset["shield"] == initial["shield"] == 50
        assert reset["ammo"] == 25 and reset["shots"] == reset["kills"] == reset["waypoint"] == 0
    result = {"death": final, "reset": reset}
    write_json(output / "damage-check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/stage1-validation")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    config.update(start_jitter=0, yaw_jitter=0, enemy_probability=1, enemy_fire=False, initial_ammo=1, max_steps=400)
    scenario = output / "mechanics-scenario.json"
    write_json(scenario, config)
    records = []
    with ValorantEnv(scenario) as env:
        for split, seed in (("train", 42), ("validation", 100042)):
            frame, info = env.reset(options={"engine_seed": seed})
            assert info["shots"] == info["kills"] == info["waypoint"] == 0
            assert info["ammo"] == 1 and info["enemies_spawned"] == 1 and info["hp"] == 100
            writer = EpisodeWriter(output / "pipeline-data", config, seed, source="script_test", split=split)
            Image.fromarray(frame).save(output / f"{split}-start.png")
            killed_at = None
            saw_reload = False
            for step in range(config["max_steps"]):
                firing = info["waypoint"] >= 1 and info["kills"] < info["enemies_spawned"]
                if firing:
                    target = np.array(config["enemies"][0], dtype=float)
                    target[1] += 1.1
                else:
                    target = config["waypoints"][min(info["waypoint"], len(config["waypoints"]) - 1)]
                action = steer(env, info, target, fire=firing, move=not firing)
                at = time.monotonic()
                after_frame, _, done, truncated, after = env.step(action)
                writer.append(frame, action, before=info, after=after,
                              timing={"input_sampled": at, "response_received": time.monotonic()})
                frame, info = after_frame, after
                saw_reload |= info["reloading"]
                if info["kills"] and killed_at is None:
                    killed_at = info["waypoint"]
                    Image.fromarray(frame).save(output / f"{split}-kill.png")
                if step % 40 == 0:
                    print(split, step, "waypoint", info["waypoint"], "hits", info["hits"],
                          "ammo", info["ammo"], "position", info["position"], flush=True)
                if done or truncated:
                    break
            writer.finish(info, accepted=info["is_success"])
            assert info["is_success"], info
            assert killed_at is not None and info["waypoint"] > killed_at, "Must continue the route after the kill"
            assert saw_reload and info["kills"] == 1 and info["hits"] > 0
            Image.fromarray(frame).save(output / f"{split}-finish.png")
            records.append({"split": split, "seed": seed, "reloaded": saw_reload, "kill_waypoint": killed_at, "final": info})
        # Check pitch signs and reset cleanup through the same action interface.
        _, info = env.reset(options={"engine_seed": 42})
        action = env.noop.copy(); action[4] = 5
        _, _, _, _, looked = env.step(action)
        assert .24 < looked["pitch"] < .26
        _, reset = env.reset(options={"engine_seed": 42})
        assert reset["pitch"] == 0 and reset["kills"] == 0 and reset["ammo"] == 1
    records.append({"enemy_damage": check_enemy_damage(output, config)})
    realtime_config = {**config, "enemy_probability": 0, "max_steps": 80}
    realtime_path = output / "realtime-scenario.json"
    write_json(realtime_path, realtime_config)
    with ValorantEnv(realtime_path, realtime=True) as env:
        env.reset(options={"engine_seed": 42})
        action = env.noop.copy(); action[0] = 1
        _, _, done, truncated, first = env.step(action)
        assert not (done or truncated)
        time.sleep(.6)
        _, _, _, _, later = env.step(env.noop)
        assert later["simulation_time"] - first["simulation_time"] > .3, "Realtime world stalled during client delay"
        assert later["watchdog_releases"] >= 1, "Held controls must expire"
        records.append({"realtime_probe": later})
    write_json(output / "checks.json", {"checks": records, "controller": "privileged_script_test_not_a_model"})
    print("Rendered integration checks passed; generated data is SCRIPT TEST ONLY.", flush=True)


if __name__ == "__main__":
    main()
