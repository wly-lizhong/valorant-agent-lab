"""Rendered regression for forgiving route goals. Privileged test control, no human data."""
import json
from pathlib import Path
import uuid
from PIL import Image
from valorant_lab.env import ROOT, ValorantEnv
from valorant_lab.dataset import write_json
from valorant_lab.scenario import load_scenario
from validate_stage1 import steer


def main():
    output = ROOT / "runs" / ("route-check-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    base = load_scenario(ROOT / "configs/lotus_route.json")
    results = []
    cases = [
        ("goal_without_checkpoints", {"start": base["goal"]}, True, 5),
        ("legacy_still_ordered", {"start": base["goal"], "completion_rule": "ordered_waypoints"}, False, 0),
        ("goal_with_enemy_remaining", {"start": base["goal"], "enemies": [[-14, 0, 18]], "enemy_probability": 1, "enemy_fire": False}, False, 5),
        ("skip_earlier_checkpoint", {"start": base["waypoints"][2]}, False, 3),
    ]
    for name, changes, expected, waypoint in cases:
        config = {**base, "start_jitter": 0, "yaw_jitter": 0, **changes}
        path = output / (name + ".json")
        write_json(path, config)
        with ValorantEnv(path) as env:
            env.reset(options={"engine_seed": 42})
            frame, _, done, truncated, info = env.step(env.noop)
            assert info["is_success"] == expected and done == expected and not truncated, info
            assert info["waypoint"] == waypoint, info
            Image.fromarray(frame).save(output / (name + ".png"))
            results.append({"case": name, "info": info})
            print(name, "passed", flush=True)
    with ValorantEnv(ROOT / "configs/lotus_route.json") as env:
        for seed in (42, 43, 100142):
            _, info = env.reset(options={"engine_seed": seed})
            for _ in range(base["max_steps"]):
                target = base["waypoints"][min(info["waypoint"], len(base["waypoints"]) - 1)]
                _, _, done, truncated, info = env.step(steer(env, info, target, move=True))
                if done or truncated:
                    break
            assert info["is_success"], info
            results.append({"case": "full_route", "seed": seed, "info": info})
            print("full_route", seed, "passed", info["step"], flush=True)
    write_json(output / "checks.json", {"controller": "privileged_test_not_a_model", "checks": results})
    print(json.dumps({"passed": len(results), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
