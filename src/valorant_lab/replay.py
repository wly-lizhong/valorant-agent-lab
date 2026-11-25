"""Re-execute an episode from its seed to check recording alignment and reset behavior."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .dataset import sha256, write_json
from .env import ValorantEnv
from .scenario import contract


def replay(args):
    path = Path(args.episode)
    if path.is_dir():
        path = path / "episode.json"
    demo = json.loads(path.read_text(encoding="utf-8"))
    if demo.get("schema") != "valorant-demo-v1" or demo.get("contract") != contract(demo["scenario"]):
        raise ValueError("Demonstration schema/contract mismatch")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scenario = output / "scenario.json"
    write_json(scenario, demo["scenario"])
    differences = []
    steps = 0
    with ValorantEnv(scenario) as env, (path.parent / "diagnostics.jsonl").open(encoding="utf-8") as diagnostics:
        frame, info = env.reset(options={"engine_seed": demo["engine_seed"]})
        Image.fromarray(frame).save(output / "start.png")
        for chunk in demo["chunks"]:
            name = chunk["file"]
            if Path(name).name != name or "/" in name or "\\" in name or sha256(path.parent / name) != chunk["sha256"]:
                raise ValueError("Unsafe or corrupt demonstration chunk")
            with np.load(path.parent / name, allow_pickle=False) as data:
                actions = data["actions"].copy()
            for action in actions:
                started = time.monotonic()
                frame, _, done, truncated, info = env.step(action)
                expected = json.loads(next(diagnostics))["after"]
                if (np.linalg.norm(np.array(info["position"]) - expected["position"]) > .02
                        or any(info[k] != expected[k] for k in ("ammo", "kills", "waypoint"))):
                    differences.append(steps)
                steps += 1
                if args.paced:
                    time.sleep(max(0, env.config["action_repeat"] / env.config["physics_hz"] - (time.monotonic() - started)))
                if (done or truncated) and steps < demo["steps"]:
                    raise RuntimeError("Replay ended earlier than its recorded episode")
        Image.fromarray(frame).save(output / "final.png")
    result = {"steps": steps, "state_mismatch_steps": differences, "consistent": not differences and steps == demo["steps"],
              "success": bool(info["is_success"]), "note": "Checks position/ammo/kills/waypoint, not pixel-identical rendering."}
    write_json(output / "replay.json", result)
    print(json.dumps(result, indent=2), flush=True)
