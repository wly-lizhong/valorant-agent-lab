from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .env import ROOT, NOOP, ValorantNavigationEnv, validate_assets


def smoke(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ValorantNavigationEnv(args.scenario) as env:
        first, info = env.reset(seed=args.seed)
        if float(first.std()) < 2:
            raise RuntimeError("RGB observation appears blank; inspect the rendering display")
        Image.fromarray(first).save(output / "reset.png")
        initial_position = np.array(info["position"])
        action = NOOP.copy()
        action[0] = 1
        total_reward = 0.0
        transitions = []
        for step in range(args.steps):
            frame, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            transitions.append({"reward": reward, "terminated": terminated, "truncated": truncated, **info})
            if terminated or truncated:
                break
        moved = float(np.linalg.norm(np.array(info["position"]) - initial_position))
        if moved < 0.1:
            raise RuntimeError("Movement action did not move the player; inspect spawn/collision")
        Image.fromarray(frame).save(output / "after.png")
        expected_time = len(transitions) * env.config["action_repeat"] / env.config["physics_hz"]
        if abs(info["simulation_time"] - expected_time) > 0.002:
            raise RuntimeError("Simulation advanced outside controlled steps")
        # A second episode exercises fast reset and firing.
        env.reset(seed=args.seed)
        fire = NOOP.copy()
        fire[6] = 1
        _, _, _, _, fire_info = env.step(fire)
        if fire_info["ammo"] >= 25:
            raise RuntimeError("Fire action did not consume ammunition")
        result = {"scenario": env.config["id"], "transitions": len(transitions),
            "movement_metres": moved, "total_reward": total_reward,
            "elapsed_seconds": time.monotonic() - started,
            "success": bool(info["is_success"]), "fire_ammo": fire_info["ammo"], "trace": transitions}
        (output / "smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))


def make_vector_env(scenario, count, seed):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
    def factory():
        return Monitor(ValorantNavigationEnv(scenario))
    raw = DummyVecEnv([factory]) if count == 1 else SubprocVecEnv([factory for _ in range(count)], start_method="spawn")
    raw.seed(seed)
    # Four recent first-person frames; no position/HUD ground truth appended.
    return VecTransposeImage(VecFrameStack(raw, n_stack=4, channels_order="last"))


def train(args):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario = Path(args.scenario).resolve() if args.scenario else ROOT / "configs/lotus_navigation.json"
    manifest = {"scenario": json.loads(scenario.read_text()), "seed": args.seed,
        "device": args.device, "environments": args.envs, "additional_timesteps": args.steps,
        "resume": args.resume, "packages": {p: importlib.metadata.version(p)
            for p in ["torch", "gymnasium", "stable-baselines3", "numpy"]}}
    (output / ("run-" + str(time.time_ns()) + ".json")).write_text(json.dumps(manifest, indent=2))
    env = make_vector_env(scenario, args.envs, args.seed)
    model = None
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
            model.tensorboard_log = str(output / "tensorboard")
        else:
            model = PPO("CnnPolicy", env, n_steps=args.rollout, batch_size=args.batch_size,
                n_epochs=4, learning_rate=3e-4, gamma=0.99, seed=args.seed,
                device=args.device, verbose=1, tensorboard_log=str(output / "tensorboard"))
        callback = CheckpointCallback(save_freq=max(args.checkpoint_every // args.envs, 1),
            save_path=str(output / "checkpoints"), name_prefix="ppo_navigation")
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print("Interrupted; saving the current checkpoint.")
    finally:
        if model is not None:
            model.save(output / "latest")
        env.close()


def evaluate(args):
    from stable_baselines3 import PPO
    env = make_vector_env(args.scenario, 1, args.seed)
    try:
        model = PPO.load(args.model, env=env, device=args.device)
        observation = env.reset()
        records = []
        while len(records) < args.episodes:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, info = env.step(action)
            if done[0]:
                records.append({"success": bool(info[0].get("is_success")),
                    "episode": info[0].get("episode"), "distance": info[0].get("distance_to_goal")})
        print(json.dumps({"success_rate": sum(x["success"] for x in records) / len(records),
            "episodes": records}, indent=2))
    finally:
        env.close()


def doctor(args):
    local_path = ROOT / "local.json"
    if not local_path.exists():
        raise SystemExit("Missing local.json. Run scripts/bootstrap.py first.")
    local = json.loads(local_path.read_text())
    checks = {name: Path(local[name]).exists() for name in ("godot", "dotnet", "project")}
    checks["compiled_engine"] = bool(list(Path(local["project"]).glob(".godot/mono/temp/bin/Debug/CaValAim.dll")))
    if local.get("maps_dir"):
        checks["lotus_assets"] = validate_assets(Path(local["maps_dir"]), "lotus")
    else:
        checks["lotus_assets"] = False
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
        checks["torch"] = torch.__version__
    except ImportError:
        checks["torch"] = "not installed (pip install -e '.[train]')"
    print(json.dumps(checks, indent=2))
    if not all(checks.get(k) for k in ["godot", "dotnet", "project", "compiled_engine", "lotus_assets"]):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="独立 FPS 训练环境")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor").set_defaults(func=doctor)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--output", default="runs/smoke")
    smoke_parser.add_argument("--steps", type=int, default=16)
    smoke_parser.set_defaults(func=smoke)
    training = commands.add_parser("train")
    training.add_argument("--output", default="runs/ppo-navigation")
    training.add_argument("--steps", type=int, default=100000)
    training.add_argument("--envs", type=int, default=1)
    training.add_argument("--rollout", type=int, default=256)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--cpu-threads", type=int, default=4)
    training.add_argument("--checkpoint-every", type=int, default=10000)
    training.add_argument("--resume")
    training.add_argument("--device", default="auto")
    training.set_defaults(func=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--episodes", type=int, default=10)
    evaluation.add_argument("--device", default="auto")
    evaluation.set_defaults(func=evaluate)
    for sub in (smoke_parser, training, evaluation):
        sub.add_argument("--scenario", help="Scenario JSON (default: configs/lotus_navigation.json)")
        sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "envs", "rollout", "batch_size", "episodes", "cpu_threads", "checkpoint_every"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "train" and args.rollout * args.envs < args.batch_size:
        parser.error("rollout * envs must be >= batch-size")
    if args.command == "train" and (args.rollout * args.envs < 2 or args.batch_size < 2):
        parser.error("PPO advantage normalization requires at least two samples and batch-size >= 2")
    args.func(args)


if __name__ == "__main__":
    main()
