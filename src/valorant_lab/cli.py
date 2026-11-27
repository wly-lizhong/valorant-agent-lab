from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .env import ROOT, NOOP, ValorantNavigationEnv, validate_assets
from .scenario import load_scenario, require_contract

DEFAULT_TASK = ROOT / "configs/lotus_movement_combat.json"


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
        action = env.noop.copy()
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
        fire = env.noop.copy()
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
    from .learning import POLICY_KWARGS, TrainingCallback, metadata, save_model
    torch.set_num_threads(args.cpu_threads)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario = Path(args.scenario).resolve()
    config = load_scenario(scenario)
    parent_meta = require_contract(args.resume or args.init_bc, config) if args.resume or args.init_bc else {}
    if args.init_bc and parent_meta.get("kind") != "bc":
        raise ValueError("--init-bc requires a BC checkpoint; use --resume for PPO")
    if args.resume and parent_meta.get("kind") != "ppo":
        raise ValueError("--resume requires PPO; use --init-bc for a BC checkpoint")
    manifest = {"scenario": config, "seed": args.seed,
        "device": args.device, "environments": args.envs, "additional_timesteps": args.steps,
        "resume": args.resume, "init_bc": args.init_bc, "packages": {p: importlib.metadata.version(p)
            for p in ["torch", "gymnasium", "stable-baselines3", "numpy"]}}
    (output / ("run-" + str(time.time_ns()) + ".json")).write_text(json.dumps(manifest, indent=2))
    env = make_vector_env(scenario, args.envs, args.seed)
    meta = metadata(config, "ppo", episodes=parent_meta.get("episodes", []),
                    test_only=parent_meta.get("test_only", False), parent=args.resume or args.init_bc)
    model = None
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
            model.tensorboard_log = str(output / "tensorboard")
        else:
            model = PPO("CnnPolicy", env, n_steps=args.rollout, batch_size=args.batch_size,
                n_epochs=4, learning_rate=3e-4, gamma=0.99, seed=args.seed,
                policy_kwargs=POLICY_KWARGS, device=args.device, verbose=1, tensorboard_log=str(output / "tensorboard"))
            if args.init_bc:
                teacher = PPO.load(args.init_bc, device=args.device)
                model.policy.load_state_dict(teacher.policy.state_dict())
                del teacher
        callback = TrainingCallback(output, meta, args)
        model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(args.resume))
    except KeyboardInterrupt:
        print("Interrupted; saving the current checkpoint.")
    finally:
        if model is not None:
            save_model(model, output / "latest.zip", meta)
        env.close()


def evaluate(args):
    from .evaluation import evaluate as run
    run(args)


def record(args):
    from .recording import record as run
    run(args)


def bc_train(args):
    from .learning import bc_train as run
    run(args)


def queue(args):
    from .corrections import list_queue
    list_queue(args)


def replay(args):
    from .replay import replay as run
    run(args)


def web_control(args):
    from .web_control import serve
    serve(args)


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
    training.add_argument("--output", default="runs/ppo-movement-combat")
    training.add_argument("--steps", type=int, default=100000)
    training.add_argument("--envs", type=int, default=1)
    training.add_argument("--rollout", type=int, default=256)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--cpu-threads", type=int, default=4)
    training.add_argument("--checkpoint-every", type=int, default=10000)
    initial = training.add_mutually_exclusive_group()
    initial.add_argument("--resume")
    initial.add_argument("--init-bc")
    training.add_argument("--eval-every", type=int, default=25000, help="Development evaluation interval; 0 disables")
    training.add_argument("--eval-episodes", type=int, default=3)
    training.add_argument("--device", default="auto")
    training.set_defaults(func=train)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--episodes", type=int, default=10)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--cpu-threads", type=int, default=4)
    evaluation.add_argument("--output", default="runs/evaluation")
    evaluation.add_argument("--split", choices=["dev", "validation", "test"], default="dev")
    evaluation.add_argument("--queue", help="Deduplicated correction queue; development split only")
    evaluation.add_argument("--case", help="Re-evaluate one correction case from its original seed")
    evaluation.add_argument("--realtime", action="store_true", help="Keep simulation running during inference")
    evaluation.add_argument("--save-frames", action="store_true")
    evaluation.set_defaults(func=evaluate)
    recording = commands.add_parser("record", help="本地键鼠示范采集")
    recording.add_argument("--output", default="runs/demonstrations")
    recording.add_argument("--episodes", type=int, default=1)
    recording.add_argument("--split", choices=["train", "validation"], default="train")
    recording.add_argument("--case")
    recording.add_argument("--mouse-scale", type=float, default=.08, help="Degrees per relative mouse pixel before quantization")
    recording.set_defaults(func=record)
    native = commands.add_parser("record-native", help="本机原生窗口录制；无需远程桌面或 pygame")
    native.add_argument("--scenario", default=str(DEFAULT_TASK))
    native.add_argument("--output", default="runs/demonstrations-v215")
    native.add_argument("--seed", type=int, default=42)
    native.add_argument("--split", choices=["train", "validation"], default="train")
    native.add_argument("--case")
    native.set_defaults(func=lambda args: __import__("valorant_lab.native_capture", fromlist=["run"]).run(args))
    bc = commands.add_parser("bc-train", help="从画面与动作训练，可在无游戏进程的云服务器运行")
    bc.add_argument("--data", required=True)
    bc.add_argument("--output", default="runs/bc")
    bc.add_argument("--epochs", type=int, default=10)
    bc.add_argument("--batch-size", type=int, default=32)
    bc.add_argument("--cpu-threads", type=int, default=4)
    bc.add_argument("--learning-rate", type=float, default=3e-4)
    bc.add_argument("--device", default="auto")
    bc.add_argument("--allow-test-data", action="store_true", help="Pipeline checks only; marks model test_only")
    bc.set_defaults(func=bc_train)
    questions = commands.add_parser("queue", help="列出系统生成的人工纠错任务")
    questions.add_argument("--queue", default="runs/ppo-movement-combat/corrections")
    questions.set_defaults(func=queue)
    playback = commands.add_parser("replay", help="按原始种子和动作重放示范，检查状态一致性")
    playback.add_argument("--episode", required=True)
    playback.add_argument("--output", default="runs/replay")
    playback.add_argument("--paced", action="store_true")
    playback.set_defaults(func=replay)
    browser = commands.add_parser("web", help="中文浏览器示范与纠错入口；云端运行游戏")
    browser.add_argument("--host", default="127.0.0.1")
    browser.add_argument("--port", type=int, default=8765)
    browser.add_argument("--origin", action="append", help="Exact browser origin; repeat for multiple hosts")
    browser.add_argument("--output", default="runs/demonstrations")
    browser.add_argument("--queue", default="runs/ppo-movement-combat/corrections")
    browser.add_argument("--state-dir", default="runs/web")
    browser.add_argument("--token-file", default="runs/web/access-token")
    browser.add_argument("--max-frame-age-ms", type=float, default=250,
                         help="Lockstep round-trip warning threshold; does not reject an episode")
    browser.add_argument("--max-input-hold-ms", type=float, default=250,
                         help="Maximum display-to-input sample time before excluding a demonstration")
    browser.add_argument("--control-timeout", type=float, default=3)
    browser.add_argument("--allow-software", action="store_true", help="CPU rendering diagnostics only, not cloud performance acceptance")
    browser.set_defaults(func=web_control)
    for sub in (smoke_parser, training, evaluation, recording, bc):
        sub.add_argument("--scenario", default=str(ROOT / "configs/lotus_navigation.json") if sub is smoke_parser else str(DEFAULT_TASK),
                         help="Scenario JSON; smoke defaults to navigation self-test, other commands to movement/combat")
        sub.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("steps", "envs", "rollout", "batch_size", "episodes", "cpu_threads", "checkpoint_every", "epochs", "eval_episodes"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.command == "train" and args.rollout * args.envs < args.batch_size:
        parser.error("rollout * envs must be >= batch-size")
    if args.command == "train" and (args.rollout * args.envs < 2 or args.batch_size < 2):
        parser.error("PPO advantage normalization requires at least two samples and batch-size >= 2")
    if hasattr(args, "eval_every") and args.eval_every < 0:
        parser.error("--eval-every must be nonnegative")
    if hasattr(args, "seed") and not 0 <= args.seed < 100000:
        parser.error("--seed must be in [0, 100000)")
    for name in ("learning_rate", "mouse_scale", "max_frame_age_ms", "max_input_hold_ms", "control_timeout"):
        if hasattr(args, name) and (not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.command == "web" and not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    args.func(args)


if __name__ == "__main__":
    main()
