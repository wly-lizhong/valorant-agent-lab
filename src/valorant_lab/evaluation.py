from __future__ import annotations
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .corrections import enqueue_failure, load_case, update_case
from .dataset import sha256, split_seed, write_json
from .env import ValorantEnv
from .scenario import FRAME_STACK, load_scenario, require_contract


def stack_frame(history, frame):
    if not history:
        history.extend(np.zeros_like(frame) for _ in range(FRAME_STACK - 1))
    history.append(frame)
    return np.concatenate(list(history), axis=2).transpose(2, 0, 1)


def evaluate(args):
    if getattr(args, "workers", 1) > 1:
        from .evaluation_parallel import evaluate_parallel
        return evaluate_parallel(args)
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(args.cpu_threads)
    if args.queue and args.split != "dev":
        raise ValueError("Only --split dev may generate correction tasks")
    case = load_case(args.case) if args.case else None
    if case and args.split != "dev":
        raise ValueError("Correction cases belong to development evaluation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    scenario = args.scenario
    if case:
        scenario = output / "case-scenario.json"
        write_json(scenario, case["scenario"])
    config = load_scenario(scenario)
    model_meta = require_contract(args.model, config)
    model = PPO.load(args.model, device=args.device)
    model_hash = sha256(Path(args.model))
    results = []
    with ValorantEnv(scenario, realtime=args.realtime) as env:
        for episode in range(1 if case else args.episodes):
            engine_seed = case["engine_seed"] if case else split_seed(args.split, args.seed + episode)
            frame, info = env.reset(options={"engine_seed": engine_seed})
            # Validation/dev seeds can be deliberately promoted; report training overlap explicitly.
            from .scenario import digest
            overlap = any(e["case_key"] == digest([config, engine_seed]) and e["split"] == "train"
                          for e in model_meta.get("episodes", []))
            if args.split in ("test", "validation") and overlap:
                raise ValueError("Evaluation case occurs in this model's training data")
            history = deque(maxlen=FRAME_STACK)
            observation = stack_frame(history, frame)
            trace, inference_ms, intervals = [], [], []
            total_reward = 0.
            directory = output / f"episode-{episode:04d}"
            directory.mkdir(exist_ok=True)
            Image.fromarray(frame).save(directory / "start.png")
            started = time.monotonic()
            previous_received = started
            with (directory / "trace.jsonl").open("w", encoding="utf-8") as log:
                while True:
                    begin = time.monotonic()
                    action, _ = model.predict(observation, deterministic=True)
                    infer = (time.monotonic() - begin) * 1000
                    frame, reward, terminated, truncated, info = env.step(action)
                    received = time.monotonic()
                    inference_ms.append(infer)
                    intervals.append((received - previous_received) * 1000)
                    previous_received = received
                    trace.append(info)
                    total_reward += reward
                    log.write(json.dumps({"observation_frame": len(trace) - 1, "requested_action": action.tolist(),
                                          "inference_ms": infer, "received_seconds": received - started,
                                          "reward": reward, "info": info}) + "\n")
                    if args.save_frames:
                        Image.fromarray(frame).save(directory / f"frame-{len(trace):06d}.png")
                    observation = stack_frame(history, frame)
                    if terminated or truncated:
                        break
            Image.fromarray(frame).save(directory / "final.png")
            period_ms = config["action_repeat"] / config["physics_hz"] * 1000
            p95 = float(np.percentile(inference_ms, 95))
            interval95 = float(np.percentile(intervals, 95))
            realtime_ok = (args.realtime and info.get("realtime_factor", 0) >= .9
                           and info.get("dropped_seconds", 1) < .05
                           and info.get("watchdog_releases", 1) == 0
                           and p95 <= period_ms and interval95 <= 2 * period_ms)
            record = {"engine_seed": engine_seed, "success": bool(info["is_success"]),
                      "reward": total_reward, "termination_reason": info.get("termination_reason"),
                      "simulation_seconds": info["simulation_time"], "elapsed_seconds": time.monotonic() - started,
                      "inference_p95_ms": p95, "observation_interval_p95_ms": interval95,
                      "realtime_timing_pass": bool(realtime_ok),
                      "realtime_task_pass": bool(realtime_ok and info["is_success"]),
                      "training_overlap": overlap, "final": info}
            if args.queue and not info["is_success"]:
                record["correction_case"] = str(enqueue_failure(args.queue, config, engine_seed, trace, frame,
                                                                 split=args.split, model_hash=model_hash))
            if case:
                changes = {"last_model_result": record, "last_model_hash": model_hash}
                if info["is_success"]:
                    changes["status"] = "model_passed_once"
                update_case(args.case, **changes)
            results.append(record)
    report = {"model_sha256": model_hash, "model_test_only": model_meta.get("test_only", False),
              "split": args.split, "clock": "realtime" if args.realtime else "lockstep",
              "scenario": config, "success_rate": float(np.mean([r["success"] for r in results])),
              "realtime_task_pass_rate": float(np.mean([r["realtime_task_pass"] for r in results])),
              "episodes": results}
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "scenario")}, indent=2), flush=True)
    return report
