"""Compare BC input modes on real demos without writing trained checkpoints."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from valorant_lab.bc_data import make_bc_batches
from valorant_lab.dataset import load_episodes, write_json
from valorant_lab.learning import OfflineSpaces, POLICY_KWARGS, training_distribution
from valorant_lab.scenario import load_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scenario", default="configs/lotus_movement_combat.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp-fp16"], default="fp32")
    parser.add_argument("--modes", nargs="+", choices=["stream", "cpu", "cuda", "auto"], default=["stream", "auto"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--full-batches", action="store_true",
                        help="Measure this many full batches across shuffled passes, excluding short final batches")
    parser.add_argument("--cache-mib", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/bc-input-benchmark.json")
    args = parser.parse_args()
    if min(args.batch_size, args.batches, args.cache_mib, args.cpu_threads) < 1:
        parser.error("Counts and memory budget must be positive")
    torch.set_num_threads(args.cpu_threads)
    config = load_scenario(args.scenario)
    episodes = load_episodes(args.data, config)
    results = []
    for mode in args.modes:
        model = PPO("CnnPolicy", OfflineSpaces(config), n_steps=2, batch_size=2,
                    policy_kwargs=POLICY_KWARGS, seed=args.seed, device=args.device, verbose=0)
        mixed = args.precision == "amp-fp16"
        if mixed and model.device.type != "cuda":
            parser.error("amp-fp16 requires a CUDA device")
        scaler = torch.amp.GradScaler("cuda", enabled=mixed)
        train, validation, cache = make_bc_batches(episodes, device=model.device,
            batch_size=args.batch_size, mode=mode, budget_mib=args.cache_mib)
        if args.full_batches and args.batch_size > len(train.dataset):
            parser.error("--full-batches requires batch size <= number of training samples")
        model.policy.set_training_mode(True)
        def synchronize():
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
        def step(images, actions):
            dist = training_distribution(model.policy, images, mixed_precision=mixed)
            loss = -dist.log_prob(actions).mean() / 8
            model.policy.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(model.policy.optimizer)
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            scaler.step(model.policy.optimizer)
            scaler.update()
        # Identical warmup and sampler seed in each mode. These temporary policy
        # updates are profiling only, never published as trained checkpoints.
        images = torch.zeros((args.batch_size, 12, config["height"], config["width"]),
                             dtype=torch.uint8, device=model.device)
        actions = torch.zeros((args.batch_size, 8), dtype=torch.int64, device=model.device)
        step(images, actions)
        synchronize()
        torch.manual_seed(args.seed + 1)
        iterator = iter(train)
        rows = []
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        measured_batches = args.batches if args.full_batches else min(args.batches, len(train))
        for number in range(measured_batches):
            started = time.perf_counter()
            while True:
                try:
                    images, actions = next(iterator)
                except StopIteration:
                    iterator = iter(train)
                    continue
                if not args.full_batches or len(images) == args.batch_size:
                    break
            synchronize()
            loaded = time.perf_counter()
            images, actions = images.to(model.device), actions.to(model.device)
            synchronize()
            transferred = time.perf_counter()
            step(images, actions)
            synchronize()
            finished = time.perf_counter()
            rows.append({"load_ms": (loaded - started) * 1000,
                         "transfer_ms": (transferred - loaded) * 1000,
                         "train_ms": (finished - transferred) * 1000,
                         "samples": len(images)})
        means = {k: float(np.mean([r[k] for r in rows])) for k in ("load_ms", "transfer_ms", "train_ms")}
        result = {"requested_mode": mode, "cache": cache, "device": str(model.device),
                  "precision": args.precision,
                  "parameter_device": str(next(model.policy.parameters()).device),
                  "batch_size": args.batch_size, "full_batches": args.full_batches,
                  "batches": rows, "mean_ms": means,
                  "samples_per_second": sum(r["samples"] for r in rows) / (len(rows) * sum(means.values()) / 1000)}
        if model.device.type == "cuda":
            result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(model.device) / 1024**2
            result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(model.device) / 1024**2
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "batches"}), flush=True)
        model.get_env().close()
        del model, train, validation, iterator, images, actions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"results": results,
        "note": "Same real data and temporary policy steps; no trained artifact or gameplay score is produced."})


if __name__ == "__main__":
    main()
