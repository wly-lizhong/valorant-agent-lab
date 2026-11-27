"""Independent lockstep evaluators with deterministic seed partitions and one queue writer."""
from __future__ import annotations
from argparse import Namespace
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .corrections import enqueue_failure
from .dataset import sha256, split_seed, write_json
from .scenario import load_scenario, require_contract


def partitions(episodes, workers):
    if episodes < 1 or workers < 1:
        raise ValueError("Episode and worker counts must be positive")
    count = min(episodes, workers)
    size, remainder = divmod(episodes, count)
    start = 0
    result = []
    for worker in range(count):
        length = size + int(worker < remainder)
        result.append((start, length))
        start += length
    return result


def _worker(arguments):
    from .evaluation import evaluate
    directory = Path(arguments["output"])
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            return evaluate(Namespace(**arguments))


def merge_reports(args, pieces, model_hash, config, elapsed):
    """Validate every shard before writing any correction task or final report."""
    output = Path(args.output)
    records = []
    for offset, directory, report in sorted(pieces, key=lambda piece: piece[0]):
        if (offset != len(records) or report["model_sha256"] != model_hash
                or report["split"] != args.split or report["clock"] != "lockstep"
                or report["scenario"] != config):
            raise ValueError("Parallel evaluation shard identity/order mismatch")
        for local_index, record in enumerate(report["episodes"]):
            index = offset + local_index
            if record["engine_seed"] != split_seed(args.split, args.seed + index):
                raise ValueError("Parallel evaluation seed mismatch")
            relative = Path(directory) / f"episode-{local_index:04d}"
            if not (output / relative).resolve().is_relative_to(output.resolve()):
                raise ValueError("Evaluation artifacts must remain within the output directory")
            records.append({**record, "episode_index": index, "artifact_directory": str(relative)})
    if len(records) != args.episodes:
        raise ValueError("Incomplete parallel evaluation; no aggregate result published")
    if args.queue:
        if args.split != "dev":
            raise ValueError("Only --split dev may generate correction tasks")
        for record in records:
            if record["success"]:
                continue
            directory = output / record["artifact_directory"]
            # Classification needs the last40 frames; case evidence keeps80.
            with (directory / "trace.jsonl").open(encoding="utf-8") as stream:
                trace = list(deque((json.loads(line)["info"] for line in stream), maxlen=80))
            with Image.open(directory / "final.png") as image:
                frame = np.array(image.convert("RGB"))
            record["correction_case"] = str(enqueue_failure(
                args.queue, config, record["engine_seed"], trace, frame,
                split=args.split, model_hash=model_hash))
    report = {"model_sha256": model_hash, "model_test_only": pieces[0][2]["model_test_only"],
              "split": args.split, "clock": "lockstep", "scenario": config,
              "evaluation_workers": len(pieces), "elapsed_seconds": elapsed,
              "success_rate": float(np.mean([r["success"] for r in records])),
              "realtime_task_pass_rate": 0., "episodes": records}
    write_json(output / "report.json", report)
    return report


def evaluate_parallel(args):
    if args.realtime or args.case:
        raise ValueError("Use --workers 1 for realtime or single correction-case evaluation")
    if args.queue and args.split != "dev":
        raise ValueError("Only --split dev may generate correction tasks")
    groups = partitions(args.episodes, args.workers)
    config = load_scenario(args.scenario)
    require_contract(args.model, config)
    model_hash = sha256(Path(args.model))
    output = Path(args.output).resolve()
    if (output / "report.json").exists() or (output / "workers").exists():
        raise FileExistsError("Use a fresh output directory for parallel evaluation")
    (output / "workers").mkdir(parents=True)
    arguments = {k: v for k, v in vars(args).items() if k != "func"}
    # Only the parent may mutate the shared correction queue.
    arguments.update(workers=1, queue=None, cpu_threads=max(1, args.cpu_threads // len(groups)))
    started = time.monotonic()
    pieces = []
    with ProcessPoolExecutor(max_workers=len(groups), mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = []
        for worker, (offset, count) in enumerate(groups):
            directory = f"workers/worker-{worker:03d}"
            child = {**arguments, "seed": args.seed + offset, "episodes": count,
                     "output": str(output / directory)}
            pending.append((offset, directory, count, pool.submit(_worker, child)))
        for offset, directory, count, future in pending:
            result = future.result()
            if len(result["episodes"]) != count:
                raise ValueError("Parallel worker returned an incomplete shard")
            pieces.append((offset, directory, result))
    report = merge_reports(args, pieces, model_hash, config, time.monotonic() - started)
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "scenario")}, indent=2), flush=True)
    return report
