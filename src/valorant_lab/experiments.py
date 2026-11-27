"""Bounded BC experiments, isolated subprocesses, resumable receipts and honest summaries.

This module deliberately has no torch import or language-model client. Training,
diagnostics and gameplay are separate CLI processes, with their own log files.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "valorant-bc-experiments-v1"
HEADS = ("forward", "strafe", "movement_mode", "yaw", "pitch", "jump", "fire", "reload")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact/config path escapes its root")
    return path


def validate_manifest(manifest):
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown experiment manifest schema")
    allowed = {"schema", "scenario", "training", "candidates", "evaluation", "limits"}
    if set(manifest) != allowed:
        raise ValueError("Unexpected or missing experiment manifest field")
    train, evaluation, limits = (manifest[k] for k in ("training", "evaluation", "limits"))
    if set(train) != {"epochs", "batch_size", "learning_rate", "seed", "cpu_threads", "precision",
                       "data_cache", "cache_mib", "checkpoint_every_epochs", "class_weight_cap"}:
        raise ValueError("Unexpected training options; free-form CLI commands are not accepted")
    if set(evaluation) != {"split", "seed", "episodes", "workers", "cpu_threads", "device", "scenarios"}:
        raise ValueError("Unexpected evaluation options")
    if evaluation["split"] != "dev":
        raise ValueError("Automatic experiments evaluate dev only; final test data stays untouched")
    if set(evaluation["scenarios"]) != {"route", "shooting"}:
        raise ValueError("Both route and shooting evaluations are required")
    if set(limits) != {"total_seconds", "train_seconds", "probe_seconds", "evaluate_seconds", "attempts_per_stage"}:
        raise ValueError("Explicit total, stage and attempt limits are required")
    integers = [(train, k) for k in ("epochs", "batch_size", "cpu_threads", "cache_mib", "checkpoint_every_epochs")]
    integers += [(evaluation, k) for k in ("episodes", "workers", "cpu_threads")]
    integers += [(limits, k) for k in limits]
    if any(type(d[k]) is not int or d[k] < 1 for d, k in integers):
        raise ValueError("Counts and timeouts must be positive integers")
    if limits["attempts_per_stage"] > 3:
        raise ValueError("At most 3 explicit attempts per stage")
    if any(type(d["seed"]) is not int or not 0 <= d["seed"] < 100000 for d in (train, evaluation)):
        raise ValueError("Seed indices must be in [0, 100000)")
    if evaluation["seed"] + evaluation["episodes"] > 100000:
        raise ValueError("Evaluation seeds exceed the dev partition")
    for key in ("learning_rate", "class_weight_cap"):
        if not isinstance(train[key], (int, float)) or not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError("Learning rate and class weight cap must be finite and positive")
    if train["class_weight_cap"] < 1 or train["class_weight_cap"] > 8:
        raise ValueError("This campaign caps class weights at 8")
    if train["precision"] not in ("fp32", "amp-fp16") or train["data_cache"] not in ("auto", "cpu", "cuda", "stream"):
        raise ValueError("Invalid precision/cache mode")
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", evaluation["device"]):
        raise ValueError("Invalid evaluation device")
    candidates = manifest["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("Use 1 to 3 candidate configurations, never an unbounded search")
    ids, variants = set(), set()
    for candidate in candidates:
        if set(candidate) != {"id", "policy_variant", "class_balance"}:
            raise ValueError("Unexpected candidate fields")
        name = candidate["id"]
        pair = (candidate["policy_variant"], candidate["class_balance"])
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) or name in ids:
            raise ValueError("Candidate IDs must be unique safe directory names")
        if pair[0] not in ("legacy", "normalized", "normalized-relu") or pair[1] not in ("none", "sqrt"):
            raise ValueError("Unsupported policy/loss variant")
        if pair in variants:
            raise ValueError("Duplicate candidate configuration")
        ids.add(name)
        variants.add(pair)
    return manifest


def campaign_identity(root, data, manifest, *, device, gpu_display, gpu):
    """Hash actual data, not only cached manifest claims, before launching anything."""
    from .dataset import load_episodes
    from .scenario import contract, load_scenario
    config = load_scenario(contained(root, manifest["scenario"]))
    episodes = load_episodes(data, config)
    if not all(any(meta["split"] == split for _, meta in episodes) for split in ("train", "validation")):
        raise ValueError("Accepted human train AND validation episodes are required")
    for path in manifest["evaluation"]["scenarios"].values():
        if contract(load_scenario(contained(root, path))) != contract(config):
            raise ValueError("Evaluation must preserve the model's scenario/action contract")
    source = {}
    files = set((root / "src").rglob("*.py")) | set((root / "engine" / "overlay").rglob("*.cs"))
    files.update(root / "scripts" / name for name in ("run_bc_experiments.py", "check_bc_policy.py"))
    files.update(contained(root, name) for name in [manifest["scenario"], *manifest["evaluation"]["scenarios"].values()])
    files.update(root / name for name in ("dependencies.lock.json", "local.json", "local.linux.json", "local.windows.json") if (root / name).exists())
    for path in sorted(files):
        source[str(path.relative_to(root)).replace("\\", "/")] = file_hash(path)
    dataset = [{"episode": path.name, "manifest": file_hash(path / "episode.json"),
                "split": meta["split"], "case_key": meta["case_key"], "steps": meta["steps"],
                "chunks": [{"file": c["file"], "sha256": c["sha256"]} for c in meta["chunks"]]}
               for path, meta in episodes]
    packages = {}
    for name in ("torch", "stable-baselines3", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"manifest": manifest, "source": source, "data": dataset, "data_path": str(data),
            "runtime": {"python": sys.version, "packages": packages, "device": device,
                        "gpu_display": gpu_display, "gpu": gpu}}


@contextmanager
def output_lock(directory):
    """OS lock releases on crash; a stale PID never authorizes killing a process."""
    path = directory / "runner.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            stream.write(b"0")
            stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another runner owns this output directory") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another runner owns this output directory, or its filesystem cannot lock") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def stop_process_tree(process, grace=3):
    """Stop only the child tree created for this stage, never a shared X server."""
    if os.name == "nt":
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    else:
        # The parent may already have exited while a grandchild is still alive.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_process(command, *, cwd, env, log_path, timeout, on_started=None):
    started = time.monotonic()
    with Path(log_path).open("wb") as log:
        options = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, **options)
        try:
            if on_started:
                on_started(process.pid)
            try:
                code = process.wait(timeout=timeout)
                return {"returncode": code, "timed_out": False, "elapsed_seconds": time.monotonic() - started}
            except subprocess.TimeoutExpired:
                stop_process_tree(process)
                return {"returncode": process.returncode, "timed_out": True,
                        "elapsed_seconds": time.monotonic() - started}
        finally:
            # Also clean up descendants after a failed process exits on its own.
            stop_process_tree(process)


class GPUSampler:
    def __init__(self, path, gpu, stage, interval=1):
        self.path, self.gpu, self.stage, self.interval = Path(path), gpu, stage, interval
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self.thread.start()
        return self

    def _run(self):
        with self.path.open("a", encoding="utf-8") as stream:
            while not self.stop.is_set():
                row = {"time_ns": time.time_ns(), "stage": self.stage()}
                try:
                    result = subprocess.run(["nvidia-smi", "-i", str(self.gpu),
                        "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=3)
                    if result.returncode:
                        raise RuntimeError(result.stderr.strip()[-300:])
                    values = [float(value.strip()) for value in result.stdout.strip().split(",")]
                    if len(values) != 3 or not all(math.isfinite(value) for value in values):
                        raise ValueError("Unavailable GPU counters")
                    row.update(utilization_percent=values[0], memory_mib=values[1], power_w=values[2])
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    row["error"] = str(exc)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                if "error" in row:
                    # Missing GPU tooling is reported, not retried on every second.
                    break
                self.stop.wait(self.interval)

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=5)


@dataclass
class Stage:
    key: str
    kind: str
    candidate: str
    directory: Path
    command: list[str]
    required: tuple[str, ...]
    timeout: int


def stage_plan(root, output, data, manifest, device):
    train, evaluation, limits = (manifest[k] for k in ("training", "evaluation", "limits"))
    for candidate in manifest["candidates"]:
        name = candidate["id"]
        base = output / name
        training = base / "training"
        command = [sys.executable, "-m", "valorant_lab.cli", "bc-train", "--scenario", str(contained(root, manifest["scenario"])),
                   "--data", str(data), "--output", str(training), "--device", device]
        for key, value in {**train, **{k: v for k, v in candidate.items() if k != "id"}}.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        yield Stage(name + "/train", "train", name, training, command,
                    ("best.zip", "best.json", "latest.zip", "latest.json", "resume.pt", "metrics.json"), limits["train_seconds"])
        for selection in ("best", "latest"):
            directory = base / ("probe-" + selection)
            command = [sys.executable, str(root / "scripts/check_bc_policy.py"), "--model", str(training / (selection + ".zip")),
                       "--data", str(data), "--scenario", str(contained(root, manifest["scenario"])),
                       "--output", str(directory / "report.json"), "--device", device,
                       "--batch-size", str(train["batch_size"]), "--cpu-threads", str(train["cpu_threads"])]
            yield Stage(name + "/probe-" + selection, "probe", name, directory, command,
                        ("report.json",), limits["probe_seconds"])
        for task, scenario in evaluation["scenarios"].items():
            directory = base / ("eval-" + task)
            command = [sys.executable, "-m", "valorant_lab.cli", "evaluate", "--model", str(training / "best.zip"),
                       "--scenario", str(contained(root, scenario)), "--output", str(directory), "--split", "dev"]
            for key in ("seed", "episodes", "workers", "cpu_threads", "device"):
                command.extend(["--" + key.replace("_", "-"), str(evaluation[key])])
            yield Stage(name + "/eval-" + task, "evaluate", name, directory, command,
                        ("report.json",), limits["evaluate_seconds"])


def artifacts_for(root, stage):
    for name in stage.required:
        if not (stage.directory / name).is_file():
            raise ValueError(f"Stage did not produce required artifact: {stage.directory / name}")
    files = [p for p in stage.directory.rglob("*") if p.is_file() and p.suffix not in (".log", ".tmp")]
    return {str(path.relative_to(root)).replace("\\", "/"): file_hash(path) for path in sorted(files)}


def validate_artifacts(root, artifacts):
    if not artifacts:
        raise ValueError("Completed stage is missing its artifact receipt")
    for name, expected in artifacts.items():
        path = contained(root, name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Completed artifact changed or disappeared: {name}; preserve it and use a new campaign")


def validate_stage(stage, manifest):
    if stage.kind == "train":
        metrics = read_json(stage.directory / "metrics.json")
        if not metrics.get("epochs") or metrics["epochs"][-1]["epoch"] != manifest["training"]["epochs"]:
            raise ValueError("Training exited before the requested final epoch")
        for row in metrics["epochs"]:
            if not all(math.isfinite(row[key]) for key in ("train_loss", "validation_loss")):
                raise ValueError("Non-finite training/validation loss")
    elif stage.kind == "probe":
        report = read_json(stage.directory / "report.json")
        model = Path(stage.command[stage.command.index("--model") + 1])
        if (report.get("schema") != "bc-policy-diagnostic-v1"
                or report.get("model_sha256") != file_hash(model)
                or set(report.get("splits", {})) != {"train", "validation"}
                or any(report["splits"][name].get("samples", 0) <= 0 for name in ("train", "validation"))):
            raise ValueError("Policy diagnostics must preserve train/validation separation")
    elif stage.kind == "evaluate":
        report = read_json(stage.directory / "report.json")
        evaluation = manifest["evaluation"]
        expected = list(range(200000 + evaluation["seed"], 200000 + evaluation["seed"] + evaluation["episodes"]))
        if (report.get("split") != "dev" or report.get("clock") != "lockstep"
                or report.get("model_test_only") is not False
                or report.get("scenario") != read_json(stage.command[stage.command.index("--scenario") + 1])
                or [row["engine_seed"] for row in report["episodes"]] != expected
                or any(row.get("training_overlap") is not False for row in report["episodes"])):
            raise ValueError("Wrong evaluation split/seeds or training overlap")
        model = Path(stage.command[stage.command.index("--model") + 1])
        if report["model_sha256"] != file_hash(model):
            raise ValueError("Evaluation used a different model")
        summarize_gameplay(stage.directory / "report.json")  # Verifies trace evidence before stage success.


def summarize_gameplay(path):
    path = Path(path)
    report = read_json(path)
    heads, joint = [Counter() for _ in HEADS], Counter()
    total, stuck, look_limit, longest = 0, 0, 0, 0
    episodes = []
    for index, record in enumerate(report["episodes"]):
        directory = contained(path.parent, record.get("artifact_directory", f"episode-{index:04d}"))
        tail, last_action, streak, count, blocked = deque(maxlen=40), None, 0, 0, 0
        with (directory / "trace.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                action, info = row["requested_action"], row["info"]
                if len(action) != 8:
                    raise ValueError("Bad action trace")
                joint[",".join(map(str, action))] += 1
                for head, value in zip(heads, action):
                    head[str(value)] += 1
                streak = streak + 1 if action == last_action else 1
                last_action = action
                longest = max(longest, streak)
                count += 1
                look_limit += int(abs(info.get("pitch", 0)) > 70)
                tail.append((info["position"], bool(action[0] or action[1])))
                if len(tail) == 40 and sum(moving for _, moving in tail) > 28:
                    origin = tail[0][0]
                    if max(math.dist(position, origin) for position, _ in tail) < .5:
                        blocked += 1
        if count == 0:
            raise ValueError("Empty gameplay trace")
        final = record["final"]
        if any(type(final.get(key)) is not int or final[key] < 0 for key in ("shots", "hits", "kills")):
            raise ValueError("Actual gameplay shot/hit/kill counters are missing or invalid")
        episodes.append({"engine_seed": record["engine_seed"], "success": record["success"],
                         "shots": final["shots"], "hits": final["hits"], "kills": final["kills"],
                         "steps": count, "blocked_windows": blocked, "termination_reason": record.get("termination_reason"),
                         "inference_p95_ms": record.get("inference_p95_ms"), "artifacts": str(directory)})
        total += count
        stuck += blocked
    return {"report": str(path), "split": report["split"], "clock": report["clock"],
            "success_rate": report["success_rate"], "episodes": episodes,
            "shots": sum(row["shots"] for row in episodes), "hits": sum(row["hits"] for row in episodes),
            "kills": sum(row["kills"] for row in episodes), "steps": total,
            "unique_actions": len(joint), "action_counts": dict(joint),
            "head_counts": dict(zip(HEADS, map(dict, heads))), "blocked_windows": stuck,
            "blocked_window_fraction_of_steps": stuck / total if total else None,
            "look_limit_steps": look_limit, "longest_constant_action_steps": longest,
            "stuck_definition": "40-step window: movement requested >70%, maximum displacement from its first position <0.5m; overlapping windows, not seconds",
            "note": "Lockstep gameplay, not realtime latency certification. No hidden diagnostics are policy inputs."}


def gpu_summary(path):
    groups = {}
    if not Path(path).exists():
        return {"available": False}
    errors = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended last row is not evidence.
            if "error" in row:
                errors.append(row["error"])
                continue
            values = groups.setdefault(row.get("stage") or "idle", [])
            values.append(row)
    return {"available": bool(groups), "errors": errors, "scope": "whole selected GPU, not per-process attribution",
            "stages": {name: {"samples": len(rows),
                "mean_utilization_percent": sum(r["utilization_percent"] for r in rows) / len(rows),
                "peak_memory_mib": max(r["memory_mib"] for r in rows),
                "mean_power_w": sum(r["power_w"] for r in rows) / len(rows)} for name, rows in groups.items()}}


class Campaign:
    def __init__(self, root, data, output, manifest, *, device="cuda", resume=False, gpu_display=False, gpu=0):
        self.root, self.data, self.output = Path(root).resolve(), Path(data).resolve(), Path(output).resolve()
        self.manifest = validate_manifest(manifest)
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) or type(gpu) is not int or gpu < 0:
            raise ValueError("Invalid training device/GPU")
        if device == "cpu" and manifest["training"]["precision"] != "fp32":
            raise ValueError("CPU training requires a manifest with fp32 precision")
        self.device, self.resume, self.use_display, self.gpu = device, resume, gpu_display, gpu
        self.state = None

    def save(self):
        atomic_json(self.output / "state.json", self.state)

    def initialize(self):
        identity = campaign_identity(self.root, self.data, self.manifest, device=self.device,
                                     gpu_display=self.use_display, gpu=self.gpu)
        digest = value_hash(identity)
        state_path = self.output / "state.json"
        if state_path.exists():
            if not self.resume:
                raise ValueError("Output already has a campaign; use --resume or a new directory")
            self.state = read_json(state_path)
            if self.state["fingerprint"] != digest:
                raise ValueError("Source, data, runtime or manifest changed; refusing a mixed resume")
            if value_hash(read_json(self.output / "identity.json")) != digest:
                raise ValueError("Frozen experiment identity changed; refusing resume")
            for entry in self.state["stages"].values():
                if entry["status"] == "completed":
                    validate_artifacts(self.output, entry["artifacts"])
                elif entry["status"] == "running":
                    attempt = entry["attempts"][-1]
                    pid = attempt.get("pid")
                    if pid and attempt.get("host") == socket.gethostname():
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f"Previous child PID {pid} may still be running; inspect it before resuming")
                    elapsed = min(attempt["timeout_seconds"], max(0, (time.time_ns() - attempt["started_ns"]) / 1e9))
                    self.state["consumed_seconds"] += elapsed
                    attempt.update(status="interrupted", elapsed_seconds=elapsed, error="Runner ended before recording completion")
                    entry["status"] = "interrupted"
            self.save()
            return
        if self.resume:
            raise ValueError("No existing state to resume")
        if any(path.name != "runner.lock" for path in self.output.iterdir()):
            raise ValueError("A new campaign needs an empty output directory")
        self.state = {"schema": SCHEMA, "fingerprint": digest, "created_ns": time.time_ns(),
                      "status": "pending", "current_stage": None, "consumed_seconds": 0., "stages": {}}
        atomic_json(self.output / "identity.json", identity)
        self.save()

    def summary(self):
        result = {"schema": SCHEMA, "status": self.state["status"], "current_stage": self.state["current_stage"],
                  "consumed_seconds": self.state["consumed_seconds"], "state": str(self.output / "state.json"),
                  "fingerprint": self.state["fingerprint"], "candidates": {}, "stages": self.state["stages"],
                  "gpu": gpu_summary(self.output / "gpu.jsonl"),
                  "note": "Experiment completion is not first-stage gameplay acceptance. Final test split was not evaluated."}
        for candidate in self.manifest["candidates"]:
            name = candidate["id"]
            value = {"configuration": candidate}
            for key, stage in self.state["stages"].items():
                if not key.startswith(name + "/") or stage["status"] != "completed":
                    continue
                directory = contained(self.output, stage["artifact_directory"])
                if stage["kind"] == "evaluate":
                    value[key.split("/")[1]] = summarize_gameplay(directory / "report.json")
                elif stage["kind"] == "train":
                    metrics = read_json(directory / "metrics.json")
                    value["training"] = {"directory": str(directory), "epochs": len(metrics["epochs"]),
                        "last_epoch": metrics["epochs"][-1], "runtime": metrics.get("runtime"),
                        "train_compute_seconds": sum(row.get("train_seconds", 0) for row in metrics["epochs"])}
                elif stage["kind"] == "probe":
                    value[key.split("/")[1]] = {"report": str(directory / "report.json"),
                                               "splits": read_json(directory / "report.json").get("splits")}
            result["candidates"][name] = value
        atomic_json(self.output / "summary.json", result)

    def execute(self, stage, env):
        entry = self.state["stages"].setdefault(stage.key, {"kind": stage.kind, "status": "pending", "attempts": []})
        if entry["status"] == "completed":
            return
        limits = self.manifest["limits"]
        remaining = limits["total_seconds"] - self.state["consumed_seconds"]
        if remaining <= 0:
            raise RuntimeError("Campaign total time budget exhausted; no further stage launched")
        if len(entry["attempts"]) >= limits["attempts_per_stage"]:
            raise RuntimeError(f"Attempt limit reached for {stage.key}; no automatic retry")
        number = len(entry["attempts"]) + 1
        attempt_dir = self.output / stage.candidate / "attempts" / stage.key.split("/")[1] / f"attempt-{number:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        command = list(stage.command)
        if stage.kind == "train":
            if stage.directory.exists() and any(stage.directory.iterdir()):
                if not (stage.directory / "resume.pt").exists():
                    raise RuntimeError("Interrupted training has no resumable checkpoint; preserve it and use a new campaign")
                # Keep the checkpoint consumed by this attempt for independent audit/recovery.
                backup = attempt_dir / "recovery-input"
                backup.mkdir()
                for file in stage.directory.iterdir():
                    if file.is_file() and file.suffix in (".zip", ".json", ".pt"):
                        shutil.copy2(file, backup / file.name)
                command.append("--resume")
        else:
            # A failed evaluator's partial reports/frames are never overwritten.
            old = stage.directory
            stage.directory = attempt_dir / "artifacts"
            command = [str(arg).replace(str(old), str(stage.directory)) for arg in command]
        stage.directory.mkdir(parents=True, exist_ok=True)
        log_path = attempt_dir / "process.log"
        attempt = {"status": "running", "started_ns": time.time_ns(), "host": socket.gethostname(),
                   "timeout_seconds": min(stage.timeout, remaining), "command": command, "log": str(log_path)}
        entry.update(status="running", artifact_directory=str(stage.directory.relative_to(self.output)))
        entry["attempts"].append(attempt)
        self.state["current_stage"] = stage.key
        self.save()
        self.summary()
        started = time.monotonic()

        def on_started(pid):
            attempt["pid"] = pid
            self.save()

        try:
            outcome = run_process(command, cwd=self.root, env=env, log_path=log_path,
                                  timeout=attempt["timeout_seconds"], on_started=on_started)
            attempt.update(outcome)
            if outcome["timed_out"] or outcome["returncode"]:
                raise RuntimeError(f"{stage.key} {'timed out' if outcome['timed_out'] else 'failed'}; inspect {log_path}")
            validate_stage(stage, self.manifest)
            entry["artifacts"] = artifacts_for(self.output, stage)
            entry["status"] = attempt["status"] = "completed"
        except BaseException as exc:
            entry["status"] = attempt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            attempt["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            attempt["finished_ns"] = time.time_ns()
            attempt["elapsed_seconds"] = time.monotonic() - started
            self.state["consumed_seconds"] += attempt["elapsed_seconds"]
            self.save()
            self.summary()

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with output_lock(self.output):
            self.initialize()
            if self.state["status"] == "completed":
                self.summary()
                return self.state
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            env["PYTHONPATH"] = str(self.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
            self.state["status"] = "running"
            self.state.pop("error", None)
            self.save()
            try:
                with ExitStack() as stack:
                    stack.enter_context(GPUSampler(self.output / "gpu.jsonl", self.gpu, lambda: self.state["current_stage"]))
                    display_ready = False
                    for stage in stage_plan(self.root, self.output, self.data, self.manifest, self.device):
                        if self.state["stages"].get(stage.key, {}).get("status") == "completed":
                            continue
                        if stage.kind == "evaluate" and self.use_display and not display_ready:
                            from .cloud import gpu_display
                            graphics_env = stack.enter_context(gpu_display(gpu=self.gpu))
                            env.update(DISPLAY=graphics_env["DISPLAY"], XAUTHORITY=graphics_env["XAUTHORITY"],
                                       VALORANT_LAB_GPU_REQUIRED="1")
                            env.pop("LIBGL_ALWAYS_SOFTWARE", None)
                            display_ready = True
                        self.execute(stage, env)
                    self.state["status"] = "completed"
                    self.state["current_stage"] = None
            except BaseException as exc:
                self.state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                self.state["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self.save()
                self.summary()
            return self.state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "configs/bc-experiments-round01.json"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu-display", action="store_true", help="Create/clean up a private NVIDIA X display on Linux")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if os.name != "nt":
        def interrupted(_sig, _frame):
            raise KeyboardInterrupt("Runner interrupted; stage process tree will be stopped")
        signal.signal(signal.SIGTERM, interrupted)
    campaign = Campaign(ROOT, args.data, args.output, read_json(args.manifest), device=args.device,
                        resume=args.resume, gpu_display=args.gpu_display, gpu=args.gpu)
    try:
        state = campaign.run()
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "stopped", "error": str(exc), "output": str(campaign.output)}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps({"status": state["status"], "summary": str(campaign.output / "summary.json")}), flush=True)
    return 0
