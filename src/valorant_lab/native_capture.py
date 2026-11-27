"""Native Godot human capture and strict conversion to the existing BC dataset."""
from __future__ import annotations
import gzip
import json
import os
from pathlib import Path
import subprocess
import uuid
import numpy as np
from .env import ROOT, validate_assets, validate_runtime
from .scenario import load_scenario, digest, validate_scenario, contract
from .dataset import EpisodeWriter, split_seed, sha256, write_json, validate_native_alignment
from .corrections import load_case, update_case


def local_config(root=ROOT):
    root = Path(root)
    path = root / ("local.windows.json" if os.name == "nt" else "local.linux.json")
    if not path.is_file():
        path = root / "local.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))


def import_capture(job, output):
    job, output = Path(job), Path(output)
    request = json.loads((job / "request.json").read_text(encoding="utf-8"))
    config = load_scenario(job / "scenario.json")
    raw = job / "capture"
    manifest = json.loads((raw / "native.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != "valorant-native-v1" or manifest.get("source") not in ("human", "script_test")
            or request.get("engine_contract") != contract(config)
            or manifest.get("engine") != contract(config)["engine"]
            or manifest.get("engine_seed") != request["engine_seed"]
            or request["split"] not in ("train", "validation")
            or type(manifest.get("steps")) is not int or not 0 <= manifest["steps"] <= config["max_steps"]):
        raise ValueError("Invalid native capture manifest")
    rows_path = raw / "steps.jsonl"
    capture_id = digest([request, config, manifest, sha256(rows_path)])
    for existing in output.glob("episode-*/episode.json"):
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("native_capture_id") == capture_id:
            return existing.parent
    writer = EpisodeWriter(output, config, request["engine_seed"], source=manifest["source"],
                           split=request["split"], case_id=request.get("case_id"))
    writer.manifest.update(native_capture_id=capture_id, collection={"transport": "native_window_v1",
        "alignment": "rendered_pre_action_v1", "input": "scenario_action_bins", "physics_hz": 60})
    try:
        frame_size = config["width"] * config["height"] * 3
        count = 0
        with rows_path.open(encoding="utf-8-sig") as stream:
            for index, line in enumerate(stream):
                if index >= config["max_steps"]:
                    raise ValueError("Too many native steps")
                row = json.loads(line)
                if row["index"] != index or row["before"]["step"] != index:
                    raise ValueError("Native step order mismatch")
                validate_native_alignment(config, row["before"], row["after"], row["timing"])
                with gzip.open(raw / f"frame-{index:06d}.rgb.gz", "rb") as f:
                    pixels = f.read(frame_size + 1)
                if len(pixels) != frame_size:
                    raise ValueError("Native pixel length mismatch")
                frame = np.frombuffer(pixels, np.uint8).reshape(config["height"], config["width"], 3)
                writer.append(frame, np.asarray(row["action"]), before=row["before"], after=row["after"], timing=row["timing"])
                count += 1
        if count != manifest["steps"]:
            raise ValueError("Native step count mismatch")
        for issue in manifest.get("quality_issues", []):
            writer.issue(str(issue))
        return writer.finish(manifest["final"], accepted=manifest.get("accepted") is True,
                             complete=manifest.get("complete") is True)
    finally:
        if not writer.closed:
            writer.issue("native_import_failed")
            writer.finish({}, accepted=False, complete=False)


def record_native(*, scenario, output=ROOT / "runs/demonstrations-v215", seed=42, split="train",
                  case=None, root=ROOT, script_test=False):
    root = Path(root)
    correction = load_case(case) if case else None
    config = correction["scenario"] if correction else load_scenario(scenario)
    validate_scenario(config)
    if correction and split != "train":
        raise ValueError("Correction tasks belong to training only")
    engine_seed = correction["engine_seed"] if correction else split_seed(split, seed)
    local = local_config(root)
    validate_runtime(local)
    if config.get("require_real_map", True):
        validate_assets(Path(local["maps_dir"]), config["map"])
    job = root / "runs/native" / ("capture-" + uuid.uuid4().hex[:16])
    job.mkdir(parents=True)
    write_json(job / "scenario.json", config)
    write_json(job / "request.json", {"engine_seed": engine_seed, "split": split, "engine_contract": contract(config),
                                      "case_id": correction["id"] if correction else None})
    env = os.environ.copy()
    dotnet = Path(local["dotnet"]).parent
    env.update(DOTNET_ROOT=str(dotnet), DOTNET_CLI_TELEMETRY_OPTOUT="1",
               PATH=str(dotnet) + os.pathsep + env.get("PATH", ""))
    args = [local["godot"], "--path", local["project"], "--rendering-method", "gl_compatibility",
            "--audio-driver", "Dummy", "--resolution", f'{config["render_width"]}x{config["render_height"]}',
            "--windowed", "--", "--dev", "--data-dir", str(job / "settings"),
            "--agent-config", str(job / "scenario.json"), "--agent-human", str(job / "capture"),
            "--agent-seed", str(engine_seed)]
    if local.get("maps_dir"):
        args += ["--maps-dir", local["maps_dir"]]
    if script_test:
        args.append("--agent-human-test")
    with (job / "engine.log").open("wb") as log:
        subprocess.run(args, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                       timeout=90 if script_test else None, check=True)
    episode = import_capture(job, output)
    result = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    if correction and result["accepted"]:
        update_case(case, status="demonstrated", demonstration=episode.name)
    return episode


def run(args):
    print(record_native(scenario=args.scenario, output=args.output, seed=args.seed,
                        split=args.split, case=args.case), flush=True)
