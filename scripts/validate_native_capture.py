"""Real rendered recorder QA; synthetic inputs are always tagged script_test and rejected.

Run on a machine with a working local runtime/display. Never run alongside a human recording.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import uuid

from valorant_lab.dataset import write_json
from valorant_lab.env import ROOT, validate_runtime
from valorant_lab.native_capture import import_capture, local_config
from valorant_lab.scenario import contract, load_scenario

CASES = {
    "steady": [],
    "action-pause": ["--agent-human-test-stall-ms", "350"],
    "capture-pause": ["--agent-human-test-stall-ms", "350", "--agent-human-test-stall-phase", "capture"],
    "repeated": ["--agent-human-test-stall-ms", "350", "--agent-human-test-stall-every", "2"],
    "severe": ["--agent-human-test-stall-ms", "900"],
}


def check(name, steps, *, guide=True):
    local = local_config()
    validate_runtime(local)
    config = load_scenario(ROOT / "configs/lotus_route.json")
    config["max_steps"] = steps
    job = ROOT / "runs/native" / ("qa-" + name + "-" + uuid.uuid4().hex[:8])
    job.mkdir(parents=True)
    write_json(job / "scenario.json", config)
    write_json(job / "request.json", {"engine_seed": 42, "split": "train",
                                     "engine_contract": contract(config), "case_id": None})
    env = os.environ.copy()
    env["DOTNET_ROOT"] = str(Path(local["dotnet"]).parent)
    env["PATH"] = env["DOTNET_ROOT"] + os.pathsep + env.get("PATH", "")
    args = [local["godot"], "--path", local["project"], "--rendering-method", "gl_compatibility",
            "--audio-driver", "Dummy", "--windowed", "--resolution", "960x540", "--", "--dev",
            "--data-dir", str(job / "settings"), "--maps-dir", local["maps_dir"],
            "--agent-config", str(job / "scenario.json"), "--agent-human", str(job / "capture"),
            "--agent-seed", "42", "--agent-human-test", "--agent-human-test-steps", str(steps),
            "--agent-human-test-pan",
            "--agent-human-test-guide-images" if guide else "--agent-human-test-no-guide", *CASES[name]]
    with (job / "engine.log").open("wb") as log:
        subprocess.run(args, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=max(90, steps * .15))
    manifest = json.loads((job / "capture/native.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "script_test" and not manifest["accepted"], job
    log_text = (job / "engine.log").read_text(encoding="utf-8-sig")
    if guide:
        assert "[native-guide] separate_viewport=true" in log_text, job
        assert (job / "capture/teacher-guide.png").is_file(), job
    assert manifest["performance"]["fps_cap"] == 60, job
    if name in ("repeated", "severe"):
        assert manifest["quality_issues"] == ["sustained_render_lag"], job
        assert 0 < manifest["steps"] < steps, job
    else:
        assert not manifest["quality_issues"] and manifest["steps"] == steps, job
        if name != "steady":
            assert manifest["performance"]["recovered_stalls"] >= 1, job
        final = manifest["final"]
        # The discarded backlog must remain missing from simulation, not catch up later.
        assert abs(final["wall_seconds"] - final["dropped_seconds"] - final["simulation_time"]) < .2, job
    # Strict import checks every pre-action frame, label, clock delta and bounded pause.
    episode = import_capture(job, ROOT / "runs/native-qa-demonstrations")
    converted = json.loads((episode / "episode.json").read_text(encoding="utf-8"))
    assert not converted["accepted"] and converted["source"] == "script_test", job
    rows = [json.loads(line) for line in (job / "capture/steps.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    for prior, current in zip(rows, rows[1:]):
        for key in ("step", "simulation_time", "position", "yaw", "pitch", "ammo"):
            assert prior["after"][key] == current["before"][key], (job, key)
    if name == "action-pause":
        assert any(row["timing"]["paused_seconds"] >= .3 for row in rows), job
    result = {"case": name, "guide": guide, "job": str(job), "steps": manifest["steps"],
              "wall_seconds": manifest["final"]["wall_seconds"],
              "simulation_seconds": manifest["final"]["simulation_time"],
              "quality_issues": manifest["quality_issues"], **manifest["performance"]}
    write_json(job / "validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--without-guide", action="store_true", help="Diagnostic A/B baseline only")
    args = parser.parse_args()
    if not 50 <= args.steps <= 1200:
        parser.error("--steps must be 50..1200")
    for name in CASES if args.case == "all" else [args.case]:
        check(name, args.steps, guide=not args.without_guide)
