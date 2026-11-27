import copy
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

import pytest

from valorant_lab import experiments as ex


@pytest.fixture
def manifest():
    value = ex.read_json(ex.ROOT / "configs/bc-experiments-round01.json")
    value["training"]["precision"] = "fp32"
    return value


@pytest.mark.parametrize("fault", ["four", "duplicate", "path", "test", "validation", "seeds", "unbounded", "command"])
def test_rejects_unbounded_search_and_protected_split(manifest, fault):
    if fault == "four":
        manifest["candidates"].append({"id": "fourth", "policy_variant": "legacy", "class_balance": "none"})
    elif fault == "duplicate":
        manifest["candidates"][2]["policy_variant"] = "normalized"
    elif fault == "path":
        manifest["candidates"][0]["id"] = "../escape"
    elif fault in ("test", "validation"):
        manifest["evaluation"]["split"] = fault
    elif fault == "seeds":
        manifest["evaluation"]["seed"] = 99999
    elif fault == "unbounded":
        manifest["limits"]["attempts_per_stage"] = 999
    else:
        manifest["training"]["command"] = "echo not-an-experiment"
    with pytest.raises(ValueError):
        ex.validate_manifest(manifest)


def test_stage_plan_keeps_all_candidates_comparable_and_test_unopened(tmp_path, manifest):
    plan = list(ex.stage_plan(ex.ROOT, tmp_path / "out", tmp_path / "data", manifest, "cuda"))
    assert len(plan) == 15
    assert len({stage.candidate for stage in plan}) == 3
    for stage in plan:
        assert stage.timeout > 0
        if stage.kind == "evaluate":
            assert stage.command[stage.command.index("--split") + 1] == "dev"
            assert "--save-frames" not in stage.command
        elif stage.kind == "train":
            assert stage.command[stage.command.index("--epochs") + 1] == "60"
            assert stage.command[stage.command.index("--batch-size") + 1] == "128"


def test_timeout_stops_owned_parent_and_descendant(tmp_path):
    marker = tmp_path / "orphan.txt"
    child = tmp_path / "child.py"
    child.write_text("import sys,time\nfrom pathlib import Path\ntime.sleep(2)\nPath(sys.argv[1]).write_text('orphan')\n")
    parent = tmp_path / "parent.py"
    parent.write_text("import subprocess,sys,time\nsubprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\nprint('launched',flush=True)\ntime.sleep(60)\n")
    result = ex.run_process([sys.executable, str(parent), str(child), str(marker)],
        cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "timeout.log", timeout=.4)
    assert result["timed_out"] and result["returncode"] != 0
    assert result["elapsed_seconds"] < 12
    time.sleep(2.1)
    assert not marker.exists()


def test_failure_keeps_process_output(tmp_path):
    result = ex.run_process([sys.executable, "-c", "print('specific failure',flush=True);raise SystemExit(7)"],
        cwd=tmp_path, env=os.environ.copy(), log_path=tmp_path / "failure.log", timeout=10)
    assert result["returncode"] == 7 and not result["timed_out"]
    assert "specific failure" in (tmp_path / "failure.log").read_text()


@pytest.fixture
def campaign(tmp_path, manifest, monkeypatch):
    manifest["candidates"] = manifest["candidates"][:1]
    root, data, output = (tmp_path / name for name in ("source", "data", "output"))
    root.mkdir()
    data.mkdir()
    source = root / "source.txt"
    source.write_text("original source")
    helper = root / "fixture.py"
    helper.write_text("""import json,sys
from pathlib import Path
output,flag=map(Path,sys.argv[1:])
output.mkdir(parents=True,exist_ok=True)
if flag.exists():
    (output/'partial.txt').write_text('keep failed evidence')
    print('deliberate stage failure',flush=True)
    raise SystemExit(7)
(output/'report.json').write_text(json.dumps({'ok':True}))
""")
    def identity(*args, **kwargs):
        return {"source": ex.file_hash(source), "manifest": copy.deepcopy(manifest)}
    def stages(root, output, data, manifest, device):
        for name in ("first", "second"):
            directory = output / "normalized" / name
            yield ex.Stage("normalized/" + name, "fixture", "normalized", directory,
                [sys.executable, str(helper), str(directory), str(root / (name + ".fail"))], ("report.json",), 15)
    monkeypatch.setattr(ex, "campaign_identity", identity)
    monkeypatch.setattr(ex, "stage_plan", stages)
    monkeypatch.setattr(ex, "GPUSampler", lambda *a, **k: nullcontext())
    def create(resume=False):
        return ex.Campaign(root, data, output, manifest, device="cpu", resume=resume)
    return create, root, output, manifest


def test_resume_skips_verified_stages_and_preserves_failed_attempt(campaign):
    create, root, output, _ = campaign
    (root / "second.fail").touch()
    with pytest.raises(RuntimeError, match="failed"):
        create().run()
    failed = ex.read_json(output / "state.json")
    assert failed["stages"]["normalized/first"]["status"] == "completed"
    assert failed["current_stage"] == "normalized/second"
    assert failed["consumed_seconds"] > 0
    old = Path(failed["stages"]["normalized/second"]["attempts"][0]["log"])
    partial = old.parent / "artifacts/partial.txt"
    assert partial.read_text() == "keep failed evidence"
    (root / "second.fail").unlink()
    completed = create(resume=True).run()
    assert completed["status"] == "completed"
    assert len(completed["stages"]["normalized/first"]["attempts"]) == 1
    assert len(completed["stages"]["normalized/second"]["attempts"]) == 2
    assert partial.read_text() == "keep failed evidence"
    assert "deliberate stage failure" in old.read_text()
    assert completed["consumed_seconds"] > failed["consumed_seconds"]
    assert create(resume=True).run()["status"] == "completed"


@pytest.mark.parametrize("fault", ["artifact", "source", "manifest"])
def test_dirty_resume_never_launches_another_attempt(campaign, fault):
    create, root, output, manifest = campaign
    state = create().run()
    if fault == "artifact":
        path = next(iter(state["stages"]["normalized/first"]["artifacts"]))
        (output / path).write_text("tampered")
    elif fault == "source":
        (root / "source.txt").write_text("new code")
    else:
        manifest["training"]["epochs"] += 1
    with pytest.raises(ValueError, match="changed|disappeared"):
        create(resume=True).run()
    assert len(ex.read_json(output / "state.json")["stages"]["normalized/first"]["attempts"]) == 1


def test_attempt_limit_and_accumulated_budget_stop_before_launch(campaign):
    create, root, output, manifest = campaign
    (root / "first.fail").touch()
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="failed"):
            create(resume=resume).run()
    with pytest.raises(RuntimeError, match="Attempt limit"):
        create(resume=True).run()
    state = ex.read_json(output / "state.json")
    assert len(state["stages"]["normalized/first"]["attempts"]) == 2
    state["consumed_seconds"] = manifest["limits"]["total_seconds"]
    ex.atomic_json(output / "state.json", state)
    with pytest.raises(RuntimeError, match="time budget"):
        create(resume=True).run()


def test_output_lock_rejects_second_writer(tmp_path):
    with ex.output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="runner"):
            with ex.output_lock(tmp_path):
                pytest.fail("second writer entered")


def test_summary_reports_actions_and_actual_shots_not_fire_button_count(tmp_path):
    episode = tmp_path / "episode-0000"
    episode.mkdir()
    with (episode / "trace.jsonl").open("w", encoding="utf-8") as stream:
        for index in range(80):
            stream.write(json.dumps({"requested_action": [1, 0, 0, 4, 4, 0, int(index >= 60), 0],
                                    "info": {"position": [0, 0, 0], "pitch": 80}}) + "\n")
    ex.atomic_json(tmp_path / "report.json", {"split": "dev", "clock": "lockstep", "success_rate": 0.,
        "episodes": [{"engine_seed": 200000, "success": False, "final": {"shots": 2, "hits": 1, "kills": 0}}]})
    result = ex.summarize_gameplay(tmp_path / "report.json")
    assert result["shots"] == 2 and result["hits"] == 1
    assert result["unique_actions"] == 2
    assert result["head_counts"]["fire"] == {"0": 60, "1": 20}
    assert result["blocked_windows"] == 41
    assert result["look_limit_steps"] == 80 and result["longest_constant_action_steps"] == 60


def test_summary_refuses_trace_path_escape(tmp_path):
    ex.atomic_json(tmp_path / "report.json", {"episodes": [{"artifact_directory": "../outside"}]})
    with pytest.raises(ValueError, match="escapes"):
        ex.summarize_gameplay(tmp_path / "report.json")


@pytest.mark.parametrize("fault", ["overlap", "model", "seeds", "missing_trace", "empty_trace", "scenario", "test_model"])
def test_evaluation_stage_requires_exact_model_seeds_and_real_trace(tmp_path, manifest, fault):
    manifest["evaluation"]["episodes"] = 1
    model, scenario = tmp_path / "best.zip", tmp_path / "scenario.json"
    model.write_bytes(b"model-fixture")
    ex.atomic_json(scenario, {"id": "route-fixture"})
    directory = tmp_path / "evaluation"
    episode = directory / "episode-0000"
    episode.mkdir(parents=True)
    trace = episode / "trace.jsonl"
    trace.write_text(json.dumps({"requested_action": [1, 0, 0, 4, 4, 0, 0, 0],
                                 "info": {"position": [0, 0, 0]}}) + "\n")
    report = {"split": "dev", "clock": "lockstep", "model_test_only": False,
              "scenario": ex.read_json(scenario), "model_sha256": ex.file_hash(model), "success_rate": 0.,
              "episodes": [{"engine_seed": 200000, "success": False, "training_overlap": False,
                            "final": {"shots": 0, "hits": 0, "kills": 0}}]}
    ex.atomic_json(directory / "report.json", report)
    stage = ex.Stage("normalized/eval-route", "evaluate", "normalized", directory,
                     [sys.executable, "--model", str(model), "--scenario", str(scenario)], ("report.json",), 10)
    ex.validate_stage(stage, manifest)
    if fault == "overlap":
        report["episodes"][0]["training_overlap"] = True
    elif fault == "model":
        report["model_sha256"] = "wrong-model"
    elif fault == "seeds":
        report["episodes"][0]["engine_seed"] = 100142
    elif fault == "missing_trace":
        trace.unlink()
    elif fault == "empty_trace":
        trace.write_text("")
    elif fault == "scenario":
        report["scenario"]["id"] = "another-scenario"
    else:
        report["model_test_only"] = True
    ex.atomic_json(directory / "report.json", report)
    with pytest.raises((ValueError, FileNotFoundError)):
        ex.validate_stage(stage, manifest)


def test_gpu_summary_marks_scope_and_unavailable_samples(tmp_path):
    path = tmp_path / "gpu.jsonl"
    rows = [{"stage": "a/train", "utilization_percent": 50, "memory_mib": 100, "power_w": 80},
            {"stage": "a/train", "utilization_percent": 90, "memory_mib": 150, "power_w": 100},
            {"stage": "a/train", "error": "unavailable"}]
    path.write_text("\n".join(map(json.dumps, rows)) + "\n{partial", encoding="utf-8")
    result = ex.gpu_summary(path)
    assert result["stages"]["a/train"]["mean_utilization_percent"] == 70
    assert result["stages"]["a/train"]["peak_memory_mib"] == 150
    assert result["errors"] == ["unavailable"]


def test_identity_hashes_real_data_and_refuses_corrupt_chunks(tmp_path, manifest):
    import numpy as np
    from valorant_lab.dataset import EpisodeWriter
    from valorant_lab.scenario import load_scenario, noop
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    for filename in ("run_bc_experiments.py", "check_bc_policy.py"):
        (root / "scripts" / filename).write_text("# fixture")
    for name in (manifest["scenario"], *manifest["evaluation"]["scenarios"].values()):
        (root / name).write_bytes((ex.ROOT / name).read_bytes())
    config = load_scenario(root / manifest["scenario"])
    data = tmp_path / "data"
    for seed, split in ((42, "train"), (100142, "validation")):
        writer = EpisodeWriter(data, config, seed, source="human", split=split)
        writer.append(np.zeros((144, 256, 3), dtype=np.uint8), noop(config), before={}, after={}, timing={})
        writer.finish({"is_success": True}, accepted=True)
    identity = ex.campaign_identity(root, data, manifest, device="cpu", gpu_display=False, gpu=0)
    assert {row["split"] for row in identity["data"]} == {"train", "validation"}
    next(data.glob("episode-*/chunk-*.npz")).write_bytes(b"bad archive")
    with pytest.raises(ValueError, match="Corrupt"):
        ex.campaign_identity(root, data, manifest, device="cpu", gpu_display=False, gpu=0)
