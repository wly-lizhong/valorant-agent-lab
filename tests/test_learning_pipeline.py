import json
from pathlib import Path
import numpy as np
import pytest
from valorant_lab.env import ROOT
from valorant_lab.scenario import load_scenario, contract, noop, action_sizes, require_contract
from valorant_lab.dataset import EpisodeWriter, load_episodes, DemonstrationDataset, split_seed
from valorant_lab.corrections import enqueue_failure, load_case
from valorant_lab.recording import look_action


@pytest.fixture
def config():
    return load_scenario(ROOT / "configs/lotus_movement_combat.json")


def write_episode(root, config, split="train", engine_seed=42):
    writer = EpisodeWriter(root, config, engine_seed, split=split, chunk_size=2)
    for i in range(5):
        writer.append(np.full((config["height"], config["width"], 3), 10 + i, np.uint8), noop(config),
                      before={"position": [999, 999, 999], "frame_id": i}, after={"frame_id": i + 1}, timing={})
    writer.finish({"is_success": True}, accepted=True)
    return writer


def test_fine_controls_and_mouse_direction(config):
    assert action_sizes(config).tolist() == [3, 3, 3, 9, 9, 2, 2, 2]
    assert noop(config).tolist() == [0, 0, 0, 4, 4, 0, 0, 0]
    assert look_action(config, 12.5, -12.5, .08) == (6, 6)
    assert look_action(config, 0, 0, .08) == (4, 4)


def test_dataset_chronology_chunks_padding_and_no_info_leak(tmp_path, config):
    write_episode(tmp_path, config)
    write_episode(tmp_path, config, engine_seed=43)
    episodes = load_episodes(tmp_path, config)
    dataset = DemonstrationDataset(episodes)
    first, action = dataset[0]
    assert first.shape == (12, 144, 256)
    assert np.all(first[:9] == 0) and np.all(first[9:] == 10)
    cross_chunk, _ = dataset[3]
    assert [int(cross_chunk[i * 3, 0, 0]) for i in range(4)] == [10, 11, 12, 13]
    boundary, _ = dataset[5]
    assert np.all(boundary[:9] == 0)  # never carry the previous episode into a new one
    assert action.tolist() == noop(config).tolist()
    from collections import deque
    from valorant_lab.evaluation import stack_frame
    history = deque(maxlen=4)
    for i in range(4):
        online = stack_frame(history, np.full((144, 256, 3), 10 + i, np.uint8))
    assert np.array_equal(online, cross_chunk)


def test_same_case_cannot_leak_between_splits(tmp_path, config):
    write_episode(tmp_path, config)
    write_episode(tmp_path, config, split="validation")
    with pytest.raises(ValueError, match="both train and validation"):
        load_episodes(tmp_path, config)


def test_corruption_and_contract_fail_closed(tmp_path, config):
    writer = write_episode(tmp_path, config)
    different = {**config, "action_repeat": 6}
    with pytest.raises(ValueError, match="contract"):
        load_episodes(tmp_path, different)
    chunk = writer.directory / writer.manifest["chunks"][0]["file"]
    chunk.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        load_episodes(tmp_path, config)


def test_incomplete_failure_and_wrong_executed_action_are_excluded(tmp_path, config):
    writer = EpisodeWriter(tmp_path, config, 42)
    frame = np.zeros((144, 256, 3), np.uint8)
    writer.append(frame, noop(config), before={}, after={"action_applied": [0] * 8}, timing={})
    writer.finish({"is_success": True}, accepted=True)
    assert not writer.manifest["accepted"]
    assert load_episodes(tmp_path, config) == []


def test_failure_queue_deduplicates_and_protects_test(tmp_path, config):
    trace = [{"position": [1, 0, 2], "pitch": 80, "waypoint": 0}] * 40
    frame = np.zeros((144, 256, 3), np.uint8)
    with pytest.raises(ValueError, match="development"):
        enqueue_failure(tmp_path, config, 300000, trace, frame, split="test", model_hash="x")
    a = enqueue_failure(tmp_path, config, 200000, trace, frame, split="dev", model_hash="x")
    b = enqueue_failure(tmp_path, config, 200001, trace, frame, split="dev", model_hash="y")
    assert a == b
    case = load_case(a)
    assert case["occurrences"] == 2 and case["reason"] == "look_limit"
    assert case["replay"] == "restart_from_seed" and case["status"] == "pending"


def test_model_contract_rejects_semantic_changes(tmp_path, config):
    path = tmp_path / "model.zip"
    path.with_suffix(".json").write_text(json.dumps({"contract": contract(config)}))
    assert require_contract(path, config)
    with pytest.raises(ValueError, match="contract"):
        require_contract(path, {**config, "yaw_bins": [-10, -4, -1, -.25, 0, .25, 1, 4, 10]})
    assert split_seed("train", 42) != split_seed("test", 42)


def test_goal_rule_is_versioned_and_old_contract_stays_readable(tmp_path, config):
    legacy = {**config}
    legacy.pop("completion_rule")
    assert "completion_rule" not in contract(legacy)
    assert contract(legacy) == contract({**legacy, "completion_rule": "ordered_waypoints"})
    write_episode(tmp_path, legacy)
    with pytest.raises(ValueError, match="contract"):
        load_episodes(tmp_path, config)
    assert contract(config)["completion_rule"] == "goal_and_targets"
    assert contract(config) != contract({**config, "goal_radius": 2})
    for name in ("route", "shooting", "reload", "recovery"):
        assert contract(load_scenario(ROOT / "configs" / f"lotus_{name}.json")) == contract(config)


@pytest.mark.parametrize("field,value", [("yaw_bins", [-1, 1, 2]), ("start", [0, 1]),
                                         ("goal_marker", True), ("player_hud", False),
                                         ("completion_rule", "accept_anywhere")])
def test_invalid_task_configuration(tmp_path, config, field, value):
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({**config, field: value}))
    with pytest.raises(ValueError):
        load_scenario(path)
