"""Regression checks for conditional BC objectives and restart continuity."""
import json
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO
from valorant_lab import learning
from valorant_lab.bc_checkpoint import load_checkpoint
from valorant_lab.bc_objective import ActionMetrics, behavioral_loss, make_class_weights, train_action_counts
from valorant_lab.dataset import EpisodeWriter, load_episodes
from valorant_lab.env import ROOT
from valorant_lab.scenario import action_sizes, load_scenario, noop, validate_scenario


@pytest.fixture
def sample_data(tmp_path):
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    config.update(width=64, height=64)
    for episode, (split, length) in enumerate((("train", 7), ("validation", 3))):
        writer = EpisodeWriter(tmp_path / "data", config, 42 + episode,
                               source="script_test", split=split, chunk_size=3)
        for step in range(length):
            pixels = np.random.default_rng(episode * 100 + step).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            action = noop(config)
            action[0] = int(step % 3 != 0)
            action[3] = 3 if step % 2 else 5
            action[6] = int(step % 3 == 0)
            writer.append(pixels, action, before={}, after={}, timing={})
        writer.finish({"is_success": True}, accepted=True)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(config), encoding="utf-8")
    return config, scenario, load_episodes(tmp_path / "data", config, allow_test_data=True)


def arguments(tmp_path, scenario, output, **overrides):
    values = dict(cpu_threads=1, seed=42, scenario=str(scenario), data=str(tmp_path / "data"),
                  allow_test_data=True, output=str(output), learning_rate=3e-4, device="cpu", batch_size=4,
                  epochs=3, data_cache="cpu", cache_mib=16, policy_variant="normalized", class_balance="sqrt",
                  checkpoint_every_epochs=2)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_class_weights_use_train_only_preserve_negatives_and_cap(sample_data):
    config, _, episodes = sample_data
    counts = train_action_counts(episodes, action_sizes(config))
    assert counts[6].tolist() == [4, 3]  # validation's positive is excluded
    assert all(int(count.sum()) == 7 for count in counts)
    extreme = [torch.tensor([10000, 1, 0]), torch.tensor([7, 0])]
    weights = make_class_weights(extreme, "sqrt", 4)
    for count, weight in zip(extreme, weights):
        assert bool((weight > 0).all())
        assert float(weight.max() / weight.min()) <= 4.00001
        assert torch.isclose((weight * count).sum() / count.sum(), torch.tensor(1.))
    assert make_class_weights(counts, "none") is None
    with pytest.raises(ValueError):
        make_class_weights(counts, "sqrt", float("nan"))


def test_weighted_loss_keeps_negative_firing_gradient():
    from stable_baselines3.common.distributions import MultiCategoricalDistribution
    logits = torch.zeros(2, 16, requires_grad=True)
    distribution = MultiCategoricalDistribution([2] * 8).proba_distribution(logits)
    targets = torch.zeros(2, 8, dtype=torch.int64)
    targets[1, 6] = 1
    weights = make_class_weights([[99, 1]] * 8, "sqrt", 8)
    loss = behavioral_loss(distribution, targets, weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 13] > 0  # negative label penalizes firing
    assert logits.grad[1, 13] < 0  # positive label encourages firing
    assert abs(logits.grad[1, 13] / logits.grad[0, 13]) <= 8.00001


def test_metrics_expose_majority_collapse_and_absent_classes():
    sizes = [3, 3, 3, 9, 9, 2, 2, 2]
    truth = torch.tensor([[1, 0, 0, 4, 4, 0, 0, 0], [0, 0, 0, 5, 4, 0, 1, 0]])
    predicted = truth[[0, 0]]
    metric = ActionMetrics(sizes)
    metric.update(predicted, truth)
    result = metric.compute()
    assert result["events"]["fire"]["recall"] == 0
    assert result["events"]["fire"]["false_positive_rate"] == 0
    assert result["events"]["stop"]["recall"] == 0
    assert result["events"]["turn"]["recall"] == 0
    assert result["heads"][0]["class_recall"][2] is None


def test_turn_metrics_use_real_zero_index_and_exclude_pitch_only_changes():
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    config["yaw_bins"] = [-12, 0, .25, 1, 4]
    validate_scenario(config)
    zero = noop(config)
    assert zero[3] == 1 and zero[3] != len(config["yaw_bins"]) // 2
    target = torch.as_tensor(np.tile(zero, (2, 1)))
    predicted = target.clone()
    target[1, 4] = predicted[1, 4] = 0  # pitch-only motion is not a horizontal turn
    metrics = ActionMetrics(action_sizes(config), zero_yaw=zero[3])
    metrics.update(predicted, target)
    turn = metrics.compute()["events"]["turn"]
    assert turn["positive_labels"] == turn["positive_predictions"] == 0
    assert turn["recall"] is None and turn["false_positive_rate"] == 0
    with pytest.raises(ValueError, match="Zero-yaw"):
        ActionMetrics(action_sizes(config), zero_yaw=99)


def test_normalization_controls_feature_scale_and_legacy_loads(sample_data):
    config, _, _ = sample_data
    torch.set_num_threads(1)
    images = torch.randint(0, 256, (3, 12, 64, 64), dtype=torch.uint8)
    for variant in ("legacy", "normalized", "normalized-relu"):
        model = PPO("CnnPolicy", learning.OfflineSpaces(config), n_steps=2, batch_size=2,
                    policy_kwargs=learning.policy_kwargs(variant), device="cpu", seed=42)
        with torch.no_grad():
            features = model.policy.extract_features(images)
        if variant != "legacy":
            assert torch.allclose(features.mean(1), torch.zeros(3), atol=1e-5)
            assert torch.allclose(features.var(1, unbiased=False), torch.ones(3), atol=1e-3)
            assert float(features.std(dim=0).mean()) > .01
        with BytesIO() as buffer:
            model.save(buffer)
            buffer.seek(0)
            loaded = PPO.load(buffer, device="cpu")
            # Deterministic actor output survives export on every architecture.
            assert np.array_equal(model.predict(images[0].numpy(), deterministic=True)[0],
                                  loaded.predict(images[0].numpy(), deterministic=True)[0])
        model.get_env().close()


def test_resume_matches_uninterrupted_parameters_and_optimizer(tmp_path, sample_data):
    _, scenario, _ = sample_data
    uninterrupted = tmp_path / "full"
    resumed = tmp_path / "resumed"
    learning.bc_train(arguments(tmp_path, scenario, uninterrupted))
    learning.bc_train(arguments(tmp_path, scenario, resumed, epochs=1))
    learning.bc_train(arguments(tmp_path, scenario, resumed, resume=True))
    left = PPO.load(uninterrupted / "latest.zip", device="cpu")
    right = PPO.load(resumed / "latest.zip", device="cpu")
    for name, tensor in left.policy.state_dict().items():
        assert torch.equal(tensor, right.policy.state_dict()[name]), name
    for key, state in left.policy.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            assert torch.equal(value, right.policy.optimizer.state_dict()["state"][key][field])
    a = json.loads((uninterrupted / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((resumed / "metrics.json").read_text(encoding="utf-8"))
    assert [e["validation_loss"] for e in a["epochs"]] == [e["validation_loss"] for e in b["epochs"]]
    assert b["runtime"]["start_epoch"] == 1
    assert len(list(resumed.glob("resume*.pt"))) == 2
    with pytest.raises(ValueError, match="already contains"):
        learning.bc_train(arguments(tmp_path, scenario, resumed))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        learning.bc_train(arguments(tmp_path, scenario, resumed, resume=True, class_balance="none"))


def test_restart_falls_back_only_for_damaged_bundle(tmp_path, sample_data):
    _, scenario, _ = sample_data
    output = tmp_path / "corrupt"
    learning.bc_train(arguments(tmp_path, scenario, output, epochs=2, checkpoint_every_epochs=1))
    state = torch.load(output / "resume.pt", weights_only=True)
    (output / "resume.pt").write_bytes(b"interrupted bytes")
    recovered, name = load_checkpoint(output, state["identity"])
    assert name == "resume.previous.pt" and recovered["epoch"] == 1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_checkpoint(output, {**state["identity"], "batch_size": 999})
    learning.bc_train(arguments(tmp_path, scenario, output, epochs=2, resume=True))
    assert json.loads((output / "latest.json").read_text())["epochs"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual normalized CUDA AMP restart acceptance")
def test_normalized_balanced_amp_checkpoint_restores_scaler(tmp_path, sample_data):
    _, scenario, _ = sample_data
    output = tmp_path / "amp-resume"
    options = dict(device="cuda", precision="amp-fp16", data_cache="cuda")
    learning.bc_train(arguments(tmp_path, scenario, output, epochs=1, **options))
    saved = torch.load(output / "resume.pt", weights_only=True)
    assert saved["scaler"]["scale"] > 0
    assert saved["completed_updates"] == 2
    before = PPO.load(output / "latest.zip", device="cpu").policy.state_dict()
    # Resuming at the same total epoch restores exports without another update.
    learning.bc_train(arguments(tmp_path, scenario, output, epochs=1, resume=True, **options))
    after = PPO.load(output / "latest.zip", device="cpu").policy.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    learning.bc_train(arguments(tmp_path, scenario, output, epochs=2, resume=True, **options))
    resumed = torch.load(output / "resume.pt", weights_only=True)
    assert resumed["completed_updates"] == 4
    assert len(resumed["rng"]["cuda"]) == torch.cuda.device_count()
    assert all(epoch["skipped_nonfinite_updates"] == 0 for epoch in resumed["log"])
