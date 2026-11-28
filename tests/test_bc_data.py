"""Cached training must preserve the original pixel/action and sampler semantics."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from valorant_lab.bc_data import CachedDemonstrations, cache_bytes, make_bc_batches, select_cache
from valorant_lab.dataset import EpisodeWriter, DemonstrationDataset, load_episodes
from valorant_lab.env import ROOT
from valorant_lab.scenario import load_scenario, noop


def test_initial_idle_view_keeps_histories_later_stops_and_all_validation(tmp_path):
    from valorant_lab.bc_data import training_sample_indices
    from valorant_lab.bc_objective import train_action_counts
    from valorant_lab.dataset import sha256
    from valorant_lab.scenario import action_sizes
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    first_active = {42: 3, 43: 1, 100142: 3}
    for seed, split in ((42, "train"), (43, "train"), (100142, "validation")):
        writer = EpisodeWriter(tmp_path, config, seed, source="script_test", split=split, chunk_size=2)
        for step in range(8):
            frame = np.full((144, 256, 3), (seed + step) % 256, np.uint8)
            action = noop(config)
            if step == first_active[seed]:
                action[0] = 1
            if step == 6:
                action[6] = 1
            writer.append(frame, action, before={}, after={}, timing={})
        writer.finish({"is_success": True}, accepted=True)
    episodes = load_episodes(tmp_path, config, allow_test_data=True)
    before = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    indices, report = training_sample_indices(episodes, "skip")
    train = [e for e in episodes if e[1]["split"] == "train"]
    validation = [e for e in episodes if e[1]["split"] == "validation"]
    expected = [offset * 8 + i for offset, (_, m) in enumerate(train)
                for i in range(first_active[m["engine_seed"]], 8)]
    assert indices.tolist() == expected and report["excluded_initial_idle_steps"] == 4
    assert len(indices) == 12  # Later stops, including the final no-op, remain selected.
    reference = DemonstrationDataset(train)
    train_cached, val_cached, info = make_bc_batches(
        episodes, device="cpu", batch_size=5, mode="cpu", budget_mib=16, train_indices=indices)
    torch.manual_seed(42)
    actual = list(train_cached)
    train_stream, val_stream, _ = make_bc_batches(
        episodes, device="cpu", batch_size=5, mode="stream", budget_mib=16, train_indices=indices)
    torch.manual_seed(42)
    streamed = list(train_stream)
    for (x, y), (a, b) in zip(actual, streamed):
        assert torch.equal(x, a) and torch.equal(y, b)
    cached = train_cached.dataset
    for index in indices:
        x, y = cached.batch([index])
        expected_x, expected_y = reference[int(index)]
        assert np.array_equal(x[0].numpy(), expected_x)
        assert np.array_equal(y[0].numpy(), expected_y)
    assert torch.equal(torch.cat([y for _, y in val_cached]), torch.cat([y for _, y in val_stream]))
    assert sum(len(y) for _, y in val_stream) == len(DemonstrationDataset(validation)) == 8
    assert info["training_steps"] == 12 and info["original_training_steps"] == 16
    counts = train_action_counts(episodes, action_sizes(config), indices)
    assert counts[0].sum() == 12 and counts[6].tolist() == [10, 2]
    assert before == {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    kept, kept_report = training_sample_indices(episodes)
    assert kept is None and kept_report["excluded_initial_idle_steps"] == 0


@pytest.mark.parametrize("indices", [[], [1, 1], [-1, 0], [999], [2, 1], [1.5]])
def test_sampling_indices_cannot_alias_or_escape_training(demonstrations, indices):
    _, episodes = demonstrations
    with pytest.raises(ValueError, match="Training indices"):
        make_bc_batches(episodes, device="cpu", batch_size=2, train_indices=indices)


@pytest.fixture
def demonstrations(tmp_path):
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    for number, (split, count) in enumerate((("train", 5), ("train", 4), ("validation", 3))):
        writer = EpisodeWriter(tmp_path / "data", config, 42 + number,
                               source="script_test", split=split, chunk_size=2)
        for i in range(count):
            frame = np.arange(144 * 256 * 3, dtype=np.int64).reshape(144, 256, 3)
            frame = ((frame + number * 37 + i * 5) % 256).astype(np.uint8)
            action = noop(config)
            action[0] = i % 3
            action[3] = (number + i) % 9
            writer.append(frame, action, before={"position": [999, 999, 999]}, after={}, timing={})
        writer.finish({"is_success": True}, accepted=True)
    return config, load_episodes(tmp_path / "data", config, allow_test_data=True)


def test_cached_batch_matches_streaming_pixels_actions_and_boundaries(demonstrations):
    _, episodes = demonstrations
    reference = DemonstrationDataset(episodes)
    cached = CachedDemonstrations(episodes)
    indices = [0, 4, 5, 8, 9, 11, 2, 5]
    images, actions = cached.batch(indices)
    for row, index in enumerate(indices):
        expected_image, expected_action = reference[index]
        assert np.array_equal(images[row].numpy(), expected_image)
        assert np.array_equal(actions[row].numpy(), expected_action)
    tensors = (cached.frames, cached.actions, cached.histories)
    assert sum(t.numel() * t.element_size() for t in tensors) == cache_bytes(episodes)


def test_cache_decodes_once_and_keeps_shuffle_partial_batch_rng(demonstrations, monkeypatch):
    _, episodes = demonstrations
    original = np.load
    decoded = []
    def counting_load(path, **kwargs):
        decoded.append(str(path))
        return original(path, **kwargs)
    monkeypatch.setattr(np, "load", counting_load)
    cached_train, cached_val, info = make_bc_batches(
        episodes, device="cpu", batch_size=4, mode="auto", budget_mib=16)
    assert info["mode"] == "cpu"
    assert len(decoded) == sum(len(m["chunks"]) for _, m in episodes)
    initial_reads = len(decoded)
    torch.manual_seed(133)
    actual = list(cached_train) + list(cached_val)
    assert len(decoded) == initial_reads
    assert [len(x) for x, _ in actual] == [4, 4, 1, 3]
    stream_train, stream_val, stream_info = make_bc_batches(
        episodes, device="cpu", batch_size=4, mode="stream", budget_mib=16)
    torch.manual_seed(133)
    expected = list(stream_train) + list(stream_val)
    assert stream_info["resident_bytes"] == 0
    for (x, y), (a, b) in zip(actual, expected):
        assert torch.equal(x, a) and torch.equal(y, b)


def test_cache_budget_and_cuda_reserve_are_respected():
    gib = 1024**3
    needed = 300 * 1024**2
    assert select_cache("auto", needed, gib, "cuda", cuda_free=16*gib) == "cuda"
    assert select_cache("auto", needed, gib, "cuda", cuda_free=gib) == "cpu"
    assert select_cache("auto", needed, 1, "cpu") == "stream"
    assert select_cache("cpu", needed, gib, "cpu") == "cpu"
    with pytest.raises(ValueError, match="exceed"):
        select_cache("cpu", needed, 1, "cpu")
    with pytest.raises(ValueError, match="CUDA cache"):
        select_cache("cuda", needed, gib, "cuda", cuda_free=gib)
    with pytest.raises(ValueError, match="CUDA cache"):
        select_cache("cuda", needed, gib, "cpu")
    with pytest.raises(ValueError, match="budget"):
        select_cache("auto", needed, 0, "cpu")


def test_cached_decode_keeps_tensor_validation(demonstrations):
    _, episodes = demonstrations
    path, meta = episodes[0]
    chunk = path / meta["chunks"][0]["file"]
    with np.load(chunk) as data:
        frames, actions = data["frames"], data["actions"]
    actions[0, 0] = 999
    np.savez_compressed(chunk, frames=frames, actions=actions)
    with pytest.raises(ValueError, match="Invalid demonstration tensor"):
        CachedDemonstrations(episodes)


def test_bc_cached_training_matches_streaming_weights(tmp_path, demonstrations):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from valorant_lab.learning import bc_train
    config, _ = demonstrations
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(config))
    results = []
    for mode in ("stream", "cpu"):
        output = tmp_path / mode
        args = SimpleNamespace(cpu_threads=1, seed=42, scenario=str(scenario),
            data=str(tmp_path / "data"), allow_test_data=True, output=str(output),
            learning_rate=3e-4, device="cpu", batch_size=4, epochs=2,
            data_cache=mode, cache_mib=16)
        bc_train(args)
        model = PPO.load(output / "latest.zip", device="cpu")
        results.append(model.policy.state_dict())
        metrics = json.loads((output / "metrics.json").read_text())
        assert metrics["training_steps"] == 9 and metrics["validation_steps"] == 3
        assert metrics["runtime"]["data_cache"]["mode"] == mode
        assert all(e["train_samples_per_second"] > 0 for e in metrics["epochs"])
        assert json.loads((output / "latest.json").read_text())["test_only"]
    for name in results[0]:
        assert torch.equal(results[0][name], results[1][name]), name


def test_sparse_checkpoints_keep_earlier_best_weights(tmp_path, demonstrations, monkeypatch):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from valorant_lab import learning
    config, _ = demonstrations
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(config))
    real_constructor = learning.PPO

    def constructed(*args, **kwargs):
        model = real_constructor(*args, **kwargs)
        distribution = model.policy.get_distribution
        validation_calls = 0

        def controlled_validation(images):
            nonlocal validation_calls
            dist = distribution(images)
            if model.policy.training:
                return dist
            validation_calls += 1
            # One validation batch per epoch: epoch2 is best, between disk
            # checkpoints, and subsequent epochs deliberately get worse.
            cost = (2., 1., 3., 4.)[validation_calls - 1]
            return SimpleNamespace(
                log_prob=lambda actions: torch.full((len(actions),), -8 * cost),
                mode=dist.mode)

        model.policy.get_distribution = controlled_validation
        return model

    monkeypatch.setattr(learning, "PPO", constructed)
    saved = []
    save = learning.save_model

    def tracked_save(model, path, meta):
        saved.append((path.parent.name, meta["epochs"]))
        return save(model, path, meta)

    monkeypatch.setattr(learning, "save_model", tracked_save)
    snapshots = []
    for interval in (1, 3):
        output = tmp_path / f"interval-{interval}"
        args = SimpleNamespace(cpu_threads=1, seed=42, scenario=str(scenario),
            data=str(tmp_path / "data"), allow_test_data=True, output=str(output),
            learning_rate=3e-4, device="cpu", batch_size=4, epochs=4,
            data_cache="cpu", cache_mib=16, checkpoint_every_epochs=interval)
        learning.bc_train(args)
        best = PPO.load(output / "best.zip", device="cpu").policy.state_dict()
        latest = PPO.load(output / "latest.zip", device="cpu").policy.state_dict()
        snapshots.append((best, latest))
        assert json.loads((output / "best.json").read_text())["epochs"] == 2
        assert json.loads((output / "latest.json").read_text())["epochs"] == 4
        assert any(not torch.equal(best[k], latest[k]) for k in best)
    assert [epoch for name, epoch in saved if name == "interval-3"] == [1, 3, 4]
    for dense, sparse in zip(snapshots[0], snapshots[1]):
        assert all(torch.equal(dense[k], sparse[k]) for k in dense)


def test_mixed_precision_requires_cuda_before_loading_data(monkeypatch):
    pytest.importorskip("stable_baselines3")
    from valorant_lab.learning import bc_train
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="CUDA training device"):
        bc_train(SimpleNamespace(precision="amp-fp16", device="cpu"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA training device"):
        bc_train(SimpleNamespace(precision="amp-fp16", device="auto"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA AMP acceptance")
def test_cuda_amp_finite_loss_fp32_export_and_input_scaling(tmp_path, demonstrations):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from valorant_lab.learning import bc_train, training_distribution
    config, _ = demonstrations
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "amp"
    args = SimpleNamespace(cpu_threads=1, seed=42, scenario=str(scenario),
        data=str(tmp_path / "data"), allow_test_data=True, output=str(output),
        learning_rate=3e-4, device="cuda", batch_size=4, epochs=2,
        data_cache="cuda", cache_mib=16, precision="amp-fp16")
    bc_train(args)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["runtime"]["precision"] == "amp-fp16"
    for epoch in metrics["epochs"]:
        assert np.isfinite(epoch["train_loss"]) and np.isfinite(epoch["validation_loss"])
        assert epoch["optimizer_updates"] == 3
        assert epoch["skipped_nonfinite_updates"] == 0
    model = PPO.load(output / "best.zip", device="cuda")
    assert all(p.dtype == torch.float32 for p in model.policy.parameters())
    frame = torch.randint(0, 256, (2, 12, 144, 256), dtype=torch.uint8, device="cuda")
    received = []
    hook = model.policy.pi_features_extractor.register_forward_pre_hook(
        lambda module, args: received.append(args[0].detach().clone()))
    with torch.no_grad():
        dist = training_distribution(model.policy, frame, mixed_precision=True)
    hook.remove()
    assert received[0].dtype == torch.float16
    assert torch.equal(received[0], frame.to(torch.float16) / 255)
    assert all(d.logits.dtype == torch.float32 for d in dist.distribution)
    # The exported policy needs neither a scaler nor CUDA to run on Windows.
    portable = PPO.load(output / "best.zip", device="cpu")
    action, _ = portable.predict(frame[0].cpu().numpy(), deterministic=True)
    assert action.shape == (8,)
