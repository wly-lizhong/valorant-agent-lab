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
    for variant in ("legacy", "normalized", "normalized-relu", "normalized-dropout", "spatial-dropout"):
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


@pytest.mark.parametrize("idle_policy,variant", [("keep", "normalized"), ("skip", "normalized"),
                         ("skip", "normalized-dropout"), ("skip", "spatial-dropout")])
def test_resume_matches_uninterrupted_parameters_and_optimizer(tmp_path, sample_data, idle_policy, variant):
    _, scenario, _ = sample_data
    uninterrupted = tmp_path / "full"
    resumed = tmp_path / "resumed"
    learning.bc_train(arguments(tmp_path, scenario, uninterrupted, initial_idle_policy=idle_policy, policy_variant=variant))
    learning.bc_train(arguments(tmp_path, scenario, resumed, epochs=1, initial_idle_policy=idle_policy, policy_variant=variant))
    learning.bc_train(arguments(tmp_path, scenario, resumed, resume=True, initial_idle_policy=idle_policy, policy_variant=variant))
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
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        learning.bc_train(arguments(tmp_path, scenario, resumed, resume=True,
                                    initial_idle_policy="skip" if idle_policy == "keep" else "keep"))


@pytest.mark.parametrize("variant", ["normalized-dropout", "spatial-dropout"])
def test_regularized_policy_is_visual_and_deterministic_at_inference(sample_data, variant):
    config, _, _ = sample_data
    model = PPO("CnnPolicy", learning.OfflineSpaces(config), n_steps=2, batch_size=2,
                policy_kwargs=learning.policy_kwargs(variant), device="cpu", seed=42)
    images = torch.randint(0, 256, (3, 12, 64, 64), dtype=torch.uint8)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        first = model.policy.get_distribution(images).distribution[3].probs.clone()
        repeat = model.policy.get_distribution(images).distribution[3].probs.clone()
        blank = model.policy.get_distribution(torch.zeros_like(images)).distribution[3].probs.clone()
    assert torch.equal(first, repeat)
    assert not torch.equal(first, blank)
    model.policy.set_training_mode(True)
    features = model.policy.extract_features(images)
    assert not torch.equal(features, model.policy.extract_features(images))
    loss = -model.policy.get_distribution(images).log_prob(torch.as_tensor(np.tile(noop(config), (3, 1)))).mean()
    loss.backward()
    gradient = model.policy.features_extractor.network[0].weight.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all()) and bool((gradient != 0).any())
    model.get_env().close()


@pytest.mark.parametrize("variant", ["legacy", "normalized", "normalized-relu", "spatial-dropout"])
def test_ppo_initialization_preserves_bc_architecture_and_eval_probabilities(tmp_path, sample_data, variant):
    config, _, _ = sample_data
    teacher = PPO("CnnPolicy", learning.OfflineSpaces(config), n_steps=2, batch_size=2,
                  policy_kwargs=learning.policy_kwargs(variant), device="cpu", seed=42)
    # Non-default weights prove this transfers the fitted policy, not just its shape.
    with torch.no_grad():
        teacher.policy.action_net.bias.add_(torch.arange(teacher.policy.action_net.out_features) * .01)
    teacher.save(tmp_path / "bc.zip")
    model = learning.initialize_ppo_from_bc(learning.OfflineSpaces(config), tmp_path / "bc.zip",
        n_steps=8, batch_size=4, device="cpu", learning_rate=3e-5, seed=123, target_kl=.02)
    images = torch.randint(0, 256, (2, 12, 64, 64), dtype=torch.uint8)
    teacher.policy.set_training_mode(False)
    model.policy.set_training_mode(True)
    with torch.no_grad():
        expected = teacher.policy.get_distribution(images).distribution
        actual = model.policy.get_distribution(images).distribution
        assert all(torch.equal(a.probs, b.probs) for a, b in zip(expected, actual))
    assert model.n_steps == 8 and model.batch_size == 4
    assert not model.policy.optimizer.state
    assert not any(isinstance(m, torch.nn.Dropout) and m.p for m in model.policy.modules())
    model.save(tmp_path / "ppo.zip")
    loaded = PPO.load(tmp_path / "ppo.zip", device="cpu")
    assert not any(isinstance(m, torch.nn.Dropout) and m.p for m in loaded.policy.modules())
    teacher.get_env().close()
    model.get_env().close()


@pytest.mark.parametrize("rehearsal", [False, True])
def test_ppo_cli_records_real_terminal_events_and_restores_signal_handler(tmp_path, sample_data, monkeypatch, rehearsal):
    import gymnasium as gym
    import signal
    from stable_baselines3.common.vec_env import DummyVecEnv
    from valorant_lab import cli
    config, scenario, _ = sample_data
    spaces = learning.OfflineSpaces(config)
    class FixtureEnv(gym.Env):
        observation_space, action_space = spaces.observation_space, spaces.action_space
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(self.observation_space.shape, np.uint8), {}
        def step(self, action):
            self.steps += 1
            return np.zeros(self.observation_space.shape, np.uint8), 0., False, self.steps == 2, {
                "seed": 42, "is_success": False, "termination_reason": "fixture",
                "shots": 0, "hits": 0, "kills": 0, "moving_shots": 0, "simulation_time": .1}
    teacher = PPO("CnnPolicy", spaces, n_steps=2, batch_size=2,
                  policy_kwargs=learning.policy_kwargs("spatial-dropout"), device="cpu")
    checkpoint = tmp_path / "teacher.zip"
    learning.save_model(teacher, checkpoint, learning.metadata(config, "bc", test_only=True,
                       policy_variant="spatial-dropout", episodes=[]))
    monkeypatch.setattr(cli, "make_vector_env", lambda *unused: DummyVecEnv([FixtureEnv]))
    args = SimpleNamespace(cpu_threads=1, output=str(tmp_path/'ppo'), scenario=str(scenario),
        resume=None, init_bc=str(checkpoint), learning_rate=3e-5, entropy_coef=.002, target_kl=.02,
        seed=42, device="cpu", envs=1, steps=8, rollout=4, batch_size=4,
        checkpoint_every=4, eval_every=0, eval_episodes=1)
    if rehearsal:
        args.rehearsal_data = str(tmp_path / "data")
        args.separate_critic = True
    previous = signal.getsignal(signal.SIGTERM)
    cli.train(args)
    assert signal.getsignal(signal.SIGTERM) is previous
    meta = json.loads((tmp_path/'ppo/latest.json').read_text())
    assert meta['num_timesteps'] == 8 and meta['test_only'] is True
    rows = [json.loads(s) for s in (tmp_path/'ppo/training-episodes.jsonl').read_text().splitlines()]
    assert len(rows) == 4 and all(x['seed'] == 42 and x['shots'] == 0 for x in rows)
    assert meta['feature_dropout_disabled'] is True
    if rehearsal:
        assert meta['separate_critic'] is True
        assert meta['rehearsal']['completed_updates'] == 8
        assert meta['rehearsal']['samples'] == 7
        args.resume, args.init_bc = str(tmp_path/'ppo/latest.zip'), None
        args.output, args.steps = str(tmp_path/'continued'), 4
        cli.train(args)
        continued = json.loads((tmp_path/'continued/latest.json').read_text())
        assert continued['num_timesteps'] == 12
        assert continued['rehearsal']['completed_updates'] == 12
        # Inference needs neither demonstration files nor a special evaluation path.
        loaded = PPO.load(tmp_path/'continued/latest.zip', device='cpu')
        assert loaded.predict(np.zeros((12, 64, 64), np.uint8), deterministic=True)[0].shape == (8,)
    teacher.get_env().close()


@pytest.mark.parametrize("override", [False, True])
def test_ppo_resume_honors_seed_and_preserves_optimizer_and_hyperparameters(tmp_path, sample_data, monkeypatch, override):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv
    from valorant_lab import cli
    config, scenario, _ = sample_data
    spaces = learning.OfflineSpaces(config)
    seen_seeds = []

    class SeedEnv(gym.Env):
        observation_space, action_space = spaces.observation_space, spaces.action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            seen_seeds.append(seed)
            self.steps = 0
            self.engine_seed = int(self.np_random.integers(0, 100000))
            return np.full(self.observation_space.shape, 17, np.uint8), {}

        def step(self, action):
            self.steps += 1
            return np.full(self.observation_space.shape, 17 + self.steps, np.uint8), 1., False, self.steps == 2, {
                'seed': self.engine_seed, 'is_success': False, 'termination_reason': 'fixture',
                'shots': 0, 'hits': 0, 'kills': 0, 'moving_shots': 0, 'simulation_time': .1}

    torch.set_num_threads(1)
    teacher = PPO('CnnPolicy', DummyVecEnv([SeedEnv]), n_steps=4, batch_size=4, n_epochs=2,
                  policy_kwargs=learning.policy_kwargs('normalized'), device='cpu', seed=42,
                  learning_rate=3e-5, ent_coef=.002, target_kl=.02)
    teacher.learn(4)
    checkpoint = tmp_path/'parent.zip'
    learning.save_model(teacher, checkpoint, learning.metadata(config, 'ppo', test_only=True,
                       policy_variant='normalized', episodes=[]))
    optimizer = teacher.policy.optimizer.state_dict()
    assert optimizer['state']
    original_learn = PPO.learn
    expected_lr = 3e-6 if override else 3e-5
    expected_kl = .005 if override else .02

    def check_before_update(model, *args, **kwargs):
        assert model.seed == 1042
        assert (model.n_steps, model.batch_size, model.n_epochs) == (4, 4, 2)
        assert model.lr_schedule(1) == expected_lr and model.ent_coef == .002 and model.target_kl == expected_kl
        assert all(torch.equal(t, model.policy.state_dict()[n]) for n, t in teacher.policy.state_dict().items())
        actual = model.policy.optimizer.state_dict()
        assert actual['param_groups'] == [dict(group, lr=expected_lr) for group in optimizer['param_groups']]
        for key, state in optimizer['state'].items():
            assert all(torch.equal(value, actual['state'][key][name]) for name, value in state.items())
        return original_learn(model, *args, **kwargs)

    seen_seeds.clear()
    monkeypatch.setattr(PPO, 'learn', check_before_update)
    monkeypatch.setattr(cli, 'make_vector_env', lambda *unused: DummyVecEnv([SeedEnv]))
    args = SimpleNamespace(cpu_threads=1, output=str(tmp_path/'resumed'), scenario=str(scenario),
        resume=str(checkpoint), init_bc=None, learning_rate=3e-4, entropy_coef=0., target_kl=.1,
        seed=1042, device='cpu', envs=1, steps=4, rollout=8, batch_size=2,
        checkpoint_every=4, eval_every=0, eval_episodes=1)
    if override:
        args.resume_learning_rate, args.resume_target_kl = expected_lr, expected_kl
    cli.train(args)
    assert seen_seeds[0] == 1042
    meta = json.loads((tmp_path/'resumed/latest.json').read_text())
    assert meta['training_rng_seed'] == 1042 and meta['num_timesteps'] == 8
    assert meta['ppo_rollout_steps'] == meta['ppo_batch_size'] == 4
    assert meta['ppo_learning_rate'] == expected_lr and meta['target_kl'] == expected_kl
    assert meta['resume_overrides'] == ({'learning_rate':expected_lr,'target_kl':expected_kl} if override else {})
    rows = [json.loads(x) for x in (tmp_path/'resumed/training-episodes.jsonl').read_text().splitlines()]
    assert rows[0]['seed'] == int(np.random.default_rng(1042).integers(0, 100000))
    assert all(0 <= row['seed'] < 100000 for row in rows)
    teacher.get_env().close()


@pytest.mark.parametrize('option,value', [('resume_learning_rate',0.),('resume_learning_rate',float('nan')),
                                         ('resume_target_kl',-1.),('resume_target_kl',float('inf'))])
def test_invalid_resume_overrides_fail_before_creating_environment(option, value):
    from valorant_lab import cli
    args = SimpleNamespace(learning_rate=3e-5, entropy_coef=.002, target_kl=.02, resume='unused')
    setattr(args,option,value)
    with pytest.raises(ValueError,match='positive and finite'):
        cli.train(args)
    args.resume = None
    setattr(args,option,.005)
    with pytest.raises(ValueError,match='require --resume'):
        cli.train(args)


def test_separate_critic_preserves_bc_but_value_gradients_cannot_change_actor(tmp_path, sample_data):
    config, _, _ = sample_data
    torch.set_num_threads(1)
    teacher = PPO('CnnPolicy', learning.OfflineSpaces(config), n_steps=2, batch_size=2,
                  policy_kwargs=learning.policy_kwargs('spatial-dropout'), device='cpu', seed=42)
    teacher.save(tmp_path/'teacher.zip')
    model = learning.initialize_ppo_from_bc(learning.OfflineSpaces(config), tmp_path/'teacher.zip',
                separate_critic=True, n_steps=4, batch_size=4, device='cpu', seed=42)
    images = torch.randint(0, 256, (4, 12, 64, 64), dtype=torch.uint8)
    teacher.policy.set_training_mode(False)
    with torch.no_grad():
        a = teacher.policy.get_distribution(images).distribution
        b = model.policy.get_distribution(images).distribution
        assert all(torch.equal(x.probs, y.probs) for x, y in zip(a, b))
    assert model.policy.pi_features_extractor is not model.policy.vf_features_extractor
    actor_before = {n: p.detach().clone() for n, p in model.policy.pi_features_extractor.named_parameters()}
    critic_before = {n: p.detach().clone() for n, p in model.policy.vf_features_extractor.named_parameters()}
    model.policy.optimizer.zero_grad(set_to_none=True)
    (model.policy.predict_values(images) - 5.).square().mean().backward()
    assert all(p.grad is None for p in model.policy.pi_features_extractor.parameters())
    model.policy.optimizer.step()
    assert all(torch.equal(actor_before[n], p) for n,p in model.policy.pi_features_extractor.named_parameters())
    assert any(not torch.equal(critic_before[n], p) for n,p in model.policy.vf_features_extractor.named_parameters())
    model.get_env().close()
    teacher.get_env().close()


def test_rehearsal_train_only_resumes_sampler_optimizer_and_preserves_critic(tmp_path, sample_data):
    from valorant_lab.online_learning import RehearsalPPO
    config, _, episodes = sample_data
    torch.set_num_threads(1)
    teacher = PPO('CnnPolicy', learning.OfflineSpaces(config), n_steps=2, batch_size=2,
                  policy_kwargs=learning.policy_kwargs('normalized'), device='cpu', seed=42)
    teacher.save(tmp_path/'teacher.zip')
    model = learning.initialize_ppo_from_bc(learning.OfflineSpaces(config), tmp_path/'teacher.zip',
        separate_critic=True, algorithm_class=RehearsalPPO, n_steps=4, batch_size=4,
        device='cpu', learning_rate=3e-4, seed=42)
    details = model.configure_rehearsal(episodes, config, updates=8, batch_size=4)
    training_ids = [m['episode_id'] for _,m in episodes if m['split']=='train']
    assert details['episode_ids'] == training_ids and details['samples'] == 7
    assert len(model._rehearsal_dataset) == 7  # validation's three rows never enter the cache
    images, actions = model._rehearsal_dataset.batch(model._rehearsal_indices)
    with torch.no_grad():
        before = float(behavioral_loss(model.policy.get_distribution(images), actions, model._rehearsal_weights))
    critic_before = {n: p.detach().clone() for n,p in model.policy.named_parameters()
                     if n.startswith(('vf_features_extractor.', 'mlp_extractor.value_net.', 'value_net.'))}
    model._rehearse()
    with torch.no_grad():
        after = float(behavioral_loss(model.policy.get_distribution(images), actions, model._rehearsal_weights))
    assert after < before
    assert all(torch.equal(p, dict(model.policy.named_parameters())[n]) for n,p in critic_before.items())
    model.save(tmp_path/'rehearsal.zip')
    loaded = RehearsalPPO.load(tmp_path/'rehearsal.zip', device='cpu')
    assert not hasattr(loaded, '_rehearsal_dataset')
    with pytest.raises(RuntimeError, match='Reattach'):
        loaded._rehearse()
    with pytest.raises(ValueError, match='changed on resume'):
        loaded.configure_rehearsal(episodes, config, updates=9, batch_size=4)
    loaded.configure_rehearsal(episodes, config, updates=8, batch_size=4)
    model._rehearse()
    loaded._rehearse()
    assert model.rehearsal_updates == loaded.rehearsal_updates == 16
    assert all(torch.equal(t, loaded.policy.state_dict()[n]) for n,t in model.policy.state_dict().items())
    for k, state in model.policy.optimizer.state_dict()['state'].items():
        assert all(torch.equal(v, loaded.policy.optimizer.state_dict()['state'][k][n]) for n,v in state.items())
    teacher.get_env().close()
    model.get_env().close()


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
