"""Offline action and visual-dependence checks; never supplies diagnostic state to a policy."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .dataset import DemonstrationDataset, load_episodes, read_demonstration_chunk, sha256, write_json
from .scenario import action_sizes, digest, load_scenario, noop, require_contract

HEAD_NAMES = ("forward", "strafe", "move_mode", "yaw", "pitch", "jump", "fire", "reload")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def binary_metrics(target, predicted):
    target, predicted = np.asarray(target, bool), np.asarray(predicted, bool)
    tp, fp = int((target & predicted).sum()), int((~target & predicted).sum())
    fn, tn = int((target & ~predicted).sum()), int((~target & ~predicted).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": _ratio(tp, tp + fn), "precision": _ratio(tp, tp + fp),
            "false_positive_rate": _ratio(fp, fp + tn),
            "target_positive": tp + fn, "predicted_positive": tp + fp}


def action_metrics(target, predicted, sizes, zero_yaw):
    """Return counts and denominators so rare-action failure cannot hide in accuracy."""
    target, predicted = np.asarray(target), np.asarray(predicted)
    sizes = np.asarray(sizes)
    if (target.ndim != 2 or target.shape[1] != 8 or not len(target)
            or target.shape != predicted.shape or sizes.shape != (8,)
            or target.dtype.kind not in "iu" or predicted.dtype.kind not in "iu"
            or np.any(target < 0) or np.any(target >= sizes)
            or np.any(predicted < 0) or np.any(predicted >= sizes)):
        raise ValueError("Expected nonempty, valid categorical action arrays of shape (N, 8)")
    heads = {}
    for head, (name, size) in enumerate(zip(HEAD_NAMES, sizes)):
        confusion = np.bincount(target[:, head] * size + predicted[:, head],
                                minlength=int(size * size)).reshape(size, size)
        support, emitted = confusion.sum(1), confusion.sum(0)
        correct = confusion.diagonal()
        recalls = [_ratio(a, b) for a, b in zip(correct, support)]
        heads[name] = {"confusion_target_rows_prediction_columns": confusion.tolist(),
                       "support": support.tolist(), "predicted_counts": emitted.tolist(),
                       "precision": [_ratio(a, b) for a, b in zip(correct, emitted)],
                       "recall": recalls, "accuracy": float(correct.sum() / len(target)),
                       "macro_recall_observed_classes": float(np.mean([x for x in recalls if x is not None]))}
    actions, counts = np.unique(predicted, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {"samples": len(target), "heads": heads,
            "joint_accuracy": float(np.all(target == predicted, axis=1).mean()),
            "unique_joint_actions": len(actions), "constant_action": len(actions) == 1,
            "most_common_actions": [{"action": actions[i].tolist(), "count": int(counts[i])}
                                    for i in order[:12]],
            "fire": binary_metrics(target[:, 6] != 0, predicted[:, 6] != 0),
            "stop": binary_metrics(np.all(target[:, :2] == 0, axis=1),
                                   np.all(predicted[:, :2] == 0, axis=1)),
            "turn": binary_metrics(target[:, 3] != zero_yaw, predicted[:, 3] != zero_yaw)}


def derangement(count, seed):
    """A reproducible within-split permutation with no unchanged rows for N > 1."""
    if count < 1:
        raise ValueError("Cannot permute an empty set")
    order = np.random.default_rng(seed).permutation(count)
    result = np.empty(count, dtype=np.int64)
    result[order] = np.roll(order, 1)
    return result


def comparison_report(target, probabilities, sizes, zero_yaw, *, seed=42):
    """Permuting deterministic feed-forward outputs exactly matches permuting their inputs.

    Labels stay fixed and histories stay intact. This avoids a redundant model pass.
    It measures association with these samples, not semantic enemy recognition or gameplay.
    """
    target = np.asarray(target)
    sizes = np.asarray(sizes)
    if len(probabilities) != 8:
        raise ValueError("Expected eight categorical probability heads")
    predicted = np.column_stack([np.asarray(p).argmax(1) for p in probabilities])
    metrics = action_metrics(target, predicted, sizes, zero_yaw)
    permutation = derangement(len(target), seed)
    shuffled = action_metrics(target, predicted[permutation], sizes, zero_yaw)
    nll, shuffled_nll, ranges, target_probability = [], [], [], []
    rows = np.arange(len(target))
    for head, (probability, size) in enumerate(zip(probabilities, sizes)):
        p = np.asarray(probability, dtype=np.float64)
        if (p.shape != (len(target), size) or not np.isfinite(p).all() or np.any(p < 0)
                or np.any(p > 1) or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise ValueError("Invalid probability array")
        nll.append(float(-np.log(np.maximum(p[rows, target[:, head]], 1e-12)).mean()))
        shuffled_nll.append(float(-np.log(np.maximum(p[permutation, target[:, head]], 1e-12)).mean()))
        ranges.append(float(np.ptp(p, axis=0).max()))
        target_probability.append(float(p[rows, target[:, head]].mean()))
    metrics["unweighted_nll"] = float(np.mean(nll))
    for head, name in enumerate(HEAD_NAMES):
        metrics["heads"][name].update(nll=nll[head],
            mean_target_probability=target_probability[head], max_probability_range=ranges[head])
    dependency = {"method": "within_split_deranged_history_outputs_labels_fixed",
                  "seed": seed, "meaningful_sample_count": len(target) > 1,
                  "fixed_rows": int((permutation == rows).sum()),
                  "permutation_sha256": digest(permutation.tolist()),
                  "shuffled_unweighted_nll": float(np.mean(shuffled_nll)),
                  "shuffled_minus_aligned_nll": float(np.mean(shuffled_nll) - np.mean(nll)),
                  "aligned_minus_shuffled_joint_accuracy": metrics["joint_accuracy"] - shuffled["joint_accuracy"],
                  "prediction_changed_fraction": float(np.any(predicted != predicted[permutation], axis=1).mean()),
                  "max_probability_range": max(ranges),
                  "shuffled_fire": shuffled["fire"], "shuffled_stop": shuffled["stop"],
                  "shuffled_turn": shuffled["turn"],
                  "shuffled_head_nll": dict(zip(HEAD_NAMES, shuffled_nll))}
    return {"metrics": metrics, "visual_dependency": dependency}


def stratified_train_indices(actions, limit, zero_yaw, seed=42):
    """Balanced learnability probe, not a population-level validation sample."""
    if not 1 <= limit <= 128 or not len(actions):
        raise ValueError("Tiny-fit sample limit must be in [1, 128]")
    actions = np.asarray(actions)
    count = min(limit, len(actions))
    rng = np.random.default_rng(seed)
    selected = set()
    groups = [np.flatnonzero(actions[:, 6] != 0),
              np.flatnonzero(actions[:, 3] != zero_yaw),
              np.flatnonzero(np.all(actions[:, :2] == 0, axis=1)),
              np.arange(len(actions))]
    for group in groups:
        available = [int(i) for i in rng.permutation(group) if int(i) not in selected]
        selected.update(available[:max(1, count // 4)])
        if len(selected) >= count:
            break
    remaining = [int(i) for i in rng.permutation(len(actions)) if int(i) not in selected]
    selected.update(remaining[:count - len(selected)])
    return np.array(sorted(selected)[:count], dtype=np.int64)


def episode_fingerprint(episodes):
    records = [{"episode_id": m["episode_id"], "case_key": m["case_key"], "split": m["split"],
                "steps": m["steps"], "manifest_sha256": sha256(path / "episode.json"),
                "chunks": m["chunks"]} for path, m in episodes]
    return {"sha256": digest(records), "episodes": records}


def source_fingerprint():
    package = Path(__file__).parent
    names = ("bc_diagnostics.py", "learning.py", "bc_objective.py", "bc_data.py", "dataset.py", "scenario.py")
    files = {name: sha256(package / name) for name in names if (package / name).is_file()}
    return {"sha256": digest(files), "files": files}


def validation_separation(model_meta, episodes):
    """Check historical model training cases, not only the current dataset split."""
    recorded = model_meta.get("episodes")
    if not isinstance(recorded, list):
        raise ValueError("Model training-case provenance is missing; cannot verify validation separation")
    training = [record for record in recorded if record.get("split") == "train"]
    if not training or any(not isinstance(record.get("case_key"), str) or not record["case_key"]
                           for record in training):
        raise ValueError("Model training-case provenance is missing; cannot verify validation separation")
    model_keys = {record["case_key"] for record in training}
    validation_keys = {meta["case_key"] for _, meta in episodes if meta["split"] == "validation"}
    if not validation_keys:
        raise ValueError("No validation cases to check")
    overlap = sorted(model_keys & validation_keys)
    if overlap:
        raise ValueError("Validation cases occur in the model's recorded training data: " + ", ".join(overlap))
    return {"verified": True, "basis": "model_metadata_train_case_keys_vs_current_validation_case_keys",
            "model_training_case_keys": sorted(model_keys), "validation_case_keys": sorted(validation_keys),
            "model_training_case_count": len(model_keys), "validation_case_count": len(validation_keys),
            "overlap_case_keys": []}


class _Moments:
    def __init__(self):
        self.count = 0
        self.total = self.square = None
        self.absolute = self.zero = self.over_099 = 0.

    def update(self, value):
        value = value.detach().float().cpu().numpy().astype(np.float64, copy=False)
        value = value.reshape(len(value), -1)
        if self.total is None:
            self.total = np.zeros(value.shape[1])
            self.square = np.zeros(value.shape[1])
        self.total += value.sum(0)
        self.square += np.square(value).sum(0)
        self.count += len(value)
        self.absolute += np.abs(value).sum()
        self.zero += (value == 0).sum()
        self.over_099 += (np.abs(value) > .99).sum()

    def compute(self):
        if not self.count:
            return None
        variance = np.maximum(self.square / self.count - np.square(self.total / self.count), 0)
        elements = self.count * len(self.total)
        return {"abs_mean": float(self.absolute / elements), "zero_fraction": float(self.zero / elements),
                "mean_std_across_images": float(np.sqrt(variance).mean()),
                "abs_over_099_fraction": float(self.over_099 / elements)}


def inspect_policy(policy, dataset, indices, sizes, zero_yaw, *, batch_size=128, seed=42):
    import torch
    if batch_size < 1 or not len(indices):
        raise ValueError("Batch size and selected sample count must be positive")
    features, latent = _Moments(), _Moments()
    actor = policy.mlp_extractor.policy_net
    tanh_actor = bool(len(actor) and isinstance(actor[-1], torch.nn.Tanh))
    hooks = [policy.pi_features_extractor.register_forward_hook(lambda _m, _a, out: features.update(out)),
             actor.register_forward_hook(lambda _m, _a, out: latent.update(out))]
    probabilities = [[] for _ in sizes]
    targets = []
    was_training = policy.training
    policy.set_training_mode(False)
    try:
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                samples = [dataset[int(i)] for i in indices[start:start + batch_size]]
                images = torch.as_tensor(np.stack([s[0] for s in samples]), device=policy.device)
                targets.append(np.stack([s[1] for s in samples]))
                distribution = policy.get_distribution(images)
                for head, categorical in enumerate(distribution.distribution):
                    probabilities[head].append(categorical.probs.cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
        policy.set_training_mode(was_training)
    report = comparison_report(np.concatenate(targets), [np.concatenate(p) for p in probabilities],
                               sizes, zero_yaw, seed=seed)
    actor_stats = latent.compute()
    report["saturation"] = {"features": features.compute(), "actor": actor_stats,
                            "actor_final_activation": type(actor[-1]).__name__ if len(actor) else "Identity",
                            "tanh_saturated_fraction": actor_stats["abs_over_099_fraction"] if tanh_actor else None}
    return report


def diagnose_model(args):
    import torch
    from stable_baselines3 import PPO
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Diagnostic output already exists; use a new file to preserve previous evidence")
    if args.max_samples < 0 or args.cpu_threads < 1:
        raise ValueError("Invalid diagnostic sample/thread limit")
    torch.set_num_threads(args.cpu_threads)
    config = load_scenario(args.scenario)
    model_meta = require_contract(args.model, config)
    episodes = load_episodes(args.data, config, allow_test_data=getattr(args, "allow_test_data", False))
    if not all(any(m["split"] == split for _, m in episodes) for split in ("train", "validation")):
        raise ValueError("Diagnostics require separate accepted train and validation episodes")
    separation = validation_separation(model_meta, episodes)
    model = PPO.load(args.model, device=args.device)
    started = time.perf_counter()
    splits = {}
    for split in ("train", "validation"):
        records = [e for e in episodes if e[1]["split"] == split]
        dataset = DemonstrationDataset(records)
        if args.max_samples and args.max_samples < len(dataset):
            indices = np.sort(np.random.default_rng(args.seed).choice(len(dataset), args.max_samples, replace=False))
            selection = "uniform_without_replacement"
        else:
            indices, selection = np.arange(len(dataset)), "all"
        splits[split] = {"available_samples": len(dataset), "samples": len(indices), "selection": selection,
                         "selected_indices_sha256": digest(indices.tolist()), "episode_count": len(records),
                         **inspect_policy(model.policy, dataset, indices, action_sizes(config), int(noop(config)[3]),
                                          batch_size=args.batch_size, seed=args.seed)}
    report = {"schema": "bc-policy-diagnostic-v1", "model_sha256": sha256(Path(args.model)),
              "model_metadata_sha256": sha256(Path(args.model).with_suffix(".json")),
              "model_test_only": model_meta.get("test_only", False),
              "validation_separation": separation,
              "data": episode_fingerprint(episodes), "source": source_fingerprint(),
              "device": str(model.device), "elapsed_seconds": time.perf_counter() - started, "splits": splits,
              "note": "Offline association is not gameplay success; shuffled comparisons are descriptive, not semantic proof."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


def tiny_fit(args):
    """Fit at most 128 train-only samples as a pipeline check, never a gameplay claim."""
    import torch
    from stable_baselines3 import PPO
    from .bc_objective import behavioral_loss, make_class_weights
    from .learning import OfflineSpaces, metadata, policy_kwargs, save_model
    combinations = {("normalized", "none"), ("normalized", "sqrt"), ("normalized-relu", "sqrt")}
    if (args.policy_variant, args.class_balance) not in combinations:
        raise ValueError("Tiny-fit uses one of the three predeclared candidate configurations")
    limit = args.max_samples or 128
    if not 1 <= args.steps <= 300 or not 1 <= limit <= 128 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("Tiny-fit requires 1..300 updates and 1..128 samples")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Tiny-fit requires a fresh output directory")
    config = load_scenario(args.scenario)
    episodes = load_episodes(args.data, config, allow_test_data=getattr(args, "allow_test_data", False))
    train = [e for e in episodes if e[1]["split"] == "train"]
    if not train:
        raise ValueError("No accepted training demonstrations")
    # Validation records are neither decoded nor used for selection/weights/optimization.
    labels = np.concatenate([read_demonstration_chunk(path, meta, i)[1]
                             for path, meta in train for i in range(len(meta["chunks"]))])
    indices = stratified_train_indices(labels, limit, int(noop(config)[3]), args.seed)
    dataset = DemonstrationDataset(train)
    selected = [dataset[int(i)] for i in indices]
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    model = PPO("CnnPolicy", OfflineSpaces(config), n_steps=2, batch_size=2,
                policy_kwargs=policy_kwargs(args.policy_variant), learning_rate=args.learning_rate,
                seed=args.seed, device=args.device, verbose=0)
    images = torch.as_tensor(np.stack([s[0] for s in selected]), device=model.device)
    targets = torch.as_tensor(np.stack([s[1] for s in selected]), dtype=torch.long, device=model.device)
    sizes = action_sizes(config)
    counts = [np.bincount(labels[indices, head], minlength=int(size)).tolist() for head, size in enumerate(sizes)]
    weights = make_class_weights(counts, mode=args.class_balance, cap=args.class_weight_cap, device=model.device)
    before = inspect_policy(model.policy, selected, np.arange(len(selected)), sizes, int(noop(config)[3]),
                            batch_size=args.batch_size, seed=args.seed)
    output.mkdir(parents=True, exist_ok=False)
    losses = []
    started = time.perf_counter()
    try:
        model.policy.set_training_mode(True)
        for step in range(args.steps):
            batch = torch.randperm(len(selected), device=model.device)[:args.batch_size]
            distribution = model.policy.get_distribution(images[batch])
            loss = behavioral_loss(distribution, targets[batch], weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite tiny-fit loss")
            model.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5)
            model.policy.optimizer.step()
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
                losses.append({"update": step + 1, "loss": float(loss.detach())})
        after = inspect_policy(model.policy, selected, np.arange(len(selected)), sizes, int(noop(config)[3]),
                               batch_size=args.batch_size, seed=args.seed)
        meta = metadata(config, "tiny_fit_diagnostic", test_only=True, diagnostic_only=True,
                        policy_variant=args.policy_variant, class_balance=args.class_balance,
                        class_weight_cap=args.class_weight_cap, training_split="train", epochs=None,
                        batch_size=min(args.batch_size, len(selected)), training_device=str(model.device),
                        parameter_precision="fp32",
                        episodes=[{"id": m["episode_id"], "case_key": m["case_key"], "split": "train"} for _, m in train],
                        selected_train_indices=indices.tolist(), optimizer_updates=args.steps)
        save_model(model, output / "diagnostic-only.zip", meta)
        report = {"schema": "bc-tiny-fit-v1", "test_only": True, "diagnostic_only": True,
                  "training_split": "train", "validation_used": False, "sample_count": len(selected),
                  "selected_train_indices": indices.tolist(), "data": episode_fingerprint(train),
                  "source": source_fingerprint(), "policy_variant": args.policy_variant,
                  "class_balance": args.class_balance, "class_weight_cap": args.class_weight_cap,
                  "learning_rate": args.learning_rate, "seed": args.seed,
                  "batch_size": min(args.batch_size, len(selected)), "requested_batch_size": args.batch_size,
                  "device": str(model.device), "precision": "fp32",
                  "optimizer_updates": args.steps, "elapsed_seconds": time.perf_counter() - started,
                  "model_sha256": sha256(output / "diagnostic-only.zip"), "loss_samples": losses,
                  "before": before, "after": after,
                  "note": "Training-only memorization check. Not a validation result or a deployable gameplay model."}
        write_json(output / "report.json", report)
        return report
    finally:
        model.get_env().close()
