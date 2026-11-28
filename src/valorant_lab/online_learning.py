"""Opt-in PPO with bounded, train-only demonstration rehearsal between rollouts."""
from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import PPO

from .bc_checkpoint import dataset_fingerprint
from .bc_data import CachedDemonstrations, cache_bytes, select_cache, training_sample_indices
from .bc_objective import behavioral_loss, make_class_weights, train_action_counts
from .scenario import action_sizes


class RehearsalPPO(PPO):
    """Alternate normal PPO updates with a fixed number of supervised updates.

    No demonstrations, labels or diagnostics enter environment observations or
    inference. This is alternating rehearsal, not a modification of PPO's
    clipped objective. The rehearsal sampler resumes from its saved counter;
    full online trajectories are not bitwise reproducible after an env restart.
    """

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "_rehearsal_dataset", "_rehearsal_indices", "_rehearsal_weights"]

    def configure_rehearsal(self, episodes, config, *, updates=4, batch_size=256, seed=42,
                            cache_mib=1024):
        if updates < 1 or batch_size < 1 or seed < 0:
            raise ValueError("Rehearsal updates/batch must be positive and seed nonnegative")
        train = [episode for episode in episodes if episode[1]["split"] == "train"]
        if not train:
            raise ValueError("Rehearsal requires training demonstrations")
        indices, sampling = training_sample_indices(train, "skip")
        identity = {"dataset_fingerprint": dataset_fingerprint(train),
                    "episode_ids": [meta["episode_id"] for _, meta in train],
                    "samples": len(indices), "updates_per_rollout": updates,
                    "batch_size": batch_size, "seed": seed, "initial_idle_policy": "skip",
                    "class_balance": "sqrt", "class_weight_cap": 8.}
        if getattr(self, "rehearsal_identity", identity) != identity:
            raise ValueError("Rehearsal dataset or settings changed on resume")
        location = select_cache("auto", cache_bytes(train), cache_mib * 1024**2, self.device)
        if location == "stream":
            raise ValueError("Rehearsal exceeds decoded cache budget; use a larger cache budget")
        self._rehearsal_dataset = CachedDemonstrations(train, self.device if location == "cuda" else "cpu")
        self._rehearsal_indices = indices
        counts = train_action_counts(train, action_sizes(config), indices)
        self._rehearsal_weights = make_class_weights(counts, "sqrt", 8., device=self.device)
        self.rehearsal_identity = identity
        self.rehearsal_updates = getattr(self, "rehearsal_updates", 0)
        return {**identity, "cache": location, "sampling": sampling}

    def _rehearse(self):
        if not hasattr(self, "_rehearsal_dataset"):
            raise RuntimeError("Reattach the original training demonstrations before continuing rehearsal PPO")
        identity = self.rehearsal_identity
        self.policy.set_training_mode(True)
        losses = []
        for _ in range(identity["updates_per_rollout"]):
            # Counter-based sampling preserves the next batch across save/load.
            rng = np.random.default_rng(np.random.SeedSequence([identity["seed"], self.rehearsal_updates]))
            indices = rng.choice(self._rehearsal_indices,
                size=min(identity["batch_size"], len(self._rehearsal_indices)), replace=False)
            images, actions = self._rehearsal_dataset.batch(indices)
            images, actions = images.to(self.device), actions.to(self.device)
            loss = behavioral_loss(self.policy.get_distribution(images), actions, self._rehearsal_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite rehearsal loss; stop before optimizer update")
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.policy.optimizer.step()
            self.rehearsal_updates += 1
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def train(self):
        # Fail before any PPO mutation if a resumed model lost its data attachment.
        if not hasattr(self, "_rehearsal_dataset"):
            raise RuntimeError("Reattach the original training demonstrations before continuing rehearsal PPO")
        super().train()
        loss = self._rehearse()
        self.logger.record("train/rehearsal_loss", loss)
        self.logger.record("train/rehearsal_updates", self.rehearsal_updates)
