"""Train-only BC class statistics and action-aware diagnostics (no engine state inputs)."""
from __future__ import annotations

import math
import torch

from .dataset import read_demonstration_chunk


def train_action_counts(episodes, action_sizes):
    """Do not inspect validation labels when fitting the supervised objective."""
    counts = [torch.zeros(int(size), dtype=torch.int64) for size in action_sizes]
    for path, meta in episodes:
        if meta["split"] != "train":
            continue
        for chunk in range(len(meta["chunks"])):
            _, actions = read_demonstration_chunk(path, meta, chunk)
            labels = torch.as_tensor(actions.astype("int64", copy=False))
            for head, count in enumerate(counts):
                count.add_(torch.bincount(labels[:, head], minlength=len(count)))
    if not counts or any(int(count.sum()) == 0 for count in counts):
        raise ValueError("Class statistics require non-empty training demonstrations")
    return counts


def make_class_weights(counts, mode="none", cap=8., device="cpu"):
    """Bound the rare/majority ratio, then normalize expected train weight to one.

    All samples, including negative firing examples, remain in the objective.
    Unobserved classes get no artificial positive targets. The cap limits the
    ratio BEFORE normalization; normalization does not change that ratio.
    """
    if mode not in ("none", "sqrt") or not math.isfinite(cap) or cap < 1:
        raise ValueError("Class balance must be none/sqrt and weight cap must be finite and >= 1")
    if mode == "none":
        return None
    weights = []
    for values in counts:
        count = torch.as_tensor(values, dtype=torch.float32, device=device)
        if count.ndim != 1 or not bool(torch.isfinite(count).all()) or bool((count < 0).any()) or count.sum() <= 0:
            raise ValueError("Invalid training action counts")
        weight = (count.max() / count.clamp_min(1)).sqrt().clamp(1, cap)
        weight = weight / ((weight * count).sum() / count.sum())
        weights.append(weight)
    return weights


def behavioral_loss(distribution, actions, weights=None):
    """Eight categorical heads; retain the exact original objective by default."""
    if weights is None:
        return -distribution.log_prob(actions).mean() / actions.shape[1]
    if len(weights) != actions.shape[1]:
        raise ValueError("Class weight heads do not match the action space")
    losses = [-head.log_prob(actions[:, index]) * weights[index][actions[:, index]]
              for index, head in enumerate(distribution.distribution)]
    return torch.stack(losses, dim=1).mean()


class ActionMetrics:
    """Accumulate on device, transfer only at report time; absent positives are null."""

    def __init__(self, action_sizes, device="cpu", *, zero_yaw=None):
        self.sizes = [int(size) for size in action_sizes]
        # The fallback is for older callers with symmetric bins. Training must
        # pass scenario.noop(config)[3], because legal bins may be asymmetric.
        self.zero_yaw = self.sizes[3] // 2 if zero_yaw is None else int(zero_yaw)
        if not 0 <= self.zero_yaw < self.sizes[3]:
            raise ValueError("Zero-yaw index is outside the action space")
        self.confusions = [torch.zeros((size, size), dtype=torch.int64, device=device) for size in self.sizes]
        self.events = torch.zeros((3, 4), dtype=torch.int64, device=device)  # TP, FP, FN, TN

    def _events(self, actions):
        return torch.stack((actions[:, 6] == 1,
                            (actions[:, 0] == 0) & (actions[:, 1] == 0),
                            actions[:, 3] != self.zero_yaw), dim=1)

    @torch.no_grad()
    def update(self, predicted, target):
        for index, (size, matrix) in enumerate(zip(self.sizes, self.confusions)):
            matrix.add_(torch.bincount(target[:, index] * size + predicted[:, index],
                                      minlength=size * size).reshape(size, size))
        actual, guess = self._events(target), self._events(predicted)
        self.events.add_(torch.stack(((actual & guess).sum(0), (~actual & guess).sum(0),
                                      (actual & ~guess).sum(0), (~actual & ~guess).sum(0)), dim=1))

    def compute(self):
        matrices = [matrix.cpu().tolist() for matrix in self.confusions]
        result = {"heads": [], "events": {}}
        for matrix in matrices:
            truth = [sum(row) for row in matrix]
            guessed = [sum(row[index] for row in matrix) for index in range(len(matrix))]
            total = sum(truth)
            result["heads"].append({
                "label_counts": truth, "predicted_counts": guessed,
                "accuracy": sum(matrix[i][i] for i in range(len(matrix))) / total if total else None,
                "class_recall": [matrix[i][i] / count if count else None for i, count in enumerate(truth)]})
        for name, (tp, fp, fn, tn) in zip(("fire", "stop", "turn"), self.events.cpu().tolist()):
            result["events"][name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "positive_labels": tp + fn, "positive_predictions": tp + fp,
                "recall": tp / (tp + fn) if tp + fn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None}
        return result
