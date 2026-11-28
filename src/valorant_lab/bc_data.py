"""Budgeted, decoded pixel/action batches. Imported only by training commands."""
from __future__ import annotations

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import DemonstrationDataset, read_demonstration_chunk
from .scenario import FRAME_STACK, noop

MIB = 1024**2


def training_sample_indices(episodes, initial_idle_policy="keep"):
    """Optional train-only sample view; never shorten the underlying RGB histories."""
    if initial_idle_policy not in ("keep", "skip"):
        raise ValueError("Initial idle policy must be keep or skip")
    report = {"initial_idle_policy": initial_idle_policy, "excluded_initial_idle_steps": 0,
              "episodes": []}
    if initial_idle_policy == "keep":
        return None, report
    selected, offset = [], 0
    for path, meta in episodes:
        if meta["split"] != "train":
            continue
        leading = 0
        for chunk in range(len(meta["chunks"])):
            _, actions = read_demonstration_chunk(path, meta, chunk)
            active = np.flatnonzero(np.any(actions != noop(meta["scenario"]), axis=1))
            if len(active):
                leading += int(active[0])
                break
            leading += len(actions)
        selected.extend(range(offset + leading, offset + meta["steps"]))
        offset += meta["steps"]
        report["episodes"].append({"episode": path.name, "excluded_initial_idle_steps": leading})
        report["excluded_initial_idle_steps"] += leading
    if not selected:
        raise ValueError("Initial idle sampling left no training actions")
    return np.asarray(selected, dtype=np.int64), report


def cache_bytes(episodes):
    """Exact persistent tensor bytes, including one zero-padding frame per split."""
    if not episodes:
        return 0
    count = sum(m["steps"] for _, m in episodes)
    shape = episodes[0][1]["contract"]["shape"]
    return (count + 1) * int(np.prod(shape)) + count * (8 + FRAME_STACK) * 8


def select_cache(mode, needed, budget, device, *, cuda_free=None):
    if mode not in ("auto", "cpu", "cuda", "stream") or budget <= 0:
        raise ValueError("Invalid data cache mode or budget")
    device = torch.device(device)
    if mode == "stream":
        return "stream"
    if needed > budget:
        if mode == "auto":
            return "stream"
        raise ValueError("Decoded demonstrations exceed --cache-mib; raise it or use --data-cache stream")
    if mode == "cpu":
        return "cpu"
    if device.type == "cuda":
        if cuda_free is None:
            cuda_free, _ = torch.cuda.mem_get_info(device)
        # Leave room for the policy, optimizer, activations and CUDA workspaces. This is
        # a cache cap, not a promise that arbitrary training batch sizes will fit.
        cuda_budget = min(cuda_free // 4, max(0, cuda_free - 2 * 1024**3))
        if mode != "cpu" and needed <= cuda_budget:
            return "cuda"
    if mode == "cuda":
        raise ValueError("CUDA cache requires a CUDA policy and sufficient free VRAM; use auto/cpu or a smaller cache")
    return "cpu"


class CachedDemonstrations:
    """Store each RGB frame once; gather histories without crossing episodes."""

    def __init__(self, episodes, device="cpu"):
        if not episodes:
            raise ValueError("Cannot cache an empty demonstration split")
        self.count = sum(m["steps"] for _, m in episodes)
        self.shape = tuple(episodes[0][1]["contract"]["shape"])
        height, width, channels = self.shape
        self.frames = torch.empty((self.count + 1, channels, height, width),
                                  dtype=torch.uint8, device=device)
        self.frames[0].zero_()
        self.actions = torch.empty((self.count, 8), dtype=torch.int64, device=device)
        self.histories = torch.empty((self.count, FRAME_STACK), dtype=torch.int64, device=device)
        offset = 0
        for path, meta in episodes:
            if tuple(meta["contract"]["shape"]) != self.shape:
                raise ValueError("Inconsistent cached observation shape")
            start = offset
            for chunk in range(len(meta["chunks"])):
                frames, actions = read_demonstration_chunk(path, meta, chunk)
                end = offset + len(frames)
                if end > start + meta["steps"]:
                    raise ValueError("Demonstration length mismatch")
                self.frames[offset + 1:end + 1].copy_(torch.from_numpy(frames).permute(0, 3, 1, 2))
                self.actions[offset:end].copy_(torch.from_numpy(actions.astype(np.int64, copy=False)))
                offset = end
            if offset != start + meta["steps"]:
                raise ValueError("Demonstration length mismatch")
            indices = np.arange(start, offset)[:, None] - np.arange(FRAME_STACK - 1, -1, -1)
            indices = np.where(indices >= start, indices + 1, 0)
            self.histories[start:offset].copy_(torch.from_numpy(indices))

    def __len__(self):
        return self.count

    def batch(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.frames.device)
        history = self.histories.index_select(0, indices)
        images = self.frames[history].flatten(1, 2)
        return images, self.actions.index_select(0, indices)


class CachedBatches:
    def __init__(self, dataset, batch_size, shuffle, indices=None):
        self.dataset = dataset
        # Keep DataLoader's sampler/RNG behavior identical to the original path.
        self.indices = DataLoader(range(len(dataset)) if indices is None else indices, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=0)

    def __iter__(self):
        for indices in self.indices:
            yield self.dataset.batch(indices)

    def __len__(self):
        return len(self.indices)


def make_bc_batches(episodes, *, device, batch_size, mode="auto", budget_mib=1024, train_indices=None):
    train = [e for e in episodes if e[1]["split"] == "train"]
    validation = [e for e in episodes if e[1]["split"] == "validation"]
    counts = [sum(m["steps"] for _, m in group) for group in (train, validation)]
    if not all(counts):
        raise ValueError("BC requires accepted train AND validation episodes, separated by scenario/seed")
    original_train_count = counts[0]
    if train_indices is not None:
        train_indices = np.asarray(train_indices)
        if (train_indices.ndim != 1 or train_indices.dtype.kind not in "iu" or not len(train_indices)
                or train_indices[0] < 0 or train_indices[-1] >= counts[0]
                or np.any(train_indices[1:] <= train_indices[:-1])):
            raise ValueError("Training indices must be nonempty, sorted, unique and within train")
        counts[0] = len(train_indices)
    needed = cache_bytes(train) + cache_bytes(validation)
    selected = select_cache(mode, needed, int(budget_mib * MIB), device)
    started = time.perf_counter()
    if selected == "stream":
        train_dataset = DemonstrationDataset(train)
        if train_indices is not None:
            train_dataset = Subset(train_dataset, train_indices.tolist())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(DemonstrationDataset(validation), batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        cache_device = device if selected == "cuda" else "cpu"
        train_loader = CachedBatches(CachedDemonstrations(train, cache_device), batch_size, True, train_indices)
        val_loader = CachedBatches(CachedDemonstrations(validation, cache_device), batch_size, False)
        if selected == "cuda":
            torch.cuda.synchronize(device)
    info = {"requested": mode, "mode": selected, "decoded_bytes": needed,
            "resident_bytes": needed if selected != "stream" else 0,
            "budget_mib": budget_mib, "preload_seconds": time.perf_counter() - started,
            "training_steps": counts[0], "original_training_steps": original_train_count,
            "validation_steps": counts[1]}
    if selected == "stream" and mode == "auto":
        info["reason"] = "dataset_exceeds_cache_budget"
    return train_loader, val_loader, info
