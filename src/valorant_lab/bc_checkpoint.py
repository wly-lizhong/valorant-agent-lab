"""Atomic, bounded BC restart checkpoints, separate from inference model exports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import shutil
import torch
import numpy as np

from .scenario import digest


def dataset_fingerprint(episodes):
    # Verified chunk hashes bind pixel/action bytes as well as the split and case.
    return digest([{key: meta.get(key) for key in
        ("episode_id", "case_key", "split", "source", "steps", "contract", "chunks")}
        for _, meta in episodes])


def capture_rng():
    numpy_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed since the BC checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(output, state):
    """Commit the self-contained bundle last. Keep at most two generations."""
    path = Path(output) / "resume.pt"
    temporary = path.with_suffix(".pt.tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        previous = path.with_name("resume.previous.pt")
        backup = previous.with_suffix(".pt.tmp")
        shutil.copyfile(path, backup)
        backup.replace(previous)
    temporary.replace(path)


def load_checkpoint(output, identity):
    path = Path(output) / "resume.pt"
    failures = []
    for candidate in (path, path.with_name("resume.previous.pt")):
        if not candidate.exists():
            continue
        try:
            state = torch.load(candidate, map_location="cpu", weights_only=True)
        except Exception as error:
            # A killed writer or disk damage must not make the last committed
            # generation unusable. Identity mismatches below never fall back.
            failures.append(f"{candidate.name}: {type(error).__name__}")
            continue
        if state.get("schema") != "valorant-bc-resume-v1" or state.get("identity") != identity:
            raise ValueError("BC resume configuration/data fingerprint mismatch; use a fresh output directory")
        return state, candidate.name
    raise ValueError("No readable BC restart checkpoint: " + ", ".join(failures))


def plain_metadata(value):
    """Keep the restart envelope safe for torch.load(weights_only=True)."""
    return json.loads(json.dumps(value))
