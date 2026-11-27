"""Episode-separated, chunked demonstration storage. Diagnostics are never learner inputs."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np
from .scenario import FRAME_STACK, contract, digest

SCHEMA = "valorant-demo-v1"
SEED_BASE = {"train": 0, "validation": 100000, "dev": 200000, "test": 300000}


def split_seed(split: str, index: int) -> int:
    if split not in SEED_BASE or not 0 <= index < 100000:
        raise ValueError("Use a known split and a seed index in [0, 100000)")
    return SEED_BASE[split] + index


def write_json(path: Path, data: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class EpisodeWriter:
    def __init__(self, root, config: dict, engine_seed: int, *, source="human", split="train", case_id=None,
                 chunk_size=64):
        if split not in ("train", "validation") or chunk_size < 1:
            raise ValueError("Demonstrations require train/validation split and a positive chunk size")
        self.directory = Path(root) / ("episode-" + uuid.uuid4().hex[:16])
        self.directory.mkdir(parents=True)
        self.manifest = {"schema": SCHEMA, "episode_id": self.directory.name, "source": source,
                         "split": split, "case_id": case_id, "created_ns": time.time_ns(),
                         "scenario": config, "scenario_hash": digest(config), "contract": contract(config),
                         "engine_seed": engine_seed, "case_key": digest([config, engine_seed]),
                         "chunks": [], "steps": 0, "accepted": False, "complete": False,
                         "quality_issues": []}
        self.chunk_size = chunk_size
        self.frames, self.actions = [], []
        self.diagnostics = (self.directory / "diagnostics.jsonl").open("w", encoding="utf-8")
        self.closed = False
        write_json(self.directory / "episode.json", self.manifest)

    def append(self, observation, action, *, before: dict, after: dict, timing: dict):
        if self.closed:
            raise RuntimeError("Episode already closed")
        frame = np.asarray(observation)
        action = np.asarray(action)
        c = self.manifest["contract"]
        if frame.shape != tuple(c["shape"]) or frame.dtype != np.uint8:
            raise ValueError("Demonstration frame shape/dtype mismatch")
        if action.shape != (8,) or action.dtype.kind not in "iu" or np.any(action < 0) or np.any(action >= c["action_sizes"]):
            raise ValueError("Invalid demonstration action")
        # The observation precedes its label; never pair a post-action frame with that same action.
        self.frames.append(frame.copy())
        self.actions.append(action.astype(np.int64))
        entry = {"index": self.manifest["steps"], "before": before, "after": after, "timing": timing}
        self.diagnostics.write(json.dumps(entry, allow_nan=False) + "\n")
        self.manifest["steps"] += 1
        if before.get("clock") == "native_fixed_step":
            validate_native_alignment(self.manifest["scenario"], before, after, timing)
        elif before.get("clock", "lockstep") != "lockstep":
            self.issue("realtime_recording_requires_temporal_alignment_review")
        if "action_applied" in after and after["action_applied"] != action.tolist():
            self.issue("action_not_applied_as_recorded")
        if len(self.frames) >= self.chunk_size:
            self.flush()

    def issue(self, message: str):
        if message not in self.manifest["quality_issues"]:
            self.manifest["quality_issues"].append(message)

    def flush(self):
        if not self.frames:
            return
        name = f"chunk-{len(self.manifest['chunks']):05d}.npz"
        path = self.directory / name
        np.savez_compressed(path, frames=np.stack(self.frames), actions=np.stack(self.actions))
        self.manifest["chunks"].append({"file": name, "steps": len(self.frames), "sha256": sha256(path)})
        self.frames.clear(); self.actions.clear()

    def finish(self, final_info: dict, *, accepted: bool, complete=True):
        if self.closed:
            return self.directory
        self.flush()
        self.diagnostics.close()
        self.manifest.update(complete=bool(complete), final=final_info,
                             accepted=bool(accepted and complete and final_info.get("is_success")
                                           and not self.manifest["quality_issues"] and self.manifest["steps"] > 0))
        write_json(self.directory / "episode.json", self.manifest)
        self.closed = True
        return self.directory


def load_episodes(root, config: dict, *, allow_test_data=False) -> list[tuple[Path, dict]]:
    episodes = []
    allowed_sources = {"human"} | ({"script_test"} if allow_test_data else set())
    for path in sorted(Path(root).glob("episode-*/episode.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("accepted"):
            continue
        if (data.get("schema") != SCHEMA or not data.get("complete") or data.get("quality_issues")
                or data.get("source") not in allowed_sources or data.get("split") not in ("train", "validation")):
            raise ValueError(f"Unusable accepted episode: {path}")
        if data.get("contract") != contract(config):
            raise ValueError(f"Incompatible scenario/action contract: {path}")
        count = 0
        for chunk in data["chunks"]:
            name = chunk["file"]
            if Path(name).name != name or "/" in name or "\\" in name:
                raise ValueError("Unsafe demonstration chunk path")
            if sha256(path.parent / name) != chunk["sha256"]:
                raise ValueError(f"Corrupt demonstration chunk: {path.parent / name}")
            count += chunk["steps"]
        if count != data["steps"] or count < 1:
            raise ValueError("Demonstration length mismatch")
        episodes.append((path.parent, data))
    groups = {}
    for _, data in episodes:
        prior = groups.setdefault(data["case_key"], data["split"])
        if prior != data["split"]:
            raise ValueError("The same scenario/seed occurs in both train and validation")
    return episodes


def validate_native_alignment(config, before, after, timing):
    duration = config["action_repeat"] / config["physics_hz"]
    delta = after.get("simulation_time", -1) - before.get("simulation_time", -1)
    recorded_duration = timing.get("simulation_dt", -1)
    elapsed = timing.get("completed", -1) - timing.get("input_sampled", 0)
    pause = timing.get("paused_seconds", 0)
    if timing.get("pacing") == "bounded_pause_v1":
        prior_loss, loss = before.get("dropped_seconds", -1), after.get("dropped_seconds", -1)
        # A pause is only valid when the fixed-step engine explicitly discarded unsimulated
        # wall time. Preserve the old wall-time bound for active execution, and cap total loss.
        if (not all(np.isfinite(x) for x in (pause, prior_loss, loss, elapsed))
                or not 0 <= pause <= .75 or not 0 <= prior_loss <= loss <= 2
                or pause > loss - prior_loss + 1e-6 or pause > elapsed):
            raise ValueError("Native pause accounting mismatch")
    elif timing.get("pacing") is not None or pause != 0:
        raise ValueError("Unrecognized native pause accounting")
    if (timing.get("alignment") != "rendered_pre_action_v1"
            or timing.get("clock") != "native_fixed_step"
            or before.get("clock") != "native_fixed_step"
            or after.get("clock") != "native_fixed_step"
            or type(before.get("step")) is not int or type(after.get("step")) is not int
            or after.get("step") != before.get("step", -2) + 1
            or not np.isfinite(delta) or abs(delta - duration) > .002
            or not np.isfinite(recorded_duration) or abs(recorded_duration - duration) > 1e-6
            or not 0 <= elapsed - pause <= .3):
        raise ValueError("Native pre-action frame/action timing mismatch")


class DemonstrationDataset:
    """Map-style dataset; only pixels/actions are read. Bounded chunk cache, no full-dataset RAM load."""
    def __init__(self, episodes: list[tuple[Path, dict]]):
        self.episodes = episodes
        self.ends = np.cumsum([m["steps"] for _, m in episodes])
        self.chunk_ends = [np.cumsum([c["steps"] for c in m["chunks"]]) for _, m in episodes]
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def _sample(self, episode, index):
        path, metadata = self.episodes[episode]
        ends = self.chunk_ends[episode]
        chunk = int(np.searchsorted(ends, index, side="right"))
        offset = index - (int(ends[chunk - 1]) if chunk else 0)
        key = (episode, chunk)
        if key not in self.cache:
            with np.load(path / metadata["chunks"][chunk]["file"], allow_pickle=False) as z:
                frames, actions = z["frames"], z["actions"]
            count = metadata["chunks"][chunk]["steps"]
            c = metadata["contract"]
            if (frames.dtype != np.uint8 or frames.shape != (count, *c["shape"])
                    or actions.shape != (count, 8) or actions.dtype.kind not in "iu"
                    or np.any(actions < 0) or np.any(actions >= c["action_sizes"])):
                raise ValueError("Invalid demonstration tensor contents")
            self.cache[key] = (frames, actions)
            if len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        frames, actions = self.cache[key]
        return frames[offset], actions[offset]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        local = index - (int(self.ends[episode - 1]) if episode else 0)
        frame, action = self._sample(episode, local)
        frames = [np.zeros_like(frame) if i < 0 else self._sample(episode, i)[0]
                  for i in range(local - FRAME_STACK + 1, local + 1)]
        observation = np.concatenate(frames, axis=2).transpose(2, 0, 1).copy()
        return observation, action.copy()
