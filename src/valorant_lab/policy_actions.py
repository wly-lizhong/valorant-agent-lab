"""Explicit, reproducible action-selection diagnostics; default policy unchanged."""
from __future__ import annotations
import math
import numpy as np


def selection_settings(args):
    mode = getattr(args, "action_selection", "mode")
    temperature = float(getattr(args, "action_temperature", 1.))
    seed = getattr(args, "sampling_seed", 0)
    if mode not in ("mode", "sample") or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Action selection must be mode/sample with a positive finite temperature")
    if type(seed) is not int or not 0 <= seed < 100000:
        raise ValueError("Sampling seed must be in [0, 100000)")
    if mode == "mode" and (temperature != 1. or seed != 0):
        raise ValueError("Temperature and sampling seed require --action-selection sample")
    return {"mode": mode, "temperature": temperature, "sampling_seed": seed}


def episode_sampling_seed(settings, engine_seed):
    if settings["mode"] == "mode":
        return None
    return (settings["sampling_seed"] + 1) * 1000003 + int(engine_seed)


class PolicyActions:
    """Select only from the visual policy's categorical action probabilities.

    Sampling uses a per-episode CPU generator so worker scheduling and unrelated
    random calls do not affect the stream. It does not inspect game diagnostics,
    force recovery actions, or alter the inference-time observation.
    """
    def __init__(self, model, settings, engine_seed):
        self.model, self.settings = model, settings
        self.sampling_seed = episode_sampling_seed(settings, engine_seed)
        if self.sampling_seed is not None:
            import torch
            self.generator = torch.Generator(device="cpu").manual_seed(self.sampling_seed)
            self.model.policy.set_training_mode(False)

    def predict(self, observation):
        if self.sampling_seed is None:
            return self.model.predict(observation, deterministic=True)[0]
        import torch
        with torch.inference_mode():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            heads = self.model.policy.get_distribution(tensor).distribution
            if not isinstance(heads, list) or len(heads) != 8:
                raise ValueError("Action sampling requires the eight categorical FPS heads")
            action = []
            for head in heads:
                logits = head.logits.detach().float().cpu()
                if logits.ndim != 2 or logits.shape[0] != 1:
                    raise ValueError("Action sampling expects one observation at a time")
                probabilities = torch.softmax(logits / self.settings["temperature"], dim=-1)
                action.append(torch.multinomial(probabilities[0], 1, generator=self.generator).item())
            return np.asarray(action, dtype=np.int64)
