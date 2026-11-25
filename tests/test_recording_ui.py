"""Exercise the real collector event loop without taking over a user's keyboard."""
import json
from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest
from valorant_lab.env import ROOT
from valorant_lab.scenario import load_scenario, noop


def test_collector_records_displayed_frame_and_requires_save(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame = pytest.importorskip("pygame")
    import valorant_lab.recording as recording
    config = load_scenario(ROOT / "configs/lotus_movement_combat.json")
    class FakeEnv:
        def __init__(self, scenario):
            self.config = config
            self.noop = noop(config)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def reset(self, **kwargs):
            return np.full((144, 256, 3), 13, np.uint8), {"frame_id": 0}
        def step(self, action):
            return np.full((144, 256, 3), 99, np.uint8), 0., True, False, {
                "is_success": True, "frame_id": 1, "action_applied": action.tolist()}
    events = iter([[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)],
                   [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s)]])
    monkeypatch.setattr(recording, "ValorantEnv", FakeEnv)
    monkeypatch.setattr(pygame.event, "get", lambda: next(events, [pygame.event.Event(pygame.QUIT)]))
    args = Namespace(case=None, scenario="unused", split="train", seed=0, episodes=1, output=tmp_path, mouse_scale=.08)
    recording.record(args)
    files = list(tmp_path.glob("episode-*/episode.json"))
    assert len(files) == 1
    meta = json.loads(files[0].read_text())
    assert meta["accepted"] and meta["steps"] == 1
    with np.load(files[0].parent / meta["chunks"][0]["file"]) as z:
        assert np.all(z["frames"] == 13)  # action labels the pre-action frame, not the 99-valued next frame
        assert z["actions"][0, 5] == 0  # the start key must not accidentally jump
