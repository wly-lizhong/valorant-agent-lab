"""Manual browser UI check without GPU: python tests/serve_web_fixture.py.

Uses only synthetic test pixels; output cannot be loaded as ordinary human data.
The fixed token below is for this loopback-only fixture, never a production secret.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from aiohttp import web
from valorant_lab.env import ROOT
from valorant_lab.scenario import load_scenario, noop
from valorant_lab.web_control import Settings, create_app


class SyntheticEnv:
    def __init__(self, path):
        self.config = load_scenario(path)
        self.noop = noop(self.config)
        self.engine_metadata = {"render_device": "synthetic UI test"}
        self.step_count = 0

    def frame(self):
        im = Image.new("RGB", (256, 144), "#273849")
        draw = ImageDraw.Draw(im)
        draw.polygon([(0, 144), (100, 64), (156, 64), (256, 144)], fill="#576875")
        draw.rectangle((112, 25, 144, 65), fill="#6be3b8")
        draw.text((10, 10), "SYNTHETIC UI TEST", fill="white")
        draw.text((10, 125), f"steps {self.step_count}/90", fill="white")
        draw.line((124, 72, 132, 72), fill="white")
        draw.line((128, 68, 128, 76), fill="white")
        return np.array(im)

    def reset(self, **kwargs):
        return self.frame(), {"step": 0, "clock": "lockstep"}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= 90
        return self.frame(), 0., done, False, {"step": self.step_count, "is_success": done,
                                             "action_applied": action.tolist()}

    def close(self):
        pass


if __name__ == "__main__":
    settings = Settings(output=ROOT / "runs/ui-test/data", queue=ROOT / "runs/ui-test/queue",
                        state_dir=ROOT / "runs/ui-test/state", source="script_test")
    app = create_app(settings, token="local-ui-test-only-not-a-production-token",
                     origins=["http://127.0.0.1:8766", "http://localhost:8766"], env_factory=SyntheticEnv)
    web.run_app(app, host="127.0.0.1", port=8766, access_log=None)
