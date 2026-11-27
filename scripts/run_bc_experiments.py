"""Run a bounded local/cloud experiment queue; never calls a language model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from valorant_lab.experiments import main

if __name__ == "__main__":
    raise SystemExit(main())
