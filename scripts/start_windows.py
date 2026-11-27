"""One-click Windows preparation; no administrator privileges or training GPU needed."""
from pathlib import Path
import argparse
import hashlib
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup-only", action="store_true")
    a = p.parse_args()
    if os.name != "nt" or not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use 64-bit Windows with Python 3.10-3.13")
    python = ROOT / ".venv-windows/Scripts/python.exe"
    if not python.exists():
        run(sys.executable, "-m", "venv", ROOT / ".venv-windows")
    sources = [ROOT / "pyproject.toml", ROOT / "dependencies.lock.json", ROOT / "scripts/bootstrap.py",
               *sorted((ROOT / "engine/overlay").glob("*.cs"))]
    fingerprint = hashlib.sha256(b"".join(x.read_bytes() for x in sources)).hexdigest()
    stamp = ROOT / ".cache/windows-ready.sha256"
    if not stamp.exists() or stamp.read_text() != fingerprint or not (ROOT / "local.windows.json").exists():
        run(python, "-m", "pip", "install", "-e", ".")
        maps = ROOT / "assets/maps-data-v2.15.0"
        if (maps / "asset-inventory.json").exists():
            run(python, "scripts/assets.py", "verify", "--root", maps)
        else:
            checksum = (ROOT / "assets/lotus-assets.sha256").read_text().split()[0]
            run(python, "scripts/assets.py", "install", "--archive", ROOT / "assets/lotus-assets.zip",
                "--sha256", checksum, "--destination", maps)
        run(python, "scripts/bootstrap.py", "--platform", "windows", "--maps-dir", maps,
            "--local-config", ROOT / "local.windows.json")
        stamp.write_text(fingerprint)
    if not a.setup_only:
        run(python, "-m", "valorant_lab.windows_app")


if __name__ == "__main__":
    main()
