#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This bootstrap currently supports x86_64 only.' >&2
  exit 1
fi
SUDO=()
if [[ "$(id -u)" != 0 ]]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y python3-venv curl ca-certificates xvfb xauth libgl1 libx11-6 libxcursor1 libxinerama1 libxrandr2 libxi6
# Package names differ between Ubuntu 22.04 and 24.04.
if apt-cache show libasound2t64 >/dev/null 2>&1; then
  "${SUDO[@]}" apt-get install -y libasound2t64 libicu74
else
  "${SUDO[@]}" apt-get install -y libasound2 libicu70
fi
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/python scripts/bootstrap.py "$@"
printf '%s\n' 'Environment prepared. Install PyTorch 2.8.0 for CPU/CUDA, then pip install -r requirements-train.txt.'
