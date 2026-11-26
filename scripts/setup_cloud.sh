#!/usr/bin/env bash
# Run after setup_ubuntu.sh. Uses the provider's existing NVIDIA driver.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${VALORANT_LAB_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ "$(uname -s)" != Linux || ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Prepare the Ubuntu Python environment first (scripts/setup_ubuntu.sh).' >&2
  exit 1
fi
SUDO=()
if [[ "$(id -u)" != 0 ]]; then SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y xserver-xorg-core xauth mesa-utils
"$PYTHON_BIN" -m pip install -e '.[web]'
"$PYTHON_BIN" -m valorant_lab.cloud check
printf '%s\n' 'Cloud utilities installed. No NVIDIA driver was installed or changed.'
printf '%s\n' 'Follow docs/cloud-workflow.md to start and validate a private GPU display.'
