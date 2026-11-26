#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${VALORANT_LAB_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Python environment missing; follow README setup instructions.' >&2
  exit 1
fi
exec "$PYTHON_BIN" -m valorant_lab.cloud run --gpu "${VALORANT_LAB_GPU:-0}" -- "$@"
