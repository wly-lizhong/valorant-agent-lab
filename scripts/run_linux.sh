#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${VALORANT_LAB_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Python environment missing; follow README setup instructions.' >&2
  exit 1
fi
if [[ -z "${DISPLAY:-}" ]]; then
  printf '%s\n' 'No DISPLAY: using Xvfb + software rendering. CUDA policy training remains independent.' >&2
  exec env LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s '-screen 0 1280x720x24' "$PYTHON_BIN" -m valorant_lab.cli "$@"
fi
exec "$PYTHON_BIN" -m valorant_lab.cli "$@"
