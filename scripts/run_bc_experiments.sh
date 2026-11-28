#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${VALORANT_LAB_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' 'Python environment missing; set VALORANT_LAB_PYTHON.' >&2
  exit 1
fi
# Detached jobs must load instance-specific graphics libraries themselves.
# This is an explicit trusted operator configuration, not a downloaded script.
if [[ -n "${VALORANT_LAB_GRAPHICS_ENV:-}" ]]; then
  if [[ ! -f "$VALORANT_LAB_GRAPHICS_ENV" ]]; then
    printf '%s\n' 'VALORANT_LAB_GRAPHICS_ENV does not name an existing file.' >&2
    exit 1
  fi
  source "$VALORANT_LAB_GRAPHICS_ENV"
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/run_bc_experiments.py" "$@"
