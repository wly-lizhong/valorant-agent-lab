#!/usr/bin/env bash
# First install and subsequent launches use the same command.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "$(uname -s)" != Linux ]]; then
  printf '%s\n' '请在 Ubuntu 云服务器中运行此脚本。' >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  printf '%s\n' '服务器缺少 python3，请先安装：sudo apt-get install python3' >&2
  exit 1
fi
exec python3 -u scripts/start_cloud.py "$@"
