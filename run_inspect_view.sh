#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSPECT_BIN="${ROOT_DIR}/.venv/bin/inspect"
exec "${INSPECT_BIN}" view --log-dir "${ROOT_DIR}/eval_logs" --port 5010 --host 0.0.0.0 "$@"
