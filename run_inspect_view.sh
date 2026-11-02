#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${INSPECT_BIN:-}" ]]; then
  INSPECT_BIN="${ROOT_DIR}/.venv/bin/inspect"
  if [[ -x "${ROOT_DIR}/.venv-package/bin/inspect" ]]; then INSPECT_BIN="${ROOT_DIR}/.venv-package/bin/inspect"; fi
fi
exec "${INSPECT_BIN}" view --log-dir "${ROOT_DIR}/eval_logs" --port 5010 --host 0.0.0.0 "$@"
