#!/usr/bin/env bash
# Run a package task against a local OpenAI-compatible proxy.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <task-name> <model> [Inspect arguments...]" >&2
  exit 1
fi
TASK="$1"
MODEL="$2"
shift 2
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-a}"
COMPAT="${COT_USE_LITELLM_COMPAT:-auto}"
if [[ "${COMPAT}" == auto ]]; then
  COMPAT=0
  if [[ "${OPENAI_BASE_URL}" == *127.0.0.1:8000* || "${OPENAI_BASE_URL}" == *localhost:8000* ]]; then COMPAT=1; fi
fi

# Reading the factory signature does not construct datasets or call a model.
export COT_RUNNER_COMPAT="${COMPAT}" COT_RUNNER_MODEL="${MODEL}"
read -r EFFORT ROLE_CONFIG BODY < <(uv run --locked python - "${TASK}" <<'PY'
import inspect
import json
import os
import sys
from cot_monitorability import _registry

name = sys.argv[1]
if name not in _registry.__all__:
    raise SystemExit(f"Unknown task: {name}")
parameter = inspect.signature(getattr(_registry, name)).parameters.get(
    "target_reasoning_effort"
)
override = os.environ.get("REASONING_EFFORT")
if parameter is None and override:
    raise SystemExit(
        "NCRI uses its own no-CoT request; use task arguments to configure it."
    )
effort = override or (parameter.default if parameter else "disabled")
model = os.environ.get("JUDGE_MODEL")
role = {"model": model, "reasoning_effort": "high"} if model else None
body = None
if os.environ["COT_RUNNER_COMPAT"] == "1":
    body = {"allowed_openai_params": ["reasoning_effort"]}
    if effort != "disabled" and not any(
        x in os.environ["COT_RUNNER_MODEL"].lower() for x in ("minimax", "m3")
    ):
        body["reasoning"] = {"effort": effort}
    if role:
        role["extra_body"] = {
            "allowed_openai_params": ["reasoning_effort"],
            "reasoning": {"effort": "high"},
        }
encode = lambda value: json.dumps(value, separators=(",", ":")) if value else "none"
print(effort, encode(role), encode(body))
PY
)
ARGS=(eval "cot_monitorability/${TASK}" --model "${MODEL}" --log-dir "${ROOT_DIR}/eval_logs/${TASK}")
if [[ "${EFFORT}" != disabled ]]; then
  ARGS+=(--reasoning-effort "${EFFORT}" -T "target_reasoning_effort=${EFFORT}")
fi
if [[ "${BODY}" != none ]]; then
  ARGS+=(--extra-body "${BODY}")
fi
if [[ "${ROLE_CONFIG}" != none ]]; then
  ARGS+=(--model-role "judge=${ROLE_CONFIG}" --model-role "monitor=${ROLE_CONFIG}")
fi
exec uv run --locked inspect "${ARGS[@]}" "$@"
