#!/usr/bin/env bash
# Run a package task against a configured OpenAI-compatible API.
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
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL to your API endpoint}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY to your API credential}"
export OPENAI_BASE_URL OPENAI_API_KEY
export COT_RUNNER_COMPAT="${COT_USE_LITELLM_COMPAT:-0}" COT_RUNNER_MODEL="${MODEL}"

# Resolve defaults and explicit overrides before constructing datasets.
read -r EFFORT ROLE_CONFIG BODY USE_TARGET CONTROL < <(uv run --locked python - "${TASK}" "$@" <<'PY'
import argparse
import inspect
import json
import os
import sys

from inspect_ai.model import GenerateConfig
from cot_monitorability import _registry

name = sys.argv[1]
if name not in _registry.__all__:
    raise SystemExit(f"Unknown task: {name}")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--reasoning-effort")
parser.add_argument("-T", action="append", default=[])
options, _ = parser.parse_known_args(sys.argv[2:])
task_args = dict(arg.split("=", 1) for arg in options.T if "=" in arg)
parameter = inspect.signature(getattr(_registry, name)).parameters.get(
    "target_reasoning_effort"
)
task_effort = (
    task_args.get("target_reasoning_effort") if parameter else task_args.get("effort")
)
if task_effort and options.reasoning_effort and task_effort != options.reasoning_effort:
    raise SystemExit("Conflicting task and model reasoning efforts.")
effort = task_effort or options.reasoning_effort or os.environ.get("REASONING_EFFORT")
control = parameter is None and (
    task_args.get("disable_reasoning", "true").lower() == "false"
    or task_args.get("effort") not in (None, "none")
)
if parameter is None:
    if not control and effort not in (None, "none"):
        raise SystemExit(
            "NCRI no-CoT mode requires none; set disable_reasoning=false for a positive control."
        )
    effort = effort or ("high" if control else "none")
else:
    effort = effort or parameter.default
GenerateConfig(reasoning_effort=effort)

model = os.environ.get("JUDGE_MODEL") if parameter else None
role = {"model": model, "reasoning_effort": "high"} if model else None
body = None
if os.environ["COT_RUNNER_COMPAT"] == "1":
    body = {"allowed_openai_params": ["reasoning_effort"]}
    if effort != "none" and not any(
        x in os.environ["COT_RUNNER_MODEL"].lower() for x in ("minimax", "m3")
    ):
        body["reasoning"] = {"effort": effort}
    if role:
        role["extra_body"] = {
            "allowed_openai_params": ["reasoning_effort"],
            "reasoning": {"effort": "high"},
        }
encode = lambda value: json.dumps(value, separators=(",", ":")) if value else "none"
print(effort, encode(role), encode(body), int(parameter is not None), int(control))
PY
)
LOG_DIR="${ROOT_DIR}/eval_logs/${TASK}"
if [[ "${CONTROL}" == 1 ]]; then
  LOG_DIR="${ROOT_DIR}/tmp/controls/${TASK}"
fi
ARGS=(eval "cot_monitorability/${TASK}" --model "${MODEL}" --log-dir "${LOG_DIR}" --reasoning-effort "${EFFORT}")
if [[ "${USE_TARGET}" == 1 ]]; then
  ARGS+=(-T "target_reasoning_effort=${EFFORT}")
fi
if [[ "${CONTROL}" == 1 ]]; then
  ARGS+=(--tags positive-control)
fi
if [[ "${BODY}" != none ]]; then
  ARGS+=(--extra-body "${BODY}")
fi
if [[ "${ROLE_CONFIG}" != none ]]; then
  ARGS+=(--model-role "judge=${ROLE_CONFIG}" --model-role "monitor=${ROLE_CONFIG}")
fi
exec uv run --locked inspect "${ARGS[@]}" "$@"
