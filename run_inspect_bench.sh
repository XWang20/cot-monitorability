#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
if [[ -z "${INSPECT_BIN:-}" ]]; then
  INSPECT_BIN="${ROOT_DIR}/.venv/bin/inspect"
  if [[ -x "${ROOT_DIR}/.venv-package/bin/inspect" ]]; then
    INSPECT_BIN="${ROOT_DIR}/.venv-package/bin/inspect"
  fi
fi
PYTHON_BIN="$(dirname "${INSPECT_BIN}")/python"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${PROXY_API_KEY:-a}}"
export INSPECT_EVALS_JUDGE_MODEL="${INSPECT_EVALS_JUDGE_MODEL:-openai/deepseek-v4.1-flash-1017}"
BENCH="${1:-all}"
MODEL="${2:-openai/deepseek-v4.1-flash-1017}"
LIMIT="${3:-10}"
MAX_CONN="${4:-5}"
EFFORT="${5:-auto}"
shift $(( $# >= 5 ? 5 : $# )) || true
EXTRA_ARGS=("$@")
case "${EFFORT}" in auto|low|medium|high|sweep) ;; *) echo "Unsupported effort: ${EFFORT}" >&2; exit 1 ;; esac

COMPAT="${COT_USE_LITELLM_COMPAT:-auto}"
if [[ "${COMPAT}" == auto ]]; then
  COMPAT=0
  if [[ "${OPENAI_BASE_URL}" == *127.0.0.1:8000* || "${OPENAI_BASE_URL}" == *localhost:8000* ]]; then COMPAT=1; fi
fi
export COT_RUNNER_COMPAT="${COMPAT}"
ROLE_CONFIG="$("${PYTHON_BIN}" -c 'import json,os; cfg={"model":os.environ["INSPECT_EVALS_JUDGE_MODEL"],"reasoning_effort":"high"}; cfg.update({"extra_body":{"allowed_openai_params":["reasoning_effort"],"reasoning":{"effort":"high"}}} if os.environ["COT_RUNNER_COMPAT"]=="1" else {}); print(json.dumps(cfg))')"

if [[ "${EFFORT}" == sweep ]]; then
  for eff in low medium high; do "$0" "${BENCH}" "${MODEL}" "${LIMIT}" "${MAX_CONN}" "${eff}" "${EXTRA_ARGS[@]}"; done
  exit 0
fi

run_one() {
  local task="$1" log_dir="$2" default_effort="$3"
  shift 3
  local chosen="${EFFORT}"
  if [[ "${default_effort}" == disabled ]]; then chosen=disabled; elif [[ "${chosen}" == auto ]]; then chosen="${default_effort}"; fi
  local proxy_args=()
  if [[ "${COMPAT}" == 1 ]]; then
    local body='{"allowed_openai_params":["reasoning_effort"]}'
    if [[ "${chosen}" != disabled && "${MODEL,,}" != *minimax* && "${MODEL,,}" != *m3* ]]; then
      printf -v body '{"allowed_openai_params":["reasoning_effort"],"reasoning":{"effort":"%s"}}' "${chosen}"
    fi
    proxy_args=(--extra-body "${body}")
  fi
  local effort_args=()
  if [[ "${chosen}" != disabled ]]; then
    effort_args=(--reasoning-effort "${chosen}" -T "target_reasoning_effort=${chosen}")
  fi
  "${INSPECT_BIN}" eval "cot_monitorability/${task}" --model "${MODEL}" \
    --limit "${LIMIT}" --max-connections "${MAX_CONN}" --display plain \
    --log-dir "${ROOT_DIR}/eval_logs/${log_dir}" \
    --model-role "judge=${ROLE_CONFIG}" --model-role "monitor=${ROLE_CONFIG}" \
    "${effort_args[@]}" "${proxy_args[@]}" "$@" "${EXTRA_ARGS[@]}"
}

case "${BENCH}" in
  1|cot_controllability_qa) run_one cot_controllability_qa 1_cot_controllability_qa high -T dataset=gpqa -T modes=word_suppression ;;
  2|openai_process|openai_monitorability_process) run_one openai_monitorability_process 2_openai_monitorability_process medium ;;
  3|openai_intervention|openai_monitorability_intervention) run_one openai_monitorability_intervention 3_openai_monitorability_intervention medium -T dataset=scruples_suggest_wrong ;;
  4|cot_immediacy|cot_immediacy_2510) run_one cot_immediacy_2510 4_cot_immediacy_2510 high -T dataset=gpqa ;;
  5|ncri|ncri_bench) run_one ncri_bench 5_ncri_bench disabled -T banks=arithmetic ;;
  6|monitorbench) run_one monitorbench 6_monitorbench high -T subset=cue_condition.preference -T stress_test=standard ;;
  all) for b in 1 2 3 4 5 6; do "$0" "${b}" "${MODEL}" "${LIMIT}" "${MAX_CONN}" "${EFFORT}" "${EXTRA_ARGS[@]}"; done ;;
  *) echo "Usage: $0 {1|2|3|4|5|6|all} [model] [limit] [connections] [auto|low|medium|high|sweep] [Inspect args...]" >&2; exit 1 ;;
esac
