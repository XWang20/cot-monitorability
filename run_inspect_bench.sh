#!/usr/bin/env bash
# ==============================================================================
# CoT Monitorability Benchmarks — Unified Inspect Evals Runner
# Server: wangxing@8.160.177.124:/home/wangxing/cot-monitorability
# Proxy:  http://127.0.0.1:8000/v1 (API Key: a)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSPECT_BIN="${ROOT_DIR}/.venv/bin/inspect"
cd "${ROOT_DIR}"

# All benchmark models use the same local OpenAI-compatible proxy.
# Do not source the legacy .env: it may select the old external development API.
# Explicitly exported values remain available for deployments using another proxy.
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

if [[ "${EFFORT}" == "sweep" ]]; then
  for eff in low medium high; do
    echo ">>> [Effort Sweep] Running effort=${eff} for bench=${BENCH} model=${MODEL} <<<"
    "$0" "${BENCH}" "${MODEL}" "${LIMIT}" "${MAX_CONN}" "${eff}" "${EXTRA_ARGS[@]}"
  done
  exit 0
fi

run_one() {
  local bench_id="$1"
  local task_name="$2"
  local log_subdir="$3"
  local default_effort="$4"
  shift 4
  local chosen_effort="${EFFORT}"
  if [[ "${chosen_effort}" == "auto" ]]; then
    chosen_effort="${default_effort}"
  fi
  echo "=================================================================="
  echo "[Inspect Eval] Task: ${task_name} | Model: ${MODEL} | Effort: ${chosen_effort} | Limit: ${LIMIT}"
  echo "=================================================================="
  "${INSPECT_BIN}" eval "${task_name}" \
    --model "${MODEL}" \
    --reasoning-effort "${chosen_effort}" \
    --limit "${LIMIT}" \
    --max-connections "${MAX_CONN}" \
    --display plain \
    --log-dir "${ROOT_DIR}/eval_logs/${log_subdir}" \
    -T "target_reasoning_effort=${chosen_effort}" \
    "$@" "${EXTRA_ARGS[@]}"
}

case "${BENCH}" in
  1|cot_controllability_qa)
    run_one 1 "inspect_evals/cot_controllability_qa" "1_cot_controllability_qa" "high" \
      -T dataset=gpqa -T modes=word_suppression
    ;;
  2|openai_process|openai_monitorability_process)
    run_one 2 "inspect_evals/openai_monitorability_process" "2_openai_monitorability_process" "medium"
    ;;
  3|openai_intervention|openai_monitorability_intervention)
    run_one 3 "inspect_evals/openai_monitorability_intervention" "3_openai_monitorability_intervention" "medium" \
      -T dataset=scruples_suggest_wrong
    ;;
  4|cot_immediacy|cot_immediacy_2510)
    run_one 4 "inspect_evals/cot_immediacy_2510" "4_cot_immediacy_2510" "high" \
      -T dataset=gpqa
    ;;
  5|ncri|ncri_bench)
    "${INSPECT_BIN}" eval "inspect_evals/ncri_bench" \
      --model "${MODEL}" \
      --limit "${LIMIT}" \
      --max-connections "${MAX_CONN}" \
      --display plain \
      --log-dir "${ROOT_DIR}/eval_logs/5_ncri_bench" \
      -T banks=arithmetic "${EXTRA_ARGS[@]}"
    ;;
  6|monitorbench)
    run_one 6 "inspect_evals/monitorbench" "6_monitorbench" "high" \
      --model-role monitor=openai/deepseek-v4.1-flash-1017 \
      -T subset=cue_condition.preference -T stress_test=standard
    ;;
  all)
    for b in 1 2 3 4 5 6; do
      "$0" "${b}" "${MODEL}" "${LIMIT}" "${MAX_CONN}" "${EFFORT}" "${EXTRA_ARGS[@]}"
    done
    ;;
  *)
    echo "Usage: $0 {1|2|3|4|5|6|all} [model] [limit] [max_connections] [auto|low|medium|high|sweep] [extra inspect args...]"
    exit 1
    ;;
esac
