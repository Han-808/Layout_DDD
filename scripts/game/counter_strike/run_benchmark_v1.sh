#!/usr/bin/env bash
#
# Resumable five-case Counter-Strike static 3D benchmark.
#
# The proxy and LITELLM_MASTER_KEY must already exist in this terminal.  The
# script never prints the key.  Each case captures its original Three.js runtime
# once, then reuses that immutable bank for canonical L1/L3 and CS L4.

set -u
set -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1

export PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/configs/models/gpt5_6_sol_litellm_local_cs_judge.json}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-$REPO/configs/game/counter_strike/benchmark_v1.yaml}"
GAME_MODE_CONFIG="${GAME_MODE_CONFIG:-$REPO/configs/game/game_mode_canonical_v1.yaml}"
CORPUS_ROOT="${CORPUS_ROOT:-$REPO/Support/datasets/game_corpus/20260720_190732 2/cs_fps}"
RUN_ROOT="${RUN_ROOT:-$REPO/Support/artifacts/outputs/cs_benchmark_v1_gpt56_20260727}"
PHASE="${PHASE:-all}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RESUME="${RESUME:-1}"

mkdir -p "$RUN_ROOT/cases" "$RUN_ROOT/phase_logs" "$RUN_ROOT/status"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python runtime missing: $PYTHON" >&2
  exit 2
fi
if [[ "$PHASE" != "capture" && -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "LITELLM_MASTER_KEY is not set in this terminal." >&2
  exit 2
fi
if [[ ! "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_WORKERS must be a positive integer." >&2
  exit 2
fi

CASE_SLUGS=(
  "claude_opus_4_7"
  "claude_opus_4_8"
  "hy3"
  "kimi_k3"
  "minimax_m3"
)
CORPUS_DIRS=(
  "Claude-Opus-4_7"
  "Claude-Opus-4_8"
  "Hy3"
  "Kimi-K3"
  "MiniMax-M3"
)
CONTRACTS=(
  "$REPO/configs/game/counter_strike/corpus/claude_opus_4_7.json"
  "$REPO/configs/game/counter_strike/corpus/claude_opus_4_8.json"
  "$REPO/configs/game/counter_strike/corpus/hy3.json"
  "$REPO/configs/game/counter_strike/corpus/kimi_k3.json"
  "$REPO/configs/game/counter_strike/corpus/minimax_m3.json"
)

if [[ "$PHASE" != "capture" ]]; then
  PREFLIGHT_IMAGE="${PREFLIGHT_IMAGE:-$RUN_ROOT/cases/claude_opus_4_7/renders/global_global_oblique_00.png}"
  if [[ ! -f "$PREFLIGHT_IMAGE" ]]; then
    LEGACY_PREFLIGHT_IMAGE="$REPO/Support/artifacts/outputs/cs_benchmark_v1/captures/Claude-Opus-4_7/global_global_oblique_00.png"
    if [[ -f "$LEGACY_PREFLIGHT_IMAGE" ]]; then
      PREFLIGHT_IMAGE="$LEGACY_PREFLIGHT_IMAGE"
    fi
  fi
  if [[ ! -f "$PREFLIGHT_IMAGE" ]]; then
    echo "Multimodal preflight image is missing: $PREFLIGHT_IMAGE" >&2
    echo "Run PHASE=capture first, or set PREFLIGHT_IMAGE to a valid local PNG." >&2
    exit 2
  fi
  echo "==== GPT-5.6-Sol multimodal preflight ===="
  "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint http://127.0.0.1:4010/v1 \
    --model gpt-5.6-sol \
    --api-key-env LITELLM_MASTER_KEY \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal \
    --image-path "$PREFLIGHT_IMAGE" \
    >"$RUN_ROOT/model_preflight.json"
  echo "GPT-5.6-Sol multimodal preflight: passed"
fi

run_case() {
  local index="$1"
  local slug="${CASE_SLUGS[$index]}"
  local corpus="${CORPUS_DIRS[$index]}"
  local contract="${CONTRACTS[$index]}"
  local source_root="$CORPUS_ROOT/$corpus"
  local case_out="$RUN_ROOT/cases/$slug"
  local log="$RUN_ROOT/phase_logs/$slug.log"
  local status_file="$RUN_ROOT/status/$slug.status"
  local final_report="$case_out/counter_strike_evaluation_report.json"

  if [[ "$RESUME" == "1" && "$PHASE" != "capture" && -f "$final_report" ]] &&
    "$PYTHON" -c \
      'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("benchmark_score_status")=="complete" else 1)' \
      "$final_report"; then
    echo "cached_complete" >"$status_file"
    echo "[cached complete] $slug"
    return 0
  fi

  echo "[start] $slug"
  if "$PYTHON" -m benchmark.game_scene.counter_strike.runner \
    --game-root "$source_root" \
    --case-contract "$contract" \
    --out-dir "$case_out" \
    --model-config "$MODEL_CONFIG" \
    --game-mode-config "$GAME_MODE_CONFIG" \
    --benchmark-config "$BENCHMARK_CONFIG" \
    --phase "$PHASE" \
    >"$log" 2>&1; then
    echo "ok" >"$status_file"
    echo "[ok] $slug"
    return 0
  fi
  echo "failed" >"$status_file"
  echo "[failed] $slug; see $log" >&2
  return 1
}

running_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

echo "==== launch CS corpus: phase=$PHASE workers=$MAX_WORKERS ===="
index=0
while [[ "$index" -lt "${#CASE_SLUGS[@]}" ]]; do
  while [[ "$(running_jobs)" -ge "$MAX_WORKERS" ]]; do
    sleep 2
  done
  run_case "$index" &
  index=$((index + 1))
done

wait

if [[ "$PHASE" == "capture" ]]; then
  failed=0
  for slug in "${CASE_SLUGS[@]}"; do
    status="$(cat "$RUN_ROOT/status/$slug.status" 2>/dev/null || true)"
    if [[ "$status" != "ok" && "$status" != "cached_complete" ]]; then
      failed=$((failed + 1))
    fi
  done
  echo "Capture phase finished: failures=$failed"
  exit "$failed"
fi

aggregate_args=()
for contract in "${CONTRACTS[@]}"; do
  aggregate_args+=(--expected-contract "$contract")
done

if "$PYTHON" -m benchmark.game_scene.counter_strike.runner \
  --aggregate-root "$RUN_ROOT" \
  "${aggregate_args[@]}" \
  >"$RUN_ROOT/aggregate.log" 2>&1; then
  echo "==== Counter-Strike benchmark complete ===="
  echo "Summary: $RUN_ROOT/corpus_summary.tsv"
  echo "JSON:    $RUN_ROOT/corpus_summary.json"
  exit 0
fi

echo "Counter-Strike benchmark incomplete; see $RUN_ROOT/aggregate.log" >&2
exit 1
