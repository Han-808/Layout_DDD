#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
BLENDER="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
DATASET_ROOT="$REPO_ROOT/Support/datasets/cal_dataset2_non_l1_evidence"
RENDER_ROOT="$REPO_ROOT/Support/artifacts/outputs/cal_dataset2_non_l1_review_renders"
CONTOUR_ROOT="${CONTOUR_ROOT:-$REPO_ROOT/Support/artifacts/outputs/cal_dataset2_non_l1_contours}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp2_non_l1_visual_evidence_gpt56}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_non_l1_judge.json}"
GT_PATH="${GT_PATH:-$DATASET_ROOT/human_review/human_gt_20260725.tsv}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:4010/v1}"
PROXY_PORT="${LITELLM_GPT56_PORT:-4010}"
PROXY_LOG="${PROXY_LOG:-/private/tmp/layoutddd-exp2-litellm-gpt56.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-/private/tmp/layoutddd-exp2-litellm-gpt56.pid}"
REPEATS="${REPEATS:-2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_RETRY_PASSES="${MAX_RETRY_PASSES:-6}"
KEEP_PROXY="${KEEP_PROXY:-0}"
PHASE="${PHASE:-all}"

case "$PHASE" in
  all|contour|judge|analysis|plan) ;;
  *)
    echo "PHASE must be all, contour, judge, analysis, or plan" >&2
    exit 2
    ;;
esac
if [[ ! -x "$PYTHON" ]]; then
  echo "Benchmark Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -x "$BLENDER" ]]; then
  echo "Blender is not executable: $BLENDER" >&2
  exit 2
fi
if ! [[ "$REPEATS" =~ ^[2-9][0-9]*$ ]]; then
  echo "REPEATS must be an integer >= 2" >&2
  exit 2
fi
if ! [[ "$MAX_WORKERS" =~ ^[1-8]$ ]]; then
  echo "MAX_WORKERS must be an integer from 1 to 8" >&2
  exit 2
fi
if ! [[ "$MAX_RETRY_PASSES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RETRY_PASSES must be a positive integer" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"

if [[ "$PHASE" == "plan" ]]; then
  "$PYTHON" scripts/validate_cal_dataset2_non_l1.py
  "$PYTHON" scripts/prepare_cal_dataset2_non_l1_contours.py \
    --dataset-root "$DATASET_ROOT" \
    --render-root "$RENDER_ROOT" \
    --out-root "$CONTOUR_ROOT" \
    --asset-root "$REPO_ROOT/Support/Assets/imaginarium_assets" \
    --blender-bin "$BLENDER" \
    --plan-only
  echo "Plan completed. No Blender render and no model call were made."
  exit 0
fi

if [[ "$PHASE" == "all" || "$PHASE" == "contour" ]]; then
  "$PYTHON" scripts/validate_cal_dataset2_non_l1.py
  "$PYTHON" scripts/prepare_cal_dataset2_non_l1_contours.py \
    --dataset-root "$DATASET_ROOT" \
    --render-root "$RENDER_ROOT" \
    --out-root "$CONTOUR_ROOT" \
    --asset-root "$REPO_ROOT/Support/Assets/imaginarium_assets" \
    --blender-bin "$BLENDER" \
    --resume
fi

PROXY_OWNED=0
PROXY_PID=""
cleanup_proxy() {
  if [[ "$PROXY_OWNED" == "1" && "$KEEP_PROXY" != "1" && -n "$PROXY_PID" ]]; then
    if kill -0 "$PROXY_PID" 2>/dev/null; then
      kill "$PROXY_PID" 2>/dev/null || true
      wait "$PROXY_PID" 2>/dev/null || true
    fi
  fi
}
trap cleanup_proxy EXIT INT TERM

ensure_proxy() {
  if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "LITELLM_MASTER_KEY is not set in this shell." >&2
    exit 2
  fi
  if ! lsof -nP -iTCP:"$PROXY_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    for variable_name in OPENAPI_BASE_URL OPENAPI_GPT_KEY; do
      if [[ -z "${!variable_name:-}" ]]; then
        echo "Required to start LiteLLM: $variable_name is not set." >&2
        exit 2
      fi
    done
    : >"$PROXY_LOG"
    LITELLM_GPT56_PORT="$PROXY_PORT" \
      bash Support/bash/local/run_litellm_gpt56sol_proxy.sh \
      >>"$PROXY_LOG" 2>&1 &
    PROXY_PID=$!
    PROXY_OWNED=1
    echo "$PROXY_PID" >"$PROXY_PID_FILE"
    for _attempt in $(seq 1 60); do
      if curl --noproxy "*" -fsS \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        "http://127.0.0.1:$PROXY_PORT/v1/models" \
        >/dev/null 2>&1; then
        break
      fi
      if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "LiteLLM proxy exited during startup. See $PROXY_LOG" >&2
        tail -n 60 "$PROXY_LOG" >&2 || true
        exit 1
      fi
      sleep 2
    done
  fi

  local smoke_image
  smoke_image="$RENDER_ROOT/observations/obs_5e1b4e5306d4e48b/local_views/camera_00_oar_00_sw.png"
  if [[ ! -f "$smoke_image" ]]; then
    smoke_image="$(find "$RENDER_ROOT/observations" -type f -path '*/local_views/*.png' -print -quit)"
  fi
  "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint "$JUDGE_ENDPOINT" \
    --model gpt-5.6-sol \
    --api-key-env LITELLM_MASTER_KEY \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal \
    --image-path "$smoke_image" \
    >"$RUN_ROOT/model_preflight.json"
  echo "GPT-5.6-Sol multimodal preflight: passed"
}

run_repeat() {
  local repeat_index="$1"
  local out_dir="$RUN_ROOT/repeat_${repeat_index}"
  local retry_pass failure_count
  mkdir -p "$out_dir"
  for retry_pass in $(seq 1 "$MAX_RETRY_PASSES"); do
    echo "==== repeat $repeat_index / retry pass $retry_pass ===="
    "$PYTHON" scripts/judge_cal_dataset2_non_l1_evidence.py \
      --dataset-root "$DATASET_ROOT" \
      --render-root "$RENDER_ROOT" \
      --contour-root "$CONTOUR_ROOT" \
      --ground-truth "$GT_PATH" \
      --judge-config "$JUDGE_CONFIG" \
      --out-dir "$out_dir" \
      --repeat-id "repeat_${repeat_index}" \
      --max-workers "$MAX_WORKERS" \
      --resume \
      --continue-on-error
    failure_count="$(
      "$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["failure_count"])' \
        "$out_dir/run_manifest.json"
    )"
    if [[ "$failure_count" == "0" ]]; then
      return 0
    fi
    echo "repeat $repeat_index still has $failure_count failed calls; resuming"
    sleep $((retry_pass * 5))
  done
  echo "repeat $repeat_index still has failures after $MAX_RETRY_PASSES passes" >&2
  return 1
}

if [[ "$PHASE" == "all" || "$PHASE" == "judge" ]]; then
  ensure_proxy
  for repeat_index in $(seq 1 "$REPEATS"); do
    run_repeat "$repeat_index"
  done
fi

if [[ "$PHASE" == "all" || "$PHASE" == "analysis" ]]; then
  "$PYTHON" scripts/analyze_cal_dataset2_non_l1_repeats.py \
    --left-run "$RUN_ROOT/repeat_1" \
    --right-run "$RUN_ROOT/repeat_2" \
    --out-dir "$RUN_ROOT/analysis"
fi

echo "Experiment completed successfully."
echo "Run root: $RUN_ROOT"
echo "Repeat analysis: $RUN_ROOT/analysis/report.md"
echo "Proxy log: $PROXY_LOG"
