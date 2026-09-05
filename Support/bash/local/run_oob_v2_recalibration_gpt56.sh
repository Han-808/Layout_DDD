#!/usr/bin/env bash

set -u
set -o pipefail

# Re-run only OOB after the floor-contact contract changes.
# Historical outputs remain immutable. No Blender or camera selection is run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/Support/datasets/cal_dataset1}"
INVALID_EVIDENCE="${INVALID_EVIDENCE:-$REPO_ROOT/Support/artifacts/outputs/exp1_1}"
AMBIGUOUS_EVIDENCE="${AMBIGUOUS_EVIDENCE:-$REPO_ROOT/Support/artifacts/outputs/exp1_1_fine_edge}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_fine_edge_judge.json}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp3_oob_v2_gpt56}"
PHASE="${PHASE:-all}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RESUME="${RESUME:-1}"
AMBIGUOUS_REPEATS="${AMBIGUOUS_REPEATS:-2}"
AUTO_START_PROXY="${AUTO_START_PROXY:-1}"
KEEP_PROXY="${KEEP_PROXY:-1}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:4010/v1}"
PROXY_LOG="${PROXY_LOG:-/private/tmp/layoutddd-litellm-gpt56.log}"
PROXY_PID_FILE="${PROXY_PID_FILE:-/private/tmp/layoutddd-litellm-gpt56.pid}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
for path in "$DATASET_ROOT" "$INVALID_EVIDENCE" "$AMBIGUOUS_EVIDENCE" "$JUDGE_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required local input is missing: $path" >&2
    exit 2
  fi
done
if [[ ! "$MAX_WORKERS" =~ ^[1-8]$ ]]; then
  echo "MAX_WORKERS must be an integer from 1 to 8." >&2
  exit 2
fi
if [[ ! "$AMBIGUOUS_REPEATS" =~ ^[2-9]$ ]]; then
  echo "AMBIGUOUS_REPEATS must be an integer from 2 to 9." >&2
  exit 2
fi
case "$PHASE" in
  all|contract|source_valid|invalid_two_arm|invalid_nine_arm|ambiguous)
    ;;
  *)
    echo "PHASE must be all, contract, source_valid, invalid_two_arm, invalid_nine_arm, or ambiguous." >&2
    exit 2
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

resume_flag="--no-resume"
if [[ "$RESUME" == "1" ]]; then
  resume_flag="--resume"
fi

mkdir -p "$OUT_ROOT"

echo "==== verify OOB v2 contract ===="
"$PYTHON" scripts/run_cal_dataset1_oob_contract_replay.py \
  --check-contract \
  >"$OUT_ROOT/oob_contract.json"

if [[ "$PHASE" == "contract" ]]; then
  cat "$OUT_ROOT/oob_contract.json"
  exit 0
fi

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "LITELLM_MASTER_KEY is not set in this terminal." >&2
  exit 2
fi

proxy_owned=0
cleanup() {
  local rc=$?
  if [[ "$rc" -ne 0 && "$proxy_owned" == "1" && -f "$PROXY_PID_FILE" ]]; then
    local proxy_pid
    proxy_pid="$(cat "$PROXY_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
      kill "$proxy_pid" 2>/dev/null || true
    fi
  elif [[ "$KEEP_PROXY" == "0" && "$proxy_owned" == "1" && -f "$PROXY_PID_FILE" ]]; then
    local proxy_pid
    proxy_pid="$(cat "$PROXY_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$proxy_pid" ]] && kill -0 "$proxy_pid" 2>/dev/null; then
      kill "$proxy_pid" 2>/dev/null || true
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if ! curl --noproxy "*" -sS --connect-timeout 2 \
  -o /dev/null "${JUDGE_ENDPOINT%/v1}/health" 2>/dev/null; then
  if [[ "$AUTO_START_PROXY" != "1" ]]; then
    echo "Judge endpoint is unavailable: $JUDGE_ENDPOINT" >&2
    exit 2
  fi
  for variable_name in OPENAPI_BASE_URL OPENAPI_GPT_KEY; do
    if [[ -z "${!variable_name:-}" ]]; then
      echo "Cannot auto-start proxy: $variable_name is not set." >&2
      exit 2
    fi
  done
  echo "Starting the local GPT-5.6-Sol LiteLLM proxy."
  nohup Support/bash/local/run_litellm_gpt56sol_proxy.sh \
    >"$PROXY_LOG" 2>&1 < /dev/null &
  echo $! >"$PROXY_PID_FILE"
  proxy_owned=1
  endpoint_ready=0
  for _attempt in $(seq 1 30); do
    if curl --noproxy "*" -sS --connect-timeout 2 \
      -o /dev/null "${JUDGE_ENDPOINT%/v1}/health" 2>/dev/null; then
      endpoint_ready=1
      break
    fi
    sleep 2
  done
  if [[ "$endpoint_ready" != "1" ]]; then
    echo "Local judge proxy did not become ready. See $PROXY_LOG" >&2
    exit 1
  fi
fi

echo "==== multimodal endpoint preflight ===="
"$PYTHON" scripts/check_model_endpoint.py \
  --endpoint "$JUDGE_ENDPOINT" \
  --model gpt-5.6-sol \
  --api-key-env LITELLM_MASTER_KEY \
  --timeout-seconds 3000 \
  --max-tokens 200 \
  --no-send-temperature \
  --no-response-format-json \
  --multimodal \
  --image-path "$DATASET_ROOT/evaluation/mesh_geometry/source_valid_001/renders/standardized_top.png" \
  >"$OUT_ROOT/preflight.json"

failures=0

run_replay() {
  local label="$1"
  shift
  echo
  echo "==== $label ===="
  "$PYTHON" scripts/run_cal_dataset1_oob_contract_replay.py "$@"
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    failures=$((failures + 1))
    echo "$label failed with rc=$rc; continuing to the next OOB phase." >&2
  fi
}

if [[ "$PHASE" == "all" || "$PHASE" == "source_valid" ]]; then
  run_replay "source-valid OOB specificity under v2" \
    --experiment dataset \
    --dataset-root "$DATASET_ROOT" \
    --judge-config "$JUDGE_CONFIG" \
    --out-dir "$OUT_ROOT/source_valid" \
    --split source_valid \
    --gt-label valid \
    --max-workers "$MAX_WORKERS" \
    "$resume_flag" \
    --continue-on-error
fi

if [[ "$PHASE" == "all" || "$PHASE" == "invalid_two_arm" ]]; then
  run_replay "constructed-invalid OOB: fixed global vs deterministic local" \
    --experiment two_arm \
    --dataset-root "$DATASET_ROOT" \
    --evidence-root "$INVALID_EVIDENCE" \
    --judge-config "$JUDGE_CONFIG" \
    --out-dir "$OUT_ROOT/invalid_two_arm" \
    --split obvious_distortion \
    --split subtle_distortion \
    --gt-label invalid \
    --max-workers "$MAX_WORKERS" \
    "$resume_flag" \
    --continue-on-error
fi

if [[ "$PHASE" == "all" || "$PHASE" == "invalid_nine_arm" ]]; then
  run_replay "constructed-invalid OOB: nine VisualConfig arms" \
    --experiment nine_arm \
    --dataset-root "$DATASET_ROOT" \
    --evidence-root "$INVALID_EVIDENCE" \
    --judge-config "$JUDGE_CONFIG" \
    --out-dir "$OUT_ROOT/invalid_nine_arm" \
    --split obvious_distortion \
    --split subtle_distortion \
    --gt-label invalid \
    --max-workers "$MAX_WORKERS" \
    "$resume_flag" \
    --continue-on-error
fi

if [[ "$PHASE" == "all" || "$PHASE" == "ambiguous" ]]; then
  for repeat in $(seq 1 "$AMBIGUOUS_REPEATS"); do
    run_replay "ambiguous OOB exact-input repeat $repeat" \
      --experiment two_arm \
      --dataset-root "$DATASET_ROOT" \
      --evidence-root "$AMBIGUOUS_EVIDENCE" \
      --judge-config "$JUDGE_CONFIG" \
      --out-dir "$OUT_ROOT/ambiguous_two_arm_r${repeat}" \
      --split fine_edge \
      --gt-label ambiguous \
      --max-workers "$MAX_WORKERS" \
      "$resume_flag" \
      --continue-on-error
  done
fi

echo
echo "==== OOB v2 outputs ===="
echo "contract:            $OUT_ROOT/oob_contract.json"
echo "source-valid:         $OUT_ROOT/source_valid"
echo "invalid two-arm:      $OUT_ROOT/invalid_two_arm"
echo "invalid nine-arm:     $OUT_ROOT/invalid_nine_arm"
echo "ambiguous repeats:    $OUT_ROOT/ambiguous_two_arm_r*"
echo "historical outputs:   unchanged"
echo "Blender/rendering:    not invoked"

if [[ "$failures" -ne 0 ]]; then
  echo "OOB v2 replay finished all requested phases with $failures failed phase(s)." >&2
  exit 1
fi
echo "OOB v2 replay completed successfully."
