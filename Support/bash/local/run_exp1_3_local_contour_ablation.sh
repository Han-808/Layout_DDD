#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${BENCH_PY:-$REPO_ROOT/.venv/bin/python}"
BLENDER="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
SOURCE_EVIDENCE="${SOURCE_EVIDENCE:-$REPO_ROOT/Support/artifacts/outputs/exp1_1}"
CONTOUR_EVIDENCE="${CONTOUR_EVIDENCE:-$REPO_ROOT/Support/artifacts/outputs/exp1_3_local_contour_evidence}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/Support/datasets/cal_dataset1}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp1_3_local_contour_gpt56}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_fine_edge_judge.json}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:4010/v1}"
SMOKE_IMAGE="${SMOKE_IMAGE:-$SOURCE_EVIDENCE/cases/subtle_support_002/metric_local_highlight/support__obj_000__144ae3a484/final_bundle/rgb_00_support_00_contact_060.png}"

PHASE="${PHASE:-all}"
REPEATS="${REPEATS:-2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RESUME="${RESUME:-1}"
CONTINUOUS="${CONTINUOUS:-1}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-300}"
ACTIVE_CHILD_PID=""

terminate_process_tree() {
  local parent_pid=$1
  local child_pid
  while IFS= read -r child_pid; do
    if [[ "$child_pid" =~ ^[0-9]+$ ]]; then
      terminate_process_tree "$child_pid"
    fi
  done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
  kill -TERM "$parent_pid" 2>/dev/null || true
}

terminate_chain() {
  if [[ "$ACTIVE_CHILD_PID" =~ ^[0-9]+$ ]] &&
     kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
    terminate_process_tree "$ACTIVE_CHILD_PID"
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
  fi
  echo "$(date '+%F %T') exp1_3 terminated"
  exit 143
}
trap terminate_chain INT TERM HUP

for path in "$PYTHON" "$SOURCE_EVIDENCE" "$DATASET_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "Required local input does not exist: $path" >&2
    exit 2
  fi
done
if [[ "$PHASE" != "all" && "$PHASE" != "evidence" && "$PHASE" != "judge" ]]; then
  echo "PHASE must be all, evidence, or judge." >&2
  exit 2
fi
if ! [[ "$REPEATS" =~ ^[2-9][0-9]*$ ]]; then
  echo "REPEATS must be at least 2." >&2
  exit 2
fi
if ! [[ "$MAX_WORKERS" =~ ^[1-8]$ ]]; then
  echo "MAX_WORKERS must be between 1 and 8." >&2
  exit 2
fi
if ! [[ "$RETRY_DELAY_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RETRY_DELAY_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ "$PHASE" != "judge" && ! -x "$BLENDER" ]]; then
  echo "Blender is not executable: $BLENDER" >&2
  exit 2
fi
if [[ "$PHASE" != "evidence" && -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "LITELLM_MASTER_KEY is not set in this terminal." >&2
  exit 2
fi

mkdir -p "$CONTOUR_EVIDENCE" "$OUT_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

resume_flag="--resume"
if [[ "$RESUME" == "0" ]]; then
  resume_flag="--no-resume"
fi

manifest_complete() {
  local manifest=$1
  local expected=$2
  "$PYTHON" - "$manifest" "$expected" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text())
complete = value.get("complete") is True
count = value.get("result_count")
if count is None:
    count = value.get("completed_event_count")
failures = int(value.get("failure_count", value.get("failed_event_count", 0)) or 0)
raise SystemExit(0 if complete and int(count or 0) == expected and failures == 0 else 1)
PY
}

run_and_retry() {
  local label=$1
  local manifest=$2
  local expected=$3
  shift 3
  while ! manifest_complete "$manifest" "$expected"; do
    echo "$(date '+%F %T') starting/resuming $label"
    "$@" &
    ACTIVE_CHILD_PID=$!
    wait "$ACTIVE_CHILD_PID"
    local status=$?
    ACTIVE_CHILD_PID=""
    if manifest_complete "$manifest" "$expected"; then
      break
    fi
    echo "$(date '+%F %T') $label incomplete (status=$status)"
    if [[ "$CONTINUOUS" != "1" ]]; then
      return "$status"
    fi
    echo "Retrying in ${RETRY_DELAY_SECONDS}s; completed calls remain cached."
    sleep "$RETRY_DELAY_SECONDS"
  done
  echo "$(date '+%F %T') $label complete"
}

prepare_evidence() {
  "$PYTHON" scripts/prepare_cal_dataset1_contour_highlight.py \
    --evidence-root "$SOURCE_EVIDENCE" \
    --out-dir "$CONTOUR_EVIDENCE" \
    --blender-bin "$BLENDER" \
    --require-event-count 24 \
    "$resume_flag" \
    --continue-on-error
}

endpoint_preflight() {
  "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint "$JUDGE_ENDPOINT" \
    --model gpt-5.6-sol \
    --api-key-env LITELLM_MASTER_KEY \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal \
    --image-path "$SMOKE_IMAGE" \
    >"$OUT_ROOT/preflight.json"
}

judge_repeat() {
  local repeat=$1
  local repeat_out="$OUT_ROOT/repeat_${repeat}"
  mkdir -p "$repeat_out"
  "$PYTHON" scripts/judge_cal_dataset1_visual_config.py \
    --evidence-root "$SOURCE_EVIDENCE" \
    --contour-evidence-root "$CONTOUR_EVIDENCE" \
    --dataset-root "$DATASET_ROOT" \
    --judge-config "$JUDGE_CONFIG" \
    --out-dir "$repeat_out" \
    --arm presence_local_raw \
    --arm presence_local_raw_highlight \
    --arm presence_local_raw_contour \
    --max-workers "$MAX_WORKERS" \
    "$resume_flag" \
    --continue-on-error
}

echo "==== exp1_3 local highlighting strategy ablation ===="
echo "Events: 24 constructed-invalid (8 Collision / 8 OOB / 8 Support)"
echo "Arms: local_raw, local_legacy, local_contour"
echo "Global evidence: excluded"
echo "Repeats: $REPEATS"
echo "Judge calls: $((24 * 3 * REPEATS))"
echo "Output: $OUT_ROOT"

if [[ "$PHASE" == "all" || "$PHASE" == "evidence" ]]; then
  run_and_retry \
    "occlusion-aware contour evidence" \
    "$CONTOUR_EVIDENCE/run_manifest.json" \
    24 \
    prepare_evidence
fi

if [[ "$PHASE" == "all" || "$PHASE" == "judge" ]]; then
  if [[ ! -f "$CONTOUR_EVIDENCE/run_manifest.json" ]]; then
    echo "Contour evidence is missing; run PHASE=evidence first." >&2
    exit 2
  fi
  while ! endpoint_preflight; do
    echo "$(date '+%F %T') GPT-5.6-Sol preflight failed."
    if [[ "$CONTINUOUS" != "1" ]]; then
      exit 1
    fi
    echo "Retrying endpoint in ${RETRY_DELAY_SECONDS}s."
    sleep "$RETRY_DELAY_SECONDS"
  done
  echo "GPT-5.6-Sol multimodal preflight: passed"

  for repeat in $(seq 1 "$REPEATS"); do
    run_and_retry \
      "judge repeat $repeat" \
      "$OUT_ROOT/repeat_${repeat}/run_manifest.json" \
      72 \
      judge_repeat "$repeat"
  done

  "$PYTHON" scripts/compare_cal_dataset1_judge_repeats.py \
    --left-run "$OUT_ROOT/repeat_1" \
    --right-run "$OUT_ROOT/repeat_2" \
    --out-dir "$OUT_ROOT/repeatability"
fi

echo
echo "==== exp1_3 complete ===="
echo "Contour evidence: $CONTOUR_EVIDENCE"
echo "Repeat 1:        $OUT_ROOT/repeat_1"
echo "Repeat 2:        $OUT_ROOT/repeat_2"
echo "Repeatability:   $OUT_ROOT/repeatability"
