#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
EXP1_1_OUT_DIR="${EXP1_1_OUT_DIR:-$REPO_ROOT/Support/artifacts/outputs/exp1_1_gpt56_judge}"
VISUAL_OUT_DIR="${VISUAL_OUT_DIR:-$REPO_ROOT/Support/artifacts/outputs/exp1_1_visual_config_gpt56}"
MAX_WORKERS="${MAX_WORKERS:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-300}"
ACTIVE_CHILD_PID=""

terminate_chain() {
  if [[ "$ACTIVE_CHILD_PID" =~ ^[0-9]+$ ]] && kill -0 "$ACTIVE_CHILD_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_CHILD_PID" 2>/dev/null || true
    wait "$ACTIVE_CHILD_PID" 2>/dev/null || true
  fi
  echo "$(date '+%F %T') continuous chain terminated"
  exit 143
}

trap terminate_chain INT TERM HUP

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "Required environment variable is not set: LITELLM_MASTER_KEY" >&2
  exit 2
fi
if ! [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ && "$RETRY_DELAY_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "POLL_SECONDS and RETRY_DELAY_SECONDS must be positive integers" >&2
  exit 2
fi

mkdir -p "$EXP1_1_OUT_DIR" "$VISUAL_OUT_DIR"

manifest_complete() {
  local path=$1 expected=$2
  "$PYTHON_BIN" - "$path" "$expected" <<'PY' >/dev/null 2>&1
import json, sys
from pathlib import Path
path, expected = Path(sys.argv[1]), int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
value = json.loads(path.read_text())
complete = value.get("complete")
if complete is None:
    complete = (
        int(value.get("result_count") or 0) == expected
        and int(value.get("resolved_count") or 0) == expected
        and int(value.get("failure_count") or 0) == 0
    )
raise SystemExit(0 if complete else 1)
PY
}

wait_for_existing_exp1_1() {
  local pid_file="$EXP1_1_OUT_DIR/run.pid" pid command
  [[ -f "$pid_file" ]] || return 0
  pid=$(cat "$pid_file" 2>/dev/null || true)
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  while kill -0 "$pid" 2>/dev/null; do
    command=$(ps -p "$pid" -o command= 2>/dev/null || true)
    if [[ "$command" != *"run_exp1_1_gpt56_judge"* && "$command" != *"judge_cal_dataset1_camera_evidence"* ]]; then
      return 0
    fi
    echo "$(date '+%F %T') exp1_1 is already running (pid=$pid); waiting"
    sleep "$POLL_SECONDS"
  done
}

echo "$(date '+%F %T') continuous chain started"
echo "Stage 1: exp1_1 / 48 calls"
echo "Stage 2: exp1_1 VisualConfig / 216 calls"
echo "Retry delay: ${RETRY_DELAY_SECONDS}s; successful results are always cached"

wait_for_existing_exp1_1
until manifest_complete "$EXP1_1_OUT_DIR/run_manifest.json" 48; do
  echo "$(date '+%F %T') running/resuming exp1_1"
  OUT_DIR="$EXP1_1_OUT_DIR" MAX_WORKERS="$MAX_WORKERS" RESUME=1 \
    "$SCRIPT_DIR/run_exp1_1_gpt56_judge.sh" \
    >>"$EXP1_1_OUT_DIR/continuous.log" 2>&1 &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  status=$?
  ACTIVE_CHILD_PID=""
  if manifest_complete "$EXP1_1_OUT_DIR/run_manifest.json" 48; then
    break
  fi
  echo "$(date '+%F %T') exp1_1 incomplete (status=$status); retrying in ${RETRY_DELAY_SECONDS}s"
  sleep "$RETRY_DELAY_SECONDS"
done

echo "$(date '+%F %T') exp1_1 complete; entering VisualConfig replay"
until manifest_complete "$VISUAL_OUT_DIR/run_manifest.json" 216; do
  echo "$(date '+%F %T') running/resuming VisualConfig replay"
  OUT_DIR="$VISUAL_OUT_DIR" MAX_WORKERS="$MAX_WORKERS" RESUME=1 \
    "$SCRIPT_DIR/run_exp1_1_visual_config_gpt56.sh" \
    >>"$VISUAL_OUT_DIR/continuous.log" 2>&1 &
  ACTIVE_CHILD_PID=$!
  wait "$ACTIVE_CHILD_PID"
  status=$?
  ACTIVE_CHILD_PID=""
  if manifest_complete "$VISUAL_OUT_DIR/run_manifest.json" 216; then
    break
  fi
  echo "$(date '+%F %T') VisualConfig replay incomplete (status=$status); retrying in ${RETRY_DELAY_SECONDS}s"
  sleep "$RETRY_DELAY_SECONDS"
done

echo "$(date '+%F %T') continuous chain complete"
echo "Results: $VISUAL_OUT_DIR"
