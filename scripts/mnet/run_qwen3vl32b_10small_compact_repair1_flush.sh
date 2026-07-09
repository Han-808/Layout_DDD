#!/usr/bin/env bash
set -Eeuo pipefail

# Run this inside the MNET pod, not on the local Mac.
# It launches one Qwen3-VL-32B SGLang server over 8 H20 GPUs, then runs
# the 10 selected HSSD-HAB scenes sequentially with cache flush after each scene.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}
MODEL_PATH=${MODEL_PATH:-/mnt/group/cmh/models/Qwen3-VL-32B-Instruct}
SGLANG_PY=${SGLANG_PY:-/mnt/group/cmh/envs/sglang-qwen3vl/bin/python}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}

PORT=${PORT:-8298}
BASE_URL="http://127.0.0.1:${PORT}"
ENDPOINT="${BASE_URL}/v1"
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-96K}

TP_SIZE=${TP_SIZE:-8}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3,4,5,6,7}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-98304}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}

GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-64000}
REPAIR_MAX_TOKENS=${REPAIR_MAX_TOKENS:-24000}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-2048}
PROMPT_SAFETY_MARGIN_TOKENS=${PROMPT_SAFETY_MARGIN_TOKENS:-4096}
MAX_REPAIR_ITERATIONS=${MAX_REPAIR_ITERATIONS:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}

# Empty override is allowed: MAX_OBJECTS= bash ...
# Default 80 keeps 32B/96K reasonably conservative for this first unattended pass.
MAX_OBJECTS=${MAX_OBJECTS-80}

RUN_TAG=${RUN_TAG:-mnet_32b_tp8_96k_10small_compact_repair1_flush_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/${RUN_TAG}}
CASES_ROOT=${CASES_ROOT:-${REPO_ROOT}/data/external/hssd-hab-converted/${RUN_TAG}}
HSSD_ROOT=${HSSD_ROOT:-${REPO_ROOT}/data/external/hssd-hab}

DRIVER_LOG="${LOG_ROOT}/${RUN_TAG}.driver.out"
SERVER_LOG="${LOG_ROOT}/${RUN_TAG}.server.out"
SERVER_PID_FILE="${LOG_ROOT}/${RUN_TAG}.server.pid"
DRIVER_PID_FILE="${LOG_ROOT}/${RUN_TAG}.driver.pid"
SUMMARY_FILE="${OUT_DIR}/validity_results.tsv"
TAR_PATH="/mnt/group/cmh/${RUN_TAG}.tar.gz"

SCENES=(
  102343992
  102344022
  102344049
  102344094
  102344115
  102344193
  102344250
  102344280
  102344307
  102344328
)

mkdir -p "$LOG_ROOT" "$OUT_DIR" "$CASES_ROOT"
echo $$ > "$DRIVER_PID_FILE"
exec > >(tee -a "$DRIVER_LOG") 2>&1

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
}

print_gpu_health() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
}

stop_existing_sglang() {
  local pids
  pids="$(pgrep -f 'sglang.launch_server' || true)"
  if [[ -n "$pids" ]]; then
    log "stopping existing sglang server(s): ${pids//$'\n'/ }"
    pkill -f 'sglang.launch_server' || true
    sleep 8
  fi
}

server_alive() {
  curl --noproxy "*" -fsS "${ENDPOINT}/models" >/dev/null 2>&1
}

wait_for_server() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if server_alive; then
      log "server ready"
      curl --noproxy "*" -sS "${ENDPOINT}/models" || true
      echo
      return 0
    fi

    if [[ -f "$SERVER_PID_FILE" ]]; then
      local pid
      pid="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
      if [[ -n "$pid" ]] && ! ps -p "$pid" >/dev/null 2>&1; then
        log "server process exited before ready"
        tail -n 160 "$SERVER_LOG" || true
        return 1
      fi
    fi

    tail -n 8 "$SERVER_LOG" 2>/dev/null || true
    sleep 15
  done

  log "server readiness timeout"
  tail -n 200 "$SERVER_LOG" || true
  return 1
}

launch_server() {
  log "launching 32B server"
  stop_existing_sglang

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$TP_SIZE" \
    --disable-cuda-graph \
    > "$SERVER_LOG" 2>&1 &

  echo $! > "$SERVER_PID_FILE"
  log "server pid=$(cat "$SERVER_PID_FILE") log=$SERVER_LOG"
  wait_for_server
}

ensure_server() {
  if server_alive; then
    return 0
  fi
  log "server not healthy; relaunching"
  launch_server
}

flush_cache() {
  log "flush cache"
  curl --noproxy "*" -sS -X POST "${BASE_URL}/flush_cache" || true
  echo
}

run_scene() {
  local scene_id="$1"
  local scene_out="${OUT_DIR}/compact_objects/${scene_id}"

  log "scene=${scene_id} start"
  ensure_server

  local args=(
    "$REPO_ROOT/scripts/legend/legend_run_hssd_hab_10_qwen_validity.py"
    --hssd-root "$HSSD_ROOT"
    --cases-root "$CASES_ROOT"
    --scene-ids "$scene_id"
    --modes compact_objects
    --model qwen3vl_sglang_32b
    --judge-model same
    --model-endpoint "$ENDPOINT"
    --model-id "$SERVED_MODEL"
    --max-tokens "$GENERATION_MAX_TOKENS"
    --generation-max-tokens "$GENERATION_MAX_TOKENS"
    --repair-max-tokens "$REPAIR_MAX_TOKENS"
    --judge-max-tokens "$JUDGE_MAX_TOKENS"
    --context-length "$CONTEXT_LENGTH"
    --prompt-safety-margin-tokens "$PROMPT_SAFETY_MARGIN_TOKENS"
    --max-repair-iterations "$MAX_REPAIR_ITERATIONS"
    --timeout-seconds "$TIMEOUT_SECONDS"
    --valid-source overall_valid
    --no-download
    --out "$OUT_DIR"
  )

  if [[ -n "$MAX_OBJECTS" ]]; then
    args+=(--max-objects "$MAX_OBJECTS")
  fi

  "$BENCH_PY" "${args[@]}" || true

  if [[ -f "${scene_out}/task_error.txt" ]]; then
    log "scene=${scene_id} task_error"
    cat "${scene_out}/task_error.txt" || true
  fi

  flush_cache
  log "health"
  print_gpu_health

  if ! server_alive; then
    log "server unhealthy after scene=${scene_id}; relaunching before next scene"
    launch_server
  fi

  log "scene=${scene_id} done"
}

summarize_run() {
  log "summary"
  echo "RUN_TAG=$RUN_TAG"
  echo "OUT_DIR=$OUT_DIR"
  echo "DRIVER_LOG=$DRIVER_LOG"
  echo "SERVER_LOG=$SERVER_LOG"
  echo "SUMMARY_FILE=$SUMMARY_FILE"
  echo "TAR_PATH=$TAR_PATH"

  if [[ -f "$SUMMARY_FILE" ]]; then
    cat "$SUMMARY_FILE"
  else
    echo "summary file missing: $SUMMARY_FILE"
  fi

  echo "case_metrics count:"
  find "$OUT_DIR" -name case_metrics.json -print | wc -l || true
  echo "task_error count:"
  find "$OUT_DIR" -name task_error.txt -print | wc -l || true
}

package_outputs() {
  log "packaging outputs"
  tar -czf "$TAR_PATH" -C "${REPO_ROOT}/outputs" "$RUN_TAG" || true
  ls -lah "$TAR_PATH" 2>/dev/null || true
}

main() {
  cd "$REPO_ROOT"

  require_file "$SGLANG_PY"
  require_file "$BENCH_PY"
  require_file "$MODEL_PATH/config.json"
  require_file "$HSSD_ROOT"

  log "config"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SERVED_MODEL=$SERVED_MODEL"
  echo "ENDPOINT=$ENDPOINT"
  echo "CONTEXT_LENGTH=$CONTEXT_LENGTH"
  echo "MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC"
  echo "GENERATION_MAX_TOKENS=$GENERATION_MAX_TOKENS"
  echo "REPAIR_MAX_TOKENS=$REPAIR_MAX_TOKENS"
  echo "JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
  echo "MAX_OBJECTS=${MAX_OBJECTS:-<none>}"
  echo "MAX_REPAIR_ITERATIONS=$MAX_REPAIR_ITERATIONS"
  echo "SCENES=${SCENES[*]}"
  "$SGLANG_PY" - <<'PY'
import sys
import sglang
import zmq
print("sglang python:", sys.executable)
print("sglang + zmq ok")
PY

  launch_server
  print_gpu_health

  for scene in "${SCENES[@]}"; do
    run_scene "$scene"
  done

  summarize_run
  package_outputs
  log "done"
}

main "$@"
