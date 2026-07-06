#!/usr/bin/env bash
set -Eeuo pipefail

# GPU health sweep for the MNET/K8s pod.
#
# Each GPU runs the same scene/mode inputs for REPEATS rounds. A dedicated
# OpenAI-compatible Qwen3-VL server is started per GPU/port and restarted after
# each round to release CUDA memory. The durable output is TSV validity status;
# per-case pipeline artifacts are removed by default to keep the run small.

REPO_ROOT="${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}"
BENCH_PYTHON="${BENCH_PYTHON:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}"
SERVER_PYTHON="${SERVER_PYTHON:-/mnt/group/xingyifei/conda_envs/qwen3vl/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/group/xingyifei/models/Qwen3VL-8B-Instruct}"
SERVER_SCRIPT="${SERVER_SCRIPT:-${REPO_ROOT}/scripts/serve_qwen3vl_transformers_openai.py}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/gpu_health_qwen3vl_8b_10scene_40repeat}"
LOG_ROOT="${LOG_ROOT:-/mnt/group/cmh/logs/gpu_health_qwen3vl_8b}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
BASE_PORT="${BASE_PORT:-8100}"
REPEATS="${REPEATS:-40}"
MAX_TOKENS="${MAX_TOKENS:-2096}"
MAX_OBJECTS="${MAX_OBJECTS:-20}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MODES="${MODES:-compact_objects}"
SCENE_IDS="${SCENE_IDS:-102343992 102344022 102344049 102344094 102344115 102344193 102344250 102344280 102344307 102344328}"

MODEL_NAME="${MODEL_NAME:-qwen3vl_sglang_32b}"
JUDGE_MODEL="${JUDGE_MODEL:-same}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
VALID_SOURCE="${VALID_SOURCE:-overall_valid}"
HSSD_ROOT="${HSSD_ROOT:-${REPO_ROOT}/data/external/hssd-hab}"
KEEP_ARTIFACTS="${KEEP_ARTIFACTS:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}"

SUMMARY="${OUT_ROOT}/summary.tsv"
if [[ ! -f "${SUMMARY}" ]]; then
  printf "timestamp\tgpu\tround\tscene_id\tmode\tstatus\n" > "${SUMMARY}"
fi

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*"
}

port_for_gpu() {
  local gpu="$1"
  printf "%s" "$((BASE_PORT + gpu))"
}

pid_file_for_gpu() {
  local gpu="$1"
  printf "%s/gpu_%s_server.pid" "${LOG_ROOT}" "${gpu}"
}

server_log_for_gpu() {
  local gpu="$1"
  printf "%s/gpu_%s_server.out" "${LOG_ROOT}" "${gpu}"
}

stop_server() {
  local gpu="$1"
  local pid_file
  pid_file="$(pid_file_for_gpu "${gpu}")"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(cat "${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
    rm -f "${pid_file}"
  fi
}

wait_for_server() {
  local gpu="$1"
  local port="$2"
  local pid_file="$3"
  local log_file="$4"
  local deadline=$((SECONDS + 240))
  while (( SECONDS < deadline )); do
    if [[ -f "${pid_file}" ]]; then
      local pid
      pid="$(cat "${pid_file}")"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        log "gpu ${gpu} server exited during startup; see ${log_file}"
        return 1
      fi
    fi
    if curl --noproxy "*" -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  log "gpu ${gpu} server did not become ready on port ${port}; see ${log_file}"
  return 1
}

start_server() {
  local gpu="$1"
  local port="$2"
  local pid_file
  local log_file
  pid_file="$(pid_file_for_gpu "${gpu}")"
  log_file="$(server_log_for_gpu "${gpu}")"

  stop_server "${gpu}"
  pkill -f "${SERVER_SCRIPT}.*--port ${port}" 2>/dev/null || true

  log "gpu ${gpu} starting server on port ${port}"
  PYTHONDONTWRITEBYTECODE=1 \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "${SERVER_PYTHON}" "${SERVER_SCRIPT}" \
    --model-path "${MODEL_PATH}" \
    --port "${port}" \
    --attn-implementation "${ATTN_IMPLEMENTATION}" \
    > "${log_file}" 2>&1 &
  echo "$!" > "${pid_file}"

  wait_for_server "${gpu}" "${port}" "${pid_file}" "${log_file}"
}

append_status() {
  local gpu="$1"
  local round="$2"
  local scene_id="$3"
  local mode="$4"
  local status="$5"
  local gpu_summary="${OUT_ROOT}/gpu_${gpu}/summary.tsv"
  mkdir -p "$(dirname "${gpu_summary}")"
  if [[ ! -f "${gpu_summary}" ]]; then
    printf "timestamp\tgpu\tround\tscene_id\tmode\tstatus\n" > "${gpu_summary}"
  fi
  local row
  row="$(printf "%s\t%s\t%s\t%s\t%s\t%s" "$(date -Is)" "${gpu}" "${round}" "${scene_id}" "${mode}" "${status}")"
  printf "%s\n" "${row}" >> "${gpu_summary}"
  flock "${SUMMARY}.lock" bash -c 'printf "%s\n" "$1" >> "$2"' _ "${row}" "${SUMMARY}"
}

run_one_case() {
  local gpu="$1"
  local round="$2"
  local scene_id="$3"
  local mode="$4"
  local port="$5"
  local run_out="${OUT_ROOT}/gpu_${gpu}/round_${round}/${mode}/${scene_id}"
  local cases_root="${OUT_ROOT}/gpu_${gpu}/cases"
  local run_log="${LOG_ROOT}/gpu_${gpu}_round_${round}_${mode}_${scene_id}.out"

  mkdir -p "$(dirname "${run_log}")"
  log "gpu ${gpu} round ${round}/${REPEATS} scene ${scene_id} mode ${mode}"

  local status="not_valid"
  if "${BENCH_PYTHON}" "${REPO_ROOT}/scripts/run_hssd_hab_10_qwen_validity.py" \
      --hssd-root "${HSSD_ROOT}" \
      --cases-root "${cases_root}" \
      --scene-ids "${scene_id}" \
      --modes "${mode}" \
      --model "${MODEL_NAME}" \
      --judge-model "${JUDGE_MODEL}" \
      --model-endpoint "http://127.0.0.1:${port}/v1" \
      --model-id "${MODEL_PATH}" \
      --max-tokens "${MAX_TOKENS}" \
      --timeout-seconds "${TIMEOUT_SECONDS}" \
      --max-objects "${MAX_OBJECTS}" \
      --valid-source "${VALID_SOURCE}" \
      --no-download \
      --out "${run_out}" \
      > "${run_log}" 2>&1; then
    status="$(awk -F '\t' 'NR == 2 {print $3}' "${run_out}/validity_results.tsv" 2>/dev/null || true)"
  else
    status="not_valid"
  fi
  if [[ "${status}" != "valid" && "${status}" != "not_valid" ]]; then
    status="not_valid"
  fi

  append_status "${gpu}" "${round}" "${scene_id}" "${mode}" "${status}"
  log "gpu ${gpu} round ${round} scene ${scene_id} mode ${mode}: ${status}"

  if [[ "${KEEP_ARTIFACTS}" != "1" ]]; then
    rm -rf "${run_out}"
  fi
}

run_gpu_worker() {
  local gpu="$1"
  local port
  port="$(port_for_gpu "${gpu}")"

  for round in $(seq 1 "${REPEATS}"); do
    start_server "${gpu}" "${port}" || {
      for scene_id in ${SCENE_IDS}; do
        for mode in ${MODES}; do
          append_status "${gpu}" "${round}" "${scene_id}" "${mode}" "not_valid"
        done
      done
      continue
    }

    for scene_id in ${SCENE_IDS}; do
      for mode in ${MODES}; do
        run_one_case "${gpu}" "${round}" "${scene_id}" "${mode}" "${port}"
      done
    done

    log "gpu ${gpu} finished round ${round}; restarting server to free memory"
    stop_server "${gpu}"
    sleep 3
  done
  stop_server "${gpu}"
}

cleanup() {
  for gpu in ${GPUS}; do
    stop_server "${gpu}" || true
  done
}
trap cleanup EXIT INT TERM

log "starting GPU health sweep"
log "gpus=${GPUS} repeats=${REPEATS} max_tokens=${MAX_TOKENS} max_objects=${MAX_OBJECTS} attn=${ATTN_IMPLEMENTATION} modes=${MODES}"
for gpu in ${GPUS}; do
  run_gpu_worker "${gpu}" &
done
wait
log "complete: ${SUMMARY}"
