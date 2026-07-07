#!/usr/bin/env bash
set -Eeuo pipefail

# MNET H20 model sweep:
#   generator in {Qwen3-VL-8B, Qwen3-VL-32B}
#   judge     in {Qwen3-VL-8B, Qwen3-VL-32B}
#   then, if available, append generator=GLM-4.1V-9B and judge=Qwen3-VL-32B
#   instance  in cached HSSD-HAB scene-variants with total_objects < MAX_INSTANCE_OBJECTS
#   mode      in the four benchmark input modes
#
REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}
MODELS_ROOT=${MODELS_ROOT:-/mnt/group/cmh/models}
HSSD_ROOT=${HSSD_ROOT:-${REPO_ROOT}/data/external/hssd-hab}
SGLANG_PY=${SGLANG_PY:-/mnt/group/cmh/envs/sglang-qwen3vl/bin/python}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}

RUN_TAG=${RUN_TAG:-mnet_qwen8_32_matrix_le80_4mode_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/${RUN_TAG}}
CASES_ROOT=${CASES_ROOT:-${REPO_ROOT}/data/external/hssd-hab-converted/${RUN_TAG}}

QWEN8_MODEL_PATH=${QWEN8_MODEL_PATH:-${MODELS_ROOT}/Qwen3VL-8B-Instruct}
QWEN8_SERVED_MODEL=${QWEN8_SERVED_MODEL:-Qwen3VL-8B-Instruct-64K}
QWEN8_PORT=${QWEN8_PORT:-8390}
QWEN8_CUDA_DEVICES=${QWEN8_CUDA_DEVICES:-4}
QWEN8_TP_SIZE=${QWEN8_TP_SIZE:-1}
QWEN8_CONTEXT_LENGTH=${QWEN8_CONTEXT_LENGTH:-65536}
QWEN8_MEM_FRACTION_STATIC=${QWEN8_MEM_FRACTION_STATIC:-0.78}

QWEN32_MODEL_PATH=${QWEN32_MODEL_PATH:-${MODELS_ROOT}/Qwen3-VL-32B-Instruct}
QWEN32_SERVED_MODEL=${QWEN32_SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
QWEN32_PORT=${QWEN32_PORT:-8298}
QWEN32_CUDA_DEVICES=${QWEN32_CUDA_DEVICES:-0,1,2,3}
QWEN32_TP_SIZE=${QWEN32_TP_SIZE:-4}
QWEN32_CONTEXT_LENGTH=${QWEN32_CONTEXT_LENGTH:-65536}
QWEN32_MEM_FRACTION_STATIC=${QWEN32_MEM_FRACTION_STATIC:-0.68}

GLM_MODEL_PATH=${GLM_MODEL_PATH:-${MODELS_ROOT}/GLM-4.1V-9B-Thinking}
GLM_SERVED_MODEL=${GLM_SERVED_MODEL:-GLM-4.1V-9B-Thinking-64K}
GLM_PORT=${GLM_PORT:-8391}
GLM_CUDA_DEVICES=${GLM_CUDA_DEVICES:-4}
GLM_TP_SIZE=${GLM_TP_SIZE:-1}
GLM_CONTEXT_LENGTH=${GLM_CONTEXT_LENGTH:-65536}
GLM_MEM_FRACTION_STATIC=${GLM_MEM_FRACTION_STATIC:-0.78}

GENERATOR_KEYS=${GENERATOR_KEYS:-"qwen8b qwen32b"}
JUDGE_KEYS=${JUDGE_KEYS:-"qwen8b qwen32b"}
ENABLE_GLM_GENERATOR=${ENABLE_GLM_GENERATOR:-1}
MODES=${MODES:-"prompt_only compact_objects compact_objects_with_estimated_relations full_metadata_budgeted"}

GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-24000}
FULL_METADATA_GENERATION_MAX_TOKENS=${FULL_METADATA_GENERATION_MAX_TOKENS:-20000}
REPAIR_MAX_TOKENS=${REPAIR_MAX_TOKENS:-12000}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-2048}
PROMPT_SAFETY_MARGIN_TOKENS=${PROMPT_SAFETY_MARGIN_TOKENS:-4096}
MAX_REPAIR_ITERATIONS=${MAX_REPAIR_ITERATIONS:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}
TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS:-7200}

# Strictly less than this value. For <=80 objects, use 81.
MAX_INSTANCE_OBJECTS=${MAX_INSTANCE_OBJECTS:-81}
LIMIT_INSTANCES=${LIMIT_INSTANCES:-0}
VARIANT_FILTER=${VARIANT_FILTER:-}

# Set to 1 when switching from a TP8 all-GPU server to the split Qwen8/Qwen32 topology.
STOP_EXISTING_SGLANG=${STOP_EXISTING_SGLANG:-0}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-5400}

DRIVER_LOG="${LOG_ROOT}/${RUN_TAG}.driver.out"
DRIVER_PID_FILE="${LOG_ROOT}/${RUN_TAG}.driver.pid"
HEARTBEAT_FILE="${LOG_ROOT}/${RUN_TAG}.heartbeat"
SELECTED_FILE="${OUT_DIR}/selected_instances.tsv"
PAIRS_FILE="${OUT_DIR}/model_pairs.tsv"
SKIPPED_FILE="${OUT_DIR}/skipped_tasks.tsv"
SUMMARY_FILE="${OUT_DIR}/validity_results.tsv"
TAR_PATH="/mnt/group/cmh/${RUN_TAG}.tar.gz"

mkdir -p "$LOG_ROOT" "$OUT_DIR" "$CASES_ROOT"
echo $$ > "$DRIVER_PID_FILE"
exec > >(tee -a "$DRIVER_LOG") 2>&1

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
}

model_path() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_MODEL_PATH" ;;
    qwen32b) printf "%s" "$QWEN32_MODEL_PATH" ;;
    glm9b) printf "%s" "$GLM_MODEL_PATH" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_served() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_SERVED_MODEL" ;;
    qwen32b) printf "%s" "$QWEN32_SERVED_MODEL" ;;
    glm9b) printf "%s" "$GLM_SERVED_MODEL" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_port() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_PORT" ;;
    qwen32b) printf "%s" "$QWEN32_PORT" ;;
    glm9b) printf "%s" "$GLM_PORT" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_cuda_devices() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_CUDA_DEVICES" ;;
    qwen32b) printf "%s" "$QWEN32_CUDA_DEVICES" ;;
    glm9b) printf "%s" "$GLM_CUDA_DEVICES" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_tp_size() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_TP_SIZE" ;;
    qwen32b) printf "%s" "$QWEN32_TP_SIZE" ;;
    glm9b) printf "%s" "$GLM_TP_SIZE" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_context_length() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_CONTEXT_LENGTH" ;;
    qwen32b) printf "%s" "$QWEN32_CONTEXT_LENGTH" ;;
    glm9b) printf "%s" "$GLM_CONTEXT_LENGTH" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_mem_fraction() {
  case "$1" in
    qwen8b) printf "%s" "$QWEN8_MEM_FRACTION_STATIC" ;;
    qwen32b) printf "%s" "$QWEN32_MEM_FRACTION_STATIC" ;;
    glm9b) printf "%s" "$GLM_MEM_FRACTION_STATIC" ;;
    *) echo "unknown model key: $1" >&2; return 1 ;;
  esac
}

model_endpoint() {
  printf "http://127.0.0.1:%s/v1" "$(model_port "$1")"
}

model_log_file() {
  printf "%s/%s.%s_server.out" "$LOG_ROOT" "$RUN_TAG" "$1"
}

model_pid_file() {
  printf "%s/%s.%s_server.pid" "$LOG_ROOT" "$RUN_TAG" "$1"
}

uses_single_generator_gpu() {
  [[ "$1" == "qwen8b" || "$1" == "glm9b" ]]
}

stop_single_gpu_conflicts() {
  local key="$1"
  local other
  if ! uses_single_generator_gpu "$key"; then
    return 0
  fi
  for other in qwen8b glm9b; do
    if [[ "$other" != "$key" ]]; then
      stop_model_server "$other"
    fi
  done
}

model_available() {
  local key="$1"
  local path
  path="$(model_path "$key")"
  [[ -f "${path}/config.json" ]] || return 1
  if find "$path" -name "*.incomplete" -print -quit 2>/dev/null | grep -q .; then
    return 1
  fi
  if [[ -f "${path}/model.safetensors.index.json" ]]; then
    find "$path" -maxdepth 1 -name "*.safetensors" -print -quit 2>/dev/null | grep -q .
    return
  fi
  return 0
}

server_alive() {
  local port="$1"
  curl --noproxy "*" -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1
}

server_serves_model() {
  local key="$1"
  local port served
  port="$(model_port "$key")"
  served="$(model_served "$key")"
  curl --noproxy "*" -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -F "\"id\":\"${served}\"" >/dev/null 2>&1
}

model_pid_alive() {
  local key="$1"
  local pid_file pid
  pid_file="$(model_pid_file "$key")"
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1
}

tail_model_log() {
  local key="$1"
  local log_file
  log_file="$(model_log_file "$key")"
  if [[ -f "$log_file" ]]; then
    echo "---- ${key} server log tail ----"
    tail -n 80 "$log_file" || true
    echo "---- end ${key} server log tail ----"
  fi
}

wait_for_server() {
  local key="$1"
  local port
  local waited=0
  port="$(model_port "$key")"
  while (( waited < SERVER_READY_TIMEOUT_SECONDS )); do
    if server_serves_model "$key"; then
      log "${key} server ready"
      curl --noproxy "*" -sS "http://127.0.0.1:${port}/v1/models" || true
      echo
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
    if ! model_pid_alive "$key" && ! server_serves_model "$key"; then
      log "${key} server process exited before readiness"
      tail_model_log "$key"
      return 1
    fi
    if (( waited % 300 == 0 )); then
      log "waiting for ${key} server ${waited}/${SERVER_READY_TIMEOUT_SECONDS}s"
      tail_model_log "$key"
    fi
  done
  echo "${key} server did not become ready on port ${port}" >&2
  tail_model_log "$key"
  return 1
}

stop_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
      log "stopping ${label} pid=${pid}"
      kill "$pid" || true
      sleep 8
      if ps -p "$pid" >/dev/null 2>&1; then
        kill -9 "$pid" || true
      fi
    fi
  fi
}

stop_model_server() {
  local key="$1"
  local pid_file port
  pid_file="$(model_pid_file "$key")"
  port="$(model_port "$key")"
  stop_pid_file "$pid_file" "$key"
  if server_alive "$port"; then
    log "stopping unmanaged/stale ${key} server on port ${port}"
    pkill -f "sglang.launch_server.*--port ${port}" || true
    sleep 8
  fi
}

stop_existing_sglang_if_requested() {
  if [[ "$STOP_EXISTING_SGLANG" == "1" ]]; then
    log "STOP_EXISTING_SGLANG=1; stopping existing sglang.launch_server processes"
    pkill -f "sglang.launch_server" || true
    sleep 10
  fi
}

launch_model_server() {
  local key="$1"
  local path served port cuda tp ctx mem log_file pid_file
  path="$(model_path "$key")"
  served="$(model_served "$key")"
  port="$(model_port "$key")"
  cuda="$(model_cuda_devices "$key")"
  tp="$(model_tp_size "$key")"
  ctx="$(model_context_length "$key")"
  mem="$(model_mem_fraction "$key")"
  log_file="$(model_log_file "$key")"
  pid_file="$(model_pid_file "$key")"

  if server_serves_model "$key"; then
    log "${key} server already alive on port ${port}"
    return 0
  elif server_alive "$port"; then
    log "${key} port ${port} is alive but served model differs; restarting it"
    stop_model_server "$key"
  fi
  require_path "${path}/config.json"
  stop_single_gpu_conflicts "$key"
  stop_model_server "$key"
  log "launch ${key} server path=${path} served=${served} port=${port} cuda=${cuda} tp=${tp} ctx=${ctx}"
  CUDA_VISIBLE_DEVICES="$cuda" \
  PYTHONUNBUFFERED=1 \
  nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$path" \
    --served-model-name "$served" \
    --host 0.0.0.0 \
    --port "$port" \
    --context-length "$ctx" \
    --mem-fraction-static "$mem" \
    --tp-size "$tp" \
    --disable-cuda-graph \
    > "$log_file" 2>&1 &
  echo $! > "$pid_file"
  log "${key} pid=$(cat "$pid_file") log=${log_file}"
  wait_for_server "$key"
}

ensure_model_server() {
  local key="$1"
  local port
  port="$(model_port "$key")"
  if server_serves_model "$key"; then
    return 0
  fi
  log "${key} server down or serving a different model; relaunching and waiting"
  launch_model_server "$key"
}

flush_model() {
  local key="$1"
  local port
  port="$(model_port "$key")"
  if server_alive "$port"; then
    curl --noproxy "*" -sS -X POST "http://127.0.0.1:${port}/flush_cache" || true
    echo
  fi
}

flush_memory() {
  local gen_key="$1"
  local judge_key="$2"
  log "flush cache generator=${gen_key} judge=${judge_key}"
  flush_model "$gen_key"
  if [[ "$judge_key" != "$gen_key" ]]; then
    flush_model "$judge_key"
  fi
}

print_gpu_health() {
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv || true
}

heartbeat() {
  {
    echo "time=$(date '+%F %T')"
    echo "run_tag=$RUN_TAG"
    echo "driver_pid=$$"
    echo "selected_file=$SELECTED_FILE"
    echo "pairs_file=$PAIRS_FILE"
    echo "summary_file=$SUMMARY_FILE"
  } > "$HEARTBEAT_FILE"
}

write_candidate_file() {
  log "selecting HSSD-HAB instances with total_objects < ${MAX_INSTANCE_OBJECTS}"
  REPO_ROOT="$REPO_ROOT" \
  HSSD_ROOT="$HSSD_ROOT" \
  SELECTED_FILE="$SELECTED_FILE" \
  MAX_INSTANCE_OBJECTS="$MAX_INSTANCE_OBJECTS" \
  LIMIT_INSTANCES="$LIMIT_INSTANCES" \
  VARIANT_FILTER="$VARIANT_FILTER" \
  "$BENCH_PY" - <<'PY'
import csv
import json
import os
from pathlib import Path

hssd_root = Path(os.environ["HSSD_ROOT"])
out_path = Path(os.environ["SELECTED_FILE"])
max_objects = int(os.environ["MAX_INSTANCE_OBJECTS"])
limit = int(os.environ.get("LIMIT_INSTANCES", "0") or "0")
variant_filter = {item.strip() for item in os.environ.get("VARIANT_FILTER", "").split(",") if item.strip()}

def as_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None

def safe_key(variant, scene_id):
    def clean(text):
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))
    return f"{clean(variant)}__{clean(scene_id)}"

def row_from_dict(row):
    variant = row.get("variant") or row.get("folder") or row.get("split") or ""
    scene_id = row.get("scene_id") or row.get("id") or ""
    total = as_int(row.get("total_objects") or row.get("object_count") or row.get("objects"))
    rel_path = row.get("path") or row.get("relative_path") or row.get("hf_path") or ""
    if not rel_path and variant and scene_id:
        rel_path = f"{variant}/{scene_id}.scene_instance.json"
    if not variant and rel_path:
        variant = Path(rel_path).parent.name
    if not scene_id and rel_path:
        scene_id = Path(rel_path).name.replace(".scene_instance.json", "")
    if not variant or not scene_id or total is None or not rel_path:
        return None
    return {"instance_key": safe_key(variant, scene_id), "variant": variant, "scene_id": scene_id, "total_objects": total, "path": rel_path}

def read_manifest():
    for name in ["hssd_hab_le80_scene_instances.tsv", "hssd_hab_le100_scene_instances.tsv"]:
        manifest = hssd_root / "manifests" / name
        if manifest.exists():
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                rows = [item for row in reader if (item := row_from_dict(row))]
            if rows:
                return rows
    return []

def count_objects(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("object_instances", "objects"):
        objects = data.get(key)
        if isinstance(objects, list):
            return len(objects)
    return None

rows = read_manifest()
if not rows:
    rows = []
    for path in sorted(hssd_root.glob("*/*.scene_instance.json")):
        variant = path.parent.name
        scene_id = path.name.replace(".scene_instance.json", "")
        total = count_objects(path)
        if total is None:
            continue
        rows.append({"instance_key": safe_key(variant, scene_id), "variant": variant, "scene_id": scene_id, "total_objects": total, "path": str(path.relative_to(hssd_root))})

filtered = []
seen = set()
for row in rows:
    if row["total_objects"] >= max_objects:
        continue
    if variant_filter and row["variant"] not in variant_filter:
        continue
    if row["instance_key"] in seen:
        continue
    seen.add(row["instance_key"])
    filtered.append(row)

filtered.sort(key=lambda item: (item["total_objects"], item["variant"], item["scene_id"]))
if limit > 0:
    filtered = filtered[:limit]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["instance_key", "variant", "scene_id", "total_objects", "path"], delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(filtered)

print(f"selected_instances={len(filtered)}")
print(f"selected_file={out_path}")
PY
}

write_pairs_file() {
  log "writing generator/judge pairs"
  printf "generator_key\tgenerator_model_path\tgenerator_served_model\tjudge_key\tjudge_model_path\tjudge_served_model\n" > "$PAIRS_FILE"
  local gen_key judge_key
  for gen_key in $GENERATOR_KEYS; do
    require_path "$(model_path "$gen_key")/config.json"
    for judge_key in $JUDGE_KEYS; do
      require_path "$(model_path "$judge_key")/config.json"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$gen_key" "$(model_path "$gen_key")" "$(model_served "$gen_key")" \
        "$judge_key" "$(model_path "$judge_key")" "$(model_served "$judge_key")" \
        >> "$PAIRS_FILE"
    done
  done
  if [[ "$ENABLE_GLM_GENERATOR" == "1" ]]; then
    if model_available glm9b; then
      require_path "$(model_path qwen32b)/config.json"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "glm9b" "$(model_path glm9b)" "$(model_served glm9b)" \
        "qwen32b" "$(model_path qwen32b)" "$(model_served qwen32b)" \
        >> "$PAIRS_FILE"
      log "optional GLM generator lane enabled: generator=glm9b judge=qwen32b"
    else
      log "optional GLM generator lane skipped: ${GLM_MODEL_PATH} is missing, incomplete, or still downloading"
    fi
  fi
}

init_summary_files() {
  printf "generator_key\tgenerator_model_path\tgenerator_served_model\tjudge_key\tjudge_model_path\tjudge_served_model\tinstance_key\tvariant\tscene_id\ttotal_objects\tmode\tstatus\ttask_error\tvalidity_gate\toverall_valid\tvlm_score\tobject_presence_rate\tgeneration_finish_reason\trepair_finish_reason\tgeneration_completion_tokens\trepair_completion_tokens\toutput_dir\n" > "$SUMMARY_FILE"
  printf "generator_key\tjudge_key\tinstance_key\tvariant\tscene_id\ttotal_objects\tmode\treason\n" > "$SKIPPED_FILE"
}

stage_hssd_root_for_instance() {
  local instance_key="$1"
  local scene_id="$2"
  local rel_path="$3"
  local stage_root="${CASES_ROOT}/_staged_hssd/${instance_key}"
  local src_path

  if [[ "$rel_path" = /* ]]; then
    src_path="$rel_path"
  else
    src_path="${HSSD_ROOT}/${rel_path}"
  fi
  if [[ ! -f "$src_path" ]]; then
    return 1
  fi

  mkdir -p "${stage_root}/scenes"
  cp -f "$src_path" "${stage_root}/scenes/${scene_id}.scene_instance.json"

  local name
  for name in objects stages semantics metadata scene_filter_files manifests; do
    if [[ -e "${HSSD_ROOT}/${name}" ]]; then
      ln -sfn "${HSSD_ROOT}/${name}" "${stage_root}/${name}"
    fi
  done
  for src_path in "${HSSD_ROOT}"/*.scene_dataset_config.json "${HSSD_ROOT}"/scene_splits.yaml; do
    if [[ -e "$src_path" ]]; then
      ln -sfn "$src_path" "${stage_root}/$(basename "$src_path")"
    fi
  done

  printf "%s" "$stage_root"
}

append_summary() {
  local gen_key="$1"
  local gen_path="$2"
  local gen_served="$3"
  local judge_key="$4"
  local judge_path="$5"
  local judge_served="$6"
  local instance_key="$7"
  local variant="$8"
  local scene_id="$9"
  local total_objects="${10}"
  local mode="${11}"
  local run_out="${12}"
  local case_dir="${run_out}/${mode}/${scene_id}"

  "$BENCH_PY" - "$SUMMARY_FILE" "$gen_key" "$gen_path" "$gen_served" "$judge_key" "$judge_path" "$judge_served" "$instance_key" "$variant" "$scene_id" "$total_objects" "$mode" "$case_dir" <<'PY'
import json
import sys
from pathlib import Path

summary, gen_key, gen_path, gen_served, judge_key, judge_path, judge_served, instance_key, variant, scene_id, total, mode, case_dir = sys.argv[1:]
case_path = Path(case_dir)

status = "not_valid"
task_error = ""
validity_gate = ""
overall_valid = ""
vlm_score = ""
object_presence = ""
generation_finish = ""
repair_finish = ""
generation_completion = ""
repair_completion = ""

results = case_path.parents[1] / "validity_results.tsv"
if results.exists():
    for line in results.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0] == scene_id and parts[1] == mode:
            status = parts[2]

err = case_path / "task_error.txt"
if err.exists():
    task_error = err.read_text(encoding="utf-8", errors="replace").strip().replace("\t", " ").replace("\n", " ")[:500]

metrics_path = case_path / "case_metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validity_gate = str(metrics.get("validity_gate", ""))
    overall_valid = str(metrics.get("overall_valid", ""))
    vlm_score = str(metrics.get("vlm_score", ""))
    object_presence = str(metrics.get("object_presence_rate", ""))

gen_meta = case_path / "generation_request_metadata.json"
if gen_meta.exists():
    data = json.loads(gen_meta.read_text(encoding="utf-8"))
    generation_finish = str(data.get("finish_reason") or "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    generation_completion = str(usage.get("completion_tokens") or "")

repair_metas = sorted(case_path.glob("repair_request_metadata_iter_*.json"))
if repair_metas:
    data = json.loads(repair_metas[-1].read_text(encoding="utf-8"))
    repair_finish = str(data.get("finish_reason") or "")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    repair_completion = str(usage.get("completion_tokens") or "")

fields = [
    gen_key,
    gen_path,
    gen_served,
    judge_key,
    judge_path,
    judge_served,
    instance_key,
    variant,
    scene_id,
    str(total),
    mode,
    status,
    task_error,
    validity_gate,
    overall_valid,
    vlm_score,
    object_presence,
    generation_finish,
    repair_finish,
    generation_completion,
    repair_completion,
    str(case_path),
]
with Path(summary).open("a", encoding="utf-8") as handle:
    handle.write("\t".join(fields) + "\n")
PY
}

generation_tokens_for_mode() {
  local mode="$1"
  if [[ "$mode" == "full_metadata_budgeted" ]]; then
    printf "%s" "$FULL_METADATA_GENERATION_MAX_TOKENS"
  else
    printf "%s" "$GENERATION_MAX_TOKENS"
  fi
}

run_task() {
  local gen_key="$1"
  local gen_path="$2"
  local gen_served="$3"
  local judge_key="$4"
  local judge_path="$5"
  local judge_served="$6"
  local instance_key="$7"
  local variant="$8"
  local scene_id="$9"
  local total_objects="${10}"
  local rel_path="${11}"
  local mode="${12}"

  local pair_key="${gen_key}__judge_${judge_key}"
  local stage_root
  local run_out="${OUT_DIR}/${pair_key}/${variant}/${scene_id}"
  local gen_max_tokens
  gen_max_tokens="$(generation_tokens_for_mode "$mode")"

  log "task generator=${gen_key} judge=${judge_key} instance=${instance_key} objects=${total_objects} mode=${mode} start"
  heartbeat
  if ! ensure_model_server "$gen_key"; then
    log "task skipped; generator server failed to start key=${gen_key}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gen_key" "$judge_key" "$instance_key" "$variant" "$scene_id" "$total_objects" "$mode" "generator_server_unavailable" >> "$SKIPPED_FILE"
    return 0
  fi
  if ! ensure_model_server "$judge_key"; then
    log "task skipped; judge server failed to start key=${judge_key}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gen_key" "$judge_key" "$instance_key" "$variant" "$scene_id" "$total_objects" "$mode" "judge_server_unavailable" >> "$SKIPPED_FILE"
    return 0
  fi

  if ! stage_root="$(stage_hssd_root_for_instance "$instance_key" "$scene_id" "$rel_path")"; then
    log "task skipped; missing ${HSSD_ROOT}/${rel_path}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$gen_key" "$judge_key" "$instance_key" "$variant" "$scene_id" "$total_objects" "$mode" "missing_source_json" >> "$SKIPPED_FILE"
    return 0
  fi

  local task_rc=0
  local case_dir="${run_out}/${mode}/${scene_id}"
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground "$TASK_TIMEOUT_SECONDS" \
      "$BENCH_PY" "$REPO_ROOT/scripts/run_hssd_hab_10_qwen_validity.py" \
        --hssd-root "$stage_root" \
        --cases-root "${CASES_ROOT}/${pair_key}/${instance_key}/${mode}" \
        --scene-ids "$scene_id" \
        --modes "$mode" \
        --model openai_compatible \
        --judge-model openai_compatible_judge \
        --model-endpoint "$(model_endpoint "$gen_key")" \
        --model-id "$gen_served" \
        --judge-model-endpoint "$(model_endpoint "$judge_key")" \
        --judge-model-id "$judge_served" \
        --max-tokens "$gen_max_tokens" \
        --generation-max-tokens "$gen_max_tokens" \
        --repair-max-tokens "$REPAIR_MAX_TOKENS" \
        --judge-max-tokens "$JUDGE_MAX_TOKENS" \
        --context-length "$(model_context_length "$gen_key")" \
        --judge-context-length "$(model_context_length "$judge_key")" \
        --prompt-safety-margin-tokens "$PROMPT_SAFETY_MARGIN_TOKENS" \
        --max-repair-iterations "$MAX_REPAIR_ITERATIONS" \
        --timeout-seconds "$TIMEOUT_SECONDS" \
        --judge-timeout-seconds "$TIMEOUT_SECONDS" \
        --valid-source overall_valid \
        --no-download \
        --out "$run_out" || task_rc=$?
  else
    "$BENCH_PY" "$REPO_ROOT/scripts/run_hssd_hab_10_qwen_validity.py" \
      --hssd-root "$stage_root" \
      --cases-root "${CASES_ROOT}/${pair_key}/${instance_key}/${mode}" \
      --scene-ids "$scene_id" \
      --modes "$mode" \
      --model openai_compatible \
      --judge-model openai_compatible_judge \
      --model-endpoint "$(model_endpoint "$gen_key")" \
      --model-id "$gen_served" \
      --judge-model-endpoint "$(model_endpoint "$judge_key")" \
      --judge-model-id "$judge_served" \
      --max-tokens "$gen_max_tokens" \
      --generation-max-tokens "$gen_max_tokens" \
      --repair-max-tokens "$REPAIR_MAX_TOKENS" \
      --judge-max-tokens "$JUDGE_MAX_TOKENS" \
      --context-length "$(model_context_length "$gen_key")" \
      --judge-context-length "$(model_context_length "$judge_key")" \
      --prompt-safety-margin-tokens "$PROMPT_SAFETY_MARGIN_TOKENS" \
      --max-repair-iterations "$MAX_REPAIR_ITERATIONS" \
      --timeout-seconds "$TIMEOUT_SECONDS" \
      --judge-timeout-seconds "$TIMEOUT_SECONDS" \
      --valid-source overall_valid \
      --no-download \
      --out "$run_out" || task_rc=$?
  fi

  if (( task_rc != 0 )) && [[ ! -f "${case_dir}/task_error.txt" ]]; then
    mkdir -p "$case_dir"
    if [[ "$task_rc" == "124" ]]; then
      echo "DriverTaskTimeout: runner exceeded TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS}" > "${case_dir}/task_error.txt"
    else
      echo "DriverTaskError: runner exited with code ${task_rc}" > "${case_dir}/task_error.txt"
    fi
  fi

  append_summary "$gen_key" "$gen_path" "$gen_served" "$judge_key" "$judge_path" "$judge_served" "$instance_key" "$variant" "$scene_id" "$total_objects" "$mode" "$run_out"

  if [[ -f "${case_dir}/task_error.txt" ]]; then
    log "task_error generator=${gen_key} judge=${judge_key} instance=${instance_key} mode=${mode}"
    cat "${case_dir}/task_error.txt" || true
  fi

  flush_memory "$gen_key" "$judge_key"
  ensure_model_server "$gen_key" || true
  ensure_model_server "$judge_key" || true
  log "health"
  print_gpu_health
  log "task generator=${gen_key} judge=${judge_key} instance=${instance_key} mode=${mode} done"
  heartbeat
}

run_all() {
  local pair_count instance_count mode_count total_tasks
  pair_count="$(awk 'NR>1{c++} END{print c+0}' "$PAIRS_FILE")"
  instance_count="$(awk 'NR>1{c++} END{print c+0}' "$SELECTED_FILE")"
  mode_count="$(wc -w <<< "$MODES" | tr -d ' ')"
  total_tasks=$((pair_count * instance_count * mode_count))
  log "task plan: pairs=${pair_count} instances=${instance_count} modes=${mode_count} total_tasks=${total_tasks}"

  local task_index=0
  while IFS=$'\t' read -r gen_key gen_path gen_served judge_key judge_path judge_served; do
    if [[ "$gen_key" == "generator_key" ]]; then
      continue
    fi

    while IFS=$'\t' read -r instance_key variant scene_id total_objects rel_path; do
      if [[ "$instance_key" == "instance_key" ]]; then
        continue
      fi
      local mode
      for mode in $MODES; do
        task_index=$((task_index + 1))
        log "progress ${task_index}/${total_tasks}"
        run_task "$gen_key" "$gen_path" "$gen_served" "$judge_key" "$judge_path" "$judge_served" "$instance_key" "$variant" "$scene_id" "$total_objects" "$rel_path" "$mode"
      done
    done < "$SELECTED_FILE"
  done < "$PAIRS_FILE"
}

summarize_run() {
  log "summary"
  echo "RUN_TAG=$RUN_TAG"
  echo "OUT_DIR=$OUT_DIR"
  echo "DRIVER_LOG=$DRIVER_LOG"
  echo "SUMMARY_FILE=$SUMMARY_FILE"
  echo "SKIPPED_FILE=$SKIPPED_FILE"
  echo "TAR_PATH=$TAR_PATH"
  echo
  echo "== model pairs =="
  cat "$PAIRS_FILE" || true
  echo
  echo "== selected instances =="
  cat "$SELECTED_FILE" || true
  echo
  echo "== validity =="
  cat "$SUMMARY_FILE" || true
  echo
  echo "== skipped =="
  cat "$SKIPPED_FILE" || true
  echo
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
  require_path "$SGLANG_PY"
  require_path "$BENCH_PY"
  require_path "$MODELS_ROOT"
  require_path "$HSSD_ROOT"

  log "config"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "HSSD_ROOT=$HSSD_ROOT"
  echo "MODELS_ROOT=$MODELS_ROOT"
  echo "GENERATOR_KEYS=$GENERATOR_KEYS"
  echo "JUDGE_KEYS=$JUDGE_KEYS"
  echo "ENABLE_GLM_GENERATOR=$ENABLE_GLM_GENERATOR"
  echo "QWEN8_MODEL_PATH=$QWEN8_MODEL_PATH"
  echo "QWEN8_ENDPOINT=$(model_endpoint qwen8b)"
  echo "QWEN8_CUDA_DEVICES=$QWEN8_CUDA_DEVICES"
  echo "QWEN8_TP_SIZE=$QWEN8_TP_SIZE"
  echo "QWEN8_CONTEXT_LENGTH=$QWEN8_CONTEXT_LENGTH"
  echo "QWEN32_MODEL_PATH=$QWEN32_MODEL_PATH"
  echo "QWEN32_ENDPOINT=$(model_endpoint qwen32b)"
  echo "QWEN32_CUDA_DEVICES=$QWEN32_CUDA_DEVICES"
  echo "QWEN32_TP_SIZE=$QWEN32_TP_SIZE"
  echo "QWEN32_CONTEXT_LENGTH=$QWEN32_CONTEXT_LENGTH"
  echo "GLM_MODEL_PATH=$GLM_MODEL_PATH"
  echo "GLM_ENDPOINT=$(model_endpoint glm9b)"
  echo "GLM_CUDA_DEVICES=$GLM_CUDA_DEVICES"
  echo "GLM_TP_SIZE=$GLM_TP_SIZE"
  echo "GLM_CONTEXT_LENGTH=$GLM_CONTEXT_LENGTH"
  echo "MODES=$MODES"
  echo "GENERATION_MAX_TOKENS=$GENERATION_MAX_TOKENS"
  echo "FULL_METADATA_GENERATION_MAX_TOKENS=$FULL_METADATA_GENERATION_MAX_TOKENS"
  echo "REPAIR_MAX_TOKENS=$REPAIR_MAX_TOKENS"
  echo "JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
  echo "MAX_INSTANCE_OBJECTS=<${MAX_INSTANCE_OBJECTS}"
  echo "LIMIT_INSTANCES=$LIMIT_INSTANCES"
  echo "VARIANT_FILTER=${VARIANT_FILTER:-<none>}"
  echo "STOP_EXISTING_SGLANG=$STOP_EXISTING_SGLANG"
  "$SGLANG_PY" - <<'PY'
import sys
import sglang
import zmq
print("sglang python:", sys.executable)
print("sglang + zmq ok")
PY

  stop_existing_sglang_if_requested
  write_candidate_file
  write_pairs_file
  init_summary_files
  print_gpu_health
  run_all
  summarize_run
  package_outputs
  log "done"
}

main "$@"
