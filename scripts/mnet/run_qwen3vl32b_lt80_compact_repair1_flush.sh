#!/usr/bin/env bash
set -Eeuo pipefail

# Run inside the MNET H20 pod.
# This launches one Qwen3-VL-32B SGLang TP8 server, then runs every cached
# HSSD-HAB scene-variant with total_objects < MAX_INSTANCE_OBJECTS.
# Each instance is run sequentially and followed by /flush_cache to avoid
# accumulated KV/cache pressure across a long unattended sweep.

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

MODE=${MODE:-compact_objects}
GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-64000}
REPAIR_MAX_TOKENS=${REPAIR_MAX_TOKENS:-24000}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-2048}
PROMPT_SAFETY_MARGIN_TOKENS=${PROMPT_SAFETY_MARGIN_TOKENS:-4096}
MAX_REPAIR_ITERATIONS=${MAX_REPAIR_ITERATIONS:-1}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}

# Strictly less than this value. The default means total_objects < 80.
MAX_INSTANCE_OBJECTS=${MAX_INSTANCE_OBJECTS:-80}
LIMIT_INSTANCES=${LIMIT_INSTANCES:-0}
VARIANT_FILTER=${VARIANT_FILTER:-}

RUN_TAG=${RUN_TAG:-mnet_32b_tp8_96k_lt80_compact_repair1_flush_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/${RUN_TAG}}
CASES_ROOT=${CASES_ROOT:-${REPO_ROOT}/data/external/hssd-hab-converted/${RUN_TAG}}
HSSD_ROOT=${HSSD_ROOT:-${REPO_ROOT}/data/external/hssd-hab}

DRIVER_LOG="${LOG_ROOT}/${RUN_TAG}.driver.out"
SERVER_LOG="${LOG_ROOT}/${RUN_TAG}.server.out"
SERVER_PID_FILE="${LOG_ROOT}/${RUN_TAG}.server.pid"
DRIVER_PID_FILE="${LOG_ROOT}/${RUN_TAG}.driver.pid"
HEARTBEAT_FILE="${LOG_ROOT}/${RUN_TAG}.heartbeat"
SELECTED_FILE="${OUT_DIR}/selected_instances.tsv"
SKIPPED_FILE="${OUT_DIR}/skipped_instances.tsv"
SUMMARY_FILE="${OUT_DIR}/validity_results.tsv"
TAR_PATH="/mnt/group/cmh/${RUN_TAG}.tar.gz"

mkdir -p "$LOG_ROOT" "$OUT_DIR" "$CASES_ROOT"
echo $$ > "$DRIVER_PID_FILE"
exec > >(tee -a "$DRIVER_LOG") 2>&1

log() {
  echo "==== $(date '+%F %T') $* ===="
}

heartbeat() {
  {
    echo "time=$(date '+%F %T')"
    echo "run_tag=$RUN_TAG"
    echo "driver_pid=$$"
    echo "selected_file=$SELECTED_FILE"
    echo "summary_file=$SUMMARY_FILE"
  } > "$HEARTBEAT_FILE"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing required path: $path" >&2
    exit 1
  fi
}

safe_key() {
  local variant="$1"
  local scene_id="$2"
  printf "%s__%s" "${variant//[^A-Za-z0-9_]/_}" "${scene_id//[^A-Za-z0-9_]/_}"
}

print_gpu_health() {
  nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv || true
}

server_alive() {
  curl --noproxy "*" -fsS "${ENDPOINT}/models" >/dev/null 2>&1
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

wait_for_server() {
  local deadline=$((SECONDS + 3000))
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
        tail -n 200 "$SERVER_LOG" || true
        return 1
      fi
    fi

    tail -n 8 "$SERVER_LOG" 2>/dev/null || true
    sleep 15
  done

  log "server readiness timeout"
  tail -n 240 "$SERVER_LOG" || true
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
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT"])
hssd_root = Path(os.environ["HSSD_ROOT"])
out_path = Path(os.environ["SELECTED_FILE"])
max_objects = int(os.environ["MAX_INSTANCE_OBJECTS"])
limit = int(os.environ.get("LIMIT_INSTANCES", "0") or "0")
variant_filter = {
    item.strip()
    for item in os.environ.get("VARIANT_FILTER", "").split(",")
    if item.strip()
}


def _as_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _safe_key(variant, scene_id):
    def clean(text):
        return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))
    return f"{clean(variant)}__{clean(scene_id)}"


def _row_from_dict(row):
    variant = row.get("variant") or row.get("folder") or row.get("split") or ""
    scene_id = row.get("scene_id") or row.get("id") or ""
    total = _as_int(row.get("total_objects") or row.get("object_count") or row.get("objects"))
    rel_path = row.get("path") or row.get("relative_path") or row.get("hf_path") or ""
    if not rel_path and variant and scene_id:
        rel_path = f"{variant}/{scene_id}.scene_instance.json"
    if not variant and rel_path:
        variant = Path(rel_path).parent.name
    if not scene_id and rel_path:
        scene_id = Path(rel_path).name.replace(".scene_instance.json", "")
    if not variant or not scene_id or total is None or not rel_path:
        return None
    return {
        "instance_key": _safe_key(variant, scene_id),
        "variant": variant,
        "scene_id": scene_id,
        "total_objects": total,
        "path": rel_path,
    }


def _read_manifest():
    manifest = hssd_root / "manifests" / "hssd_hab_le100_scene_instances.tsv"
    if not manifest.exists():
        return []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [item for row in reader if (item := _row_from_dict(row))]


def _read_notes_hssd():
    notes = repo / "notes" / "HSSD"
    if not notes.exists():
        return []
    lines = notes.read_text(encoding="utf-8").splitlines()
    table = []
    in_table = False
    for line in lines:
        if line.startswith("variant\tscene_id\ttotal_objects"):
            in_table = True
            table.append(line)
            continue
        if in_table and line.startswith("```"):
            break
        if in_table and line.strip():
            table.append(line)
    if not table:
        return []
    reader = csv.DictReader(table, delimiter="\t")
    return [item for row in reader if (item := _row_from_dict(row))]


rows = _read_manifest() or _read_notes_hssd()
filtered = []
seen = set()
for row in rows:
    if row["total_objects"] >= max_objects:
        continue
    if variant_filter and row["variant"] not in variant_filter:
        continue
    key = row["instance_key"]
    if key in seen:
        continue
    seen.add(key)
    filtered.append(row)

filtered.sort(key=lambda item: (item["total_objects"], item["variant"], item["scene_id"]))
if limit > 0:
    filtered = filtered[:limit]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["instance_key", "variant", "scene_id", "total_objects", "path"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(filtered)

print(f"selected_instances={len(filtered)}")
print(f"selected_file={out_path}")
PY
}

init_summary_files() {
  printf "instance_key\tvariant\tscene_id\ttotal_objects\tmode\tstatus\ttask_error\tgeneration_finish_reason\trepair_finish_reason\tgeneration_completion_tokens\trepair_completion_tokens\toutput_dir\n" > "$SUMMARY_FILE"
  printf "instance_key\tvariant\tscene_id\ttotal_objects\tpath\treason\n" > "$SKIPPED_FILE"
}

append_summary() {
  local instance_key="$1"
  local variant="$2"
  local scene_id="$3"
  local total_objects="$4"
  local run_out="$5"
  local case_dir="${run_out}/${MODE}/${scene_id}"

  "$BENCH_PY" - "$SUMMARY_FILE" "$instance_key" "$variant" "$scene_id" "$total_objects" "$MODE" "$case_dir" <<'PY'
import json
import sys
from pathlib import Path

summary, instance_key, variant, scene_id, total, mode, case_dir = sys.argv[1:]
case_path = Path(case_dir)

status = "not_valid"
task_error = ""
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
    instance_key,
    variant,
    scene_id,
    str(total),
    mode,
    status,
    task_error,
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
  printf "%s" "$stage_root"
}

run_instance() {
  local instance_key="$1"
  local variant="$2"
  local scene_id="$3"
  local total_objects="$4"
  local rel_path="$5"

  local run_out="${OUT_DIR}/${variant}/${scene_id}"
  local stage_root

  log "instance=${instance_key} objects=${total_objects} start"
  heartbeat
  ensure_server

  if ! stage_root="$(stage_hssd_root_for_instance "$instance_key" "$scene_id" "$rel_path")"; then
    log "instance=${instance_key} skipped; missing ${HSSD_ROOT}/${rel_path}"
    printf "%s\t%s\t%s\t%s\t%s\tmissing_source_json\n" "$instance_key" "$variant" "$scene_id" "$total_objects" "$rel_path" >> "$SKIPPED_FILE"
    return 0
  fi

  "$BENCH_PY" "$REPO_ROOT/scripts/run_hssd_hab_10_qwen_validity.py" \
    --hssd-root "$stage_root" \
    --cases-root "${CASES_ROOT}/${instance_key}" \
    --scene-ids "$scene_id" \
    --modes "$MODE" \
    --model qwen3vl_sglang_32b \
    --judge-model same \
    --model-endpoint "$ENDPOINT" \
    --model-id "$SERVED_MODEL" \
    --max-tokens "$GENERATION_MAX_TOKENS" \
    --generation-max-tokens "$GENERATION_MAX_TOKENS" \
    --repair-max-tokens "$REPAIR_MAX_TOKENS" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --context-length "$CONTEXT_LENGTH" \
    --prompt-safety-margin-tokens "$PROMPT_SAFETY_MARGIN_TOKENS" \
    --max-repair-iterations "$MAX_REPAIR_ITERATIONS" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --valid-source overall_valid \
    --no-download \
    --out "$run_out" || true

  append_summary "$instance_key" "$variant" "$scene_id" "$total_objects" "$run_out"

  local case_dir="${run_out}/${MODE}/${scene_id}"
  if [[ -f "${case_dir}/task_error.txt" ]]; then
    log "instance=${instance_key} task_error"
    cat "${case_dir}/task_error.txt" || true
  fi

  flush_cache
  log "health"
  print_gpu_health

  if ! server_alive; then
    log "server unhealthy after instance=${instance_key}; relaunching before next instance"
    launch_server
  fi

  log "instance=${instance_key} done"
  heartbeat
}

run_all_instances() {
  local total
  total="$(awk 'NR > 1 {count++} END {print count + 0}' "$SELECTED_FILE")"
  local index=0

  while IFS=$'\t' read -r instance_key variant scene_id total_objects rel_path; do
    instance_key="${instance_key%$'\r'}"
    variant="${variant%$'\r'}"
    scene_id="${scene_id%$'\r'}"
    total_objects="${total_objects%$'\r'}"
    rel_path="${rel_path%$'\r'}"
    if [[ "$instance_key" == "instance_key" ]]; then
      continue
    fi
    index=$((index + 1))
    log "progress ${index}/${total}"
    run_instance "$instance_key" "$variant" "$scene_id" "$total_objects" "$rel_path"
  done < "$SELECTED_FILE"
}

summarize_run() {
  log "summary"
  echo "RUN_TAG=$RUN_TAG"
  echo "OUT_DIR=$OUT_DIR"
  echo "DRIVER_LOG=$DRIVER_LOG"
  echo "SERVER_LOG=$SERVER_LOG"
  echo "SUMMARY_FILE=$SUMMARY_FILE"
  echo "SKIPPED_FILE=$SKIPPED_FILE"
  echo "TAR_PATH=$TAR_PATH"
  echo
  echo "== selected =="
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

  require_file "$SGLANG_PY"
  require_file "$BENCH_PY"
  require_file "$MODEL_PATH/config.json"
  require_file "$HSSD_ROOT"
  require_file "$REPO_ROOT/notes/HSSD"

  log "config"
  echo "REPO_ROOT=$REPO_ROOT"
  echo "HSSD_ROOT=$HSSD_ROOT"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "SERVED_MODEL=$SERVED_MODEL"
  echo "ENDPOINT=$ENDPOINT"
  echo "CONTEXT_LENGTH=$CONTEXT_LENGTH"
  echo "MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC"
  echo "MODE=$MODE"
  echo "GENERATION_MAX_TOKENS=$GENERATION_MAX_TOKENS"
  echo "REPAIR_MAX_TOKENS=$REPAIR_MAX_TOKENS"
  echo "JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
  echo "MAX_REPAIR_ITERATIONS=$MAX_REPAIR_ITERATIONS"
  echo "MAX_INSTANCE_OBJECTS=<${MAX_INSTANCE_OBJECTS}"
  echo "LIMIT_INSTANCES=$LIMIT_INSTANCES"
  echo "VARIANT_FILTER=${VARIANT_FILTER:-<none>}"
  "$SGLANG_PY" - <<'PY'
import sys
import sglang
import zmq
print("sglang python:", sys.executable)
print("sglang + zmq ok")
PY

  write_candidate_file
  init_summary_files

  if ! server_alive; then
    launch_server
  else
    log "server already alive"
    curl --noproxy "*" -sS "${ENDPOINT}/models" || true
    echo
  fi
  print_gpu_health

  run_all_instances
  summarize_run
  package_outputs
  log "done"
  heartbeat
}

main "$@"
