#!/usr/bin/env bash
set -Eeuo pipefail

# Complete P0b visual-config calibration on one 8 x H20 MNET pod.
#
# Physical phases never overlap Blender with Qwen3-VL-235B:
#   A. Blender: deterministic evidence, packet variants, VLM candidate previews.
#   B. 235B: select candidate IDs only (max_steps=0), then stop the server.
#   C. Blender: render only the frozen VLM-selected final views.
#   D. 235B: judge all frozen packets, score, and aggregate.
#
# The experiment is deliberately not a Cartesian product. Presence/order/budget
# arms reuse the deterministic evidence bank, and all per-metric mixtures can be
# reconstructed from the aligned event table.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
SGLANG_PY=${SGLANG_PY:-/mnt/group/cmh/envs/sglang-qwen3vl/bin/python}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
MODEL_PATH=${MODEL_PATH:-/mnt/group/cmh/models/Qwen3-VL-235B-A22B-Instruct-FP8}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-235B-A22B-Instruct-FP8}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_235b_fp8_mnet_judge.json}

FIXTURE_ROOT=${FIXTURE_ROOT:-${REPO_ROOT}/configs/experiments/p0b_source_distortion5}
CASE_FILE=${CASE_FILE:-${FIXTURE_ROOT}/cases.json}
SOURCE_STUDY_ROOT=${SOURCE_STUDY_ROOT:-${REPO_ROOT}/outputs/p0b_combined_20260717_202533/source_distortion5}
SOURCE_ROOT=${SOURCE_ROOT:-${SOURCE_STUDY_ROOT}/source_reports}
GT_ROOT=${GT_ROOT:-${SOURCE_STUDY_ROOT}/gt}
FAMILIES=${FAMILIES:-clean,collision,oob,support}

PHASE=${PHASE:-all} # all | prepare_bank | select | finalize | judge
RUN_TAG=${RUN_TAG:-qwen235b_fp8_p0b_visual_config_$(date '+%Y%m%d_%H%M%S')}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-${OUT_ROOT}/prepared_evidence}
ABLATION_ROOT=${ABLATION_ROOT:-${OUT_ROOT}/ablation}
RESULT_ROOT=${RESULT_ROOT:-${OUT_ROOT}/results}
CASE_MANIFEST=${CASE_MANIFEST:-${OUT_ROOT}/case_manifest.tsv}
PHASE_LOG_ROOT=${PHASE_LOG_ROOT:-${OUT_ROOT}/phase_logs}
LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}

PORT=${PORT:-8298}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT}/v1}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
TP_SIZE=${TP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
RENDER_GPUS=${RENDER_GPUS:-0,1,2,3,4,5,6,7}
SGL_ENABLE_JIT_DEEPGEMM=${SGL_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-7200}
SELECTOR_CONCURRENCY=${SELECTOR_CONCURRENCY:-8}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-8}
KEEP_SERVER=${KEEP_SERVER:-0}
ALLOW_PREP_ERRORS=${ALLOW_PREP_ERRORS:-0}
RESUME=${RESUME:-1}
MAX_CASES=${MAX_CASES:-0}

BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-3600}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
CYCLES_SAMPLES=${CYCLES_SAMPLES:-8}
PREVIEW_WIDTH=${PREVIEW_WIDTH:-256}
PREVIEW_HEIGHT=${PREVIEW_HEIGHT:-256}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-1}
MAX_VIEWS=${MAX_VIEWS:-2}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-6}

ARMS=(
  fixed_global
  presence_local_raw
  presence_local_raw_highlight
  presence_global_local_raw
  deterministic_metric_local
  order_local_first_full
  budget_global_first_compact
  budget_local_first_compact
  vlm_select_from_candidates
)

SERVER_STARTED_BY_SCRIPT=0
SERVER_PID=""
SERVER_ROLE=""
SERVER_LOG=""
PID_FILE=""
BLENDER_WORKER_PIDS=()

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

validate_gpu_csv() {
  local label=$1 value=$2
  "$BENCH_PY" - "$label" "$value" <<'PY'
import sys
label, raw = sys.argv[1:]
values = [item.strip() for item in raw.split(",") if item.strip()]
if len(values) != 8 or len(set(values)) != 8 or any(not item.isdigit() for item in values):
    raise SystemExit(f"{label} must contain exactly eight unique numeric GPU IDs; got {raw!r}")
print(f"{label}: {','.join(values)}")
PY
}

server_descendants() {
  local parent=$1 child
  for child in $(pgrep -P "$parent" 2>/dev/null); do
    server_descendants "$child"
  done
  echo "$parent"
}

sglang_env_pids() {
  # SGLang multiprocessing workers can outlive a failed launch_server parent.
  # Workers initially use this experiment's dedicated interpreter, but SGLang
  # later changes scheduler/tokenizer process titles to ``sglang::*``. Include
  # both forms so re-parented workers cannot evade preflight or cleanup.
  {
    pgrep -f "^${SGLANG_PY}([[:space:]]|$)" 2>/dev/null || true
    pgrep -f '^sglang::' 2>/dev/null || true
  } | sort -u
}

stop_owned_server() {
  local force=${1:-0}
  if [[ "$SERVER_STARTED_BY_SCRIPT" != "1" ]]; then
    return
  fi
  if [[ "$force" != "1" && "$KEEP_SERVER" == "1" ]]; then
    log "KEEP_SERVER=1: leaving final owned server running pid=${SERVER_PID}"
    SERVER_STARTED_BY_SCRIPT=0
    return
  fi
  log "stopping owned ${SERVER_ROLE} server pid=${SERVER_PID}"
  curl --noproxy "*" -fsS -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null 2>&1 || true
  local server_pids remaining_pids
  server_pids=$(
    {
      server_descendants "$SERVER_PID"
      sglang_env_pids
    } | sort -u | tr '\n' ' '
  )
  kill -TERM $server_pids 2>/dev/null || true
  for _ in $(seq 1 60); do
    remaining_pids=$(sglang_env_pids)
    [[ -n "$remaining_pids" ]] || break
    sleep 1
  done
  remaining_pids=$(sglang_env_pids)
  for pid in $server_pids $remaining_pids; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
  [[ -n "$PID_FILE" ]] && rm -f "$PID_FILE"
  SERVER_STARTED_BY_SCRIPT=0
  SERVER_PID=""
  SERVER_ROLE=""
}

cleanup() {
  local worker_pid worker_tree
  if (( ${#BLENDER_WORKER_PIDS[@]} )); then
    log "stopping Blender worker process trees"
    for worker_pid in "${BLENDER_WORKER_PIDS[@]}"; do
      worker_tree=$(server_descendants "$worker_pid" | sort -u | tr '\n' ' ')
      kill -TERM $worker_tree 2>/dev/null || true
    done
    BLENDER_WORKER_PIDS=()
  fi
  stop_owned_server 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ensure_no_server() {
  local lingering_pids
  if curl --noproxy "*" -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
    echo "A model server is alive at ${ENDPOINT}; Blender phases require it to be stopped" >&2
    return 1
  fi
  lingering_pids=$(sglang_env_pids)
  if [[ -n "$lingering_pids" ]]; then
    ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$(tr '\n' ',' <<<"$lingering_pids" | sed 's/,$//')" \
      >"${OUT_ROOT}/unexpected_sglang.txt" 2>/dev/null || true
    echo "SGLang interpreter processes already occupy this pod, including possible orphan workers:" >&2
    cat "${OUT_ROOT}/unexpected_sglang.txt" >&2
    return 1
  fi
}

ensure_no_blender() {
  if pgrep -af '[b]lender.*background' >"${OUT_ROOT}/lingering_blender.txt" 2>/dev/null; then
    echo "Blender workers remain active:" >&2
    cat "${OUT_ROOT}/lingering_blender.txt" >&2
    return 1
  fi
}

ensure_clean_gpu_memory() {
  local snapshot
  snapshot=$(nvidia-smi \
    --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits)
  "$BENCH_PY" - "$snapshot" <<'PY'
import sys

rows = []
for line in sys.argv[1].splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 3:
        rows.append((int(fields[0]), float(fields[1]), float(fields[2])))
if len(rows) != 8:
    raise SystemExit(f"expected memory status for 8 GPUs, found {len(rows)}")
dirty = [
    {"gpu": index, "free_mib": free, "total_mib": total}
    for index, free, total in rows
    if free < 0.90 * total
]
free_values = [free for _, free, _ in rows]
if dirty or min(free_values) < 0.90 * max(free_values):
    raise SystemExit(
        "GPU memory is not clean and balanced before SGLang startup; "
        f"dirty={dirty}, free_mib={free_values}"
    )
print("GPU memory preflight: clean and balanced")
PY
}

start_or_reuse_server() {
  local role=$1
  SERVER_ROLE=$role
  if curl --noproxy "*" -fsS "${ENDPOINT}/models" >"${OUT_ROOT}/models_${role}.json" 2>/dev/null; then
    "$BENCH_PY" - "$SERVED_MODEL" "${OUT_ROOT}/models_${role}.json" <<'PY'
import json, sys
models = {str(item.get("id")) for item in json.load(open(sys.argv[2])).get("data", [])}
if sys.argv[1] not in models:
    raise SystemExit(f"port serves {sorted(models)}, expected {sys.argv[1]!r}")
print("reusing live server:", sys.argv[1])
PY
    SERVER_STARTED_BY_SCRIPT=0
    return
  fi

  # A dead launch_server parent may leave renamed ``sglang::*`` workers holding
  # CUDA contexts. Refuse to stack a new TP/EP process group on that state.
  ensure_no_server
  ensure_clean_gpu_memory

  SERVER_LOG=${LOG_ROOT}/sglang_qwen3vl_235b_fp8_tp${TP_SIZE}_ep${EP_SIZE}_${CONTEXT_LENGTH}_${PORT}_${role}.out
  PID_FILE=${LOG_ROOT}/sglang_qwen3vl_235b_fp8_tp${TP_SIZE}_ep${EP_SIZE}_${CONTEXT_LENGTH}_${PORT}_${role}.pid
  log "launch ${role} Qwen3-VL-235B TP${TP_SIZE}/EP${EP_SIZE}"
  export CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export SGL_ENABLE_JIT_DEEPGEMM
  export SGLANG_ENABLE_JIT_DEEPGEMM
  nohup "$SGLANG_PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --context-length "$CONTEXT_LENGTH" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --tp-size "$TP_SIZE" \
    --ep-size "$EP_SIZE" \
    --disable-cuda-graph \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  SERVER_STARTED_BY_SCRIPT=1
  echo "$SERVER_PID" >"$PID_FILE"
  echo "server pid: $SERVER_PID"
  echo "server log: $SERVER_LOG"

  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until curl --noproxy "*" -fsS "${ENDPOINT}/models" >"${OUT_ROOT}/models_${role}.json" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Qwen3-VL-235B server exited before readiness" >&2
      tail -n 200 "$SERVER_LOG" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Qwen3-VL-235B startup exceeded ${STARTUP_TIMEOUT_SECONDS}s" >&2
      tail -n 200 "$SERVER_LOG" >&2
      return 1
    fi
    sleep 15
  done
  "$BENCH_PY" - "$SERVED_MODEL" "${OUT_ROOT}/models_${role}.json" <<'PY'
import json, sys
models = {str(item.get("id")) for item in json.load(open(sys.argv[2])).get("data", [])}
if sys.argv[1] not in models:
    raise SystemExit(f"served model mismatch: {sorted(models)}")
print("served model:", sys.argv[1])
PY
}

multimodal_smoke() {
  local role=$1
  local smoke_image
  smoke_image=$(find "$EVIDENCE_ROOT" -type f -name '*.png' -print -quit)
  require_path "$smoke_image"
  log "${role}: real multimodal smoke"
  "$BENCH_PY" - "$ENDPOINT" "$SERVED_MODEL" "$smoke_image" <<'PY'
import base64, json, sys, urllib.request
from pathlib import Path
endpoint, model, image_path = sys.argv[1:]
encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
payload = {
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Reply with exactly: visual-alive"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]}],
    "temperature": 0,
    "max_tokens": 16,
}
request = urllib.request.Request(
    f"{endpoint}/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
content = str(result["choices"][0]["message"].get("content") or "").strip()
if "visual-alive" not in content.lower():
    raise SystemExit(f"multimodal smoke failed: {content!r}")
print("multimodal response:", content)
PY
  curl --noproxy "*" -fsS -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null || true
}

build_case_manifest() {
  log "build frozen case manifest"
  "$BENCH_PY" - "$CASE_FILE" "$FIXTURE_ROOT" "$SOURCE_ROOT" "$GT_ROOT" "$FAMILIES" "$MAX_CASES" >"$CASE_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
case_file, fixture_root, source_root, gt_root = map(Path, sys.argv[1:5])
allowed = {value.strip() for value in sys.argv[5].split(",") if value.strip()}
max_cases = int(sys.argv[6])
family_metric = {"clean": "support", "collision": "collision", "oob": "oob", "support": "support"}
payload = json.load(case_file.open())
print("case_id\tbase_case_id\tfamily\tmetric\tfixture_dir")
selected = 0
for case in payload.get("cases") or []:
    family = str(case.get("family") or "")
    if family not in allowed or family not in family_metric:
        continue
    case_id = str(case["case_id"])
    fixture_dir = (fixture_root / str(case["fixture_dir"])).resolve()
    render_dir = source_root / case_id / "renders"
    required = [
        fixture_dir / "generated_scene.json",
        source_root / case_id / "evaluation_report.json",
        render_dir / "scene.blend",
        render_dir / "standardized_top.png",
        render_dir / "standardized_perspective.png",
        gt_root / f"{case_id}.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"frozen replay prerequisites missing for {case_id}: {missing}")
    print("\t".join([case_id, str(case["base_case_id"]), family, family_metric[family], str(fixture_dir)]))
    selected += 1
    if max_cases and selected >= max_cases:
        break
PY
  local case_count
  case_count=$(( $(wc -l <"$CASE_MANIFEST") - 1 ))
  if (( case_count <= 0 )); then
    echo "No replay cases selected" >&2
    return 1
  fi
  echo "selected cases: $case_count"
  if [[ "$MAX_CASES" == "0" && "$FAMILIES" == "clean,collision,oob,support" ]]; then
    "$BENCH_PY" - "$CASE_MANIFEST" "$GT_ROOT" <<'PY'
import json, sys
from pathlib import Path
manifest, gt_root = Path(sys.argv[1]), Path(sys.argv[2])
rows = [line.split("\t") for line in manifest.read_text().splitlines()[1:] if line]
events = 0
for row in rows:
    case_id, metric = row[0], row[3]
    gt = json.load((gt_root / f"{case_id}.json").open())
    events += sum(str(item.get("metric")) == metric for item in gt.get("events") or [])
if len(rows) != 20 or events != 54:
    raise SystemExit(f"default frozen universe drifted: cases={len(rows)} events={events}; expected 20/54")
print("default frozen universe: 20 cases / 54 events")
PY
  fi
}

run_blender_workers() {
  local stage=$1
  ensure_no_server
  IFS=',' read -r -a render_gpu_list <<<"$RENDER_GPUS"
  local worker_pids=()
  for worker_index in $(seq 0 7); do
    (
      local gpu=${render_gpu_list[$worker_index]}
      local line_number=0 worker_status=0
      while IFS=$'\t' read -r case_id base_case_id family metric fixture_dir; do
        line_number=$((line_number + 1))
        if (( (line_number - 1) % 8 != worker_index )); then
          continue
        fi
        local source_case=${SOURCE_ROOT}/${case_id}
        local render_dir=${source_case}/renders
        local case_root=${EVIDENCE_ROOT}/${case_id}
        local case_log=${PHASE_LOG_ROOT}/${stage}__${case_id}.gpu${gpu}.log
        local common_blender_args=(
          --blend-file "${render_dir}/scene.blend"
          --blender-bin "$BLENDER_BIN"
          --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS"
          --render-width "$RENDER_WIDTH"
          --render-height "$RENDER_HEIGHT"
          --render-engine CYCLES
          --cycles-device CUDA
          --cycles-samples "$CYCLES_SAMPLES"
          --cycles-denoising
          --preview-render-engine CYCLES
          --preview-width "$PREVIEW_WIDTH"
          --preview-height "$PREVIEW_HEIGHT"
          --preview-cycles-samples "$PREVIEW_SAMPLES"
          --max-views "$MAX_VIEWS"
          --candidate-count "$CANDIDATE_COUNT"
        )
        if [[ -f "${render_dir}/collision_geometry_manifest.json" ]]; then
          common_blender_args+=(--collision-geometry "${render_dir}/collision_geometry_manifest.json")
        fi
        local resume_arg=--resume
        [[ "$RESUME" == "1" ]] || resume_arg=--no-resume
        log "[${stage} GPU ${gpu}] ${case_id} metric=${metric}"
        set +e
        if [[ "$stage" == "prepare_bank" ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
            "$BENCH_PY" scripts/run_p0b_two_phase.py prepare \
              --case-id "$case_id" \
              --scene "${fixture_dir}/generated_scene.json" \
              --source-report "${source_case}/evaluation_report.json" \
              --gt "${GT_ROOT}/${case_id}.json" \
              --metric "$metric" \
              --overview "${render_dir}/standardized_top.png" \
              --overview "${render_dir}/standardized_perspective.png" \
              --out-dir "$case_root" \
              --collision-overlay \
              --continue-on-error \
              "$resume_arg" \
              "${common_blender_args[@]}" >"$case_log" 2>&1
          local status=$?
          if (( status == 0 )); then
            CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
              "$BENCH_PY" scripts/run_p0b_visual_config_experiment.py prepare-bank \
                --case-root "$case_root" \
                --continue-on-error \
                "$resume_arg" \
                "${common_blender_args[@]}" >>"$case_log" 2>&1
            status=$?
          fi
        else
          CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
            "$BENCH_PY" scripts/run_p0b_visual_config_experiment.py finalize \
              --case-root "$case_root" \
              --continue-on-error \
              "$resume_arg" \
              "${common_blender_args[@]}" >"$case_log" 2>&1
          local status=$?
        fi
        set -e
        if (( status != 0 )); then
          worker_status=1
          echo "${stage} failed for ${case_id}; see ${case_log}" >&2
        fi
      done < <(tail -n +2 "$CASE_MANIFEST")
      exit "$worker_status"
    ) &
    worker_pids+=("$!")
  done
  BLENDER_WORKER_PIDS=("${worker_pids[@]}")
  local phase_status=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || phase_status=1
  done
  BLENDER_WORKER_PIDS=()
  ensure_no_blender
  if (( phase_status != 0 && ALLOW_PREP_ERRORS != 1 )); then
    echo "${stage} contains failures" >&2
    return 1
  fi
  log "${stage} complete"
}

run_selection_phase() {
  ensure_no_blender
  start_or_reuse_server selector
  if [[ "$PHASE" == "all" && "$SERVER_STARTED_BY_SCRIPT" != "1" ]]; then
    echo "PHASE=all cannot continue to Blender with an unowned reused server" >&2
    return 1
  fi
  multimodal_smoke selector
  local resume_arg=--resume
  [[ "$RESUME" == "1" ]] || resume_arg=--no-resume
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    "$BENCH_PY" scripts/run_p0b_visual_config_experiment.py select \
      --evidence-root "$EVIDENCE_ROOT" \
      --judge-config "$JUDGE_CONFIG" \
      --judge-endpoint "$ENDPOINT" \
      --judge-model "$SERVED_MODEL" \
      --max-workers "$SELECTOR_CONCURRENCY" \
      --continue-on-error \
      "$resume_arg" \
      2>&1 | tee "${OUT_ROOT}/phase_b_select.log"
  if [[ "$PHASE" == "all" ]]; then
    stop_owned_server 1
    ensure_no_server
  fi
}

validate_final_packets() {
  "$BENCH_PY" - "$CASE_MANIFEST" "$EVIDENCE_ROOT" "$ALLOW_PREP_ERRORS" "${ARMS[@]}" <<'PY'
import json, sys
from pathlib import Path
manifest, root = Path(sys.argv[1]), Path(sys.argv[2])
allow_errors = sys.argv[3] == "1"
arms = sys.argv[4:]
missing, failures = [], []
for line in manifest.read_text().splitlines()[1:]:
    if not line:
        continue
    case_id = line.split("\t", 1)[0]
    event_count = None
    prep = root / case_id / "preparation_manifest.json"
    if prep.is_file():
        event_count = int(json.load(prep.open()).get("event_count") or 0)
    if not event_count:
        missing.append(str(prep))
        continue
    for arm in arms:
        packets = list((root / case_id / arm / "evidence_packets").glob("*.json"))
        if len(packets) != event_count:
            missing.append(f"{case_id}/{arm}: expected {event_count}, found {len(packets)}")
        for path in packets:
            payload = json.load(path.open())
            if payload.get("preparation_error"):
                failures.append({"path": str(path), "error": payload["preparation_error"]})
if missing:
    raise SystemExit(f"final visual-config packet set incomplete: {missing}")
if failures and not allow_errors:
    raise SystemExit(f"prepared packet failures: {failures}")
print(f"visual-config packet contract OK; arms={len(arms)} failures={len(failures)}")
PY
}

run_judge_phase() {
  ensure_no_blender
  validate_final_packets
  start_or_reuse_server judge
  multimodal_smoke judge
  local judge_args=(
    judge
    --evidence-root "$EVIDENCE_ROOT"
    --judge-config "$JUDGE_CONFIG"
    --judge-endpoint "$ENDPOINT"
    --judge-model "$SERVED_MODEL"
    --out-dir "$ABLATION_ROOT"
    --max-workers "$JUDGE_CONCURRENCY"
    --continue-on-error
  )
  for arm in "${ARMS[@]}"; do
    judge_args+=(--arm "$arm")
  done
  [[ "$RESUME" == "1" ]] && judge_args+=(--resume) || judge_args+=(--no-resume)
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    "$BENCH_PY" scripts/run_p0b_two_phase.py "${judge_args[@]}" \
      2>&1 | tee "${OUT_ROOT}/phase_d_judge.log"

  while IFS=$'\t' read -r case_id base_case_id family metric fixture_dir; do
    "$BENCH_PY" scripts/score_p0b_camera_ablation.py \
      --gt "${GT_ROOT}/${case_id}.json" \
      --metric "$metric" \
      --run-dir "${ABLATION_ROOT}/${case_id}"
  done < <(tail -n +2 "$CASE_MANIFEST")

  local aggregate_args=(
    --case-manifest "$CASE_MANIFEST"
    --gt-root "$GT_ROOT"
    --ablation-root "$ABLATION_ROOT"
    --out-dir "$RESULT_ROOT"
  )
  for arm in "${ARMS[@]}"; do
    aggregate_args+=(--arm "$arm")
  done
  "$BENCH_PY" scripts/aggregate_p0b_visual_evidence_policy.py "${aggregate_args[@]}"
}

case "$PHASE" in
  all|prepare_bank|select|finalize|judge) ;;
  *) echo "PHASE must be all, prepare_bank, select, finalize, or judge; got ${PHASE}" >&2; exit 1 ;;
esac
if [[ "$TP_SIZE" != "8" || "$EP_SIZE" != "8" ]]; then
  echo "Qwen3-VL-235B FP8 requires the guarded TP8/EP8 profile" >&2
  exit 1
fi
if ! [[ "$MAX_CASES" =~ ^[0-9]+$ ]]; then
  echo "MAX_CASES must be a non-negative integer; got ${MAX_CASES}" >&2
  exit 1
fi
if (( 1536 % (TP_SIZE / EP_SIZE) != 0 || (1536 / (TP_SIZE / EP_SIZE)) % 128 != 0 )); then
  echo "Invalid block-wise FP8 MoE alignment for TP=${TP_SIZE}, EP=${EP_SIZE}" >&2
  exit 1
fi
if [[ "$SGL_ENABLE_JIT_DEEPGEMM" != "0" || "$SGLANG_ENABLE_JIT_DEEPGEMM" != "0" ]]; then
  echo "Both DeepGEMM JIT flags must remain 0 on the recorded CUDA/NVCC 12.2 image" >&2
  exit 1
fi
if [[ "$MAX_VIEWS" != "2" || "$CANDIDATE_COUNT" != "6" ]]; then
  echo "Controlled experiment freezes MAX_VIEWS=2 and CANDIDATE_COUNT=6" >&2
  exit 1
fi
if [[ "$CONTEXT_LENGTH" != "65536" || "$MEM_FRACTION_STATIC" != "0.80" ]]; then
  echo "Controlled 235B profile freezes CONTEXT_LENGTH=65536 and MEM_FRACTION_STATIC=0.80" >&2
  exit 1
fi
if [[ "$SELECTOR_CONCURRENCY" != "8" || "$JUDGE_CONCURRENCY" != "8" ]]; then
  echo "Controlled experiment freezes selector and judge request concurrency at 8" >&2
  exit 1
fi
if [[ "$RENDER_WIDTH" != "512" || "$RENDER_HEIGHT" != "512" || "$CYCLES_SAMPLES" != "8" \
  || "$PREVIEW_WIDTH" != "256" || "$PREVIEW_HEIGHT" != "256" || "$PREVIEW_SAMPLES" != "1" ]]; then
  echo "Controlled renderer profile is 512x512/8 samples and 256x256/1 preview sample" >&2
  exit 1
fi

for path in \
  "$REPO_ROOT" "$SGLANG_PY" "$BENCH_PY" "$BLENDER_BIN" "$MODEL_PATH" \
  "$JUDGE_CONFIG" "$CASE_FILE" "$SOURCE_ROOT" "$GT_ROOT" \
  "${MODEL_PATH}/config.json" \
  "${REPO_ROOT}/scripts/run_p0b_two_phase.py" \
  "${REPO_ROOT}/scripts/run_p0b_visual_config_experiment.py" \
  "${REPO_ROOT}/scripts/score_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/aggregate_p0b_visual_evidence_policy.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$EVIDENCE_ROOT" "$ABLATION_ROOT" "$RESULT_ROOT" "$PHASE_LOG_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"
validate_gpu_csv RENDER_GPUS "$RENDER_GPUS"
validate_gpu_csv SGLANG_CUDA_VISIBLE_DEVICES "$SGLANG_CUDA_VISIBLE_DEVICES"
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

checkpoint_count=$(find "$MODEL_PATH" -maxdepth 1 -type f -name '*.safetensors' | wc -l | tr -d ' ')
if (( checkpoint_count <= 0 )); then
  echo "No FP8 safetensors shards found under ${MODEL_PATH}" >&2
  exit 1
fi
echo "FP8 safetensors shards: $checkpoint_count"

"$BENCH_PY" - "$JUDGE_CONFIG" "$SERVED_MODEL" "$CANDIDATE_COUNT" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
if config.get("model") != sys.argv[2]:
    raise SystemExit(f"judge config model {config.get('model')!r} != served model {sys.argv[2]!r}")
if int(config.get("max_images") or 0) != int(sys.argv[3]):
    raise SystemExit("controlled judge config requires max_images == candidate_count == 6")
if float(config.get("temperature") or 0) != 0:
    raise SystemExit("controlled experiment requires temperature=0")
print("235B model/judge visual contract: OK")
PY

build_case_manifest

"$BENCH_PY" - "${OUT_ROOT}/experiment_contract.json" "$SERVED_MODEL" "$CASE_MANIFEST" "${ARMS[@]}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path, model, manifest, *arms = sys.argv[1:]
manifest_path = Path(manifest)
payload = {
    "schema_version": "p0b_qwen235b_visual_config_contract_v1",
    "controlled_variable": "visual_config",
    "model": model,
    "tp_size": 8,
    "ep_size": 8,
    "context_length": 65536,
    "deepgemm_jit": False,
    "physical_phases": [
        "blender_evidence_bank",
        "qwen235b_candidate_id_selection",
        "blender_selected_final_render",
        "qwen235b_frozen_packet_judgement",
    ],
    "arms": arms,
    "excluded_arm": "active_metric_local",
    "presence_matrix": {
        "presence_local_raw": "local raw; no global",
        "presence_local_raw_highlight": "local raw plus same-pose highlight; no global",
        "presence_global_local_raw": "highlighted global plus local raw",
        "deterministic_metric_local": "highlighted global plus two local raw/highlight pairs"
    },
    "order_budget_checks": {
        "order_local_first_full": "same five images, local pairs before global",
        "budget_global_first_compact": "global plus one local raw/highlight pair",
        "budget_local_first_compact": "one local raw/highlight pair plus global"
    },
    "camera_policy_checks": {
        "fixed_global": "two frozen raw overview images",
        "deterministic_metric_local": "deterministic metric-aware selector",
        "vlm_select_from_candidates": "VLM chooses up to two IDs from six frozen previews"
    },
    "candidate_count": 6,
    "max_local_views": 2,
    "camera_adjustment_allowed": False,
    "frozen": [
        "GT", "physical event", "detector evidence", "prompt", "relations",
        "canonical scene", "scene.blend", "renderer", "final judge model",
    ],
    "case_manifest": str(manifest_path),
    "case_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "generator_retriever_converter_detector_rerun": False,
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY

case "$PHASE" in
  all)
    run_blender_workers prepare_bank
    run_selection_phase
    run_blender_workers finalize
    run_judge_phase
    ;;
  prepare_bank)
    run_blender_workers prepare_bank
    ;;
  select)
    run_selection_phase
    ;;
  finalize)
    run_blender_workers finalize
    ;;
  judge)
    run_judge_phase
    ;;
esac

log "Qwen3-VL-235B visual-config experiment complete"
echo "Output: $OUT_ROOT"
echo "Prepared evidence: $EVIDENCE_ROOT"
echo "Policy summary: $RESULT_ROOT/policy_summary.tsv"
echo "Paired transitions: $RESULT_ROOT/paired_transition_summary.tsv"
echo "Master event table: $RESULT_ROOT/master_event_policy_table.tsv"
echo "Offline mixed camera policies: $RESULT_ROOT/mixed_camera_policy_candidates.tsv"
echo "Selector server log: $LOG_ROOT/sglang_qwen3vl_235b_fp8_tp8_ep8_${CONTEXT_LENGTH}_${PORT}_selector.out"
echo "Judge server log: $LOG_ROOT/sglang_qwen3vl_235b_fp8_tp8_ep8_${CONTEXT_LENGTH}_${PORT}_judge.out"
