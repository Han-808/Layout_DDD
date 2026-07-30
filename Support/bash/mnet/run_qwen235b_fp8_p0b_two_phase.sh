#!/usr/bin/env bash
set -Eeuo pipefail

# Two-phase P0b replay for one 8 x H20 node.
#
# Phase A (prepare): no SGLang process; eight independent Blender workers
# generate and freeze fixed-global and deterministic metric-local evidence.
# Phase B (judge): all Blender workers are gone; one Qwen3-VL-235B TP8/EP8
# SGLang server judges the prepared packets with bounded request concurrency.

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

PHASE=${PHASE:-all} # all | prepare | judge
RUN_TAG=${RUN_TAG:-qwen235b_fp8_p0b_two_phase_$(date '+%Y%m%d_%H%M%S')}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
EVIDENCE_ROOT=${EVIDENCE_ROOT:-${OUT_ROOT}/prepared_evidence}
ABLATION_ROOT=${ABLATION_ROOT:-${OUT_ROOT}/ablation}
RESULT_ROOT=${RESULT_ROOT:-${OUT_ROOT}/results}
CASE_MANIFEST=${CASE_MANIFEST:-${OUT_ROOT}/case_manifest.tsv}
PHASE_A_LOG_ROOT=${PHASE_A_LOG_ROOT:-${OUT_ROOT}/phase_a_logs}
LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}

PORT=${PORT:-8298}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT}/v1}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
TP_SIZE=${TP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
SGLANG_CUDA_VISIBLE_DEVICES=${SGLANG_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SGL_ENABLE_JIT_DEEPGEMM=${SGL_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
RENDER_GPUS=${RENDER_GPUS:-0,1,2,3,4,5,6,7}
JUDGE_CONCURRENCY=${JUDGE_CONCURRENCY:-8}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-7200}
STOP_SERVER_ON_EXIT=${STOP_SERVER_ON_EXIT:-1}
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

SERVER_LOG=${SERVER_LOG:-${LOG_ROOT}/sglang_qwen3vl_235b_fp8_tp${TP_SIZE}_ep${EP_SIZE}_${CONTEXT_LENGTH}_${PORT}.out}
PID_FILE=${PID_FILE:-${LOG_ROOT}/sglang_qwen3vl_235b_fp8_tp${TP_SIZE}_ep${EP_SIZE}_${CONTEXT_LENGTH}_${PORT}.pid}
SERVER_STARTED_BY_SCRIPT=0
SERVER_PID=""

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

stop_owned_server() {
  if [[ "$SERVER_STARTED_BY_SCRIPT" != "1" || "$STOP_SERVER_ON_EXIT" != "1" ]]; then
    return
  fi
  log "stopping owned Qwen3-VL-235B server pid=${SERVER_PID}"
  curl --noproxy "*" -fsS -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null 2>&1 || true
  descendants() {
    local parent=$1 child
    for child in $(pgrep -P "$parent" 2>/dev/null); do
      descendants "$child"
    done
    echo "$parent"
  }
  server_pids=$(descendants "$SERVER_PID" | sort -u | tr '\n' ' ')
  kill -TERM $server_pids 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  for pid in $server_pids; do
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  done
  rm -f "$PID_FILE"
}

trap stop_owned_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$PHASE" != "all" && "$PHASE" != "prepare" && "$PHASE" != "judge" ]]; then
  echo "PHASE must be all, prepare, or judge; got ${PHASE}" >&2
  exit 1
fi
if [[ "$TP_SIZE" != "8" || "$EP_SIZE" != "8" ]]; then
  echo "This launcher reserves all eight GPUs for Qwen3-VL-235B and requires TP_SIZE=8, EP_SIZE=8" >&2
  exit 1
fi
if (( TP_SIZE % EP_SIZE != 0 )); then
  echo "EP_SIZE=${EP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
  exit 1
fi
if ! [[ "$MAX_CASES" =~ ^[0-9]+$ ]]; then
  echo "MAX_CASES must be a non-negative integer; got ${MAX_CASES}" >&2
  exit 1
fi

validate_gpu_csv() {
  local label=$1 value=$2
  "$BENCH_PY" - "$label" "$value" <<'PY'
import sys

label, raw = sys.argv[1:]
values = [item.strip() for item in raw.split(",") if item.strip()]
if len(values) != 8 or len(set(values)) != 8:
    raise SystemExit(f"{label} must contain exactly eight unique GPU IDs; got {raw!r}")
if any(not item.isdigit() for item in values):
    raise SystemExit(f"{label} contains a non-numeric GPU ID: {raw!r}")
print(f"{label}: {','.join(values)}")
PY
}
moe_tp_size=$((TP_SIZE / EP_SIZE))
if (( 1536 % moe_tp_size != 0 || (1536 / moe_tp_size) % 128 != 0 )); then
  echo "Invalid Qwen3-VL FP8 alignment for TP=${TP_SIZE}, EP=${EP_SIZE}" >&2
  exit 1
fi

for path in \
  "$REPO_ROOT" "$SGLANG_PY" "$BENCH_PY" "$BLENDER_BIN" "$MODEL_PATH" \
  "$JUDGE_CONFIG" "$CASE_FILE" "$SOURCE_ROOT" "$GT_ROOT" \
  "${MODEL_PATH}/config.json" \
  "${REPO_ROOT}/scripts/run_p0b_two_phase.py" \
  "${REPO_ROOT}/scripts/score_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/aggregate_p0b_visual_evidence_policy.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$EVIDENCE_ROOT" "$ABLATION_ROOT" "$RESULT_ROOT" \
  "$PHASE_A_LOG_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"
validate_gpu_csv RENDER_GPUS "$RENDER_GPUS"
validate_gpu_csv SGLANG_CUDA_VISIBLE_DEVICES "$SGLANG_CUDA_VISIBLE_DEVICES"

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

log "build frozen case manifest"
"$BENCH_PY" - "$CASE_FILE" "$FIXTURE_ROOT" "$SOURCE_ROOT" "$GT_ROOT" "$FAMILIES" "$MAX_CASES" >"$CASE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

case_file = Path(sys.argv[1]).resolve()
fixture_root = Path(sys.argv[2]).resolve()
source_root = Path(sys.argv[3]).resolve()
gt_root = Path(sys.argv[4]).resolve()
allowed = {value.strip() for value in sys.argv[5].split(",") if value.strip()}
max_cases = int(sys.argv[6])
family_metric = {"clean": "support", "collision": "collision", "oob": "oob", "support": "support"}
payload = json.load(case_file.open(encoding="utf-8"))
print("case_id\tbase_case_id\tfamily\tmetric\tfixture_dir")
selected = 0
for case in payload.get("cases") or []:
    family = str(case.get("family") or "")
    if family not in allowed or family not in family_metric:
        continue
    case_id = str(case["case_id"])
    fixture_dir = (fixture_root / str(case["fixture_dir"])).resolve()
    required = [
        fixture_dir / "generated_scene.json",
        source_root / case_id / "evaluation_report.json",
        source_root / case_id / "renders" / "scene.blend",
        source_root / case_id / "renders" / "standardized_top.png",
        source_root / case_id / "renders" / "standardized_perspective.png",
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

case_count=$(( $(wc -l <"$CASE_MANIFEST") - 1 ))
if (( case_count <= 0 )); then
  echo "No replay cases selected" >&2
  exit 1
fi
echo "selected cases: $case_count"

run_prepare_phase() {
  log "Phase A: eight-GPU Blender evidence preparation; SGLang must be absent"
  if curl --noproxy "*" -fsS "${ENDPOINT}/models" >/dev/null 2>&1; then
    echo "A model server is already alive at ${ENDPOINT}; stop it before Phase A" >&2
    return 1
  fi
  if pgrep -af 'sglang.*launch_server' >/tmp/p0b_existing_sglang.txt 2>/dev/null; then
    echo "An SGLang launch_server process is already using this pod:" >&2
    cat /tmp/p0b_existing_sglang.txt >&2
    return 1
  fi

  IFS=',' read -r -a render_gpu_list <<<"$RENDER_GPUS"
  if (( ${#render_gpu_list[@]} != 8 )); then
    echo "Phase A requires exactly eight RENDER_GPUS; got ${RENDER_GPUS}" >&2
    return 1
  fi

  prepare_worker() {
    local worker_index=$1
    local gpu=$2
    local line_number=0
    local worker_status=0
    while IFS=$'\t' read -r case_id base_case_id family metric fixture_dir; do
      line_number=$((line_number + 1))
      if (( (line_number - 1) % 8 != worker_index )); then
        continue
      fi
      local source_case=${SOURCE_ROOT}/${case_id}
      local render_dir=${source_case}/renders
      local case_out=${EVIDENCE_ROOT}/${case_id}
      local case_log=${PHASE_A_LOG_ROOT}/${case_id}.gpu${gpu}.log
      local args=(
        prepare
        --case-id "$case_id"
        --scene "${fixture_dir}/generated_scene.json"
        --source-report "${source_case}/evaluation_report.json"
        --gt "${GT_ROOT}/${case_id}.json"
        --metric "$metric"
        --blend-file "${render_dir}/scene.blend"
        --overview "${render_dir}/standardized_top.png"
        --overview "${render_dir}/standardized_perspective.png"
        --out-dir "$case_out"
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
        --collision-overlay
        --continue-on-error
      )
      if [[ -f "${render_dir}/collision_geometry_manifest.json" ]]; then
        args+=(--collision-geometry "${render_dir}/collision_geometry_manifest.json")
      fi
      if [[ "$RESUME" == "1" ]]; then
        args+=(--resume)
      else
        args+=(--no-resume)
      fi
      log "[Phase A GPU ${gpu}] prepare ${case_id} metric=${metric}"
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
        "$BENCH_PY" scripts/run_p0b_two_phase.py "${args[@]}" >"$case_log" 2>&1
      local case_status=$?
      set -e
      if (( case_status != 0 )); then
        worker_status=1
        echo "Phase A failed for ${case_id}; see ${case_log}" >&2
      fi
    done < <(tail -n +2 "$CASE_MANIFEST")
    return "$worker_status"
  }

  local worker_pids=()
  for worker_index in $(seq 0 7); do
    prepare_worker "$worker_index" "${render_gpu_list[$worker_index]}" &
    worker_pids+=("$!")
  done
  local phase_status=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || phase_status=1
  done
  if pgrep -af '[b]lender.*background' >/tmp/p0b_lingering_blender.txt 2>/dev/null; then
    echo "Blender processes remain after Phase A:" >&2
    cat /tmp/p0b_lingering_blender.txt >&2
    return 1
  fi
  if (( phase_status != 0 && ALLOW_PREP_ERRORS != 1 )); then
    echo "Phase A has preparation failures; Phase B was not started" >&2
    return 1
  fi
  log "Phase A complete"
}

start_or_reuse_server() {
  if curl --noproxy "*" -fsS "${ENDPOINT}/models" >/tmp/qwen235b_models.json 2>/dev/null; then
    "$BENCH_PY" - "$SERVED_MODEL" /tmp/qwen235b_models.json <<'PY'
import json, sys
with open(sys.argv[2], encoding="utf-8") as handle:
    models = {str(item.get("id")) for item in json.load(handle).get("data", [])}
if sys.argv[1] not in models:
    raise SystemExit(f"port serves {sorted(models)}, expected {sys.argv[1]!r}")
print("reusing live server:", sys.argv[1])
PY
    return
  fi

  log "Phase B: launch Qwen3-VL-235B TP${TP_SIZE}/EP${EP_SIZE} on all GPUs"
  export CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES"
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # The MNET image exposes CUDA/NVCC 12.2. DeepGEMM's JIT compiler requires
  # NVCC >= 12.3, so use SGLang's non-JIT FP8 MoE fallback on this node.
  export SGL_ENABLE_JIT_DEEPGEMM
  export SGLANG_ENABLE_JIT_DEEPGEMM
  echo "SGL_ENABLE_JIT_DEEPGEMM=${SGL_ENABLE_JIT_DEEPGEMM}"
  echo "SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM}"
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
  until curl --noproxy "*" -fsS "${ENDPOINT}/models" >/tmp/qwen235b_models.json 2>/dev/null; do
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
}

run_judge_phase() {
  log "validate prepared evidence before allocating all GPUs to SGLang"
  "$BENCH_PY" - "$CASE_MANIFEST" "$EVIDENCE_ROOT" "$ALLOW_PREP_ERRORS" <<'PY'
import json, sys
from pathlib import Path

case_manifest = Path(sys.argv[1])
evidence_root = Path(sys.argv[2])
allow_errors = sys.argv[3] == "1"
cases = [line.split("\t", 1)[0] for line in case_manifest.read_text().splitlines()[1:] if line]
missing, failures = [], []
for case_id in cases:
    path = evidence_root / case_id / "preparation_manifest.json"
    if not path.is_file():
        missing.append(str(path))
        continue
    manifest = json.load(path.open())
    if int(manifest.get("failure_count") or 0):
        failures.append({"case_id": case_id, "failures": manifest.get("failures")})
if missing:
    raise SystemExit(f"prepared evidence manifests missing: {missing}")
if failures and not allow_errors:
    raise SystemExit(f"prepared evidence contains failures: {failures}")
print(f"prepared cases: {len(cases)}; failures: {len(failures)}")
PY

  start_or_reuse_server
  log "verify model identity and flush once"
  "$BENCH_PY" - "$SERVED_MODEL" /tmp/qwen235b_models.json <<'PY'
import json, sys
with open(sys.argv[2], encoding="utf-8") as handle:
    models = {str(item.get("id")) for item in json.load(handle).get("data", [])}
if sys.argv[1] not in models:
    raise SystemExit(f"served model mismatch: {sorted(models)}")
print("served model:", sys.argv[1])
PY

  log "verify fallback kernel with a real multimodal completion"
  SMOKE_IMAGE=$(find "$EVIDENCE_ROOT" -type f -name '*.png' -print -quit)
  require_path "$SMOKE_IMAGE"
  "$BENCH_PY" - "$ENDPOINT" "$SERVED_MODEL" "$SMOKE_IMAGE" <<'PY'
import base64
import json
import sys
import urllib.request
from pathlib import Path

endpoint, model, image_path = sys.argv[1:]
encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
payload = {
    "model": model,
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Reply with exactly: visual-alive"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ],
    }],
    "temperature": 0,
    "max_tokens": 16,
}
request = urllib.request.Request(
    f"{endpoint}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
content = str(result["choices"][0]["message"].get("content") or "").strip()
if "visual-alive" not in content.lower():
    raise SystemExit(f"multimodal fallback-kernel smoke failed: {content!r}")
print("multimodal response:", content)
PY

  curl --noproxy "*" -fsS -X POST "http://127.0.0.1:${PORT}/flush_cache" >/dev/null || true

  log "judge frozen packets with concurrency=${JUDGE_CONCURRENCY}; Blender is disabled"
  judge_args=(
    judge
    --evidence-root "$EVIDENCE_ROOT"
    --judge-config "$JUDGE_CONFIG"
    --judge-endpoint "$ENDPOINT"
    --judge-model "$SERVED_MODEL"
    --out-dir "$ABLATION_ROOT"
    --max-workers "$JUDGE_CONCURRENCY"
    --continue-on-error
  )
  if [[ "$RESUME" == "1" ]]; then
    judge_args+=(--resume)
  else
    judge_args+=(--no-resume)
  fi
  CUDA_VISIBLE_DEVICES="$SGLANG_CUDA_VISIBLE_DEVICES" \
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    "$BENCH_PY" scripts/run_p0b_two_phase.py "${judge_args[@]}" \
    2>&1 | tee "${OUT_ROOT}/phase_b_judge.log"

  log "score each frozen case"
  while IFS=$'\t' read -r case_id base_case_id family metric fixture_dir; do
    "$BENCH_PY" scripts/score_p0b_camera_ablation.py \
      --gt "${GT_ROOT}/${case_id}.json" \
      --metric "$metric" \
      --run-dir "${ABLATION_ROOT}/${case_id}"
  done < <(tail -n +2 "$CASE_MANIFEST")

  log "aggregate fixed-global versus deterministic-local"
  "$BENCH_PY" scripts/aggregate_p0b_visual_evidence_policy.py \
    --case-manifest "$CASE_MANIFEST" \
    --gt-root "$GT_ROOT" \
    --ablation-root "$ABLATION_ROOT" \
    --out-dir "$RESULT_ROOT" \
    --arm fixed_global \
    --arm deterministic_metric_local
}

cat >"${OUT_ROOT}/experiment_contract.json" <<JSON
{
  "schema_version": "p0b_qwen235b_two_phase_contract_v1",
  "phase_a": "eight GPU Blender-only frozen evidence preparation",
  "phase_b": "eight GPU Qwen3-VL-235B TP8/EP8 concurrent judgement",
  "model": "${SERVED_MODEL}",
  "context_length": ${CONTEXT_LENGTH},
  "tp_size": ${TP_SIZE},
  "ep_size": ${EP_SIZE},
  "sgl_enable_jit_deepgemm": "${SGL_ENABLE_JIT_DEEPGEMM}",
  "sglang_enable_jit_deepgemm": "${SGLANG_ENABLE_JIT_DEEPGEMM}",
  "render_gpus": "${RENDER_GPUS}",
  "judge_concurrency": ${JUDGE_CONCURRENCY},
  "max_cases": ${MAX_CASES},
  "arms": ["fixed_global", "deterministic_metric_local"],
  "generator_retriever_converter_detector_rerun": false
}
JSON

if [[ "$PHASE" == "all" || "$PHASE" == "prepare" ]]; then
  run_prepare_phase
fi
if [[ "$PHASE" == "all" || "$PHASE" == "judge" ]]; then
  run_judge_phase
fi

log "two-phase P0b replay complete"
echo "Output: $OUT_ROOT"
echo "Prepared evidence: $EVIDENCE_ROOT"
echo "Results: $RESULT_ROOT/policy_summary.tsv"
