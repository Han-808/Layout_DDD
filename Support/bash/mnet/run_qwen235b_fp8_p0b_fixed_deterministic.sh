#!/usr/bin/env bash
set -Eeuo pipefail

# Start the downloaded Qwen3-VL-235B-A22B FP8 checkpoint, validate its
# OpenAI-compatible multimodal endpoint, and replay the frozen P0b event
# universe with only fixed-global and deterministic metric-local evidence.
# Generator, retrieval, conversion, detector recomputation, and query_cov are
# deliberately excluded.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
SGLANG_PY=${SGLANG_PY:-/mnt/group/cmh/envs/sglang-qwen3vl/bin/python}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
MODEL_PATH=${MODEL_PATH:-/mnt/group/cmh/models/Qwen3-VL-235B-A22B-Instruct-FP8}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-235B-A22B-Instruct-FP8}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_235b_fp8_mnet_judge.json}

PORT=${PORT:-8298}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:${PORT}/v1}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.80}
TP_SIZE=${TP_SIZE:-8}
EP_SIZE=${EP_SIZE:-8}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
SGL_ENABLE_JIT_DEEPGEMM=${SGL_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
BLENDER_CUDA_VISIBLE_DEVICES=${BLENDER_CUDA_VISIBLE_DEVICES:-7}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-7200}
STOP_SERVER_ON_EXIT=${STOP_SERVER_ON_EXIT:-1}

RUN_TAG=${RUN_TAG:-qwen235b_fp8_p0b_fixed_deterministic_$(date '+%Y%m%d_%H%M%S')}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}
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
  curl --noproxy "*" -fsS -X POST \
    "http://127.0.0.1:${PORT}/flush_cache" >/dev/null 2>&1 || true
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

for path in \
  "$REPO_ROOT" "$SGLANG_PY" "$BENCH_PY" "$MODEL_PATH" "$JUDGE_CONFIG" \
  "${MODEL_PATH}/config.json" \
  "${REPO_ROOT}/Support/bash/mnet/run_p0b_visual_evidence_policy.sh"; do
  require_path "$path"
done

mkdir -p "$LOG_ROOT" "$OUT_ROOT"
cd "$REPO_ROOT"

log "preflight: downloaded 235B FP8 checkpoint and camera contract"
shard_count=$(find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' -type f | wc -l | tr -d ' ')
if (( shard_count == 0 )); then
  echo "No safetensors shards found under $MODEL_PATH" >&2
  exit 1
fi
echo "model shards: $shard_count"
du -sh "$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

# Qwen3-VL MoE FP8 weights use 128-column quantization blocks. SGLang
# tensor-parallelizes each expert over TP/EP ranks, so the resulting shard must
# retain an aligned intermediate dimension.
if (( TP_SIZE % EP_SIZE != 0 )); then
  echo "EP_SIZE=${EP_SIZE} must divide TP_SIZE=${TP_SIZE}" >&2
  exit 1
fi
moe_tp_size=$((TP_SIZE / EP_SIZE))
moe_intermediate_size=1536
weight_block_size_n=128
if (( moe_intermediate_size % moe_tp_size != 0 || (moe_intermediate_size / moe_tp_size) % weight_block_size_n != 0 )); then
  echo "Invalid Qwen3-VL FP8 parallelism: (1536 / (TP_SIZE/EP_SIZE)) must be divisible by 128; TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE}" >&2
  exit 1
fi
echo "MoE FP8 parallelism: TP_SIZE=${TP_SIZE}, EP_SIZE=${EP_SIZE}, moe_tp_size=${moe_tp_size}"

PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" -c \
  "from benchmark.rendering.camera_pose import CAMERA_POSE_MODES,resolve_camera_pose_mode; assert 'support_contact_plane' in CAMERA_POSE_MODES; assert resolve_camera_pose_mode('auto','support',metric_modes={'support':'support_contact_plane'}) == 'support_contact_plane'; print('camera policy contract: OK')"

if curl --noproxy "*" -fsS "${ENDPOINT}/models" >/tmp/qwen235b_models.json 2>/dev/null; then
  "$BENCH_PY" - "$SERVED_MODEL" /tmp/qwen235b_models.json <<'PY'
import json, sys
with open(sys.argv[2], encoding="utf-8") as handle:
    payload = json.load(handle)
expected = sys.argv[1]
models = {str(item.get("id")) for item in payload.get("data", [])}
if expected not in models:
    raise SystemExit(f"port already serves {sorted(models)}, expected {expected!r}")
print("reusing live server:", expected)
PY
else
  log "launch Qwen3-VL-235B-A22B-Instruct-FP8 TP${TP_SIZE} EP${EP_SIZE} context=${CONTEXT_LENGTH}"
  export CUDA_VISIBLE_DEVICES
  export PYTHONUNBUFFERED=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  # MNET exposes CUDA/NVCC 12.2. DeepGEMM JIT requires NVCC >= 12.3, so use
  # SGLang's non-JIT FP8 MoE fallback and verify it with an image request below.
  export SGL_ENABLE_JIT_DEEPGEMM
  export SGLANG_ENABLE_JIT_DEEPGEMM
  export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
  export no_proxy=${no_proxy:-127.0.0.1,localhost}
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

  deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until curl --noproxy "*" -fsS "${ENDPOINT}/models" >/tmp/qwen235b_models.json 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Qwen3-VL-235B server exited before readiness" >&2
      tail -n 200 "$SERVER_LOG" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Qwen3-VL-235B startup exceeded ${STARTUP_TIMEOUT_SECONDS}s" >&2
      tail -n 200 "$SERVER_LOG" >&2
      exit 1
    fi
    sleep 15
  done
fi

log "verify served model identity"
"$BENCH_PY" - "$SERVED_MODEL" /tmp/qwen235b_models.json <<'PY'
import json, sys
with open(sys.argv[2], encoding="utf-8") as handle:
    payload = json.load(handle)
expected = sys.argv[1]
models = {str(item.get("id")) for item in payload.get("data", [])}
if expected not in models:
    raise SystemExit(f"served model mismatch: {sorted(models)} != {expected!r}")
print(json.dumps(payload))
PY

log "multimodal endpoint smoke test"
SMOKE_IMAGE=$(find \
  "${REPO_ROOT}/outputs/p0b_combined_20260717_202533/source_distortion5/source_reports" \
  -name standardized_top.png -type f -print -quit)
require_path "$SMOKE_IMAGE"
"$BENCH_PY" - "$ENDPOINT" "$SERVED_MODEL" "$SMOKE_IMAGE" <<'PY'
import base64, json, sys, urllib.request
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

log "run frozen P0b replay: fixed global + deterministic local only"
ARMS_CSV=fixed_global,deterministic_metric_local \
CUDA_VISIBLE_DEVICES="$BLENDER_CUDA_VISIBLE_DEVICES" \
JUDGE_CONFIG="$JUDGE_CONFIG" \
SERVED_MODEL="$SERVED_MODEL" \
PORT="$PORT" \
ENDPOINT="$ENDPOINT" \
RUN_TAG="$RUN_TAG" \
OUT_ROOT="$OUT_ROOT" \
RESUME=1 \
bash Support/bash/mnet/run_p0b_visual_evidence_policy.sh

log "235B fixed/deterministic replay complete"
echo "Output: $OUT_ROOT"
echo "Summary: $OUT_ROOT/results/policy_summary.tsv"
echo "Paired transitions: $OUT_ROOT/results/paired_transition_summary.tsv"
