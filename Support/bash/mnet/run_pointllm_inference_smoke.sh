#!/usr/bin/env bash
set -Eeuo pipefail

# STAGE 3 of 3 - real point-cloud inference.
#
# Loads each PointLLM checkpoint on one free GPU and requires two different
# real point clouds to produce two different answers. Stage 1 (bytes on disk)
# and stage 2 (environment imports) must already pass; this script re-runs the
# cheap size-only form of stage 1 as a precondition so a half-deleted
# checkpoint cannot be misread as a model failure.
#
# GPU memory being in use does not mean a PointLLM process is running. The
# preflight prints what actually holds each GPU so an unrelated training job or
# a stale process is never mistaken for this run.
#
# PointLLM is a 7B dense model in bf16, roughly 14 GB. It needs one GPU. The
# TP8/EP8 and DeepGEMM constraints that govern the Qwen3-VL-235B FP8 MoE
# checkpoint do not apply here, and neither does the SGLang serving path:
# `model_type: pointllm` is a custom architecture that SGLang and vLLM cannot
# load, so there is no /v1/models endpoint to check.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
MODELS_ROOT=${MODELS_ROOT:-/mnt/group/cmh/models}
REGISTRY=${REGISTRY:-${REPO_ROOT}/configs/models/pointllm_mnet_registry.json}

ENV_PY=${ENV_PY:-/mnt/group/cmh/envs/pointllm/bin/python}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
POINTLLM_DIR=${POINTLLM_DIR:-/mnt/group/cmh/tools/PointLLM}
POINTLLM_R_DIR=${POINTLLM_R_DIR:-/mnt/group/cmh/tools/PointLLM-R}

MODELNET_DAT=${MODELNET_DAT:-${MODELS_ROOT}/pointllm_data/modelnet40_data/modelnet40_test_8192pts_fps.dat}
MODELNET_INDEX_A=${MODELNET_INDEX_A:-0}
MODELNET_INDEX_B=${MODELNET_INDEX_B:-1234}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
PROMPT=${PROMPT:-What is this?}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

GPU=${GPU:-}
MIN_FREE_MIB=${MIN_FREE_MIB:-40000}

RUN_TAG=${RUN_TAG:-pointllm_smoke_$(date '+%Y%m%d_%H%M%S')}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}

log() {
  echo "==== $(date '+%F %T') $* ===="
}

fail() {
  echo "$*" >&2
  exit 1
}

for path in "$REPO_ROOT" "$REGISTRY" "$ENV_PY" "$BENCH_PY" "$POINTLLM_DIR" "$POINTLLM_R_DIR"; do
  [[ -e "$path" ]] || fail "Missing required path: $path"
done
[[ -f "$MODELNET_DAT" ]] || fail "Missing ModelNet test split: $MODELNET_DAT (stage 1 downloads it)"

mkdir -p "$OUT_ROOT"
cd "$REPO_ROOT"

log "stage 3a: re-check stage 1 (sizes only)"
"$BENCH_PY" scripts/verify_pointllm_checkpoint.py \
  --registry "$REGISTRY" --models-root "$MODELS_ROOT" --sha256 skip \
  || fail "Stage 1 no longer passes; fix the checkpoint before interpreting any inference result"

log "stage 3b: what is actually using the GPUs"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv
echo "--- processes holding GPU memory ---"
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv || true
echo "--- competing workloads ---"
pgrep -af 'blender|sglang.launch_server|run_p0b|run_scene_harness' || echo "none found"

# Blender and a resident model must not share the node; the two-phase contract
# exists so evidence generation and model inference never contend for memory.
if pgrep -af 'blender' >/dev/null 2>&1; then
  fail "Blender workers are still running. Stop them before loading a model (two-phase contract)."
fi

if [[ -z "$GPU" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' -v need="$MIN_FREE_MIB" '$2 >= need {print $1; exit}')
  [[ -n "$GPU" ]] || fail "No GPU has ${MIN_FREE_MIB} MiB free. Something else owns them; do not assume it is ours."
fi
log "selected GPU ${GPU}"

run_one() {
  local key=$1 code_dir=$2
  log "stage 3c: real inference for ${key} (code: ${code_dir})"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$code_dir" "$ENV_PY" \
    scripts/smoke_pointllm_inference.py \
    --registry "$REGISTRY" \
    --model-key "$key" \
    --model-dir "${MODELS_ROOT}/${key}" \
    --modelnet-dat "$MODELNET_DAT" \
    --modelnet-index "$MODELNET_INDEX_A" \
    --modelnet-index "$MODELNET_INDEX_B" \
    --prompt "$PROMPT" \
    --torch-dtype "$TORCH_DTYPE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --json-out "${OUT_ROOT}/stage3_${key}.json" \
    2>&1 | tee "${OUT_ROOT}/stage3_${key}.log"
  return "${PIPESTATUS[0]}"
}

run_one "PointLLM_7B_v1.2" "$POINTLLM_DIR"
run_one "PointLLM-R-7B" "$POINTLLM_R_DIR"

log "stage 3 complete"
echo "records: ${OUT_ROOT}"
ls -1 "${OUT_ROOT}"
