#!/usr/bin/env bash
set -Eeuo pipefail

# Five simple CSV-grounded prompts for a current P0a/P0b/P0c smoke test.
# Missing room dimensions intentionally exercise the benchmark 7m x 5m x 3m
# fallback before retrieval, generation, rendering, and evaluation.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}

export PROMPT_FILE=${PROMPT_FILE:-${REPO_ROOT}/configs/experiments/asset_grounded_simple_smoke5_prompts.json}
export RUN_TAG=${RUN_TAG:-qwen32b_asset_grounded_simple_smoke5_$(date +%Y%m%d_%H%M%S)}
export OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}

export MAX_CASES=${MAX_CASES:-0}
export RESUME=${RESUME:-1}
export CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
export FLUSH_CACHE=${FLUSH_CACHE:-1}

export BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
export BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CUDA}
export BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-32}
export BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-1}
export BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-1800}
export RENDER_WIDTH=${RENDER_WIDTH:-768}
export RENDER_HEIGHT=${RENDER_HEIGHT:-768}

exec bash "${REPO_ROOT}/Support/bash/mnet/run_qwen32b_asset_blender_smoke.sh"
