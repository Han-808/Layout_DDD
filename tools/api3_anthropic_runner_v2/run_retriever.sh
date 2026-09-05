#!/usr/bin/env bash
set -euo pipefail

RUNTIME_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec /root/miniconda3/envs/lf/bin/python \
  "$RUNTIME_ROOT/retriever_runtime.py" \
  --config "$RUNTIME_ROOT/retriever_runtime.pod.json" \
  "$@"
