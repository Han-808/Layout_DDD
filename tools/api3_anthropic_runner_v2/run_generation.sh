#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec "$REPO_ROOT/.venv/bin/python" \
  "$RUNNER_ROOT/generation_runner.py" \
  --briefs "$RUNNER_ROOT/briefs.json" \
  --models "$RUNNER_ROOT/models.pod.json" \
  "$@"
