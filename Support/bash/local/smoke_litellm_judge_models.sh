#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
ENDPOINT="${LITELLM_ENDPOINT:-http://127.0.0.1:4000/v1}"

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "Required environment variable is not set: LITELLM_MASTER_KEY" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Layout_DDD Python is missing: $PYTHON" >&2
  exit 2
fi

cd "$REPO_ROOT"
for model in claude-fable-5 gpt-5.6-sol hy3; do
  echo "==== multimodal smoke: $model ===="
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint "$ENDPOINT" \
    --model "$model" \
    --api-key-env LITELLM_MASTER_KEY \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal
done
