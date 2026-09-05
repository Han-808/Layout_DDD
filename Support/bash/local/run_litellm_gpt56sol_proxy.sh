#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LITELLM_ROOT="$REPO_ROOT/Support/third_party/uni_llm_hhr_cursor_snapshot"
CONFIG="$REPO_ROOT/configs/inference/litellm_gpt56sol_local.yaml"
HOST="${LITELLM_HOST:-127.0.0.1}"
PORT="${LITELLM_GPT56_PORT:-4010}"

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]]; then
  echo "Refusing to expose the local judge proxy on non-loopback host: $HOST" >&2
  exit 2
fi

for variable_name in OPENAPI_BASE_URL OPENAPI_GPT_KEY LITELLM_MASTER_KEY; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Required environment variable is not set: $variable_name" >&2
    exit 2
  fi
done

if [[ ! -d "$LITELLM_ROOT" ]]; then
  echo "Cloned LiteLLM repository is missing: $LITELLM_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "LiteLLM config is missing: $CONFIG" >&2
  exit 2
fi
if [[ ! -x "$LITELLM_ROOT/.venv/bin/litellm" ]]; then
  echo "LiteLLM runtime is not installed under $LITELLM_ROOT/.venv" >&2
  exit 2
fi
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Refusing to start: http://$HOST:$PORT is already occupied." >&2
  echo "Stop the existing listener or set LITELLM_GPT56_PORT explicitly." >&2
  exit 2
fi

export OPENAPI_BASE_URL="${OPENAPI_BASE_URL%/}"
export OPENAPI_BASE_URL_V1="${OPENAPI_BASE_URL_V1:-${OPENAPI_BASE_URL}/v1}"
export LITELLM_MODE=PRODUCTION
export LITELLM_LOCAL_MODEL_COST_MAP=True

echo "Starting GPT-5.6-Sol-only LiteLLM proxy on http://$HOST:$PORT"
echo "Credential values are sourced from environment variables and are not logged."

cd "$LITELLM_ROOT"
exec .venv/bin/litellm \
  --config "$CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --telemetry False
