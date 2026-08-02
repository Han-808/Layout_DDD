#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LITELLM_ROOT="$REPO_ROOT/Support/third_party/uni_llm_hhr_cursor_snapshot"
CONFIG="$REPO_ROOT/configs/inference/litellm_fable5_gpt56sol_hy3_local.yaml"
HOST="${LITELLM_HOST:-127.0.0.1}"
PORT="${LITELLM_PORT:-4000}"
HY3_ROUTE="${HY3_ROUTE:-tokenhub}"

if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]]; then
  echo "Refusing to expose the local judge proxy on non-loopback host: $HOST" >&2
  exit 2
fi

for variable_name in \
  OPENAPI_BASE_URL \
  OPENAPI_API_KEY \
  OPENAPI_GPT_KEY \
  LITELLM_MASTER_KEY; do
  if [[ -z "${!variable_name:-}" ]]; then
    echo "Required environment variable is not set: $variable_name" >&2
    exit 2
  fi
done

case "$HY3_ROUTE" in
  gateway)
    if [[ -z "${OPENAPI_HY3_KEY:-}" ]]; then
      echo "Required environment variable is not set for HY3_ROUTE=gateway: OPENAPI_HY3_KEY" >&2
      exit 2
    fi
    export HY3_RESOLVED_API_KEY="$OPENAPI_HY3_KEY"
    export HY3_RESOLVED_API_BASE="${OPENAPI_BASE_URL%/}"
    ;;
  tokenhub)
    if [[ -z "${TOKENHUB_API_KEY:-}" ]]; then
      echo "Required environment variable is not set for HY3_ROUTE=tokenhub: TOKENHUB_API_KEY" >&2
      exit 2
    fi
    export HY3_RESOLVED_API_KEY="$TOKENHUB_API_KEY"
    export HY3_RESOLVED_API_BASE="https://tokenhub.tencentmaas.com"
    ;;
  *)
    echo "Unsupported HY3_ROUTE: $HY3_ROUTE (expected gateway or tokenhub)" >&2
    exit 2
    ;;
esac

if [[ ! -d "$LITELLM_ROOT" ]]; then
  echo "Cloned LiteLLM repository is missing: $LITELLM_ROOT" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "LiteLLM config is missing: $CONFIG" >&2
  exit 2
fi
if [[ ! -x "$LITELLM_ROOT/.venv/bin/litellm" ]]; then
  echo "LiteLLM runtime is not installed." >&2
  echo "Run: cd '$LITELLM_ROOT' && uv sync --frozen --extra proxy" >&2
  exit 2
fi

export OPENAPI_BASE_URL="${OPENAPI_BASE_URL%/}"
export OPENAPI_BASE_URL_V1="${OPENAPI_BASE_URL_V1:-${OPENAPI_BASE_URL}/v1}"
# Disable automatic .env loading from the cloned third-party repository.
export LITELLM_MODE=PRODUCTION
# Use the model metadata bundled with the pinned fork. This prevents an
# unrelated GitHub fetch during startup and keeps outbound traffic scoped to
# the three configured model upstreams.
export LITELLM_LOCAL_MODEL_COST_MAP=True

echo "Starting local LiteLLM judge proxy on http://$HOST:$PORT"
echo "Models: claude-fable-5, gpt-5.6-sol, hy3"
echo "HY3 route: $HY3_ROUTE"
echo "Credential values are sourced from environment variables and are not logged."

cd "$LITELLM_ROOT"
exec .venv/bin/litellm \
  --config "$CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --telemetry False
