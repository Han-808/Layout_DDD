#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
CONFIG="${GROUPING_EXPERIMENT_CONFIG:-$REPO_ROOT/configs/experiments/grouping_blind30_gpt56_v1.yaml}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/Support/artifacts/outputs/grouping_blind30_gpt56_20260730_r1}"
BLENDER="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
ASSET_ROOT="${ASSET_ROOT:-$REPO_ROOT/Support/Assets/imaginarium_assets}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:4010/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-sol}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-LITELLM_MASTER_KEY}"
PHASE="${PHASE:-all}"
RESUME="${RESUME:-1}"

case "$PHASE" in
  all|prepare|render|group|review) ;;
  *)
    echo "PHASE must be all, prepare, render, group, or review" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Experiment config is missing: $CONFIG" >&2
  exit 2
fi
if [[ "$PHASE" != "prepare" && "$PHASE" != "review" ]]; then
  if [[ ! -x "$BLENDER" ]]; then
    echo "Blender is not executable: $BLENDER" >&2
    exit 2
  fi
  if [[ ! -d "$ASSET_ROOT" ]]; then
    echo "Asset root is missing: $ASSET_ROOT" >&2
    exit 2
  fi
fi
if ! [[ "$JUDGE_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
  echo "JUDGE_API_KEY_ENV must be an environment-variable name" >&2
  exit 2
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
mkdir -p "$OUT_ROOT"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi

"$PYTHON" scripts/run_grouping_blind30.py \
  --config "$CONFIG" \
  --output-root "$OUT_ROOT" \
  --stage prepare \
  "${resume_args[@]}"

if [[ "$PHASE" == "prepare" ]]; then
  exit 0
fi

if [[ "$PHASE" == "all" || "$PHASE" == "render" || "$PHASE" == "group" ]]; then
  "$PYTHON" scripts/run_grouping_blind30.py \
    --config "$CONFIG" \
    --output-root "$OUT_ROOT" \
    --stage render \
    --blender-bin "$BLENDER" \
    --asset-root "$ASSET_ROOT" \
    "${resume_args[@]}"
fi

if [[ "$PHASE" == "render" ]]; then
  exit 0
fi

if [[ "$PHASE" == "all" || "$PHASE" == "group" ]]; then
  if [[ -z "${!JUDGE_API_KEY_ENV:-}" ]]; then
    echo "Required local proxy key is not set: $JUDGE_API_KEY_ENV" >&2
    exit 2
  fi
  proxy_port="${LITELLM_GPT56_PORT:-4010}"
  listener_pid="$(
    lsof -nP -iTCP:"$proxy_port" -sTCP:LISTEN -Fp 2>/dev/null \
      | sed -n 's/^p//p' \
      | head -n 1
  )"
  if [[ -z "$listener_pid" ]]; then
    echo "No listener on local LiteLLM port $proxy_port." >&2
    echo "Start the proxy in this same Terminal session first." >&2
    exit 2
  fi
  listener_command="$(ps -p "$listener_pid" -o command=)"
  if ! grep -qi 'litellm' <<<"$listener_command"; then
    echo "Port $proxy_port is occupied by a non-LiteLLM process:" >&2
    echo "$listener_command" >&2
    exit 2
  fi
  smoke_image="$(
    find "$OUT_ROOT/cases" -type f -name identity_map.png -print -quit
  )"
  if [[ -z "$smoke_image" || ! -s "$smoke_image" ]]; then
    echo "No valid experiment image is available for preflight." >&2
    exit 2
  fi
  "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint "$JUDGE_ENDPOINT" \
    --model "$JUDGE_MODEL" \
    --api-key-env "$JUDGE_API_KEY_ENV" \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal \
    --image-path "$smoke_image" \
    >"$OUT_ROOT/model_preflight.json"
  "$PYTHON" - "$OUT_ROOT/model_preflight.json" "$JUDGE_ENDPOINT" "$JUDGE_MODEL" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
required = (
    result.get("ok") is True,
    result.get("models_ok") is True,
    result.get("text_chat_ok") is True,
    result.get("multimodal_chat_ok") is True,
    result.get("endpoint") == sys.argv[2],
    result.get("model_id") == sys.argv[3],
)
if not all(required):
    raise SystemExit("GPT-5.6-Sol multimodal preflight did not satisfy the contract")
PY
  echo "GPT-5.6-Sol multimodal preflight: passed"
  "$PYTHON" scripts/run_grouping_blind30.py \
    --config "$CONFIG" \
    --output-root "$OUT_ROOT" \
    --stage group \
    --blender-bin "$BLENDER" \
    --asset-root "$ASSET_ROOT" \
    --endpoint "$JUDGE_ENDPOINT" \
    --model "$JUDGE_MODEL" \
    --api-key-env "$JUDGE_API_KEY_ENV" \
    --continue-on-error \
    "${resume_args[@]}"
fi

if [[ "$PHASE" == "all" || "$PHASE" == "group" || "$PHASE" == "review" ]]; then
  "$PYTHON" scripts/build_grouping_blind30_review.py \
    --config "$CONFIG" \
    --output-root "$OUT_ROOT"
fi

echo "Grouping blind experiment completed."
echo "Output root: $OUT_ROOT"
echo "Review index: $OUT_ROOT/blind_review/index.html"
echo "Serve review:"
echo "$PYTHON scripts/serve_grouping_blind30_review.py --output-root $OUT_ROOT"
