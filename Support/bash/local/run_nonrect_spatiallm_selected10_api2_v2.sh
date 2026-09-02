#!/bin/zsh

set -euo pipefail
set +x

SCRIPT_PATH="${0:A}"
REPO_ROOT="${SCRIPT_PATH:h:h:h:h}"
PYTHON_BIN="${LAYOUT_DDD_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
PROFILE="$REPO_ROOT/configs/generation_extensions/non_rectangular_multi_room_v1/fullruns/spatiallm_selected10_api2_retry5_v2.json"
DEFAULT_OUTPUT="$REPO_ROOT/Support/artifacts/outputs/e2e_multi_room/nonrect_spatiallm_selected10_api2_retry5_v2_r1"
DEFAULT_RESOURCE_BINDINGS="$REPO_ROOT/.runtime/retrieval_bindings.local.json"

usage() {
  print -r -- "Usage: $SCRIPT_PATH [--check | --preflight-only] [options]"
  print -r -- ""
  print -r -- "API2-only non-rectangular generation: GLM 5.3 -> GPT-5.6-Sol -> Kimi K3."
  print -r -- "Each model runs the same 10 scenes sequentially with v2 global prompts."
  print -r -- "Stage A timeout: 2400s; Stage C timeout: 3600s."
  print -r -- ""
  print -r -- "  --check                 Static validation only; no credential or network"
  print -r -- "  --preflight-only        Live preflight only; no generation"
  print -r -- "  --fresh                 Refuse any existing output root"
  print -r -- "  --output-base PATH      Provider-isolated output root"
  print -r -- "  --generation-bindings P Private route binding JSON"
  print -r -- "  --resource-bindings P   Private retrieval binding JSON"
  print -r -- "  --gpt56-endpoint URL    Optional GPT-5.6-Sol endpoint override"
  print -r -- "  --kimi-endpoint URL     Optional Kimi K3 endpoint override"
  print -r -- "  --glm-endpoint URL      Optional GLM-5.3 endpoint override"
}

MODE="run"
FRESH=0
OUTPUT_BASE="$DEFAULT_OUTPUT"
GENERATION_BINDINGS=""
RESOURCE_BINDINGS="$DEFAULT_RESOURCE_BINDINGS"
GPT56_ENDPOINT="${NONRECT_GPT56_ENDPOINT:-}"
KIMI_ENDPOINT="${NONRECT_KIMI_ENDPOINT:-}"
GLM_ENDPOINT="${NONRECT_GLM_ENDPOINT:-}"
TEMP_BINDING_DIR=""

while (( $# > 0 )); do
  case "$1" in
    --check) MODE="check"; shift ;;
    --preflight-only) MODE="preflight"; shift ;;
    --fresh) FRESH=1; shift ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --generation-bindings) GENERATION_BINDINGS="$2"; shift 2 ;;
    --resource-bindings) RESOURCE_BINDINGS="$2"; shift 2 ;;
    --gpt56-endpoint) GPT56_ENDPOINT="$2"; shift 2 ;;
    --kimi-endpoint) KIMI_ENDPOINT="$2"; shift 2 ;;
    --glm-endpoint) GLM_ENDPOINT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 -- "Unknown argument: $1"; usage >&2; exit 2 ;;
  esac
done

[[ -x "$PYTHON_BIN" && -f "$PROFILE" ]] || {
  print -u2 -- "Required non-rectangular runtime is unavailable"
  exit 2
}
(( FRESH == 0 || MODE == "run" )) || {
  print -u2 -- "--fresh is only valid for a generation run"
  exit 2
}

typeset -a ARGS
ARGS=("$MODE" --profile "$PROFILE")
if [[ "$MODE" != "check" ]]; then
  ARGS+=(--resource-bindings "$RESOURCE_BINDINGS")
  [[ -z "$GENERATION_BINDINGS" ]] || ARGS+=(--generation-bindings "$GENERATION_BINDINGS")
fi
if [[ "$MODE" == "run" ]]; then
  ARGS+=(--output-base "$OUTPUT_BASE")
  (( FRESH == 0 )) || ARGS+=(--fresh)
fi

cleanup() {
  unset API2_APP_CREDENTIAL
  if [[ -n "$TEMP_BINDING_DIR" && -d "$TEMP_BINDING_DIR" ]]; then
    command rm -f -- "$TEMP_BINDING_DIR/generation_bindings.json"
    command rmdir -- "$TEMP_BINDING_DIR" 2>/dev/null || true
  fi
}
interrupt_run() { cleanup; exit 130; }
terminate_run() { cleanup; exit 143; }
trap cleanup EXIT
trap interrupt_run INT
trap terminate_run TERM HUP

if [[ "$MODE" != "check" ]]; then
  if [[ -z "${API2_APP_CREDENTIAL:-}" ]]; then
    read -r -s "API2_APP_CREDENTIAL?API2 APP_ID:APP_KEY (hidden; shared by GLM/GPT/Kimi): "
    print
  fi
  [[ -n "$API2_APP_CREDENTIAL" && "$API2_APP_CREDENTIAL" == *:* ]] || {
    print -u2 -- "API2 credential must have APP_ID:APP_KEY form"
    exit 2
  }
  export API2_APP_CREDENTIAL
  if [[ -z "$GENERATION_BINDINGS" ]]; then
    [[ -n "$GPT56_ENDPOINT" && -n "$KIMI_ENDPOINT" && -n "$GLM_ENDPOINT" ]] || {
      print -u2 -- "Set NONRECT_GPT56_ENDPOINT, NONRECT_KIMI_ENDPOINT, and NONRECT_GLM_ENDPOINT, or pass --generation-bindings"
      exit 2
    }
    [[ "$GPT56_ENDPOINT" == http://* || "$GPT56_ENDPOINT" == https://* ]] || exit 2
    [[ "$KIMI_ENDPOINT" == http://* || "$KIMI_ENDPOINT" == https://* ]] || exit 2
    [[ "$GLM_ENDPOINT" == http://* || "$GLM_ENDPOINT" == https://* ]] || exit 2
    TEMP_BINDING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/layoutddd-nonrect-api2.XXXXXX")
    GENERATION_BINDINGS="$TEMP_BINDING_DIR/generation_bindings.json"
    GPT56_ENDPOINT="$GPT56_ENDPOINT" KIMI_ENDPOINT="$KIMI_ENDPOINT" \
      GLM_ENDPOINT="$GLM_ENDPOINT" GENERATION_BINDINGS="$GENERATION_BINDINGS" \
      "$PYTHON_BIN" -c 'import json, os; from pathlib import Path; Path(os.environ["GENERATION_BINDINGS"]).write_text(json.dumps({"schema_version":"generation_route_bindings_v2","bindings":{"api2-standard-chat-reasoning-v1":{"endpoint":os.environ["GPT56_ENDPOINT"],"credential_env":"API2_APP_CREDENTIAL"},"api2-chat-top-level-reasoning-v1":{"endpoint":os.environ["KIMI_ENDPOINT"],"credential_env":"API2_APP_CREDENTIAL"},"api2-responses-reasoning-v1":{"endpoint":os.environ["GLM_ENDPOINT"],"credential_env":"API2_APP_CREDENTIAL"}}}, indent=2, sort_keys=True)+"\n", encoding="utf-8")'
    ARGS+=(--generation-bindings "$GENERATION_BINDINGS")
  fi
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
  -m benchmark.scene_generation.non_rectangular_multi_room.cohort_runner \
  "${ARGS[@]}"
