#!/bin/zsh

set -euo pipefail
set +x

SCRIPT_PATH="${0:A}"
REPO_ROOT="${SCRIPT_PATH:h:h:h:h}"
PYTHON_BIN="${LAYOUT_DDD_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$(command -v python3)"
PROFILE="$REPO_ROOT/configs/generation_extensions/non_rectangular_multi_room_v1/fullruns/spatiallm_selected10_api3_retry5_v2.json"
DEFAULT_OUTPUT="$REPO_ROOT/Support/artifacts/outputs/e2e_multi_room/nonrect_spatiallm_selected10_api3_retry5_v2_r1"
DEFAULT_RESOURCE_BINDINGS="$REPO_ROOT/.runtime/retrieval_bindings.local.json"

usage() {
  print -r -- "Usage: $SCRIPT_PATH [--check | --preflight-only] [options]"
  print -r -- ""
  print -r -- "API3-only non-rectangular generation: Sonnet 5 -> Opus 5 -> Fable 5."
  print -r -- "Each model runs the same 10 scenes sequentially with v2 global prompts."
  print -r -- "Stage A timeout: 2400s; Stage C timeout: 3600s."
  print -r -- ""
  print -r -- "  --check                 Static validation only; no credential or network"
  print -r -- "  --preflight-only        Live preflight only; no generation"
  print -r -- "  --fresh                 Refuse any existing output root"
  print -r -- "  --output-base PATH      Provider-isolated output root"
  print -r -- "  --generation-bindings P Private route binding JSON"
  print -r -- "  --resource-bindings P   Private retrieval binding JSON"
  print -r -- "  --api3-endpoint URL     Optional shared API3 endpoint override"
}

MODE="run"
FRESH=0
OUTPUT_BASE="$DEFAULT_OUTPUT"
GENERATION_BINDINGS=""
RESOURCE_BINDINGS="$DEFAULT_RESOURCE_BINDINGS"
API3_ENDPOINT="${NONRECT_API3_ENDPOINT:-}"
TEMP_BINDING_DIR=""

while (( $# > 0 )); do
  case "$1" in
    --check) MODE="check"; shift ;;
    --preflight-only) MODE="preflight"; shift ;;
    --fresh) FRESH=1; shift ;;
    --output-base) OUTPUT_BASE="$2"; shift 2 ;;
    --generation-bindings) GENERATION_BINDINGS="$2"; shift 2 ;;
    --resource-bindings) RESOURCE_BINDINGS="$2"; shift 2 ;;
    --api3-endpoint) API3_ENDPOINT="$2"; shift 2 ;;
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
  unset API3_API_KEY
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
  if [[ -z "${API3_API_KEY:-}" ]]; then
    read -r -s "API3_API_KEY?API3 key (hidden; shared by Sonnet/Opus/Fable): "
    print
  fi
  [[ -n "$API3_API_KEY" ]] || {
    print -u2 -- "API3 key is empty"
    exit 2
  }
  export API3_API_KEY
  if [[ -z "$GENERATION_BINDINGS" ]]; then
    [[ -n "$API3_ENDPOINT" ]] || {
      print -u2 -- "Set NONRECT_API3_ENDPOINT or pass --generation-bindings"
      exit 2
    }
    [[ "$API3_ENDPOINT" == http://* || "$API3_ENDPOINT" == https://* ]] || exit 2
    TEMP_BINDING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/layoutddd-nonrect-api3.XXXXXX")
    GENERATION_BINDINGS="$TEMP_BINDING_DIR/generation_bindings.json"
    API3_ENDPOINT="$API3_ENDPOINT" GENERATION_BINDINGS="$GENERATION_BINDINGS" \
      "$PYTHON_BIN" -c 'import json, os; from pathlib import Path; Path(os.environ["GENERATION_BINDINGS"]).write_text(json.dumps({"schema_version":"generation_route_bindings_v2","bindings":{"api3-chat-legacy-core-v1":{"endpoint":os.environ["API3_ENDPOINT"],"credential_env":"API3_API_KEY"}}}, indent=2, sort_keys=True)+"\n", encoding="utf-8")'
    ARGS+=(--generation-bindings "$GENERATION_BINDINGS")
  fi
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" \
  -m benchmark.scene_generation.non_rectangular_multi_room.cohort_runner \
  "${ARGS[@]}"
