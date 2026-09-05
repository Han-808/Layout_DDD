#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROFILE=""
OUTPUT_BASE=""
RESOURCE_BINDINGS=""
FRESH=0
CHECK_ONLY=0

usage() {
  echo "Usage: $0 --profile FILE [--output-base DIR]" \
    "[--resource-bindings FILE] [--fresh] [--check-only]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --resource-bindings)
      RESOURCE_BINDINGS="$2"
      shift 2
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${LAYOUT_DDD_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python runtime is not executable: $PYTHON_BIN" >&2
  exit 2
fi

cd "$REPO_ROOT"
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m \
  benchmark.scene_generation.non_rectangular_agent check \
  --profile "$PROFILE"

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  exit 0
fi
if [[ -z "$OUTPUT_BASE" || -z "$RESOURCE_BINDINGS" ]]; then
  echo "Generation requires --output-base and --resource-bindings." >&2
  exit 2
fi

PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" -m \
  benchmark.scene_generation.non_rectangular_agent resource-gate \
  --profile "$PROFILE" \
  --resource-bindings "$RESOURCE_BINDINGS"

RUN_ARGS=(
  -m benchmark.scene_generation.non_rectangular_agent run
  --profile "$PROFILE"
  --resource-bindings "$RESOURCE_BINDINGS"
  --output-base "$OUTPUT_BASE"
)
if [[ "$FRESH" -eq 1 ]]; then
  RUN_ARGS+=(--fresh)
fi
PYTHONPATH="$REPO_ROOT/src" "$PYTHON_BIN" "${RUN_ARGS[@]}"
