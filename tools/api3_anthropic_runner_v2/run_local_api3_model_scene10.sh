#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MODEL_KEY OUTPUT_NAME" >&2
  exit 2
fi
if [[ -z "${API3_API_KEY:-}" ]]; then
  echo "Required credential environment variable is not set: API3_API_KEY" >&2
  exit 2
fi

MODEL_KEY="$1"
OUTPUT_NAME="$2"
RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
OUTPUT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro/runs/$OUTPUT_NAME"

case "$MODEL_KEY" in
  api3-claude-opus-4-8|api3-claude-sonnet-5|api3-claude-opus-5|api3-claude-fable-5) ;;
  *)
    echo "Unsupported fixed model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi
if pgrep -af generation_runner.py | grep -F -- "--output-dir $OUTPUT_ROOT" >/dev/null; then
  echo "Refusing duplicate generation process for: $OUTPUT_ROOT" >&2
  exit 2
fi

exec "$RUNNER_ROOT/run_generation.sh" run \
  --model "$MODEL_KEY" \
  --output-dir "$OUTPUT_ROOT" \
  --retriever-root "$RUNNER_ROOT"
