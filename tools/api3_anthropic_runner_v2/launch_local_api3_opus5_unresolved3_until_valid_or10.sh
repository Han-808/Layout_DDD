#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"
CONTROLLER_NAME="api3_claude_opus_5_unresolved3_until_valid_or10_v1"
CONTROLLER_ROOT="$ARTIFACT_ROOT/runs/$CONTROLLER_NAME"
CONTROLLER_LOG="$ARTIFACT_ROOT/logs/$CONTROLLER_NAME.log"
CONTROLLER_PID="$ARTIFACT_ROOT/pids/$CONTROLLER_NAME.pid"

cleanup() {
  unset API3_API_KEY
}
trap cleanup EXIT INT TERM

if [[ -z "${API3_API_KEY:-}" ]]; then
  read -r -s -p "API3 key: " API3_API_KEY
  printf '\n'
  if [[ -z "$API3_API_KEY" ]]; then
    echo "API3 key input was empty" >&2
    exit 2
  fi
  export API3_API_KEY
fi

mkdir -p "$ARTIFACT_ROOT/runs" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"
if [[ -e "$CONTROLLER_ROOT" || -e "$CONTROLLER_LOG" || -e "$CONTROLLER_PID" ]]; then
  echo "Refusing existing controller target: $CONTROLLER_NAME" >&2
  exit 2
fi
if pgrep -af run_local_api3_opus5_unresolved3_until_valid_or10.py >/dev/null; then
  echo "Refusing duplicate Opus 5 unresolved-scene controller" >&2
  exit 2
fi
for brief_id in brief_04 brief_06 brief_08; do
  for retry_ordinal in {7..16}; do
    output_root="$ARTIFACT_ROOT/runs/api3_claude_opus_5_${brief_id}_retry$(printf '%02d' "$retry_ordinal")_v1"
    if [[ -e "$output_root" ]]; then
      echo "Refusing existing opportunity output: $output_root" >&2
      exit 2
    fi
  done
done

"$RUNNER_ROOT/run_generation.sh" check --retriever-root "$RUNNER_ROOT" >/dev/null
"$RUNNER_ROOT/preflight_api3.py" --model api3-claude-opus-5

nohup "$REPO_ROOT/.venv/bin/python" \
  "$RUNNER_ROOT/run_local_api3_opus5_unresolved3_until_valid_or10.py" \
  >"$CONTROLLER_LOG" 2>&1 < /dev/null &
process_id=$!
printf '%s\n' "$process_id" >"$CONTROLLER_PID"
sleep 2
if ! kill -0 "$process_id" 2>/dev/null && [[ ! -f "$CONTROLLER_ROOT/summary.json" ]]; then
  echo "Controller exited without terminal summary: pid=$process_id" >&2
  exit 3
fi
echo "launched Opus 5 controller pid=$process_id scene_concurrency=3 target=one-valid-per-brief max_opportunities_per_brief=10 log=$CONTROLLER_LOG"
