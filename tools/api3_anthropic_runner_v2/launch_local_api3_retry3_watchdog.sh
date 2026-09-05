#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"
WATCHDOG_NAME="api3_anthropic_failed_additional3_watchdog_v1"
WATCHDOG_LOG="$ARTIFACT_ROOT/logs/$WATCHDOG_NAME.log"
WATCHDOG_PID="$ARTIFACT_ROOT/pids/$WATCHDOG_NAME.pid"

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

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"
if [[ -e "$WATCHDOG_LOG" || -e "$WATCHDOG_PID" ]]; then
    echo "Refusing existing additional-retry watchdog target: $WATCHDOG_NAME" >&2
  exit 2
fi
if pgrep -af run_local_api3_retry3_watchdog.sh >/dev/null; then
  echo "Refusing duplicate additional-retry watchdog process" >&2
  exit 2
fi

nohup "$RUNNER_ROOT/run_local_api3_retry3_watchdog.sh" \
  >"$WATCHDOG_LOG" 2>&1 < /dev/null &
process_id=$!
printf '%s\n' "$process_id" >"$WATCHDOG_PID"
sleep 1
if ! kill -0 "$process_id" 2>/dev/null; then
  echo "Additional-retry watchdog exited immediately: pid=$process_id" >&2
  exit 3
fi
echo "launched retry-watchdog pid=$process_id additional_retries=3 total_retry_max=5 log=$WATCHDOG_LOG"
