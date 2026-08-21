#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${API3_API_KEY:-}" ]]; then
  echo "Required credential environment variable is not set: API3_API_KEY" >&2
  exit 2
fi

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"
WATCHDOG_NAME="api3_anthropic_failed_retry_max2_watchdog_v1"
WATCHDOG_LOG="$ARTIFACT_ROOT/logs/$WATCHDOG_NAME.log"
WATCHDOG_PID="$ARTIFACT_ROOT/pids/$WATCHDOG_NAME.pid"

mkdir -p "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"
if [[ -e "$WATCHDOG_LOG" || -e "$WATCHDOG_PID" ]]; then
  echo "Refusing existing retry watchdog target: $WATCHDOG_NAME" >&2
  exit 2
fi

nohup "$RUNNER_ROOT/run_local_api3_retry_watchdog.sh" \
  >"$WATCHDOG_LOG" 2>&1 < /dev/null &
process_id=$!
printf '%s\n' "$process_id" >"$WATCHDOG_PID"
echo "launched retry-watchdog pid=$process_id max_retry=2 log=$WATCHDOG_LOG"

