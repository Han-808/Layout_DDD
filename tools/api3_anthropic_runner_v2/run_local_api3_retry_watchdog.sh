#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${API3_API_KEY:-}" ]]; then
  echo "Required credential environment variable is not set: API3_API_KEY" >&2
  exit 2
fi

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
RUN_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro/runs"
DEADLINE_SECONDS=$(( $(date +%s) + 21600 ))

main_names=(
  api3_claude_opus_4_8_paired10_v1
  api3_claude_sonnet_5_paired10_v1
  api3_claude_opus_5_paired10_v1
  api3_claude_fable_5_paired10_v1
)

while true; do
  all_terminal=true
  for main_name in "${main_names[@]}"; do
    summary_file="$RUN_ROOT/$main_name/summary.json"
    if [[ ! -f "$summary_file" ]]; then
      all_terminal=false
      continue
    fi
    if [[ "$(jq -r '.processed_briefs' "$summary_file")" != 10 ]]; then
      echo "Terminal summary did not process exactly 10 briefs: $summary_file" >&2
      exit 2
    fi
  done
  if [[ "$all_terminal" == true ]]; then
    echo "all four main runs are terminal; launching isolated max-2 retry controllers"
    exec "$RUNNER_ROOT/launch_all_local_api3_failed_retry_max2.sh"
  fi
  if (( $(date +%s) >= DEADLINE_SECONDS )); then
    echo "Timed out waiting six hours for all main summaries; no retries launched" >&2
    exit 2
  fi
  sleep 30
done

