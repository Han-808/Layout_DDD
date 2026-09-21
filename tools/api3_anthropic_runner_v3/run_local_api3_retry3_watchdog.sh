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
retry_stems=(
  api3_claude_opus_4_8_failed_cases
  api3_claude_sonnet_5_failed_cases
  api3_claude_opus_5_failed_cases
  api3_claude_fable_5_failed_cases
)

while true; do
  all_max2_terminal=true
  for index in "${!main_names[@]}"; do
    main_root="$RUN_ROOT/${main_names[$index]}"
    retry1_root="$RUN_ROOT/${retry_stems[$index]}_retry1_v1"
    retry2_root="$RUN_ROOT/${retry_stems[$index]}_retry2_v1"

    if [[ ! -f "$main_root/summary.json" ]] || \
       [[ "$(jq -r '.processed_briefs' "$main_root/summary.json")" != 10 ]]; then
      echo "Main run is not terminal 10/10: $main_root" >&2
      exit 2
    fi
    if [[ ! -f "$retry1_root/summary.json" ]]; then
      all_max2_terminal=false
      continue
    fi

    needs_retry2=false
    for brief_dir in "$main_root"/brief_0[0-9]; do
      brief_id=${brief_dir##*/}
      main_result="$brief_dir/case.result.json"
      if jq -e '.status == "complete" and .eligible_for_strict_one_shot_evaluation == true' "$main_result" >/dev/null; then
        continue
      fi
      retry1_result="$retry1_root/$brief_id/case.result.json"
      if [[ ! -f "$retry1_result" ]] || \
         ! jq -e '.status == "complete" and .eligible_for_strict_one_shot_evaluation == true' "$retry1_result" >/dev/null; then
        needs_retry2=true
      fi
    done
    if [[ "$needs_retry2" == true && ! -f "$retry2_root/summary.json" ]]; then
      all_max2_terminal=false
    fi
  done

  if [[ "$all_max2_terminal" == true ]]; then
    echo "all required max-2 retries are terminal; launching isolated retry3/retry4/retry5 controllers"
    exec "$RUNNER_ROOT/launch_all_local_api3_failed_retry3.sh"
  fi
  if (( $(date +%s) >= DEADLINE_SECONDS )); then
    echo "Timed out waiting six hours for max-2 retries; additional retries not launched" >&2
    exit 2
  fi
  sleep 30
done
