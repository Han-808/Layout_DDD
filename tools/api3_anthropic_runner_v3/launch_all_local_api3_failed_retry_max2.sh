#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${API3_API_KEY:-}" ]]; then
  echo "Required credential environment variable is not set: API3_API_KEY" >&2
  exit 2
fi

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"

models=(
  api3-claude-opus-4-8
  api3-claude-sonnet-5
  api3-claude-opus-5
  api3-claude-fable-5
)
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

mkdir -p "$ARTIFACT_ROOT/runs" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"

for index in "${!models[@]}"; do
  main_root="$ARTIFACT_ROOT/runs/${main_names[$index]}"
  if [[ ! -f "$main_root/summary.json" ]]; then
    echo "Main run is not terminal; launching no retries: $main_root" >&2
    exit 2
  fi
  if [[ "$(jq -r '.processed_briefs' "$main_root/summary.json")" != 10 ]]; then
    echo "Main run did not process exactly 10 briefs; launching no retries: $main_root" >&2
    exit 2
  fi

  controller_name="${retry_stems[$index]}_max2_v1"
  controller_log="$ARTIFACT_ROOT/logs/$controller_name.log"
  controller_pid="$ARTIFACT_ROOT/pids/$controller_name.pid"
  if [[ -e "$controller_log" || -e "$controller_pid" ]]; then
    echo "Refusing existing retry controller target: $controller_name" >&2
    exit 2
  fi
  for retry_number in 1 2; do
    retry_root="$ARTIFACT_ROOT/runs/${retry_stems[$index]}_retry${retry_number}_v1"
    if [[ -e "$retry_root" ]]; then
      echo "Refusing existing retry output: $retry_root" >&2
      exit 2
    fi
  done
done

"$RUNNER_ROOT/run_generation.sh" check --retriever-root "$RUNNER_ROOT" >/dev/null
for model in "${models[@]}"; do
  "$RUNNER_ROOT/preflight_api3.py" --model "$model"
done

for index in "${!models[@]}"; do
  controller_name="${retry_stems[$index]}_max2_v1"
  controller_log="$ARTIFACT_ROOT/logs/$controller_name.log"
  controller_pid="$ARTIFACT_ROOT/pids/$controller_name.pid"
  nohup "$RUNNER_ROOT/run_local_api3_failed_retry_max2.sh" "${models[$index]}" \
    >"$controller_log" 2>&1 < /dev/null &
  process_id=$!
  printf '%s\n' "$process_id" >"$controller_pid"
  echo "launched retry-controller model=${models[$index]} max_retry=2 pid=$process_id log=$controller_log"
done

