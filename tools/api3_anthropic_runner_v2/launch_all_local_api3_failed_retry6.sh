#!/usr/bin/env bash
set -euo pipefail

RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"

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

models=(
  api3-claude-opus-4-8
  api3-claude-sonnet-5
  api3-claude-opus-5
)
retry_stems=(
  api3_claude_opus_4_8_failed_cases
  api3_claude_sonnet_5_failed_cases
  api3_claude_opus_5_failed_cases
)

mkdir -p "$ARTIFACT_ROOT/runs" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"

for index in "${!models[@]}"; do
  output_root="$ARTIFACT_ROOT/runs/${retry_stems[$index]}_retry6_v1"
  controller_name="${retry_stems[$index]}_retry6_controller_v1"
  controller_log="$ARTIFACT_ROOT/logs/$controller_name.log"
  controller_pid="$ARTIFACT_ROOT/pids/$controller_name.pid"
  if [[ -e "$output_root" ]]; then
    echo "Refusing existing retry6 output: $output_root" >&2
    exit 2
  fi
  if [[ -e "$controller_log" || -e "$controller_pid" ]]; then
    echo "Refusing existing retry6 controller target: $controller_name" >&2
    exit 2
  fi
  if pgrep -af generation_runner.py | grep -F -- "--output-dir $output_root" >/dev/null; then
    echo "Refusing duplicate retry6 process for: $output_root" >&2
    exit 2
  fi
done

"$RUNNER_ROOT/run_generation.sh" check --retriever-root "$RUNNER_ROOT" >/dev/null

for index in "${!models[@]}"; do
  controller_name="${retry_stems[$index]}_retry6_controller_v1"
  controller_log="$ARTIFACT_ROOT/logs/$controller_name.log"
  controller_pid="$ARTIFACT_ROOT/pids/$controller_name.pid"
  nohup "$RUNNER_ROOT/run_local_api3_failed_retry6.sh" "${models[$index]}" \
    >"$controller_log" 2>&1 < /dev/null &
  process_id=$!
  printf '%s\n' "$process_id" >"$controller_pid"
  echo "launched retry6-controller model=${models[$index]} pid=$process_id log=$controller_log"
done

sleep 2
for index in "${!models[@]}"; do
  controller_name="${retry_stems[$index]}_retry6_controller_v1"
  controller_pid="$ARTIFACT_ROOT/pids/$controller_name.pid"
  output_root="$ARTIFACT_ROOT/runs/${retry_stems[$index]}_retry6_v1"
  process_id=$(<"$controller_pid")
  if ! kill -0 "$process_id" 2>/dev/null && [[ ! -f "$output_root/summary.json" ]]; then
    echo "Retry6 controller exited without terminal summary: model=${models[$index]} pid=$process_id" >&2
    exit 3
  fi
done
