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
outputs=(
  api3_claude_opus_4_8_paired10_v1
  api3_claude_sonnet_5_paired10_v1
  api3_claude_opus_5_paired10_v1
  api3_claude_fable_5_paired10_v1
)

mkdir -p "$ARTIFACT_ROOT/runs" "$ARTIFACT_ROOT/logs" "$ARTIFACT_ROOT/pids"

for index in "${!models[@]}"; do
  output_root="$ARTIFACT_ROOT/runs/${outputs[$index]}"
  log_path="$ARTIFACT_ROOT/logs/${outputs[$index]}.log"
  pid_path="$ARTIFACT_ROOT/pids/${outputs[$index]}.pid"
  if [[ -e "$output_root" ]]; then
    echo "Refusing to overwrite existing output: $output_root" >&2
    exit 2
  fi
  if [[ -e "$log_path" ]]; then
    echo "Refusing to overwrite existing log: $log_path" >&2
    exit 2
  fi
  if [[ -e "$pid_path" ]]; then
    echo "Refusing to overwrite existing PID file: $pid_path" >&2
    exit 2
  fi
  if pgrep -af generation_runner.py | grep -F -- "--output-dir $output_root" >/dev/null; then
    echo "Refusing duplicate generation process for: $output_root" >&2
    exit 2
  fi
done

"$RUNNER_ROOT/run_generation.sh" check --retriever-root "$RUNNER_ROOT" >/dev/null

preflight_ok=(false false false false)
for index in "${!models[@]}"; do
  if "$RUNNER_ROOT/preflight_api3.py" --model "${models[$index]}"; then
    preflight_ok[$index]=true
  elif [[ "$index" -eq 0 ]]; then
    echo "optional model skipped after failed preflight: ${models[$index]}" >&2
  else
    echo "required model preflight failed; launching nothing: ${models[$index]}" >&2
    exit 2
  fi
done

for index in "${!models[@]}"; do
  if [[ "${preflight_ok[$index]}" != true ]]; then
    continue
  fi
  output_name="${outputs[$index]}"
  log_path="$ARTIFACT_ROOT/logs/$output_name.log"
  pid_path="$ARTIFACT_ROOT/pids/$output_name.pid"
  nohup "$RUNNER_ROOT/run_local_api3_model_scene10.sh" \
    "${models[$index]}" "$output_name" \
    >"$log_path" 2>&1 < /dev/null &
  process_id=$!
  printf '%s\n' "$process_id" >"$pid_path"
  echo "launched model=${models[$index]} pid=$process_id output=$ARTIFACT_ROOT/runs/$output_name log=$log_path"
done
