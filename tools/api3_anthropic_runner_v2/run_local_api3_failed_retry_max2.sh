#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 MODEL_KEY" >&2
  exit 2
fi
if [[ -z "${API3_API_KEY:-}" ]]; then
  echo "Required credential environment variable is not set: API3_API_KEY" >&2
  exit 2
fi

MODEL_KEY="$1"
RUNNER_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$RUNNER_ROOT/../.." && pwd)
ARTIFACT_ROOT="$REPO_ROOT/Support/artifacts/outputs/e2e_scenegen_repro"

case "$MODEL_KEY" in
  api3-claude-opus-4-8)
    MAIN_NAME="api3_claude_opus_4_8_paired10_v1"
    RETRY_STEM="api3_claude_opus_4_8_failed_cases"
    ;;
  api3-claude-sonnet-5)
    MAIN_NAME="api3_claude_sonnet_5_paired10_v1"
    RETRY_STEM="api3_claude_sonnet_5_failed_cases"
    ;;
  api3-claude-opus-5)
    MAIN_NAME="api3_claude_opus_5_paired10_v1"
    RETRY_STEM="api3_claude_opus_5_failed_cases"
    ;;
  api3-claude-fable-5)
    MAIN_NAME="api3_claude_fable_5_paired10_v1"
    RETRY_STEM="api3_claude_fable_5_failed_cases"
    ;;
  *)
    echo "Unsupported fixed model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

MAIN_ROOT="$ARTIFACT_ROOT/runs/$MAIN_NAME"
if [[ ! -f "$MAIN_ROOT/summary.json" ]]; then
  echo "Main run is not terminal: $MAIN_ROOT" >&2
  exit 2
fi
if [[ "$(jq -r '.processed_briefs' "$MAIN_ROOT/summary.json")" != 10 ]]; then
  echo "Main run did not process exactly 10 briefs: $MAIN_ROOT" >&2
  exit 2
fi

selected=()
for index in {0..9}; do
  printf -v brief_id 'brief_%02d' "$index"
  result_file="$MAIN_ROOT/$brief_id/case.result.json"
  if [[ ! -f "$result_file" ]]; then
    echo "Terminal main result is missing: $result_file" >&2
    exit 2
  fi
  if ! jq -e '.status == "complete" and .eligible_for_strict_one_shot_evaluation == true' "$result_file" >/dev/null; then
    selected+=("$brief_id")
  fi
done

if [[ "${#selected[@]}" -eq 0 ]]; then
  echo "No failed cases require retry: model=$MODEL_KEY"
  exit 0
fi

echo "derived main failure set: model=$MODEL_KEY cases=${selected[*]}"

for retry_number in 1 2; do
  output_name="${RETRY_STEM}_retry${retry_number}_v1"
  output_root="$ARTIFACT_ROOT/runs/$output_name"
  if [[ -e "$output_root" ]]; then
    echo "Refusing to overwrite retry output: $output_root" >&2
    exit 2
  fi
  if pgrep -af generation_runner.py | grep -F -- "--output-dir $output_root" >/dev/null; then
    echo "Refusing duplicate retry process for: $output_root" >&2
    exit 2
  fi

  run_args=()
  for brief_id in "${selected[@]}"; do
    if [[ ! "$brief_id" =~ ^brief_0[0-9]$ ]]; then
      echo "Invalid derived public brief ID: $brief_id" >&2
      exit 2
    fi
    run_args+=(--brief-id "$brief_id")
  done

  run_status=0
  "$RUNNER_ROOT/run_generation.sh" run \
    --model "$MODEL_KEY" \
    --output-dir "$output_root" \
    --retriever-root "$RUNNER_ROOT" \
    "${run_args[@]}" || run_status=$?

  if [[ "$run_status" -ne 0 && "$run_status" -ne 2 ]]; then
    echo "Retry runner failed outside normal case failure semantics: model=$MODEL_KEY retry=$retry_number status=$run_status" >&2
    exit "$run_status"
  fi

  if [[ "$retry_number" -eq 2 ]]; then
    echo "maximum semantic retries reached: model=$MODEL_KEY"
    exit 0
  fi

  remaining=()
  for brief_id in "${selected[@]}"; do
    result_file="$output_root/$brief_id/case.result.json"
    if [[ ! -f "$result_file" ]] || ! jq -e '.status == "complete" and .eligible_for_strict_one_shot_evaluation == true' "$result_file" >/dev/null; then
      remaining+=("$brief_id")
    fi
  done
  selected=("${remaining[@]}")
  if [[ "${#selected[@]}" -eq 0 ]]; then
    echo "all failed cases recovered on retry1: model=$MODEL_KEY"
    exit 0
  fi
  echo "derived retry2 set: model=$MODEL_KEY cases=${selected[*]}"
done

