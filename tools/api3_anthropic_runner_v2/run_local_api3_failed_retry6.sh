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
    EXPECTED=(brief_05)
    ;;
  api3-claude-sonnet-5)
    MAIN_NAME="api3_claude_sonnet_5_paired10_v1"
    RETRY_STEM="api3_claude_sonnet_5_failed_cases"
    EXPECTED=(brief_09)
    ;;
  api3-claude-opus-5)
    MAIN_NAME="api3_claude_opus_5_paired10_v1"
    RETRY_STEM="api3_claude_opus_5_failed_cases"
    EXPECTED=(brief_04 brief_06 brief_08)
    ;;
  *)
    echo "Unsupported fixed retry6 model key: $MODEL_KEY" >&2
    exit 2
    ;;
esac

MAIN_ROOT="$ARTIFACT_ROOT/runs/$MAIN_NAME"
OUTPUT_ROOT="$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry6_v1"

if [[ ! -f "$MAIN_ROOT/summary.json" ]] || \
   [[ "$(jq -r '.processed_briefs' "$MAIN_ROOT/summary.json")" != 10 ]]; then
  echo "Main run is not terminal 10/10: $MAIN_ROOT" >&2
  exit 2
fi
for retry_number in 1 2 3 4 5; do
  prior_root="$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry${retry_number}_v1"
  if [[ ! -f "$prior_root/summary.json" ]]; then
    echo "Prior retry is not terminal: $prior_root" >&2
    exit 2
  fi
done

selected=()
for index in {0..9}; do
  printf -v brief_id 'brief_%02d' "$index"
  recovered=false
  for candidate in \
    "$MAIN_ROOT/$brief_id/case.result.json" \
    "$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry1_v1/$brief_id/case.result.json" \
    "$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry2_v1/$brief_id/case.result.json" \
    "$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry3_v1/$brief_id/case.result.json" \
    "$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry4_v1/$brief_id/case.result.json" \
    "$ARTIFACT_ROOT/runs/${RETRY_STEM}_retry5_v1/$brief_id/case.result.json"; do
    if [[ -f "$candidate" ]] && \
       jq -e '.status == "complete" and .eligible_for_strict_one_shot_evaluation == true' "$candidate" >/dev/null; then
      recovered=true
      break
    fi
  done
  if [[ "$recovered" == false ]]; then
    selected+=("$brief_id")
  fi
done

if [[ "${selected[*]}" != "${EXPECTED[*]}" ]]; then
  echo "Refusing changed retry6 set: model=$MODEL_KEY expected=${EXPECTED[*]} actual=${selected[*]}" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite retry6 output: $OUTPUT_ROOT" >&2
  exit 2
fi
if pgrep -af generation_runner.py | grep -F -- "--output-dir $OUTPUT_ROOT" >/dev/null; then
  echo "Refusing duplicate retry6 process for: $OUTPUT_ROOT" >&2
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

echo "validated retry6 set: model=$MODEL_KEY cases=${selected[*]}"
"$RUNNER_ROOT/preflight_api3.py" --model "$MODEL_KEY"

run_status=0
"$RUNNER_ROOT/run_generation.sh" run \
  --model "$MODEL_KEY" \
  --output-dir "$OUTPUT_ROOT" \
  --retriever-root "$RUNNER_ROOT" \
  "${run_args[@]}" || run_status=$?

if [[ "$run_status" -ne 0 && "$run_status" -ne 2 ]]; then
  echo "Retry6 runner failed outside normal case failure semantics: model=$MODEL_KEY status=$run_status" >&2
  exit "$run_status"
fi

echo "retry6 terminal: model=$MODEL_KEY"
