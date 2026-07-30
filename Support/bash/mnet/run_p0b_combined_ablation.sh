#!/usr/bin/env bash
set -Eeuo pipefail

# One submission entry point for both controlled studies:
#   1. 10 natural-language generated scenes (human event GT required).
#   2. 5 clean repository scenes x 5 controlled output variants (generator skipped).

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
PHASE=${PHASE:-all}
RUN_TAG=${RUN_TAG:-p0b_combined_camera_distortion}
MASTER_OUT_ROOT=${MASTER_OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
GENERATED_OUT_ROOT=${GENERATED_OUT_ROOT:-${MASTER_OUT_ROOT}/generated_ablation10}
SOURCE_OUT_ROOT=${SOURCE_OUT_ROOT:-${MASTER_OUT_ROOT}/source_distortion5}

case "$PHASE" in
  prepare|ablate|all) ;;
  *) echo "PHASE must be prepare, ablate, or all" >&2; exit 2 ;;
esac

mkdir -p "$MASTER_OUT_ROOT"
cd "$REPO_ROOT"

echo "==== $(date '+%F %T') combined job phase=${PHASE} ===="

run_generated_study() {
  local child_phase=$1
  OUT_ROOT="$GENERATED_OUT_ROOT" \
  FROZEN_ROOT="${GENERATED_OUT_ROOT}/frozen_cases" \
  GT_ROOT="${GENERATED_OUT_ROOT}/gt" \
  ABLATION_ROOT="${GENERATED_OUT_ROOT}/ablation" \
  ROOM_ROOT="${GENERATED_OUT_ROOT}/rooms" \
  CASE_MANIFEST="${GENERATED_OUT_ROOT}/case_manifest.tsv" \
  PREPARE_SUMMARY="${GENERATED_OUT_ROOT}/prepare_summary.tsv" \
  REVIEW_QUEUE="${GENERATED_OUT_ROOT}/gt_review_queue.tsv" \
  ADAPTER_CONFIG="${GENERATED_OUT_ROOT}/layout_json_adapter.json" \
  PHASE="$child_phase" \
    bash Support/bash/mnet/run_p0b_ablation10_highlight_camera.sh
}

run_source_study() {
  local child_phase=$1
  OUT_ROOT="$SOURCE_OUT_ROOT" \
  SOURCE_ROOT="${SOURCE_OUT_ROOT}/source_reports" \
  GT_ROOT="${SOURCE_OUT_ROOT}/gt" \
  ABLATION_ROOT="${SOURCE_OUT_ROOT}/ablation" \
  RESULT_ROOT="${SOURCE_OUT_ROOT}/results" \
  CASE_MANIFEST="${SOURCE_OUT_ROOT}/case_manifest.tsv" \
  SOURCE_SUMMARY="${SOURCE_OUT_ROOT}/source_summary.tsv" \
  PHASE="$child_phase" \
    bash Support/bash/mnet/run_p0b_source_distortion5.sh
}

if [[ "$PHASE" == "prepare" ]]; then
  run_generated_study prepare
  run_source_study evaluate
elif [[ "$PHASE" == "ablate" ]]; then
  run_generated_study ablate
  run_source_study ablate
else
  # The generated arm preserves its human-GT gate and will not fabricate labels.
  # The source-scene arm has transform-derived GT and can complete end to end.
  run_generated_study all
  run_source_study all
fi

echo "==== $(date '+%F %T') combined job complete ===="
echo "Generated 10-case study: $GENERATED_OUT_ROOT"
echo "Source-scene distortion study: $SOURCE_OUT_ROOT"
