#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON="${BENCH_PY:-$REPO_ROOT/.venv/bin/python}"
BLENDER="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"
DATASET_ROOT="$REPO_ROOT/Support/datasets/cal_dataset1"
ASSET_ROOT="$REPO_ROOT/Support/Assets/imaginarium_assets"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp1_1_fine_edge}"
JUDGE_ROOT="${JUDGE_ROOT:-$REPO_ROOT/Support/artifacts/outputs}"
REPEATABILITY_ROOT="${REPEATABILITY_ROOT:-$JUDGE_ROOT/exp1_1_fine_edge_gpt56_repeatability}"
MERGED_ROOT="${MERGED_ROOT:-$JUDGE_ROOT/exp1_1_extended_gpt56}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_fine_edge_judge.json}"
JUDGE_ENDPOINT="${JUDGE_ENDPOINT:-http://127.0.0.1:4010/v1}"
REPEATS="${REPEATS:-2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
PHASE="${PHASE:-all}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Benchmark Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -x "$BLENDER" ]]; then
  echo "Blender is not executable: $BLENDER" >&2
  exit 2
fi
if [[ ! -d "$ASSET_ROOT" ]]; then
  echo "Asset root is missing: $ASSET_ROOT" >&2
  exit 2
fi
if [[ "$REPEATS" -lt 2 ]]; then
  echo "REPEATS must be at least 2 for exact-input stability measurement." >&2
  exit 2
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1

if [[ "$PHASE" == "all" || "$PHASE" == "evidence" ]]; then
  "$PYTHON" scripts/run_cal_dataset1_camera_evidence.py \
    --dataset-root "$DATASET_ROOT" \
    --out-dir "$EVIDENCE_ROOT" \
    --experiment-id exp1_1_fine_edge \
    --blender-bin "$BLENDER" \
    --asset-root "$ASSET_ROOT" \
    --split fine_edge \
    --candidate-policy local \
    --render-engine BLENDER_WORKBENCH \
    --cycles-device CPU \
    --resume
fi

if [[ "$PHASE" == "all" || "$PHASE" == "judge" ]]; then
  if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
    echo "LITELLM_MASTER_KEY is not set in this shell." >&2
    exit 2
  fi
  "$PYTHON" scripts/check_model_endpoint.py \
    --endpoint "$JUDGE_ENDPOINT" \
    --model gpt-5.6-sol \
    --api-key-env LITELLM_MASTER_KEY \
    --timeout-seconds 3000 \
    --max-tokens 200 \
    --no-send-temperature \
    --no-response-format-json \
    --multimodal \
    --image-path "$DATASET_ROOT/evaluation/mesh_geometry/fine_edge_easy_001/renders/standardized_top.png"

  for repeat in $(seq 1 "$REPEATS"); do
    out_dir="$JUDGE_ROOT/exp1_1_fine_edge_gpt56_judge_r${repeat}"
    "$PYTHON" scripts/judge_cal_dataset1_camera_evidence.py \
      --evidence-root "$EVIDENCE_ROOT" \
      --dataset-root "$DATASET_ROOT" \
      --judge-config "$JUDGE_CONFIG" \
      --out-dir "$out_dir" \
      --severity edge \
      --max-workers "$MAX_WORKERS" \
      --resume
  done

  "$PYTHON" scripts/compare_cal_dataset1_judge_repeats.py \
    --left-run "$JUDGE_ROOT/exp1_1_fine_edge_gpt56_judge_r1" \
    --right-run "$JUDGE_ROOT/exp1_1_fine_edge_gpt56_judge_r2" \
    --out-dir "$REPEATABILITY_ROOT"

  merge_args=()
  for repeat in $(seq 1 "$REPEATS"); do
    merge_args+=(
      --ambiguous-run
      "$JUDGE_ROOT/exp1_1_fine_edge_gpt56_judge_r${repeat}"
    )
  done
  "$PYTHON" scripts/merge_cal_dataset1_job1_results.py \
    --invalid-run "$JUDGE_ROOT/exp1_1_gpt56_judge" \
    "${merge_args[@]}" \
    --out-dir "$MERGED_ROOT"
fi

echo "Fine-edge evidence: $EVIDENCE_ROOT"
echo "Repeatability: $REPEATABILITY_ROOT"
echo "Merged 39-event report: $MERGED_ROOT"
