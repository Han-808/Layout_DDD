#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
EVIDENCE_ROOT="${EVIDENCE_ROOT:-$REPO_ROOT/Support/artifacts/outputs/exp1_1}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/Support/datasets/cal_dataset1}"
JUDGE_CONFIG="${JUDGE_CONFIG:-$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_judge.json}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/Support/artifacts/outputs/exp1_1_gpt56_judge}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RESUME="${RESUME:-1}"
SMOKE_IMAGE="${SMOKE_IMAGE:-$EVIDENCE_ROOT/cases/subtle_support_002/metric_local_highlight/support__obj_000__144ae3a484/final_bundle/rgb_00_support_00_contact_060.png}"

if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "Required environment variable is not set: LITELLM_MASTER_KEY" >&2
  exit 2
fi
for path in "$PYTHON_BIN" "$EVIDENCE_ROOT" "$DATASET_ROOT" "$JUDGE_CONFIG" "$SMOKE_IMAGE"; do
  if [[ ! -e "$path" ]]; then
    echo "Required local input does not exist: $path" >&2
    exit 2
  fi
done
if ! [[ "$MAX_WORKERS" =~ ^[1-8]$ ]]; then
  echo "MAX_WORKERS must be an integer between 1 and 8" >&2
  exit 2
fi

resume_flag="--resume"
if [[ "$RESUME" == "0" ]]; then
  resume_flag="--no-resume"
fi

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting GPT-5.6-Sol judgement-only replay"
echo "Evidence: $EVIDENCE_ROOT"
echo "Output:   $OUT_DIR"
echo "Arms: fixed_global_highlight, metric_local_highlight"
echo "Workers: $MAX_WORKERS"
echo "Blender/rendering/camera selection: disabled"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"
"$PYTHON_BIN" scripts/check_model_endpoint.py \
  --endpoint http://127.0.0.1:4000/v1 \
  --model gpt-5.6-sol \
  --api-key-env LITELLM_MASTER_KEY \
  --timeout-seconds 3000 \
  --max-tokens 200 \
  --no-send-temperature \
  --no-response-format-json \
  --multimodal \
  --image-path "$SMOKE_IMAGE" \
  >"$OUT_DIR/preflight.json"
echo "GPT-5.6-Sol multimodal preflight: passed"

exec "$PYTHON_BIN" scripts/judge_cal_dataset1_camera_evidence.py \
  --evidence-root "$EVIDENCE_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --judge-config "$JUDGE_CONFIG" \
  --out-dir "$OUT_DIR" \
  --max-workers "$MAX_WORKERS" \
  "$resume_flag" \
  --continue-on-error
