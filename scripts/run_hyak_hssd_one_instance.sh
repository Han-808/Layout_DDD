#!/usr/bin/env bash
set -Eeuo pipefail

# Run one HSSD-HAB instance through the Hyak Qwen3-VL endpoint.
#
# Usage:
#   bash scripts/run_hyak_hssd_one_instance.sh <instance_index:0-9> <mode_flag:0|1>
#
# Mode flag:
#   0 = compact_objects
#   1 = compact_objects_with_estimated_relations

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_hyak_hssd_one_instance.sh <instance_index:0-9> <mode_flag:0|1>" >&2
  exit 2
fi

INSTANCE_INDEX="$1"
MODE_FLAG="$2"

if ! [[ "${INSTANCE_INDEX}" =~ ^[0-9]$ ]]; then
  echo "ERROR: instance_index must be an integer from 0 to 9." >&2
  exit 2
fi

if [[ "${MODE_FLAG}" != "0" && "${MODE_FLAG}" != "1" ]]; then
  echo "ERROR: mode_flag must be 0 or 1." >&2
  exit 2
fi

SCENE_IDS=(
  "102343992"
  "102344022"
  "102344049"
  "102344094"
  "102344115"
  "102344193"
  "102344250"
  "102344280"
  "102344307"
  "102344328"
)

if [[ "${MODE_FLAG}" == "0" ]]; then
  MODE="compact_objects"
else
  MODE="compact_objects_with_estimated_relations"
fi

SCENE_ID="${SCENE_IDS[${INSTANCE_INDEX}]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BENCH_PYTHON="${BENCH_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "${BENCH_PYTHON}" ]]; then
  BENCH_PYTHON="${PYTHON:-python}"
fi

HSSD_ROOT="${HSSD_ROOT:-${REPO_ROOT}/data/external/hssd-hab}"
CASES_ROOT="${CASES_ROOT:-${REPO_ROOT}/data/external/hssd-hab-converted/hyak_single}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs/hyak_hssd_single}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_ROOT}/${SCENE_ID}/${MODE}/${RUN_TAG}"

MODEL_NAME="${MODEL_NAME:-qwen3vl_sglang_32b}"
JUDGE_MODEL="${JUDGE_MODEL:-same}"
MODEL_ENDPOINT="${MODEL_ENDPOINT:-http://127.0.0.1:8000/v1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-32B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
MAX_REPAIR_ITERATIONS="${MAX_REPAIR_ITERATIONS:-0}"
VALID_SOURCE="${VALID_SOURCE:-overall_valid}"
TEMPERATURE="${TEMPERATURE:-0.0}"

extra_args=()
if [[ -n "${MAX_OBJECTS:-}" ]]; then
  extra_args+=(--max-objects "${MAX_OBJECTS}")
fi
if [[ "${NO_DOWNLOAD:-0}" == "1" ]]; then
  extra_args+=(--no-download)
fi

mkdir -p "${OUT_DIR}"
cd "${REPO_ROOT}"

echo "scene_index=${INSTANCE_INDEX}"
echo "scene_id=${SCENE_ID}"
echo "mode_flag=${MODE_FLAG}"
echo "mode=${MODE}"
echo "endpoint=${MODEL_ENDPOINT}"
echo "out=${OUT_DIR}"

cmd=(
  "${BENCH_PYTHON}" "${REPO_ROOT}/scripts/run_hssd_hab_10_qwen_validity.py"
  --hssd-root "${HSSD_ROOT}" \
  --cases-root "${CASES_ROOT}" \
  --scene-ids "${SCENE_ID}" \
  --modes "${MODE}" \
  --model "${MODEL_NAME}" \
  --judge-model "${JUDGE_MODEL}" \
  --model-endpoint "${MODEL_ENDPOINT}" \
  --model-id "${MODEL_ID}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --max-repair-iterations "${MAX_REPAIR_ITERATIONS}" \
  --valid-source "${VALID_SOURCE}" \
  --out "${OUT_DIR}"
)

if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

"${cmd[@]}"

echo "result:"
cat "${OUT_DIR}/validity_results.tsv"
