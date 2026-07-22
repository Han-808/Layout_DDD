#!/usr/bin/env bash
set -Eeuo pipefail

# Ten-case P0b calibration smoke using diverse prompts from the frozen NL50
# set. Runs semantic top-1 retrieval, asset-backed Blender evidence, and the
# current official Collision/OOB/Support adjudication path.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
SOURCE_PROMPT_FILE=${SOURCE_PROMPT_FILE:-${REPO_ROOT}/configs/experiments/qwen32b_nl50_prompts.json}

RUN_TAG=${RUN_TAG:-qwen32b_p0b_10_asset_blender_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
TASK_FILE=${TASK_FILE:-${OUT_ROOT}/p0b_10_prompts.json}

CASE_IDS=(
  prompt_001
  prompt_005
  prompt_010
  prompt_015
  prompt_018
  prompt_022
  prompt_027
  prompt_033
  prompt_038
  prompt_050
)

mkdir -p "$OUT_ROOT"

"$BENCH_PY" - "$SOURCE_PROMPT_FILE" "$TASK_FILE" "${CASE_IDS[@]}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
requested_ids = sys.argv[3:]
payload = json.loads(source_path.read_text(encoding="utf-8"))
by_id = {case["case_id"]: case for case in payload.get("cases", [])}
missing = [case_id for case_id in requested_ids if case_id not in by_id]
if missing:
    raise SystemExit(f"Missing case IDs in {source_path}: {missing}")

task = {
    "experiment_id": "qwen32b_p0b_10_asset_blender",
    "input_mode": "natural_language_plus_public_generator_structure",
    "prompt_granularity": "fine_grained",
    "asset_mode": "retrieve",
    "notes": [
        "Ten diverse cases reused from qwen32b_nl50_prompts.json.",
        "Runs top-1 retrieval, asset-backed Blender mesh evidence, and official P0b adjudication.",
        "Granularity, public generator structures, and private references are frozen before runtime.",
    ],
    "cases": [by_id[case_id] for case_id in requested_ids],
}
task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
print(f"Prepared {len(task['cases'])} cases at {task_path}")
PY

export RUN_TAG OUT_ROOT
export PROMPT_FILE="$TASK_FILE"
export PROMPT_GRANULARITY=fine_grained
export MAX_CASES=0
export RESUME=${RESUME:-1}
export CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
export BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
export BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CUDA}
export BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-32}
export BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-1}
export RENDER_WIDTH=${RENDER_WIDTH:-768}
export RENDER_HEIGHT=${RENDER_HEIGHT:-768}

exec bash "${REPO_ROOT}/Support/bash/mnet/run_qwen32b_asset_blender_smoke.sh"
