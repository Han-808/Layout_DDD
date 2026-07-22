#!/usr/bin/env bash
set -Eeuo pipefail

# Fifty-case P0c relation smoke using the existing NL50 prompt set. The task
# keeps the current asset-backed Blender/P0b path active so unknown explicit
# OOR/OAR predicates have visual evidence for mandatory binary VLM judgement.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
SOURCE_PROMPT_FILE=${SOURCE_PROMPT_FILE:-${REPO_ROOT}/configs/experiments/qwen32b_nl50_prompts.json}

RUN_TAG=${RUN_TAG:-qwen32b_p0c_50_asset_blender_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
TASK_FILE=${TASK_FILE:-${OUT_ROOT}/p0c_50_prompts.json}

mkdir -p "$OUT_ROOT"

"$BENCH_PY" - "$SOURCE_PROMPT_FILE" "$TASK_FILE" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
task_path = Path(sys.argv[2])
payload = json.loads(source_path.read_text(encoding="utf-8"))
cases = payload.get("cases")
if not isinstance(cases, list) or len(cases) != 50:
    raise SystemExit(f"Expected exactly 50 cases in {source_path}, found {len(cases) if isinstance(cases, list) else 'invalid'}")

case_ids = [str(case.get("case_id") or "") for case in cases if isinstance(case, dict)]
if len(case_ids) != 50 or any(not case_id for case_id in case_ids) or len(set(case_ids)) != 50:
    raise SystemExit("NL50 task requires 50 unique non-empty case_id values")

task = {
    "experiment_id": "qwen32b_p0c_50_asset_blender",
    "input_mode": "natural_language_plus_public_generator_structure",
    "prompt_granularity": "fine_grained",
    "asset_mode": "retrieve",
    "evaluation_scope": [
        "prompt_fidelity_oor",
        "prompt_fidelity_oar",
        "structural_validity_p0b",
        "visual_quality",
    ],
    "notes": [
        "Reuses all 50 prompts from qwen32b_nl50_prompts.json without rewriting them.",
        "Known OOR/OAR predicates use frozen deterministic handlers.",
        "Unknown explicit predicates require prompt, structured claim, canonical scene, and Blender renders for binary VLM adjudication.",
        "Reviewed frozen references are the only relation source for scoring; runtime conversion is forbidden.",
    ],
    "cases": cases,
}
task_path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
print(f"Prepared {len(cases)} P0c cases at {task_path}")
print(f"First/last case: {case_ids[0]} / {case_ids[-1]}")
PY

export RUN_TAG OUT_ROOT
export PROMPT_FILE="$TASK_FILE"
export PROMPT_GRANULARITY=fine_grained
export MAX_CASES=${MAX_CASES:-0}
export RESUME=${RESUME:-1}
export CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
export BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
export BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CUDA}
export BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-32}
export BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-1}
export RENDER_WIDTH=${RENDER_WIDTH:-768}
export RENDER_HEIGHT=${RENDER_HEIGHT:-768}

exec bash "${REPO_ROOT}/Support/bash/mnet/run_qwen32b_asset_blender_smoke.sh"
