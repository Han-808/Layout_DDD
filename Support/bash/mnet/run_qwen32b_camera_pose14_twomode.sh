#!/usr/bin/env bash
set -Eeuo pipefail

# Paired camera-policy calibration. Each canonical scene is generated once
# with bbox_track, then reused unchanged by query_cov. Runs are intentionally
# sequential and resumable; completion is preferred over throughput.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
ASSET_ROOT=${ASSET_ROOT:-${REPO_ROOT}/Assets/imaginarium_assets}
ASSET_CSV=${ASSET_CSV:-${ASSET_ROOT}/imaginarium_asset_info.csv}
ASSET_INDEX=${ASSET_INDEX:-${ASSET_ROOT}/.benchmark_index/qwen3_embedding_0_6b}
FIXTURE_ROOT=${FIXTURE_ROOT:-${REPO_ROOT}/configs/experiments/camera_pose14}
PROMPT_FILE=${PROMPT_FILE:-${FIXTURE_ROOT}/cases.json}
GENERATOR_STRUCTURE_DIR=${GENERATOR_STRUCTURE_DIR:-${FIXTURE_ROOT}/generator_structures}
REFERENCE_ANNOTATION_DIR=${REFERENCE_ANNOTATION_DIR:-${FIXTURE_ROOT}/reference_annotations}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}

PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
RUN_TAG=${RUN_TAG:-qwen32b_camera_pose14_twomode_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
RUN_ROOT=$OUT_ROOT
BBOX_ROOT=${OUT_ROOT}/bbox_track
QUERY_ROOT=${OUT_ROOT}/query_cov
COMBINED_SUMMARY=${OUT_ROOT}/summary_28.tsv

RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CUDA}
BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-8}
BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-1}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-3600}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}
CAMERA_POSE_MAX_VIEWS=${CAMERA_POSE_MAX_VIEWS:-2}
QUERY_COV_MAX_STEPS=${QUERY_COV_MAX_STEPS:-1}

export REPO_ROOT BENCH_PY BLENDER_BIN ASSET_ROOT ASSET_CSV ASSET_INDEX
export PROMPT_FILE GENERATOR_STRUCTURE_DIR REFERENCE_ANNOTATION_DIR JUDGE_CONFIG
export PORT BASE_URL ENDPOINT SERVED_MODEL RESUME CONTINUE_ON_ERROR FLUSH_CACHE
export CUDA_VISIBLE_DEVICES BLENDER_RENDER_ENGINE BLENDER_CYCLES_DEVICE
export BLENDER_CYCLES_SAMPLES BLENDER_CYCLES_DENOISING RENDER_WIDTH RENDER_HEIGHT
export BLENDER_TIMEOUT_SECONDS TIMEOUT_SECONDS CAMERA_POSE_MAX_VIEWS
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required path: $1" >&2
    exit 1
  fi
}

flush_cache() {
  if [[ "$FLUSH_CACHE" == "1" ]]; then
    curl --noproxy "*" -fsS -X POST "${BASE_URL}/flush_cache" >/dev/null || true
  fi
}

for path in \
  "$REPO_ROOT" \
  "$BENCH_PY" \
  "$BLENDER_BIN" \
  "$PROMPT_FILE" \
  "$GENERATOR_STRUCTURE_DIR" \
  "$REFERENCE_ANNOTATION_DIR" \
  "$JUDGE_CONFIG" \
  "$ASSET_ROOT" \
  "$ASSET_CSV" \
  "${ASSET_INDEX}.json" \
  "${ASSET_INDEX}.npy"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$BBOX_ROOT" "$QUERY_ROOT"
cd "$REPO_ROOT"

log "preflight: endpoint, fixtures, and Blender"
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
"$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"
echo "$models_json"
"$BENCH_PY" - "$PROMPT_FILE" "$GENERATOR_STRUCTURE_DIR" "$REFERENCE_ANNOTATION_DIR" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
cases = payload.get("cases", [])
if len(cases) != 14:
    raise SystemExit(f"expected 14 cases, found {len(cases)}")
for case in cases:
    case_id = case["case_id"]
    for root in sys.argv[2:]:
        path = Path(root) / f"{case_id}.json"
        if not path.is_file():
            raise SystemExit(f"missing fixture: {path}")
print("cases:", len(cases))
print("moderate:", sum(case.get("difficulty") == "moderate" for case in cases))
print("easy:", sum(case.get("difficulty") == "easy" for case in cases))
PY
"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='${BLENDER_RENDER_ENGINE}'; print('render engine:', bpy.context.scene.render.engine)"

log "phase 1/2: generate each scene once and evaluate bbox_track"
export RUN_TAG="${RUN_TAG}_bbox_track"
export OUT_ROOT="$BBOX_ROOT"
export MAX_CASES=0
export PROMPT_GRANULARITY=fine_grained
export CAMERA_POSE_MODE=bbox_track
export CAMERA_POSE_MAX_STEPS=0
bash "${REPO_ROOT}/Support/bash/mnet/run_qwen32b_asset_blender_smoke.sh"

log "phase 2/2: reuse each bbox_track scene and evaluate query_cov"
CASE_MANIFEST=${RUN_ROOT}/camera_pose14_case_manifest.tsv
"$BENCH_PY" - "$PROMPT_FILE" > "$CASE_MANIFEST" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for case in payload.get("cases", []):
    print(f"{case['case_id']}\t{case.get('difficulty', '')}")
PY

while IFS=$'\t' read -r case_id difficulty; do
  [[ -n "$case_id" ]] || continue
  bbox_case=${BBOX_ROOT}/${case_id}
  query_case=${QUERY_ROOT}/${case_id}
  query_report=${query_case}/evaluation_report.json
  query_log=${QUERY_ROOT}/${case_id}.log
  scene=${bbox_case}/generated_scene.json
  scene_request=${bbox_case}/scene_request.json
  blend_file=${bbox_case}/renders/scene.blend
  collision_geometry=${bbox_case}/renders/collision_geometry_manifest.json
  top_view=${bbox_case}/renders/standardized_top.png
  perspective_view=${bbox_case}/renders/standardized_perspective.png
  generator_structure=${GENERATOR_STRUCTURE_DIR}/${case_id}.json
  reference_annotation=${REFERENCE_ANNOTATION_DIR}/${case_id}.json

  if [[ "$RESUME" == "1" && -f "$query_report" ]]; then
    log "query_cov skipping existing ${case_id} (${difficulty})"
    continue
  fi
  if [[ ! -f "$scene" || ! -f "$scene_request" || ! -f "$blend_file" || ! -f "$top_view" || ! -f "$perspective_view" ]]; then
    log "query_cov cannot run ${case_id}: bbox_track artifacts are incomplete"
    continue
  fi

  mkdir -p "$query_case"
  flush_cache
  log "query_cov running ${case_id} (${difficulty})"
  EVAL_ARGS=(
    --scene "$scene"
    --scene-request "$scene_request"
    --generator-structure "$generator_structure"
    --reference-annotation "$reference_annotation"
    --asset-csv "$ASSET_CSV"
    --asset-root "$ASSET_ROOT"
    --enrich-assets
    --eval-generic-validity
    --eval-oor
    --eval-oar
    --support-enabled
    --p0b-official-mode
    --render-evidence "$top_view"
    --render-evidence "$perspective_view"
    --vlm-judge-config "$JUDGE_CONFIG"
    --camera-pose-mode query_cov
    --camera-blend-file "$blend_file"
    --camera-evidence-dir "$query_case/camera_evidence"
    --camera-pose-max-views "$CAMERA_POSE_MAX_VIEWS"
    --camera-pose-max-steps "$QUERY_COV_MAX_STEPS"
    --blender-bin "$BLENDER_BIN"
    --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS"
    --camera-render-width "$RENDER_WIDTH"
    --camera-render-height "$RENDER_HEIGHT"
    --camera-render-engine "$BLENDER_RENDER_ENGINE"
    --camera-cycles-device "$BLENDER_CYCLES_DEVICE"
    --camera-cycles-samples "$BLENDER_CYCLES_SAMPLES"
    --out "$query_report"
  )
  if [[ -f "$collision_geometry" ]]; then
    EVAL_ARGS+=(--collision-geometry "$collision_geometry")
  fi
  if [[ "$BLENDER_CYCLES_DENOISING" == "1" ]]; then
    EVAL_ARGS+=(--camera-cycles-denoising)
  fi

  set +e
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" evaluate.py "${EVAL_ARGS[@]}" \
    2>&1 | tee "$query_log"
  query_status=${PIPESTATUS[0]}
  set -e
  if (( query_status == 0 )); then
    log "query_cov completed ${case_id}"
  else
    log "query_cov failed ${case_id} exit=${query_status}; see ${query_log}"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$query_status"
    fi
  fi
done < "$CASE_MANIFEST"

log "building paired 28-row summary"
"$BENCH_PY" - "$PROMPT_FILE" "$BBOX_ROOT" "$QUERY_ROOT" "$COMBINED_SUMMARY" <<'PY'
import csv
import json
import sys
from pathlib import Path

prompt_file, bbox_root, query_root, output = map(Path, sys.argv[1:])
cases = json.load(open(prompt_file, encoding="utf-8")).get("cases", [])
rows = []
for case in cases:
    case_id = case["case_id"]
    for mode, root in (("bbox_track", bbox_root), ("query_cov", query_root)):
        report_path = root / case_id / "evaluation_report.json"
        if report_path.is_file():
            report = json.load(open(report_path, encoding="utf-8"))
            categories = report.get("category_reports", {})
            rows.append({
                "case_id": case_id,
                "difficulty": case.get("difficulty", ""),
                "camera_mode": mode,
                "status": "completed",
                "benchmark_score": report.get("benchmark_score"),
                "score_status": report.get("benchmark_score_status"),
                "prompt_fidelity": categories.get("prompt_fidelity", {}).get("score"),
                "structural_validity": categories.get("structural_validity", {}).get("score"),
                "visual_quality": categories.get("visual_quality", {}).get("score"),
                "report": str(report_path),
            })
        else:
            rows.append({
                "case_id": case_id,
                "difficulty": case.get("difficulty", ""),
                "camera_mode": mode,
                "status": "missing_or_failed",
                "benchmark_score": "",
                "score_status": "",
                "prompt_fidelity": "",
                "structural_validity": "",
                "visual_quality": "",
                "report": str(report_path),
            })
fields = [
    "case_id", "difficulty", "camera_mode", "status", "benchmark_score",
    "score_status", "prompt_fidelity", "structural_validity", "visual_quality", "report",
]
with open(output, "w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"completed: {sum(row['status'] == 'completed' for row in rows)} / {len(rows)}")
print(f"summary: {output}")
PY

flush_cache
log "camera_pose14 paired run complete"
echo "Outputs: $RUN_ROOT"
echo "Summary: $COMBINED_SUMMARY"
cat "$COMBINED_SUMMARY"
