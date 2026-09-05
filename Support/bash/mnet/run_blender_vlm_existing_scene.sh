#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke-test the canonical scene -> Blender -> PNG -> VLM judge path without
# rerunning generation or entering asset retrieval.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CPU}
BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-16}
BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-0}
RENDER_WIDTH=${RENDER_WIDTH:-768}
RENDER_HEIGHT=${RENDER_HEIGHT:-768}
ASSET_ROOT=${ASSET_ROOT:-}
REQUIRE_ASSET_MESH=${REQUIRE_ASSET_MESH:-0}
ENDPOINT=${ENDPOINT:-http://127.0.0.1:8298/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}
CAMERA_POSE_MODE=${CAMERA_POSE_MODE:-}
CAMERA_POSE_METRIC_MODES=${CAMERA_POSE_METRIC_MODES:-}
CAMERA_POSE_MAX_VIEWS=${CAMERA_POSE_MAX_VIEWS:-2}
CAMERA_POSE_MAX_STEPS=${CAMERA_POSE_MAX_STEPS:-1}

# The MNET pod has no display server. These defaults make Blender use Mesa's
# headless EGL path and are inherited by the Blender subprocess.
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CASE_DIR" >&2
  echo "Example: $0 outputs/qwen32b_nl5_smoke_20260713_155634/prompt_01" >&2
  exit 2
fi

CASE_DIR=$1
if [[ "$CASE_DIR" != /* ]]; then
  CASE_DIR=${REPO_ROOT}/${CASE_DIR}
fi
SCENE_JSON=${CASE_DIR}/generated_scene.json
SCENE_REQUEST=${CASE_DIR}/scene_request.json
GENERATOR_STRUCTURE=${GENERATOR_STRUCTURE:-${CASE_DIR}/generator_structure.json}
REFERENCE_ANNOTATION=${REFERENCE_ANNOTATION:-${CASE_DIR}/reference_annotation.json}
OUT_DIR=${OUT_DIR:-${CASE_DIR}/blender_vlm_smoke}
RENDER_DIR=${OUT_DIR}/renders
CURRENT_STAGE=preflight

on_error() {
  local status=$?
  echo "==== $(date '+%F %T') FAILED during ${CURRENT_STAGE} (exit ${status}) ====" >&2
  if [[ -f "${RENDER_DIR}/blender.stderr.log" ]]; then
    echo "---- blender.stderr.log (tail) ----" >&2
    tail -n 80 "${RENDER_DIR}/blender.stderr.log" >&2
  fi
  if [[ -f "${RENDER_DIR}/blender.stdout.log" ]]; then
    echo "---- blender.stdout.log (tail) ----" >&2
    tail -n 80 "${RENDER_DIR}/blender.stdout.log" >&2
  fi
  exit "$status"
}
trap on_error ERR

for path in "$REPO_ROOT" "$BENCH_PY" "$BLENDER_BIN" "$JUDGE_CONFIG" "$SCENE_JSON" "$SCENE_REQUEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done
if [[ -n "$ASSET_ROOT" && ! -d "$ASSET_ROOT" ]]; then
  echo "Asset root does not exist: $ASSET_ROOT" >&2
  exit 1
fi
if [[ ! -x "$BLENDER_BIN" ]]; then
  echo "Blender is not executable: $BLENDER_BIN" >&2
  exit 1
fi
case "$CAMERA_POSE_MODE" in
  ""|global_only|bbox_track|visibility_ranked|support_contact_plane|query_cov|auto) ;;
  *)
    echo "Invalid CAMERA_POSE_MODE: $CAMERA_POSE_MODE" >&2
    exit 2 ;;
esac

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

echo "==== $(date '+%F %T') endpoint preflight ===="
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
echo "$models_json"
if ! "$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
expected = sys.argv[1]
raise SystemExit(0 if any(item.get("id") == expected for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"; then
  echo "Expected model ${SERVED_MODEL} was not returned by ${ENDPOINT}/models" >&2
  exit 1
fi

echo "==== $(date '+%F %T') Blender preflight ===="
echo "LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE}"
echo "EGL_PLATFORM=${EGL_PLATFORM}"
echo "BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE}"
echo "BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE}"
echo "BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES}"
echo "BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING}"
echo "RENDER_SIZE=${RENDER_WIDTH}x${RENDER_HEIGHT}"
echo "CAMERA_POSE_MODE=${CAMERA_POSE_MODE:-disabled}"
echo "CAMERA_POSE_METRIC_MODES=${CAMERA_POSE_METRIC_MODES:-none}"
echo "CAMERA_POSE_MAX_VIEWS=${CAMERA_POSE_MAX_VIEWS}"
echo "CAMERA_POSE_MAX_STEPS=${CAMERA_POSE_MAX_STEPS}"
"$BLENDER_BIN" --background --version

CURRENT_STAGE=render
echo "==== $(date '+%F %T') rendering canonical scene ===="
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" - \
  "$BLENDER_BIN" "$SCENE_JSON" "$RENDER_DIR" "$BLENDER_RENDER_ENGINE" \
  "$ASSET_ROOT" "$RENDER_WIDTH" "$RENDER_HEIGHT" "$BLENDER_CYCLES_DEVICE" \
  "$BLENDER_CYCLES_SAMPLES" "$BLENDER_CYCLES_DENOISING" "$REQUIRE_ASSET_MESH" <<'PY'
import json
import sys

from benchmark.rendering import BlenderRenderer

renderer = BlenderRenderer(
    blender_bin=sys.argv[1],
    timeout_seconds=900,
    width=int(sys.argv[6]),
    height=int(sys.argv[7]),
    render_engine=sys.argv[4],
    cycles_device=sys.argv[8],
    cycles_samples=int(sys.argv[9]),
    cycles_denoising=sys.argv[10] == "1",
    require_asset_mesh=sys.argv[11] == "1",
)
manifest = renderer.render_scene(
    scene_path=sys.argv[2],
    out_dir=sys.argv[3],
    asset_root=sys.argv[5] or None,
)
print(json.dumps(manifest, indent=2))
PY

TOP_VIEW=${RENDER_DIR}/standardized_top.png
PERSPECTIVE_VIEW=${RENDER_DIR}/standardized_perspective.png
COLLISION_GEOMETRY=${RENDER_DIR}/collision_geometry_manifest.json
BLEND_FILE=${RENDER_DIR}/scene.blend

CURRENT_STAGE=judge
echo "==== $(date '+%F %T') judging rendered evidence ===="
EVAL_ARGS=(
  --scene "$SCENE_JSON"
  --scene-request "$SCENE_REQUEST"
  --eval-generic-validity
  --eval-oor
  --eval-oar
  --render-evidence "$TOP_VIEW"
  --render-evidence "$PERSPECTIVE_VIEW"
  --vlm-judge-config "$JUDGE_CONFIG"
  --p0b-official-mode
  --out "$OUT_DIR/evaluation_report.json"
)
if [[ -f "$GENERATOR_STRUCTURE" ]]; then
  EVAL_ARGS+=(--generator-structure "$GENERATOR_STRUCTURE")
fi
if [[ -f "$REFERENCE_ANNOTATION" ]]; then
  EVAL_ARGS+=(--reference-annotation "$REFERENCE_ANNOTATION")
else
  echo "Warning: no reviewed reference annotation; fine-grained prompt fidelity will be unscored." >&2
fi
if [[ -f "$COLLISION_GEOMETRY" ]]; then
  EVAL_ARGS+=(--collision-geometry "$COLLISION_GEOMETRY")
else
  echo "Warning: no collision geometry manifest; P0b will use OBB evidence only." >&2
fi
if [[ -n "$CAMERA_POSE_MODE" || -n "$CAMERA_POSE_METRIC_MODES" ]]; then
  EVAL_ARGS+=(
    --camera-selector-config "$JUDGE_CONFIG"
    --camera-blend-file "$BLEND_FILE"
    --camera-evidence-dir "$OUT_DIR/camera_evidence"
    --camera-pose-max-views "$CAMERA_POSE_MAX_VIEWS"
    --camera-pose-max-steps "$CAMERA_POSE_MAX_STEPS"
    --blender-bin "$BLENDER_BIN"
    --camera-render-width "$RENDER_WIDTH"
    --camera-render-height "$RENDER_HEIGHT"
    --camera-render-engine "$BLENDER_RENDER_ENGINE"
    --camera-cycles-device "$BLENDER_CYCLES_DEVICE"
    --camera-cycles-samples "$BLENDER_CYCLES_SAMPLES"
  )
  if [[ -n "$CAMERA_POSE_MODE" ]]; then
    EVAL_ARGS+=(--camera-pose-mode "$CAMERA_POSE_MODE")
  fi
  IFS=',' read -r -a CAMERA_METRIC_OVERRIDES <<< "$CAMERA_POSE_METRIC_MODES"
  for override in "${CAMERA_METRIC_OVERRIDES[@]}"; do
    if [[ -n "$override" ]]; then
      EVAL_ARGS+=(--camera-pose-metric-mode "$override")
    fi
  done
  if [[ "$BLENDER_CYCLES_DENOISING" == "1" ]]; then
    EVAL_ARGS+=(--camera-cycles-denoising)
  fi
fi
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" evaluate.py "${EVAL_ARGS[@]}"

CURRENT_STAGE=summary
echo "==== $(date '+%F %T') smoke complete ===="
echo "Output: $OUT_DIR"
"$BENCH_PY" - "$OUT_DIR/evaluation_report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print("benchmark_score:", report.get("benchmark_score"))
print("benchmark_score_status:", report.get("benchmark_score_status"))
for name, category in report.get("category_reports", {}).items():
    print(
        f"{name}: status={category.get('status')} score={category.get('score')} "
        f"reason={category.get('reason')}"
    )
    judgement = category.get("judgement")
    if isinstance(judgement, dict):
        print(f"  judge_summary: {judgement.get('summary')}")
        print(f"  judge_confidence: {judgement.get('confidence')}")
        print(f"  judge_model: {judgement.get('model')}")
        print(f"  images_used: {len(judgement.get('images_used', []))}")
PY
