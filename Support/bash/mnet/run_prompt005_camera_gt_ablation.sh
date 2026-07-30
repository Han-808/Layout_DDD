#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled camera-policy ablation on one frozen scene and 13 frozen P0b
# events. No generator, converter, retriever, asset binding, or detector is run.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
PY=${PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
SOURCE_CASE=${SOURCE_CASE:-${REPO_ROOT}/outputs/qwen32b_camera_pose14_twomode_20260716_202945/bbox_track/prompt_005}
GT=${GT:-${REPO_ROOT}/configs/experiments/camera_pose14/prompt_005_p0b_gt.json}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}
RUN_TAG=${RUN_TAG:-prompt005_camera_gt_ablation_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/outputs/${RUN_TAG}}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
CYCLES_SAMPLES=${CYCLES_SAMPLES:-8}
PREVIEW_WIDTH=${PREVIEW_WIDTH:-256}
PREVIEW_HEIGHT=${PREVIEW_HEIGHT:-256}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-1}
BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-1800}

export CUDA_VISIBLE_DEVICES
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

SCENE=${SOURCE_CASE}/generated_scene.json
REPORT=${SOURCE_CASE}/evaluation_report.json
BLEND=${SOURCE_CASE}/renders/scene.blend
TOP=${SOURCE_CASE}/renders/standardized_top.png
PERSPECTIVE=${SOURCE_CASE}/renders/standardized_perspective.png

for path in "$PY" "$BLENDER_BIN" "$SCENE" "$REPORT" "$BLEND" "$TOP" "$PERSPECTIVE" "$GT" "$JUDGE_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

echo "==== $(date '+%F %T') endpoint preflight ===="
curl --noproxy "*" -fsS http://127.0.0.1:8298/v1/models
echo
echo "==== $(date '+%F %T') Blender preflight ===="
"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='CYCLES'; print('Cycles preflight OK')"

echo "==== $(date '+%F %T') controlled 13-event x 3-mode ablation ===="
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$PY" scripts/run_p0b_camera_ablation.py \
  --scene "$SCENE" \
  --source-report "$REPORT" \
  --gt "$GT" \
  --blend-file "$BLEND" \
  --overview "$TOP" \
  --overview "$PERSPECTIVE" \
  --judge-config "$JUDGE_CONFIG" \
  --out-dir "$OUT_DIR" \
  --mode global_only \
  --mode bbox_track \
  --mode visibility_ranked \
  --blender-bin "$BLENDER_BIN" \
  --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS" \
  --render-width "$RENDER_WIDTH" \
  --render-height "$RENDER_HEIGHT" \
  --render-engine CYCLES \
  --cycles-device CUDA \
  --cycles-samples "$CYCLES_SAMPLES" \
  --cycles-denoising \
  --preview-render-engine CYCLES \
  --preview-width "$PREVIEW_WIDTH" \
  --preview-height "$PREVIEW_HEIGHT" \
  --preview-cycles-samples "$PREVIEW_SAMPLES" \
  --max-views 2 \
  --candidate-count 6 \
  2>&1 | tee "$OUT_DIR/run.log"

echo "==== $(date '+%F %T') scoring against frozen GT ===="
PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$PY" scripts/score_p0b_camera_ablation.py \
  --gt "$GT" \
  --run-dir "$OUT_DIR" \
  | tee "$OUT_DIR/score.log"

echo "==== complete ===="
echo "Output: $OUT_DIR"
column -t -s $'\t' "$OUT_DIR/summary.tsv" 2>/dev/null || cat "$OUT_DIR/summary.tsv"
