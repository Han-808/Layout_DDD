#!/usr/bin/env bash
set -Eeuo pipefail

# Five clean repository scenes x {clean, Collision, OOB, Support, OAR}.
# Generation, conversion, and retrieval are deliberately skipped.  Each frozen
# canonical scene is rendered, evaluated, assigned transform-derived P0b event
# GT, and optionally replayed through the four camera/highlight ablation arms.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
ASSET_ROOT=${ASSET_ROOT:-${REPO_ROOT}/Assets/imaginarium_assets}
ASSET_CSV=${ASSET_CSV:-${ASSET_ROOT}/imaginarium_asset_info.csv}
FIXTURE_ROOT=${FIXTURE_ROOT:-${REPO_ROOT}/configs/experiments/p0b_source_distortion5}
CASE_FILE=${CASE_FILE:-${FIXTURE_ROOT}/cases.json}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}

PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}

RUN_TAG=${RUN_TAG:-p0b_source_distortion5}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
SOURCE_ROOT=${SOURCE_ROOT:-${OUT_ROOT}/source_reports}
GT_ROOT=${GT_ROOT:-${OUT_ROOT}/gt}
ABLATION_ROOT=${ABLATION_ROOT:-${OUT_ROOT}/ablation}
RESULT_ROOT=${RESULT_ROOT:-${OUT_ROOT}/results}
CASE_MANIFEST=${CASE_MANIFEST:-${OUT_ROOT}/case_manifest.tsv}
SOURCE_SUMMARY=${SOURCE_SUMMARY:-${OUT_ROOT}/source_summary.tsv}

PHASE=${PHASE:-all}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}
MAX_CASES=${MAX_CASES:-0}
FAMILIES=${FAMILIES:-clean,collision,oob,support,oar}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-3600}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
CYCLES_SAMPLES=${CYCLES_SAMPLES:-8}
PREVIEW_WIDTH=${PREVIEW_WIDTH:-256}
PREVIEW_HEIGHT=${PREVIEW_HEIGHT:-256}
PREVIEW_SAMPLES=${PREVIEW_SAMPLES:-1}
MAX_VIEWS=${MAX_VIEWS:-2}
CANDIDATE_COUNT=${CANDIDATE_COUNT:-6}

export CUDA_VISIBLE_DEVICES
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

case "$PHASE" in
  evaluate|ablate|all) ;;
  *) echo "PHASE must be evaluate, ablate, or all" >&2; exit 2 ;;
esac

for path in \
  "$REPO_ROOT" "$BENCH_PY" "$BLENDER_BIN" "$ASSET_ROOT" "$ASSET_CSV" \
  "$CASE_FILE" "$JUDGE_CONFIG" \
  "${REPO_ROOT}/evaluate.py" \
  "${REPO_ROOT}/scripts/render_canonical_scene.py" \
  "${REPO_ROOT}/scripts/build_programmatic_distortion_gt.py" \
  "${REPO_ROOT}/scripts/run_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/score_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/aggregate_source_distortion_experiment.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$SOURCE_ROOT" "$GT_ROOT" "$ABLATION_ROOT" "$RESULT_ROOT"
cd "$REPO_ROOT"

log "preflight: endpoint, 25 frozen fixtures, asset database, and Blender"
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
"$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"
echo "$models_json"

"$BENCH_PY" - "$CASE_FILE" "$FIXTURE_ROOT" "$FAMILIES" > "$CASE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
allowed = {value.strip() for value in sys.argv[3].split(",") if value.strip()}
cases = payload.get("cases", [])
if payload.get("total_case_count") != 25 or len(cases) != 25:
    raise SystemExit(f"expected 25 frozen variants, found {len(cases)}")
for case in cases:
    if case["family"] not in allowed:
        continue
    fixture = (Path(sys.argv[2]) / case["fixture_dir"]).resolve()
    for name in ("generated_scene.json", "scene_request.json", "reference_annotation.json", "distortion_manifest.json"):
        if not (fixture / name).is_file():
            raise SystemExit(f"missing fixture artifact: {fixture / name}")
    print("\t".join([case["case_id"], case["base_case_id"], case["family"], str(fixture)]))
PY

"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='CYCLES'; print('Cycles preflight OK')"

evaluate_cases() {
  log "direct evaluation: frozen canonical scenes; generator skipped"
  printf 'case_id\tbase_case_id\tfamily\tstatus\trouted_events\toutput_dir\n' > "$SOURCE_SUMMARY"
  local total_cases
  total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
  local case_number=0

  while IFS=$'\t' read -r case_id base_case_id family fixture_dir; do
    case_number=$((case_number + 1))
    if (( MAX_CASES > 0 && case_number > MAX_CASES )); then
      log "reached MAX_CASES=${MAX_CASES}"
      break
    fi
    local case_out=${SOURCE_ROOT}/${case_id}
    local render_dir=${case_out}/renders
    local report=${case_out}/evaluation_report.json
    local gt_path=${GT_ROOT}/${case_id}.json
    local scene=${fixture_dir}/generated_scene.json
    local request=${fixture_dir}/scene_request.json
    local annotation=${fixture_dir}/reference_annotation.json
    local distortion=${fixture_dir}/distortion_manifest.json
    mkdir -p "$case_out"

    if [[ "$RESUME" == "1" \
      && -f "$report" \
      && -f "${render_dir}/scene.blend" \
      && -f "${render_dir}/standardized_top.png" \
      && -f "${render_dir}/standardized_perspective.png" ]]; then
      log "[${case_number}/${total_cases}] existing direct report ${case_id}"
    else
      flush_cache
      log "[${case_number}/${total_cases}] render + direct evaluate ${case_id} family=${family}"
      set +e
      PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" scripts/render_canonical_scene.py \
        --scene "$scene" \
        --out-dir "$render_dir" \
        --blender-bin "$BLENDER_BIN" \
        --asset-root "$ASSET_ROOT" \
        --timeout-seconds "$BLENDER_TIMEOUT_SECONDS" \
        --width "$RENDER_WIDTH" \
        --height "$RENDER_HEIGHT" \
        --render-engine CYCLES \
        --cycles-device CUDA \
        --cycles-samples "$CYCLES_SAMPLES" \
        --cycles-denoising \
        --require-asset-mesh \
        > "${case_out}/render.log" 2>&1
      render_status=$?
      set -e
      if (( render_status != 0 )); then
        printf '%s\t%s\t%s\trender_failed_%s\t\t%s\n' "$case_id" "$base_case_id" "$family" "$render_status" "$case_out" >> "$SOURCE_SUMMARY"
        log "render failed ${case_id}; see ${case_out}/render.log"
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$render_status"; fi
        continue
      fi

      eval_args=(
        --scene "$scene"
        --scene-request "$request"
        --reference-annotation "$annotation"
        --asset-csv "$ASSET_CSV"
        --asset-root "$ASSET_ROOT"
        --eval-generic-validity
        --eval-oor
        --eval-oar
        --support-enabled
        --p0b-official-mode
        --render-evidence "${render_dir}/standardized_top.png"
        --render-evidence "${render_dir}/standardized_perspective.png"
        --vlm-judge-config "$JUDGE_CONFIG"
        --out "$report"
      )
      if [[ -f "${render_dir}/collision_geometry_manifest.json" ]]; then
        eval_args+=(--collision-geometry "${render_dir}/collision_geometry_manifest.json")
      fi
      set +e
      PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" evaluate.py "${eval_args[@]}" \
        > "${case_out}/evaluate.log" 2>&1
      evaluate_status=$?
      set -e
      if (( evaluate_status != 0 )); then
        printf '%s\t%s\t%s\tevaluate_failed_%s\t\t%s\n' "$case_id" "$base_case_id" "$family" "$evaluate_status" "$case_out" >> "$SOURCE_SUMMARY"
        log "evaluation failed ${case_id}; see ${case_out}/evaluate.log"
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$evaluate_status"; fi
        continue
      fi
    fi

    set +e
    "$BENCH_PY" scripts/build_programmatic_distortion_gt.py \
      --source-report "$report" \
      --distortion-manifest "$distortion" \
      --out "$gt_path" \
      > "${case_out}/gt.log" 2>&1
    gt_status=$?
    set -e
    if (( gt_status != 0 )); then
      printf '%s\t%s\t%s\tgt_failed_%s\t\t%s\n' "$case_id" "$base_case_id" "$family" "$gt_status" "$case_out" >> "$SOURCE_SUMMARY"
      log "GT coverage failed ${case_id}; see ${case_out}/gt.log"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$gt_status"; fi
      continue
    fi
    routed=$("$BENCH_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["events"]))' "$gt_path")
    printf '%s\t%s\t%s\tcompleted\t%s\t%s\n' "$case_id" "$base_case_id" "$family" "$routed" "$case_out" >> "$SOURCE_SUMMARY"
  done < "$CASE_MANIFEST"
  cat "$SOURCE_SUMMARY"
}

ablate_cases() {
  log "camera/highlight ablation over transform-labeled P0b events"
  local case_number=0
  local total_cases
  total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
  while IFS=$'\t' read -r case_id base_case_id family fixture_dir; do
    case_number=$((case_number + 1))
    if (( MAX_CASES > 0 && case_number > MAX_CASES )); then break; fi
    local case_out=${SOURCE_ROOT}/${case_id}
    local report=${case_out}/evaluation_report.json
    local render_dir=${case_out}/renders
    local gt_path=${GT_ROOT}/${case_id}.json
    local ablation_out=${ABLATION_ROOT}/${case_id}
    local missing_paths=()
    for path in "$report" "$gt_path" "${render_dir}/scene.blend" "${render_dir}/standardized_top.png" "${render_dir}/standardized_perspective.png"; do
      if [[ ! -e "$path" ]]; then
        missing_paths+=("$path")
      fi
    done
    if (( ${#missing_paths[@]} > 0 )); then
      log "[${case_number}/${total_cases}] skip ${case_id}: missing ablation prerequisites"
      printf ' - %s\n' "${missing_paths[@]}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit 1; fi
      continue
    fi
    event_count=$("$BENCH_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["events"]))' "$gt_path")
    if (( event_count == 0 )); then
      log "[${case_number}/${total_cases}] ${case_id}: no routed P0b event; source OAR report retained"
      continue
    fi
    mkdir -p "$ablation_out"
    flush_cache
    log "[${case_number}/${total_cases}] ablate ${case_id} events=${event_count}"
    args=(
      --scene "${fixture_dir}/generated_scene.json"
      --source-report "$report"
      --gt "$gt_path"
      --blend-file "${render_dir}/scene.blend"
      --overview "${render_dir}/standardized_top.png"
      --overview "${render_dir}/standardized_perspective.png"
      --judge-config "$JUDGE_CONFIG"
      --out-dir "$ablation_out"
      --arm global_raw
      --arm visibility_raw
      --arm visibility_highlight
      --arm visibility_highlight_global
      --collision-overlay
      --continue-on-error
      --blender-bin "$BLENDER_BIN"
      --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS"
      --render-width "$RENDER_WIDTH"
      --render-height "$RENDER_HEIGHT"
      --render-engine CYCLES
      --cycles-device CUDA
      --cycles-samples "$CYCLES_SAMPLES"
      --cycles-denoising
      --preview-render-engine CYCLES
      --preview-width "$PREVIEW_WIDTH"
      --preview-height "$PREVIEW_HEIGHT"
      --preview-cycles-samples "$PREVIEW_SAMPLES"
      --max-views "$MAX_VIEWS"
      --candidate-count "$CANDIDATE_COUNT"
    )
    if [[ -f "${render_dir}/collision_geometry_manifest.json" ]]; then
      args+=(--collision-geometry "${render_dir}/collision_geometry_manifest.json")
    fi
    if [[ "$RESUME" == "1" ]]; then args+=(--resume); else args+=(--no-resume); fi

    set +e
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" scripts/run_p0b_camera_ablation.py "${args[@]}" \
      2>&1 | tee "${ablation_out}/run.log"
    ablation_status=${PIPESTATUS[0]}
    set -e
    if (( ablation_status != 0 )); then
      log "ablation failed ${case_id} exit=${ablation_status}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$ablation_status"; fi
      continue
    fi
    set +e
    "$BENCH_PY" scripts/score_p0b_camera_ablation.py \
      --gt "$gt_path" \
      --run-dir "$ablation_out" \
      | tee "${ablation_out}/score.log"
    score_status=${PIPESTATUS[0]}
    set -e
    if (( score_status != 0 )); then
      log "scoring failed ${case_id} exit=${score_status}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then exit "$score_status"; fi
    fi
  done < "$CASE_MANIFEST"
}

aggregate() {
  "$BENCH_PY" scripts/aggregate_source_distortion_experiment.py \
    --cases "$CASE_FILE" \
    --source-root "$SOURCE_ROOT" \
    --ablation-root "$ABLATION_ROOT" \
    --out-dir "$RESULT_ROOT"
}

if [[ "$PHASE" == "evaluate" || "$PHASE" == "all" ]]; then
  evaluate_cases
fi
if [[ "$PHASE" == "ablate" || "$PHASE" == "all" ]]; then
  ablate_cases
fi
aggregate

log "source-scene distortion job complete"
echo "Raw direct reports: $SOURCE_ROOT"
echo "Programmatic event GT: $GT_ROOT"
echo "Camera ablation: $ABLATION_ROOT"
echo "Aggregates: $RESULT_ROOT"
