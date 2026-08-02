#!/usr/bin/env bash
set -Eeuo pipefail

# Asset-backed I2 smoke: reviewed public generator structure -> semantic top-1
# retrieval -> layout JSON generation -> Blender Cycles -> private-reference
# evaluation. Runtime NL conversion is intentionally forbidden.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
ASSET_ROOT=${ASSET_ROOT:-${REPO_ROOT}/Assets/imaginarium_assets}
ASSET_CSV=${ASSET_CSV:-${ASSET_ROOT}/imaginarium_asset_info.csv}
ASSET_INDEX=${ASSET_INDEX:-${ASSET_ROOT}/.benchmark_index/qwen3_embedding_0_6b}
PROMPT_FILE=${PROMPT_FILE:-${REPO_ROOT}/configs/experiments/qwen32b_nl50_prompts.json}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}
GENERATOR_STRUCTURE_DIR=${GENERATOR_STRUCTURE_DIR:-}
REFERENCE_ANNOTATION_DIR=${REFERENCE_ANNOTATION_DIR:-}
PROMPT_GRANULARITY=${PROMPT_GRANULARITY:-fine_grained}

PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
BLENDER_CYCLES_DEVICE=${BLENDER_CYCLES_DEVICE:-CPU}
BLENDER_CYCLES_SAMPLES=${BLENDER_CYCLES_SAMPLES:-16}
BLENDER_CYCLES_DENOISING=${BLENDER_CYCLES_DENOISING:-0}
GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}
BLENDER_TIMEOUT_SECONDS=${BLENDER_TIMEOUT_SECONDS:-1800}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
CAMERA_POSE_MODE=${CAMERA_POSE_MODE:-}
CAMERA_POSE_METRIC_MODES=${CAMERA_POSE_METRIC_MODES:-}
CAMERA_POSE_MAX_VIEWS=${CAMERA_POSE_MAX_VIEWS:-2}
CAMERA_POSE_MAX_STEPS=${CAMERA_POSE_MAX_STEPS:-1}
MAX_CASES=${MAX_CASES:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}
RESUME=${RESUME:-1}

BLENDER_DENOISING_FLAG=--no-blender-cycles-denoising
if [[ "$BLENDER_CYCLES_DENOISING" == "1" ]]; then
  BLENDER_DENOISING_FLAG=--blender-cycles-denoising
fi

CAMERA_POSE_ARGS=()
case "$CAMERA_POSE_MODE" in
  ""|global_only|bbox_track|visibility_ranked|support_contact_plane|query_cov|auto) ;;
  *)
    echo "Invalid CAMERA_POSE_MODE: $CAMERA_POSE_MODE" >&2
    exit 2 ;;
esac
if [[ -n "$CAMERA_POSE_MODE" || -n "$CAMERA_POSE_METRIC_MODES" ]]; then
  CAMERA_POSE_ARGS=(
    --camera-pose-max-views "$CAMERA_POSE_MAX_VIEWS"
    --camera-pose-max-steps "$CAMERA_POSE_MAX_STEPS"
  )
  if [[ -n "$CAMERA_POSE_MODE" ]]; then
    CAMERA_POSE_ARGS+=(--camera-pose-mode "$CAMERA_POSE_MODE")
  fi
  IFS=',' read -r -a CAMERA_METRIC_OVERRIDES <<< "$CAMERA_POSE_METRIC_MODES"
  for override in "${CAMERA_METRIC_OVERRIDES[@]}"; do
    if [[ -n "$override" ]]; then
      CAMERA_POSE_ARGS+=(--camera-pose-metric-mode "$override")
    fi
  done
fi

RUN_TAG=${RUN_TAG:-qwen32b_asset_blender_smoke_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
ROOM_ROOT=${OUT_ROOT}/rooms
CASE_MANIFEST=${OUT_ROOT}/case_manifest.tsv
SUMMARY_FILE=${OUT_ROOT}/summary.tsv
ADAPTER_CONFIG=${OUT_ROOT}/layout_json_adapter.json

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export LIBGL_ALWAYS_SOFTWARE=${LIBGL_ALWAYS_SOFTWARE:-1}
export EGL_PLATFORM=${EGL_PLATFORM:-surfaceless}

log() {
  echo "==== $(date '+%F %T') $* ===="
}

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    if [[ "$path" == "${ASSET_INDEX}.json" || "$path" == "${ASSET_INDEX}.npy" ]]; then
      echo "Build it first with scripts/build_asset_index.py; ASSET_INDEX is a prefix, not a directory." >&2
    fi
    exit 1
  fi
}

flush_cache() {
  if [[ "$FLUSH_CACHE" == "1" ]]; then
    log "flushing SGLang request cache"
    curl --noproxy "*" -fsS -X POST "${BASE_URL}/flush_cache" || true
    echo
  fi
}

case_fields() {
  local case_out=$1
  "$BENCH_PY" - "$case_out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scene = json.load(open(root / "generated_scene.json", encoding="utf-8"))
render = json.load(open(root / "renders" / "render_manifest.json", encoding="utf-8"))
report = json.load(open(root / "evaluation_report.json", encoding="utf-8"))
binding = scene.get("metadata", {}).get("asset_binding", {})
coverage = render.get("asset_coverage", {})
categories = report.get("category_reports", {})
structural_report = categories.get("structural_validity", {}).get("report", {})
validity_metrics = structural_report.get("metrics", {}) if isinstance(structural_report, dict) else {}
reports = report.get("reports", {})

def score(name):
    value = categories.get(name, {}).get("score")
    return "" if value is None else str(value)

def metric_field(name, field):
    metric = validity_metrics.get(name, {})
    value = metric.get(field) if isinstance(metric, dict) else None
    return "" if value is None else str(value)

def relation_field(family, field):
    family_report = reports.get(family, {})
    if not isinstance(family_report, dict):
        return ""
    if field == "vlm_resolved":
        value = family_report.get("runtime", {}).get("vlm_fallback", {}).get("num_calls_resolved")
    elif field == "vlm_pending":
        value = family_report.get("coverage", {}).get("vlm_pending_count")
    else:
        value = family_report.get(field)
    return "" if value is None else str(value)

values = [
    binding.get("bound_object_count"),
    binding.get("unresolved_object_count"),
    coverage.get("asset_mesh_count"),
    coverage.get("bbox_proxy_count"),
    coverage.get("asset_mesh_rate"),
    report.get("benchmark_score"),
    report.get("benchmark_score_status"),
    score("prompt_fidelity"),
    score("structural_validity"),
    score("visual_quality"),
    metric_field("collision", "status"),
    metric_field("collision", "score"),
    metric_field("oob", "status"),
    metric_field("oob", "score"),
    metric_field("support", "status"),
    metric_field("support", "score"),
    relation_field("oor", "status"),
    relation_field("oor", "score"),
    relation_field("oor", "num_checks_called"),
    relation_field("oor", "num_passed"),
    relation_field("oor", "num_failed"),
    relation_field("oor", "vlm_resolved"),
    relation_field("oor", "vlm_pending"),
    relation_field("oar", "status"),
    relation_field("oar", "score"),
    relation_field("oar", "num_checks_called"),
    relation_field("oar", "num_passed"),
    relation_field("oar", "num_failed"),
    relation_field("oar", "vlm_resolved"),
    relation_field("oar", "vlm_pending"),
]
print("\t".join("" if value is None else str(value) for value in values))
PY
}

if [[ -z "$GENERATOR_STRUCTURE_DIR" || -z "$REFERENCE_ANNOTATION_DIR" ]]; then
  echo "Asset-backed benchmark runs require GENERATOR_STRUCTURE_DIR and REFERENCE_ANNOTATION_DIR." >&2
  echo "The first is public I2 input; the second is private, reviewed scoring ground truth." >&2
  exit 1
fi

for path in \
  "$REPO_ROOT" \
  "$BENCH_PY" \
  "$BLENDER_BIN" \
  "$ASSET_ROOT" \
  "$ASSET_CSV" \
  "${ASSET_INDEX}.json" \
  "${ASSET_INDEX}.npy" \
  "$PROMPT_FILE" \
  "$JUDGE_CONFIG" \
  "$GENERATOR_STRUCTURE_DIR" \
  "$REFERENCE_ANNOTATION_DIR" \
  "${REPO_ROOT}/scripts/run_scene_harness.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$ROOM_ROOT"
cd "$REPO_ROOT"

log "checking server ${ENDPOINT} model=${SERVED_MODEL}"
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
"$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"
echo "$models_json"

log "checking asset index and Blender"
"$BENCH_PY" - "$ASSET_INDEX" <<'PY'
import json, sys
from pathlib import Path
prefix = Path(sys.argv[1])
meta = json.load(open(prefix.with_suffix(".json"), encoding="utf-8"))
print("indexed assets:", len(meta.get("jid_list", [])))
print("embedding bytes:", prefix.with_suffix(".npy").stat().st_size)
PY
"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='${BLENDER_RENDER_ENGINE}'; print('render engine:', bpy.context.scene.render.engine)"

cat > "$ADAPTER_CONFIG" <<JSON
{
  "temperature": 0.0,
  "max_tokens": ${GENERATION_MAX_TOKENS},
  "context_length": ${CONTEXT_LENGTH},
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "response_format_json": true,
  "max_retries": 1
}
JSON

"$BENCH_PY" - "$PROMPT_FILE" > "$CASE_MANIFEST" <<'PY'
import base64, json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
for case in payload["cases"]:
    instruction = base64.b64encode(case["instruction"].encode("utf-8")).decode("ascii")
    room = base64.b64encode(json.dumps(case.get("room"), separators=(",", ":")).encode("utf-8")).decode("ascii")
    print(f'{case["case_id"]}\t{case["scene_type"]}\t{instruction}\t{room}')
PY

printf "case_id\tstatus\tbound\tunresolved\tasset_meshes\tproxies\tasset_mesh_rate\tbenchmark_score\tscore_status\tprompt_fidelity\tstructural_validity\tvisual_quality\tcollision_status\tcollision_score\toob_status\toob_score\tsupport_status\tsupport_score\toor_status\toor_score\toor_checks\toor_passed\toor_failed\toor_vlm_resolved\toor_vlm_pending\toar_status\toar_score\toar_checks\toar_passed\toar_failed\toar_vlm_resolved\toar_vlm_pending\toutput_dir\n" > "$SUMMARY_FILE"
total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
case_number=0

while IFS=$'\t' read -r case_id scene_type instruction_b64 room_b64; do
  case_number=$((case_number + 1))
  if (( MAX_CASES > 0 && case_number > MAX_CASES )); then
    break
  fi
  instruction=$("$BENCH_PY" -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode("utf-8"))' "$instruction_b64")
  case_out="${OUT_ROOT}/${case_id}"
  case_log="${OUT_ROOT}/${case_id}.log"
  room_file="${ROOM_ROOT}/${case_id}.json"
  generator_structure_file="${GENERATOR_STRUCTURE_DIR}/${case_id}.json"
  reference_annotation_file="${REFERENCE_ANNOTATION_DIR}/${case_id}.json"
  report_path="${case_out}/evaluation_report.json"
  require_path "$generator_structure_file"
  require_path "$reference_annotation_file"
  "$BENCH_PY" -c 'import base64,sys; open(sys.argv[2], "wb").write(base64.b64decode(sys.argv[1]))' "$room_b64" "$room_file"

  if [[ "$RESUME" == "1" && -f "$report_path" ]]; then
    fields=$(case_fields "$case_out")
    printf "%s\tskipped_existing\t%s\t%s\n" "$case_id" "$fields" "$case_out" >> "$SUMMARY_FILE"
    log "[${case_number}/${total_cases}] skipping existing ${case_id}"
    continue
  fi

  log "[${case_number}/${total_cases}] running ${case_id} with semantic retrieval + assets"
  flush_cache
  set +e
  "$BENCH_PY" scripts/run_scene_harness.py \
    --instruction "$instruction" \
    --physical-wall-policy always_enclosed \
    --scene-type "$scene_type" \
    --room-json "$room_file" \
    --prompt-granularity "$PROMPT_GRANULARITY" \
    --structure \
    --generator-structure "$generator_structure_file" \
    --reference-annotation "$reference_annotation_file" \
    --asset-mode retrieve \
    --asset-index-path "$ASSET_INDEX" \
    --retrieval-k 1 \
    --asset-root "$ASSET_ROOT" \
    --asset-csv "$ASSET_CSV" \
    --enrich-assets \
    --adapter layout_json \
    --adapter-config "$ADAPTER_CONFIG" \
    --run-generation \
    --generator-endpoint "$ENDPOINT" \
    --generator-model "$SERVED_MODEL" \
    --iteration-limit 0 \
    --eval-generic-validity \
    --eval-oor \
    --eval-oar \
    --support-enabled \
    --blender-bin "$BLENDER_BIN" \
    --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS" \
    --blender-render-engine "$BLENDER_RENDER_ENGINE" \
    --blender-cycles-device "$BLENDER_CYCLES_DEVICE" \
    --blender-cycles-samples "$BLENDER_CYCLES_SAMPLES" \
    "$BLENDER_DENOISING_FLAG" \
    --render-width "$RENDER_WIDTH" \
    --render-height "$RENDER_HEIGHT" \
    --vlm-judge-config "$JUDGE_CONFIG" \
    --p0b-official-mode \
    "${CAMERA_POSE_ARGS[@]}" \
    --out-dir "$case_out" \
    2>&1 | tee "$case_log"
  command_status=${PIPESTATUS[0]}
  set -e

  if (( command_status == 0 )) && [[ -f "$report_path" ]]; then
    fields=$(case_fields "$case_out")
    printf "%s\tcompleted\t%s\t%s\n" "$case_id" "$fields" "$case_out" >> "$SUMMARY_FILE"
    log "completed ${case_id}"
  else
    printf "%s\tfailed_%s" "$case_id" "$command_status" >> "$SUMMARY_FILE"
    for ((field_index = 0; field_index < 30; field_index++)); do
      printf "\t" >> "$SUMMARY_FILE"
    done
    printf "%s\n" "$case_out" >> "$SUMMARY_FILE"
    log "failed ${case_id} exit=${command_status}; see ${case_log}"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$command_status"
    fi
  fi
done < "$CASE_MANIFEST"

flush_cache
log "asset-backed smoke complete"
echo "Outputs: $OUT_ROOT"
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
