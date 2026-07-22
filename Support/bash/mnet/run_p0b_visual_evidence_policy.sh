#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled P0b visual-evidence-policy replay.
#
# Frozen across arms:
#   GT, physical event universe, canonical scene/.blend, detector evidence,
#   natural-language prompt, extracted relationships, final judge, and the
#   pose-selector model.
# Changed across arms:
#   fixed global views vs deterministic metric-local views vs VLM-selected
#   candidates vs query_cov views with bounded camera adjustment.
#
# This script never runs generation, retrieval, conversion, or detectors. It
# requires the completed source_distortion5 source reports and replays only the
# judge requests frozen inside those reports.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}
FIXTURE_ROOT=${FIXTURE_ROOT:-${REPO_ROOT}/configs/experiments/p0b_source_distortion5}
CASE_FILE=${CASE_FILE:-${FIXTURE_ROOT}/cases.json}

SOURCE_STUDY_ROOT=${SOURCE_STUDY_ROOT:-${REPO_ROOT}/outputs/p0b_combined_20260717_202533/source_distortion5}
SOURCE_ROOT=${SOURCE_ROOT:-${SOURCE_STUDY_ROOT}/source_reports}
GT_ROOT=${GT_ROOT:-${SOURCE_STUDY_ROOT}/gt}

RUN_TAG=${RUN_TAG:-p0b_visual_evidence_policy_$(date '+%Y%m%d_%H%M%S')}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
ABLATION_ROOT=${ABLATION_ROOT:-${OUT_ROOT}/ablation}
RESULT_ROOT=${RESULT_ROOT:-${OUT_ROOT}/results}
CASE_MANIFEST=${CASE_MANIFEST:-${OUT_ROOT}/case_manifest.tsv}
RUN_SUMMARY=${RUN_SUMMARY:-${OUT_ROOT}/run_summary.tsv}
RUN_ORDER=${RUN_ORDER:-${OUT_ROOT}/arm_order.tsv}

PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}

# clean contributes valid Support examples; the other families contribute the
# metric named by the family. OAR is intentionally outside this P0b experiment.
FAMILIES=${FAMILIES:-clean,collision,oob,support}
MAX_CASES=${MAX_CASES:-0}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}
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
ACTIVE_MAX_STEPS=${ACTIVE_MAX_STEPS:-1}
ARMS_CSV=${ARMS_CSV:-fixed_global,deterministic_metric_local,vlm_select_from_candidates,active_metric_local}

IFS=',' read -r -a SELECTED_ARMS <<< "$ARMS_CSV"
if (( ${#SELECTED_ARMS[@]} == 0 )); then
  echo "ARMS_CSV selected no visual-evidence policies" >&2
  exit 1
fi

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

for path in \
  "$REPO_ROOT" "$BENCH_PY" "$BLENDER_BIN" "$JUDGE_CONFIG" "$CASE_FILE" \
  "$SOURCE_ROOT" "$GT_ROOT" \
  "${REPO_ROOT}/scripts/run_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/score_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/aggregate_p0b_visual_evidence_policy.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$ABLATION_ROOT" "$RESULT_ROOT"
cd "$REPO_ROOT"

if [[ ",${ARMS_CSV}," == *",vlm_select_from_candidates,"* ]] \
  || [[ ",${ARMS_CSV}," == *",active_metric_local,"* ]]; then
  log "preflight: one Qwen model serves both pose selection and final judgement"
else
  log "preflight: Qwen serves final judgement; active pose selection is disabled"
fi
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
"$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"
echo "$models_json"

PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" - \
  "$JUDGE_CONFIG" "$SERVED_MODEL" "$ARMS_CSV" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
if config.get("model") != sys.argv[2]:
    raise SystemExit(
        f"judge config model {config.get('model')!r} does not match served model {sys.argv[2]!r}"
    )
from scripts.run_p0b_camera_ablation import ARM_CONFIGS

expected = {value.strip() for value in sys.argv[3].split(",") if value.strip()}
if not expected.issubset(ARM_CONFIGS):
    raise SystemExit(f"visual policy arms missing: {sorted(expected - set(ARM_CONFIGS))}")
if (
    "deterministic_metric_local" in expected
    and ARM_CONFIGS["deterministic_metric_local"]["metric_modes"]["support"]
    != "support_contact_plane"
):
    raise SystemExit("deterministic Support arm is not support_contact_plane")
if (
    "vlm_select_from_candidates" in expected
    and ARM_CONFIGS["vlm_select_from_candidates"]["metric_modes"]["support"] != "query_cov"
):
    raise SystemExit("VLM-select-only Support arm is not query_cov")
if (
    "vlm_select_from_candidates" in expected
    and ARM_CONFIGS["vlm_select_from_candidates"].get("max_steps") != 0
):
    raise SystemExit("VLM-select-only arm must disable camera adjustment")
if (
    "active_metric_local" in expected
    and ARM_CONFIGS["active_metric_local"]["metric_modes"]["support"] != "query_cov"
):
    raise SystemExit("active Support arm is not query_cov")
print("visual evidence policy contract: OK", sorted(expected))
PY

"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='CYCLES'; print('Cycles preflight OK')"

log "build targeted frozen-event manifest"
"$BENCH_PY" - "$CASE_FILE" "$FIXTURE_ROOT" "$SOURCE_ROOT" "$GT_ROOT" "$FAMILIES" > "$CASE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

case_file = Path(sys.argv[1]).resolve()
fixture_root = Path(sys.argv[2]).resolve()
source_root = Path(sys.argv[3]).resolve()
gt_root = Path(sys.argv[4]).resolve()
allowed = {value.strip() for value in sys.argv[5].split(",") if value.strip()}
family_metric = {
    "clean": "support",
    "collision": "collision",
    "oob": "oob",
    "support": "support",
}
payload = json.load(case_file.open(encoding="utf-8"))
print("case_id\tbase_case_id\tfamily\tmetric\tfixture_dir")
for case in payload.get("cases") or []:
    family = str(case.get("family") or "")
    if family not in allowed or family not in family_metric:
        continue
    case_id = str(case["case_id"])
    fixture_dir = (fixture_root / str(case["fixture_dir"])).resolve()
    required = [
        fixture_dir / "generated_scene.json",
        source_root / case_id / "evaluation_report.json",
        source_root / case_id / "renders" / "scene.blend",
        source_root / case_id / "renders" / "standardized_top.png",
        source_root / case_id / "renders" / "standardized_perspective.png",
        gt_root / f"{case_id}.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"frozen replay prerequisites missing for {case_id}: {missing}")
    print("\t".join([
        case_id,
        str(case["base_case_id"]),
        family,
        family_metric[family],
        str(fixture_dir),
    ]))
PY

case_count=$(( $(wc -l < "$CASE_MANIFEST") - 1 ))
if (( case_count <= 0 )); then
  echo "No replay cases selected" >&2
  exit 1
fi
echo "selected cases: $case_count"

"$BENCH_PY" - "${OUT_ROOT}/experiment_contract.json" "$SERVED_MODEL" \
  "$ARMS_CSV" "$ACTIVE_MAX_STEPS" "$MAX_VIEWS" "$CANDIDATE_COUNT" <<'PY'
import json
import sys

path, served_model, arms_csv, max_steps, max_views, candidate_count = sys.argv[1:]
selected = [value.strip() for value in arms_csv.split(",") if value.strip()]
descriptions = {
    "fixed_global": "frozen standardized top and perspective views",
    "deterministic_metric_local": (
        "highlighted global plus deterministic local; Support uses support_contact_plane"
    ),
    "vlm_select_from_candidates": (
        "highlighted global plus deterministic candidates selected by the VLM; no camera adjustment"
    ),
    "active_metric_local": (
        "highlighted global plus query_cov local; same model selects poses and judges"
    ),
}
frozen = [
    "GT",
    "physical event",
    "detector evidence",
    "natural-language prompt",
    "extracted relationships",
    "canonical scene and scene.blend",
    "final judge model",
]
selector_arms = {"vlm_select_from_candidates", "active_metric_local"}
if selector_arms.intersection(selected):
    frozen.append("pose-selector model")
payload = {
    "schema_version": "p0b_visual_evidence_policy_contract_v1",
    "controlled_variable": "visual_evidence_policy",
    "selected_arms": selected,
    "frozen": frozen,
    "arms": {arm: descriptions[arm] for arm in selected},
    "judge_model": served_model,
    "pose_selector_model": served_model if selector_arms.intersection(selected) else None,
    "selection_only_max_steps": 0,
    "active_max_steps": int(max_steps) if "active_metric_local" in selected else 0,
    "max_views": int(max_views),
    "candidate_count": int(candidate_count),
    "gt_status": {
        "collision": "usable_routed_event_gt",
        "oob": "usable_recall_focused_missing_broad_negative_universe",
        "support": "provisional_until_rendered_mesh_gt_contract_is_frozen",
    },
    "does_not_run": ["generator", "retriever", "converter", "detector"],
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
PY

printf 'case_id\tbase_case_id\tfamily\tmetric\tstatus\tcompleted_arms\tphysical_events\toutput_dir\n' > "$RUN_SUMMARY"
printf 'case_id\tposition\tarm\n' > "$RUN_ORDER"

case_number=0
while IFS=$'\t' read -r case_id base_case_id family metric fixture_dir; do
  case_number=$((case_number + 1))
  if (( MAX_CASES > 0 && case_number > MAX_CASES )); then
    log "reached MAX_CASES=${MAX_CASES}"
    break
  fi

  source_case=${SOURCE_ROOT}/${case_id}
  report=${source_case}/evaluation_report.json
  render_dir=${source_case}/renders
  gt_path=${GT_ROOT}/${case_id}.json
  scene=${fixture_dir}/generated_scene.json
  case_out=${ABLATION_ROOT}/${case_id}
  mkdir -p "$case_out"

  event_count=$("$BENCH_PY" - "$gt_path" "$metric" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(isinstance(event, dict) and event.get("metric") == sys.argv[2] for event in payload.get("events") or []))
PY
  )
  if (( event_count == 0 )); then
    printf '%s\t%s\t%s\t%s\tno_frozen_events\t0\t0\t%s\n' \
      "$case_id" "$base_case_id" "$family" "$metric" "$case_out" >> "$RUN_SUMMARY"
    continue
  fi

  # Alternate the three local-policy orders across cases to reduce persistent
  # Blender/GPU warm-cache order bias. Fixed global does not launch Blender.
  if [[ "$ARMS_CSV" == "fixed_global,deterministic_metric_local,vlm_select_from_candidates,active_metric_local" ]] \
    && (( case_number % 2 == 0 )); then
    arms=(fixed_global active_metric_local vlm_select_from_candidates deterministic_metric_local)
  else
    arms=("${SELECTED_ARMS[@]}")
  fi

  position=0
  for arm in "${arms[@]}"; do
    position=$((position + 1))
    printf '%s\t%s\t%s\n' "$case_id" "$position" "$arm" >> "$RUN_ORDER"
    if [[ "$RESUME" == "1" && -f "${case_out}/${arm}/mode_results.json" ]]; then
      log "[${case_number}/${case_count}] cached ${case_id} metric=${metric} arm=${arm}"
      continue
    fi

    flush_cache
    log "[${case_number}/${case_count}] replay ${case_id} metric=${metric} arm=${arm} events=${event_count}"
    args=(
      --scene "$scene"
      --source-report "$report"
      --gt "$gt_path"
      --metric "$metric"
      --blend-file "${render_dir}/scene.blend"
      --overview "${render_dir}/standardized_top.png"
      --overview "${render_dir}/standardized_perspective.png"
      --judge-config "$JUDGE_CONFIG"
      --out-dir "$case_out"
      --arm "$arm"
      --max-steps "$ACTIVE_MAX_STEPS"
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
    if [[ "$RESUME" == "1" ]]; then
      args+=(--resume)
    else
      args+=(--no-resume)
    fi

    set +e
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" \
      scripts/run_p0b_camera_ablation.py "${args[@]}" \
      2>&1 | tee "${case_out}/run__${arm}.log"
    arm_status=${PIPESTATUS[0]}
    set -e
    if [[ -f "${case_out}/run_manifest.json" ]]; then
      cp "${case_out}/run_manifest.json" "${case_out}/run_manifest__${arm}.json"
    fi
    if (( arm_status != 0 )); then
      log "arm failed ${case_id} ${arm} exit=${arm_status}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$arm_status"
      fi
    fi
  done

  set +e
  "$BENCH_PY" scripts/score_p0b_camera_ablation.py \
    --gt "$gt_path" \
    --metric "$metric" \
    --run-dir "$case_out" \
    2>&1 | tee "${case_out}/score.log"
  score_status=${PIPESTATUS[0]}
  set -e
  completed_arms=$(find "$case_out" -mindepth 2 -maxdepth 2 -name mode_results.json | wc -l | tr -d ' ')
  status=completed
  if (( score_status != 0 )); then
    status=score_failed_${score_status}
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$score_status"
    fi
  elif (( completed_arms < ${#SELECTED_ARMS[@]} )); then
    status=partial
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_id" "$base_case_id" "$family" "$metric" "$status" \
    "$completed_arms" "$event_count" "$case_out" >> "$RUN_SUMMARY"
done < <(tail -n +2 "$CASE_MANIFEST")

log "aggregate aligned policy packets and paired transitions"
aggregate_args=(
  --case-manifest "$CASE_MANIFEST"
  --gt-root "$GT_ROOT"
  --ablation-root "$ABLATION_ROOT"
  --out-dir "$RESULT_ROOT"
)
for arm in "${SELECTED_ARMS[@]}"; do
  aggregate_args+=(--arm "$arm")
done
"$BENCH_PY" scripts/aggregate_p0b_visual_evidence_policy.py \
  "${aggregate_args[@]}"

log "visual evidence policy experiment complete"
cat "$RUN_SUMMARY"
echo "Contract: $OUT_ROOT/experiment_contract.json"
echo "Master table: $RESULT_ROOT/master_event_policy_table.tsv"
echo "Policy summary: $RESULT_ROOT/policy_summary.tsv"
echo "Paired transitions: $RESULT_ROOT/paired_transition_summary.tsv"
