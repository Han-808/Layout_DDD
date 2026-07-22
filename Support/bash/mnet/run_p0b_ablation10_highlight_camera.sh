#!/usr/bin/env bash
set -Eeuo pipefail

# Controlled 10-case P0b evidence ablation.
#
# Phase 1 (prepare): generate each scene once, render the fixed overview, run
# the detector once, and export only VLM-routed Collision/OOB/Support events as
# human-review GT drafts.
#
# Phase 2 (ablate): reuse the frozen scene, detector request, prompt, judge,
# and reviewed event GT. The four arms isolate exactly one change per adjacent
# comparison:
#   global_raw -> visibility_raw                 camera selection
#   visibility_raw -> visibility_highlight      highlighting
#   visibility_highlight -> visibility_highlight_global  global context
#
# Accuracy is intentionally unavailable until every routed event has a human
# label of exactly "valid" or "invalid".

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
FIXTURE_ROOT=${FIXTURE_ROOT:-${REPO_ROOT}/configs/experiments/p0b_ablation10}
PROMPT_FILE=${PROMPT_FILE:-${FIXTURE_ROOT}/cases.json}
REFERENCE_ANNOTATION_DIR=${REFERENCE_ANNOTATION_DIR:-${FIXTURE_ROOT}/reference_annotations}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}

PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}

RUN_TAG=${RUN_TAG:-p0b_ablation10_highlight_camera}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
FROZEN_ROOT=${FROZEN_ROOT:-${OUT_ROOT}/frozen_cases}
GT_ROOT=${GT_ROOT:-${OUT_ROOT}/gt}
ABLATION_ROOT=${ABLATION_ROOT:-${OUT_ROOT}/ablation}
ROOM_ROOT=${ROOM_ROOT:-${OUT_ROOT}/rooms}
CASE_MANIFEST=${CASE_MANIFEST:-${OUT_ROOT}/case_manifest.tsv}
PREPARE_SUMMARY=${PREPARE_SUMMARY:-${OUT_ROOT}/prepare_summary.tsv}
REVIEW_QUEUE=${REVIEW_QUEUE:-${OUT_ROOT}/gt_review_queue.tsv}
ADAPTER_CONFIG=${ADAPTER_CONFIG:-${OUT_ROOT}/layout_json_adapter.json}

PHASE=${PHASE:-prepare}
RESUME=${RESUME:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
FLUSH_CACHE=${FLUSH_CACHE:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
GENERATOR_TIMEOUT_SECONDS=${GENERATOR_TIMEOUT_SECONDS:-5400}
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
  prepare|ablate|all) ;;
  *) echo "PHASE must be prepare, ablate, or all" >&2; exit 2 ;;
esac

for path in \
  "$REPO_ROOT" \
  "$BENCH_PY" \
  "$BLENDER_BIN" \
  "$PROMPT_FILE" \
  "$REFERENCE_ANNOTATION_DIR" \
  "$JUDGE_CONFIG" \
  "${REPO_ROOT}/scripts/run_scene_harness.py" \
  "${REPO_ROOT}/scripts/run_p0b_camera_ablation.py" \
  "${REPO_ROOT}/scripts/score_p0b_camera_ablation.py"; do
  require_path "$path"
done

mkdir -p "$OUT_ROOT" "$FROZEN_ROOT" "$GT_ROOT" "$ABLATION_ROOT" "$ROOM_ROOT"
cd "$REPO_ROOT"

log "preflight: endpoint, fixture contract, and Blender"
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
"$BENCH_PY" -c '
import json, sys
payload = json.load(sys.stdin)
raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"
echo "$models_json"

"$BENCH_PY" - "$PROMPT_FILE" "$REFERENCE_ANNOTATION_DIR" > "$CASE_MANIFEST" <<'PY'
import base64
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
cases = payload.get("cases", [])
if len(cases) != 10:
    raise SystemExit(f"expected 10 cases, found {len(cases)}")
for case in cases:
    case_id = case["case_id"]
    annotation = Path(sys.argv[2]) / f"{case_id}.json"
    if not annotation.is_file():
        raise SystemExit(f"missing reference annotation: {annotation}")
    instruction = base64.b64encode(case["instruction"].encode("utf-8")).decode("ascii")
    room = base64.b64encode(
        json.dumps(case["room"], separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    print(
        "\t".join(
            [case_id, case["difficulty"], case["scene_type"], instruction, room]
        )
    )
PY

"$BLENDER_BIN" --background --factory-startup --python-expr \
  "import bpy; bpy.context.scene.render.engine='CYCLES'; print('Cycles preflight OK')"

cat > "$ADAPTER_CONFIG" <<JSON
{
  "temperature": 0.0,
  "max_tokens": ${GENERATION_MAX_TOKENS},
  "context_length": ${CONTEXT_LENGTH},
  "timeout_seconds": ${GENERATOR_TIMEOUT_SECONDS},
  "response_format_json": true,
  "max_retries": 1,
  "schema_repair_attempts": 0
}
JSON

export_gt_draft() {
  local case_id=$1
  local difficulty=$2
  local case_dir=$3
  local gt_path=$4

  if [[ -f "$gt_path" ]]; then
    return
  fi
  "$BENCH_PY" - "$case_id" "$difficulty" "$case_dir/evaluation_report.json" "$gt_path" <<'PY'
import json
import sys
from pathlib import Path

case_id, difficulty, report_arg, output_arg = sys.argv[1:]
report_path = Path(report_arg).resolve()
output_path = Path(output_arg).resolve()
report = json.load(open(report_path, encoding="utf-8"))
metrics = report.get("reports", {}).get("generic_validity", {}).get("metrics", {})
sources = {
    "collision": metrics.get("collision", {}).get("pairs", []),
    "oob": metrics.get("oob", {}).get("objects", []),
    "support": metrics.get("support", {}).get("objects", []),
}
events = []
for metric in ("collision", "oob", "support"):
    for item in sources[metric]:
        if not isinstance(item, dict):
            continue
        judge_result = item.get("judge_result")
        request = judge_result.get("request") if isinstance(judge_result, dict) else None
        if not isinstance(request, dict):
            continue
        if metric == "collision":
            object_ids = [str(item.get("object_a")), str(item.get("object_b"))]
            event_id = "|".join(object_ids)
        else:
            object_ids = [str(item.get("object_id"))]
            event_id = object_ids[0]
        events.append(
            {
                "metric": metric,
                "event_id": event_id,
                "object_ids": object_ids,
                "label": None,
                "reason_code": "",
                "review_notes": "",
                "frozen_request": request,
            }
        )
payload = {
    "schema_version": "p0b_camera_ablation_gt_draft_v1",
    "status": "needs_human_review" if events else "no_routed_events",
    "case_id": case_id,
    "difficulty": difficulty,
    "source_report": str(report_path),
    "review_contract": {
        "allowed_labels": ["valid", "invalid"],
        "positive_class": "invalid",
        "instruction": "Review the frozen detector event and scene evidence; replace every null label.",
    },
    "events": events,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"GT draft: {output_path} events={len(events)}")
PY
}

write_review_queue() {
  "$BENCH_PY" - "$PROMPT_FILE" "$GT_ROOT" "$REVIEW_QUEUE" <<'PY'
import csv
import json
import sys
from pathlib import Path

cases = json.load(open(sys.argv[1], encoding="utf-8")).get("cases", [])
gt_root = Path(sys.argv[2])
output = Path(sys.argv[3])
rows = []
for case in cases:
    case_id = case["case_id"]
    path = gt_root / f"{case_id}.json"
    if not path.is_file():
        rows.append({"case_id": case_id, "difficulty": case["difficulty"], "metric": "", "event_id": "", "label": "MISSING_GT", "gt_path": str(path)})
        continue
    payload = json.load(open(path, encoding="utf-8"))
    for event in payload.get("events", []):
        rows.append({
            "case_id": case_id,
            "difficulty": case["difficulty"],
            "metric": event.get("metric", ""),
            "event_id": event.get("event_id", ""),
            "label": event.get("label"),
            "gt_path": str(path),
        })
fields = ["case_id", "difficulty", "metric", "event_id", "label", "gt_path"]
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print(f"review queue: {output} rows={len(rows)}")
PY
}

prepare_cases() {
  log "phase prepare: freeze 10 scenes, detector requests, and GT drafts"
  printf 'case_id\tdifficulty\tstatus\trouted_events\tcase_dir\n' > "$PREPARE_SUMMARY"
  local total_cases
  total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
  local case_number=0

  while IFS=$'\t' read -r case_id difficulty scene_type instruction_b64 room_b64; do
    case_number=$((case_number + 1))
    local case_dir=${FROZEN_ROOT}/${case_id}
    local case_log=${FROZEN_ROOT}/${case_id}.log
    local room_file=${ROOM_ROOT}/${case_id}.json
    local annotation=${REFERENCE_ANNOTATION_DIR}/${case_id}.json
    local gt_path=${GT_ROOT}/${case_id}.json
    local report=${case_dir}/evaluation_report.json
    local instruction
    instruction=$("$BENCH_PY" -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode("utf-8"))' "$instruction_b64")
    "$BENCH_PY" -c 'import base64,sys; open(sys.argv[2], "wb").write(base64.b64decode(sys.argv[1]))' "$room_b64" "$room_file"

    if [[ "$RESUME" == "1" \
      && -f "$report" \
      && -f "${case_dir}/generated_scene.json" \
      && -f "${case_dir}/renders/scene.blend" \
      && -f "${case_dir}/renders/standardized_top.png" \
      && -f "${case_dir}/renders/standardized_perspective.png" ]]; then
      log "[${case_number}/${total_cases}] frozen case exists: ${case_id}"
      export_gt_draft "$case_id" "$difficulty" "$case_dir" "$gt_path"
      routed=$("$BENCH_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["events"]))' "$gt_path")
      printf '%s\t%s\tskipped_existing\t%s\t%s\n' "$case_id" "$difficulty" "$routed" "$case_dir" >> "$PREPARE_SUMMARY"
      continue
    fi

    flush_cache
    log "[${case_number}/${total_cases}] generating frozen case ${case_id} (${difficulty})"
    set +e
    "$BENCH_PY" scripts/run_scene_harness.py \
      --instruction "$instruction" \
      --scene-type "$scene_type" \
      --room-json "$room_file" \
      --prompt-granularity fine_grained \
      --no-structure \
      --reference-annotation "$annotation" \
      --asset-mode off \
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
      --p0b-official-mode \
      --blender-bin "$BLENDER_BIN" \
      --blender-timeout-seconds "$BLENDER_TIMEOUT_SECONDS" \
      --blender-render-engine CYCLES \
      --blender-cycles-device CUDA \
      --blender-cycles-samples "$CYCLES_SAMPLES" \
      --blender-cycles-denoising \
      --render-width "$RENDER_WIDTH" \
      --render-height "$RENDER_HEIGHT" \
      --vlm-judge-config "$JUDGE_CONFIG" \
      --out-dir "$case_dir" \
      2>&1 | tee "$case_log"
    command_status=${PIPESTATUS[0]}
    set -e

    if (( command_status == 0 )) && [[ -f "$report" ]]; then
      export_gt_draft "$case_id" "$difficulty" "$case_dir" "$gt_path"
      routed=$("$BENCH_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["events"]))' "$gt_path")
      printf '%s\t%s\tcompleted\t%s\t%s\n' "$case_id" "$difficulty" "$routed" "$case_dir" >> "$PREPARE_SUMMARY"
    else
      printf '%s\t%s\tfailed_%s\t\t%s\n' "$case_id" "$difficulty" "$command_status" "$case_dir" >> "$PREPARE_SUMMARY"
      log "prepare failed ${case_id}; see ${case_log}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$command_status"
      fi
    fi
  done < "$CASE_MANIFEST"

  write_review_queue
  log "prepare complete"
  cat "$PREPARE_SUMMARY"
  echo "Review every null label in: $GT_ROOT"
  echo "Compact queue: $REVIEW_QUEUE"
}

gt_ready() {
  "$BENCH_PY" - "$PROMPT_FILE" "$GT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

cases = json.load(open(sys.argv[1], encoding="utf-8")).get("cases", [])
root = Path(sys.argv[2])
missing = []
unreviewed = []
for case in cases:
    case_id = case["case_id"]
    path = root / f"{case_id}.json"
    if not path.is_file():
        missing.append(case_id)
        continue
    payload = json.load(open(path, encoding="utf-8"))
    for event in payload.get("events", []):
        if event.get("label") not in {"valid", "invalid"}:
            unreviewed.append(f"{case_id}:{event.get('metric')}:{event.get('event_id')}")
print("missing GT files:", len(missing))
print("unreviewed events:", len(unreviewed))
for value in (missing + unreviewed)[:30]:
    print(" -", value)
raise SystemExit(0 if not missing and not unreviewed else 3)
PY
}

aggregate_results() {
  "$BENCH_PY" - "$PROMPT_FILE" "$ABLATION_ROOT" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

prompt_file = Path(sys.argv[1])
root = Path(sys.argv[2])
cases = json.load(open(prompt_file, encoding="utf-8")).get("cases", [])
difficulty = {case["case_id"]: case["difficulty"] for case in cases}
rows = []
for case_id, level in difficulty.items():
    path = root / case_id / "per_event.tsv"
    if not path.is_file():
        continue
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append({"case_id": case_id, "difficulty": level, **row})

combined_path = root / "combined_per_event.tsv"
if rows:
    with combined_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def number(row, key):
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0

summary = []
for subset in ("all", "easy", "moderate"):
    subset_rows = rows if subset == "all" else [row for row in rows if row["difficulty"] == subset]
    for arm in ("global_raw", "visibility_raw", "visibility_highlight", "visibility_highlight_global"):
        arm_rows = [row for row in subset_rows if row.get("arm") == arm]
        for metric in ("overall", "collision", "oob", "support"):
            selected = arm_rows if metric == "overall" else [row for row in arm_rows if row.get("metric") == metric]
            total = len(selected)
            resolved = sum(int(number(row, "resolved")) for row in selected)
            correct = sum(int(number(row, "match")) for row in selected)
            estimated = sum(number(row, "estimated_uncached_seconds") for row in selected)
            images = sum(number(row, "image_count") for row in selected)
            summary.append({
                "subset": subset,
                "arm": arm,
                "metric": metric,
                "total": total,
                "resolved": resolved,
                "correct": correct,
                "accuracy_all": correct / total if total else 0.0,
                "coverage": resolved / total if total else 0.0,
                "tp": sum(row.get("gt_label") == "invalid" and row.get("predicted_label") == "invalid" for row in selected),
                "fp": sum(row.get("gt_label") == "valid" and row.get("predicted_label") == "invalid" for row in selected),
                "fn": sum(row.get("gt_label") == "invalid" and row.get("predicted_label") == "valid" for row in selected),
                "tn": sum(row.get("gt_label") == "valid" and row.get("predicted_label") == "valid" for row in selected),
                "camera_evidence_seconds": sum(number(row, "camera_evidence_seconds") for row in selected),
                "judge_seconds": sum(number(row, "judge_seconds") for row in selected),
                "measured_seconds": sum(number(row, "elapsed_seconds") for row in selected),
                "estimated_uncached_seconds": estimated,
                "mean_estimated_seconds": estimated / total if total else 0.0,
                "image_count": images,
                "mean_images": images / total if total else 0.0,
            })

summary_path = root / "combined_summary.tsv"
with summary_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(summary)

lookup = {(row["subset"], row["arm"], row["metric"]): row for row in summary}
contrasts = []
comparisons = [
    ("camera_selection", "global_raw", "visibility_raw"),
    ("highlight", "visibility_raw", "visibility_highlight"),
    ("highlighted_global_context", "visibility_highlight", "visibility_highlight_global"),
]
for subset in ("all", "easy", "moderate"):
    for metric in ("overall", "collision", "oob", "support"):
        for variable, baseline_name, treatment_name in comparisons:
            baseline = lookup[(subset, baseline_name, metric)]
            treatment = lookup[(subset, treatment_name, metric)]
            contrasts.append({
                "subset": subset,
                "metric": metric,
                "changed_variable": variable,
                "baseline": baseline_name,
                "treatment": treatment_name,
                "event_count": treatment["total"],
                "accuracy_delta": treatment["accuracy_all"] - baseline["accuracy_all"],
                "coverage_delta": treatment["coverage"] - baseline["coverage"],
                "mean_estimated_seconds_delta": treatment["mean_estimated_seconds"] - baseline["mean_estimated_seconds"],
                "mean_images_delta": treatment["mean_images"] - baseline["mean_images"],
            })
contrast_path = root / "controlled_contrasts.tsv"
with contrast_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(contrasts)

(root / "combined_results.json").write_text(
    json.dumps({"per_event": rows, "summary": summary, "contrasts": contrasts}, indent=2),
    encoding="utf-8",
)
print("combined event rows:", len(rows))
print("summary:", summary_path)
print("controlled contrasts:", contrast_path)
PY
}

run_ablation() {
  log "GT gate"
  gt_ready
  log "phase ablate: frozen events x four controlled arms"

  while IFS=$'\t' read -r case_id difficulty scene_type instruction_b64 room_b64; do
    local case_dir=${FROZEN_ROOT}/${case_id}
    local gt_path=${GT_ROOT}/${case_id}.json
    local case_out=${ABLATION_ROOT}/${case_id}
    local scene=${case_dir}/generated_scene.json
    local report=${case_dir}/evaluation_report.json
    local blend=${case_dir}/renders/scene.blend
    local top=${case_dir}/renders/standardized_top.png
    local perspective=${case_dir}/renders/standardized_perspective.png
    local collision_geometry=${case_dir}/renders/collision_geometry_manifest.json
    local event_count
    event_count=$("$BENCH_PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["events"]))' "$gt_path")
    if (( event_count == 0 )); then
      log "skipping ${case_id}: detector routed no P0b event"
      continue
    fi
    for path in "$scene" "$report" "$blend" "$top" "$perspective" "$gt_path"; do
      require_path "$path"
    done
    mkdir -p "$case_out"
    flush_cache
    log "ablation ${case_id} (${difficulty}) events=${event_count}"

    args=(
      --scene "$scene"
      --source-report "$report"
      --gt "$gt_path"
      --blend-file "$blend"
      --overview "$top"
      --overview "$perspective"
      --judge-config "$JUDGE_CONFIG"
      --out-dir "$case_out"
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
    if [[ -f "$collision_geometry" ]]; then
      args+=(--collision-geometry "$collision_geometry")
    fi
    if [[ "$RESUME" == "1" ]]; then
      args+=(--resume)
    else
      args+=(--no-resume)
    fi

    set +e
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" scripts/run_p0b_camera_ablation.py "${args[@]}" \
      2>&1 | tee "$case_out/run.log"
    command_status=${PIPESTATUS[0]}
    set -e
    if (( command_status != 0 )); then
      log "ablation runner failed ${case_id} exit=${command_status}"
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$command_status"
      fi
      continue
    fi
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" "$BENCH_PY" scripts/score_p0b_camera_ablation.py \
      --gt "$gt_path" \
      --run-dir "$case_out" \
      | tee "$case_out/score.log"
  done < "$CASE_MANIFEST"

  aggregate_results
  log "ablation complete"
  echo "Per-event results: ${ABLATION_ROOT}/combined_per_event.tsv"
  echo "Accuracy/cost summary: ${ABLATION_ROOT}/combined_summary.tsv"
  echo "Single-variable contrasts: ${ABLATION_ROOT}/controlled_contrasts.tsv"
}

if [[ "$PHASE" == "prepare" || "$PHASE" == "all" ]]; then
  prepare_cases
fi

if [[ "$PHASE" == "ablate" ]]; then
  run_ablation
elif [[ "$PHASE" == "all" ]]; then
  if gt_ready; then
    run_ablation
  else
    log "awaiting human GT; rerun with PHASE=ablate after labels are frozen"
  fi
fi
