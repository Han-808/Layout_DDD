#!/usr/bin/env bash
set -Eeuo pipefail

# Run 50 independent fine-grained I1 cases through layout JSON generation,
# deterministic evaluation, Blender renders, and VLM categories. Runtime
# conversion/classification is forbidden; reviewed references are optional for
# diagnostics and required for official fine-grained fidelity scores.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
BLENDER_BIN=${BLENDER_BIN:-/mnt/group/cmh/tools/blender/blender}
BLENDER_RENDER_ENGINE=${BLENDER_RENDER_ENGINE:-CYCLES}
PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
PROMPT_FILE=${PROMPT_FILE:-${REPO_ROOT}/configs/experiments/qwen32b_nl50_prompts.json}
JUDGE_CONFIG=${JUDGE_CONFIG:-${REPO_ROOT}/configs/models/qwen3vl_mnet_judge.json}
PROMPT_GRANULARITY=${PROMPT_GRANULARITY:-fine_grained}
REFERENCE_ANNOTATION_DIR=${REFERENCE_ANNOTATION_DIR:-}
REQUIRE_REFERENCE_ANNOTATION=${REQUIRE_REFERENCE_ANNOTATION:-0}

GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}
RENDER_WIDTH=${RENDER_WIDTH:-512}
RENDER_HEIGHT=${RENDER_HEIGHT:-512}
FLUSH_CACHE=${FLUSH_CACHE:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-1}
RESUME=${RESUME:-1}
MAX_CASES=${MAX_CASES:-0}

RUN_TAG=${RUN_TAG:-qwen32b_nl50_cycles_vlm_$(date +%Y%m%d_%H%M%S)}
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
    exit 1
  fi
}

flush_cache() {
  if [[ "$FLUSH_CACHE" != "1" ]]; then
    return 0
  fi
  log "flushing SGLang request cache"
  curl --noproxy "*" -fsS -X POST "${BASE_URL}/flush_cache" || true
  echo
}

report_fields() {
  local report_path=$1
  "$BENCH_PY" - "$report_path" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
categories = report.get("category_reports", {})

def score(name):
    value = categories.get(name, {}).get("score")
    return "" if value is None else str(value)

values = [
    report.get("benchmark_score"),
    report.get("benchmark_score_status"),
    score("prompt_fidelity"),
    score("structural_validity"),
    score("visual_quality"),
]
print("\t".join("" if value is None else str(value) for value in values))
PY
}

for path in \
  "$REPO_ROOT" \
  "$BENCH_PY" \
  "$BLENDER_BIN" \
  "$PROMPT_FILE" \
  "$JUDGE_CONFIG" \
  "${REPO_ROOT}/scripts/run_scene_harness.py"; do
  require_path "$path"
done

if [[ ! -x "$BLENDER_BIN" ]]; then
  echo "Blender is not executable: $BLENDER_BIN" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$ROOM_ROOT"
cd "$REPO_ROOT"

log "checking server ${ENDPOINT} model=${SERVED_MODEL}"
models_json=$(curl --noproxy "*" -fsS "${ENDPOINT}/models")
if ! "$BENCH_PY" -c '
import json
import sys
payload = json.load(sys.stdin)
expected = sys.argv[1]
raise SystemExit(0 if any(item.get("id") == expected for item in payload.get("data", [])) else 1)
' "$SERVED_MODEL" <<<"$models_json"; then
  echo "Expected model '${SERVED_MODEL}' was not returned by ${ENDPOINT}/models" >&2
  echo "$models_json" >&2
  exit 1
fi
echo "$models_json"

log "checking Blender ${BLENDER_RENDER_ENGINE}"
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
import base64
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for case in payload["cases"]:
    instruction = base64.b64encode(case["instruction"].encode("utf-8")).decode("ascii")
    room = base64.b64encode(json.dumps(case["room"], separators=(",", ":")).encode("utf-8")).decode("ascii")
    print(f'{case["case_id"]}\t{case["scene_type"]}\t{instruction}\t{room}')
PY

printf "case_id\tstatus\tbenchmark_score\tscore_status\tprompt_fidelity\tstructural_validity\tvisual_quality\toutput_dir\n" > "$SUMMARY_FILE"
total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
case_number=0

while IFS=$'\t' read -r case_id scene_type instruction_b64 room_b64; do
  case_number=$((case_number + 1))
  if (( MAX_CASES > 0 && case_number > MAX_CASES )); then
    log "reached MAX_CASES=${MAX_CASES}; stopping after preflight subset"
    break
  fi
  instruction=$("$BENCH_PY" -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode("utf-8"))' "$instruction_b64")
  case_out="${OUT_ROOT}/${case_id}"
  case_log="${OUT_ROOT}/${case_id}.log"
  room_file="${ROOM_ROOT}/${case_id}.json"
  report_path="${case_out}/evaluation_report.json"
  "$BENCH_PY" -c 'import base64,sys; open(sys.argv[2], "wb").write(base64.b64decode(sys.argv[1]))' "$room_b64" "$room_file"
  reference_args=()
  if [[ -n "$REFERENCE_ANNOTATION_DIR" ]]; then
    reference_file="${REFERENCE_ANNOTATION_DIR}/${case_id}.json"
    if [[ -f "$reference_file" ]]; then
      reference_args=(--reference-annotation "$reference_file")
    elif [[ "$REQUIRE_REFERENCE_ANNOTATION" == "1" ]]; then
      echo "Missing reviewed reference annotation: $reference_file" >&2
      exit 1
    fi
  elif [[ "$REQUIRE_REFERENCE_ANNOTATION" == "1" ]]; then
    echo "REQUIRE_REFERENCE_ANNOTATION=1 requires REFERENCE_ANNOTATION_DIR." >&2
    exit 1
  fi

  if [[ "$RESUME" == "1" && -f "$report_path" ]]; then
    metrics=$(report_fields "$report_path")
    printf "%s\tskipped_existing\t%s\t%s\n" "$case_id" "$metrics" "$case_out" >> "$SUMMARY_FILE"
    log "[${case_number}/${total_cases}] skipping existing ${case_id}"
    continue
  fi

  log "[${case_number}/${total_cases}] running ${case_id} scene_type=${scene_type}"
  flush_cache

  set +e
  "$BENCH_PY" scripts/run_scene_harness.py \
    --instruction "$instruction" \
    --physical-wall-policy always_enclosed \
    --scene-type "$scene_type" \
    --room-json "$room_file" \
    --prompt-granularity "$PROMPT_GRANULARITY" \
    --no-structure \
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
    --blender-bin "$BLENDER_BIN" \
    --blender-render-engine "$BLENDER_RENDER_ENGINE" \
    --render-width "$RENDER_WIDTH" \
    --render-height "$RENDER_HEIGHT" \
    --vlm-judge-config "$JUDGE_CONFIG" \
    "${reference_args[@]}" \
    --p0b-official-mode \
    --out-dir "$case_out" \
    2>&1 | tee "$case_log"
  command_status=${PIPESTATUS[0]}
  set -e

  if (( command_status == 0 )) && [[ -f "$report_path" ]]; then
    metrics=$(report_fields "$report_path")
    printf "%s\tcompleted\t%s\t%s\n" "$case_id" "$metrics" "$case_out" >> "$SUMMARY_FILE"
    log "completed ${case_id}"
  else
    printf "%s\tfailed_%s\t\t\t\t\t\t%s\n" "$case_id" "$command_status" "$case_out" >> "$SUMMARY_FILE"
    log "failed ${case_id} exit=${command_status}; see ${case_log}"
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit "$command_status"
    fi
  fi
done < "$CASE_MANIFEST"

flush_cache
log "batch complete"
echo "Outputs: $OUT_ROOT"
echo "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
