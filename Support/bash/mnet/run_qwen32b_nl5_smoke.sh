#!/usr/bin/env bash
set -Eeuo pipefail

# Run five independent I1 NL-only cases against an already-running endpoint.
# Runtime conversion/classification is forbidden. A reviewed private reference
# can be supplied per case without exposing it to the generator.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
BENCH_PY=${BENCH_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
PORT=${PORT:-8298}
BASE_URL=${BASE_URL:-http://127.0.0.1:${PORT}}
ENDPOINT=${ENDPOINT:-${BASE_URL}/v1}
SERVED_MODEL=${SERVED_MODEL:-Qwen3-VL-32B-Instruct-64K}
PROMPT_FILE=${PROMPT_FILE:-${REPO_ROOT}/configs/experiments/qwen32b_nl5_smoke_prompts.json}
PROMPT_GRANULARITY=${PROMPT_GRANULARITY:-fine_grained}
REFERENCE_ANNOTATION_DIR=${REFERENCE_ANNOTATION_DIR:-}
REQUIRE_REFERENCE_ANNOTATION=${REQUIRE_REFERENCE_ANNOTATION:-0}

GENERATION_MAX_TOKENS=${GENERATION_MAX_TOKENS:-8192}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-5400}
FLUSH_CACHE=${FLUSH_CACHE:-1}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-0}

RUN_TAG=${RUN_TAG:-qwen32b_nl5_smoke_$(date +%Y%m%d_%H%M%S)}
OUT_ROOT=${OUT_ROOT:-${REPO_ROOT}/outputs/${RUN_TAG}}
ROOM_ROOT=${OUT_ROOT}/rooms
CASE_MANIFEST=${OUT_ROOT}/case_manifest.tsv
SUMMARY_FILE=${OUT_ROOT}/summary.tsv
ADAPTER_CONFIG=${OUT_ROOT}/layout_json_adapter.json

export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}

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

room_file_for_case() {
  case "$1" in
    prompt_01|prompt_03) echo "${ROOM_ROOT}/room_8x8x2.8.json" ;;
    prompt_04) echo "${ROOM_ROOT}/room_7x9x3.0.json" ;;
    prompt_05) echo "${ROOM_ROOT}/room_10x10x3.2.json" ;;
    *) echo "" ;;
  esac
}

require_path "$REPO_ROOT"
require_path "$BENCH_PY"
require_path "$PROMPT_FILE"
require_path "${REPO_ROOT}/scripts/run_scene_harness.py"

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
  echo "Expected served model '${SERVED_MODEL}' was not returned by ${ENDPOINT}/models" >&2
  echo "$models_json" >&2
  exit 1
fi
echo "$models_json"

cat > "${ROOM_ROOT}/room_8x8x2.8.json" <<'JSON'
{"boundary":[[0,0],[8,0],[8,8],[0,8]],"height":2.8,"unit":"meter"}
JSON
cat > "${ROOM_ROOT}/room_7x9x3.0.json" <<'JSON'
{"boundary":[[0,0],[7,0],[7,9],[0,9]],"height":3.0,"unit":"meter"}
JSON
cat > "${ROOM_ROOT}/room_10x10x3.2.json" <<'JSON'
{"boundary":[[0,0],[10,0],[10,10],[0,10]],"height":3.2,"unit":"meter"}
JSON

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
    print(f'{case["case_id"]}\t{case["scene_type"]}\t{instruction}')
PY

printf "case_id\tstatus\toutput_dir\n" > "$SUMMARY_FILE"
total_cases=$(wc -l < "$CASE_MANIFEST" | tr -d ' ')
case_number=0

while IFS=$'\t' read -r case_id scene_type instruction_b64; do
  case_number=$((case_number + 1))
  instruction=$("$BENCH_PY" -c 'import base64,sys; print(base64.b64decode(sys.argv[1]).decode("utf-8"))' "$instruction_b64")
  case_out="${OUT_ROOT}/${case_id}"
  case_log="${OUT_ROOT}/${case_id}.log"
  room_file=$(room_file_for_case "$case_id")
  room_args=()
  if [[ -n "$room_file" ]]; then
    room_args=(--room-json "$room_file")
  fi
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

  log "[${case_number}/${total_cases}] running ${case_id} scene_type=${scene_type}"
  flush_cache

  set +e
  "$BENCH_PY" scripts/run_scene_harness.py \
    --instruction "$instruction" \
    --physical-wall-policy always_enclosed \
    --scene-type "$scene_type" \
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
    "${room_args[@]}" \
    "${reference_args[@]}" \
    --out-dir "$case_out" \
    2>&1 | tee "$case_log"
  command_status=${PIPESTATUS[0]}
  set -e

  if (( command_status == 0 )); then
    printf "%s\tcompleted\t%s\n" "$case_id" "$case_out" >> "$SUMMARY_FILE"
    log "completed ${case_id}"
  else
    printf "%s\tfailed_%s\t%s\n" "$case_id" "$command_status" "$case_out" >> "$SUMMARY_FILE"
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
