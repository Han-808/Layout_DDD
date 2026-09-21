#!/bin/zsh

set -euo pipefail
set +x

# Open-space +10: run the eight `*-plus10-v1` campaigns (brief_10..19) grouped by
# credential family, one terminal per family.  Generation policy is owned by the
# registered campaigns; this launcher only decides which campaigns run, binds
# endpoints, and runs every campaign of a family concurrently (one process and
# one output directory per model, because the frozen core runs briefs serially
# and refuses to share an output root).
#
#   Terminal 1:  run_open_space_plus10.sh api2      (Kimi K3, GLM 5.3, GPT-5.6-Sol, Astra)
#   Terminal 2:  run_open_space_plus10.sh api3      (Sonnet 5, Opus 5, Fable 5)
#   Terminal 3:  run_open_space_plus10.sh tokenhub  (Hy4-preview via isolated LiteLLM)
#
# Credentials are read hidden once per family and live only in this process.

SCRIPT_PATH="$0"
SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h:h}"
# A git worktree carries the tracked tree (configs, src) but not the local-only
# runtime: .venv, .runtime bindings, untracked proxy launchers, artifact
# outputs.  Those live in the primary checkout, so fall back to it for them and
# keep generated outputs there, where they outlive the worktree.
if [[ "$REPO_ROOT" == */.claude/worktrees/* ]]; then
  PRIMARY_ROOT="${REPO_ROOT%%/.claude/worktrees/*}"
else
  PRIMARY_ROOT="$REPO_ROOT"
fi
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN="$PRIMARY_ROOT/.venv/bin/python"
RESOURCE_BINDINGS="${LAYOUT_DDD_RETRIEVAL_BINDINGS:-$REPO_ROOT/.runtime/retrieval_bindings.local.json}"
[[ -f "$RESOURCE_BINDINGS" ]] || RESOURCE_BINDINGS="$PRIMARY_ROOT/.runtime/retrieval_bindings.local.json"
PROXY_LAUNCHER="$REPO_ROOT/Support/bash/local/run_litellm_hy4_preview_tokenhub_proxy.sh"
[[ -x "$PROXY_LAUNCHER" ]] || PROXY_LAUNCHER="$PRIMARY_ROOT/Support/bash/local/run_litellm_hy4_preview_tokenhub_proxy.sh"
DEFAULT_OUTPUT_BASE="$PRIMARY_ROOT/Support/artifacts/outputs/e2e_scenegen_repro/runs/open_space_plus10_r1"

FAMILY=""
OUTPUT_BASE="$DEFAULT_OUTPUT_BASE"
MODE="run"            # run | preflight | check
PROXY_PORT=4025
typeset -a ONLY_CAMPAIGNS
ONLY_CAMPAIGNS=()

usage() {
  print -r -- "Usage: $SCRIPT_PATH <api2|api3|tokenhub> [options]"
  print -r -- ""
  print -r -- "Runs every registered Open-space plus10 campaign of one credential family"
  print -r -- "concurrently (brief_10..19, one process per model)."
  print -r -- ""
  print -r -- "Options:"
  print -r -- "  --check-only         Validate public contracts + resource gate; no credential, no network"
  print -r -- "  --preflight-only     Bind credential and run one live probe per campaign; generate nothing"
  print -r -- "  --output-base PATH   Parent directory of per-campaign outputs (default: $DEFAULT_OUTPUT_BASE)"
  print -r -- "  --campaign ID        Restrict to this campaign id; repeatable"
  print -r -- "  --proxy-port PORT    tokenhub only: isolated loopback LiteLLM port (default: 4025)"
  print -r -- "  -h, --help           Show this help"
  print -r -- ""
  print -r -- "Credential env names (read hidden when absent): api2 -> API2_APP_CREDENTIAL,"
  print -r -- "api3 -> API3_API_KEY, tokenhub -> TOKENHUB_API_KEY."
}

if (( $# == 0 )); then usage >&2; exit 2; fi
FAMILY="$1"; shift
while (( $# > 0 )); do
  case "$1" in
    --check-only) MODE="check"; shift ;;
    --preflight-only) MODE="preflight"; shift ;;
    --output-base) (( $# >= 2 )) || { print -u2 -- "--output-base requires a value"; exit 2; }; OUTPUT_BASE="$2"; shift 2 ;;
    --campaign) (( $# >= 2 )) || { print -u2 -- "--campaign requires a value"; exit 2; }; ONLY_CAMPAIGNS+=("$2"); shift 2 ;;
    --proxy-port) (( $# >= 2 )) || { print -u2 -- "--proxy-port requires a value"; exit 2; }; PROXY_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) print -u2 -- "Unknown argument: $1"; usage >&2; exit 2 ;;
  esac
done

# Campaign -> route -> upstream endpoint.  Endpoints mirror the existing
# multi-room launcher (run_multi_room_generation_v1.sh) and the Astra binding
# that produced Support/Astra_outputs; the credential is only ever named.
typeset -a CAMPAIGNS ROUTES ENDPOINTS
case "$FAMILY" in
  api2)
    CREDENTIAL_ENV="API2_APP_CREDENTIAL"
    CREDENTIAL_PROMPT="API2 APP_ID:APP_KEY (hidden; shared by Kimi/GLM/Sol/Astra)"
    # The four `*-plus10-v1` campaigns are round 1.  The three `*-t1800`/`-rerun`
    # ids are the consolidated make-up campaigns: Kimi/GLM re-bound to 1800s
    # request+stage_c timeouts (round 1 halted at brief_11 on a Stage C
    # wait_response ambiguous timeout at 600s/1200s), and Sol re-run unchanged
    # (its 3000s timeout was ample; brief_16 was a stage_a schema violation).
    # Select the make-up ids explicitly with --campaign and a fresh --output-base.
    CAMPAIGNS=(api2-kimi-k3-plus10-v1 api2-glm53-plus10-v1 api2-gpt56sol-plus10-v1 api2-gpt6-astra-high-plus10-v1 api2-kimi-k3-plus10-t1800-v1 api2-glm53-plus10-t1800-v1 api2-gpt56sol-plus10-rerun-v1)
    ROUTES=(api2-chat-top-level-reasoning-v1 api2-responses-reasoning-v1 api2-standard-chat-reasoning-v1 api2-chat-top-level-reasoning-azure-v1 api2-chat-top-level-reasoning-v1 api2-responses-reasoning-v1 api2-standard-chat-reasoning-v1)
    ENDPOINTS=(
      "http://trpc-gpt-eval.production.polaris:8080/v1/chat/completions"
      "http://trpc-gpt-eval.production.polaris:8080/api/v1/responses"
      "http://llm-api.model-eval.woa.com/v1/chat/completions"
      "http://trpc-gpt-eval.production.polaris:8080/openai/v1/chat/completions"
      "http://trpc-gpt-eval.production.polaris:8080/v1/chat/completions"
      "http://trpc-gpt-eval.production.polaris:8080/api/v1/responses"
      "http://llm-api.model-eval.woa.com/v1/chat/completions"
    )
    ;;
  api3)
    CREDENTIAL_ENV="API3_API_KEY"
    CREDENTIAL_PROMPT="API3 key (hidden; shared by Sonnet 5/Opus 5/Fable 5)"
    CAMPAIGNS=(api3-sonnet5-plus10-v1 api3-opus5-plus10-v1 api3-fable5-plus10-v1)
    ROUTES=(api3-chat-legacy-core-v1 api3-chat-legacy-core-v1 api3-chat-legacy-core-v1)
    ENDPOINTS=(
      "http://21.214.33.175:4000/v1/chat/completions"
      "http://21.214.33.175:4000/v1/chat/completions"
      "http://21.214.33.175:4000/v1/chat/completions"
    )
    ;;
  tokenhub)
    CREDENTIAL_ENV="TOKENHUB_API_KEY"
    CREDENTIAL_PROMPT="TOKENHUB_API_KEY (hidden)"
    CAMPAIGNS=(tokenhub-hy4-preview-plus10-v1)
    ROUTES=(tokenhub-chat-top-level-reasoning-v1)
    ENDPOINTS=("http://127.0.0.1:${PROXY_PORT}/v1/chat/completions")
    ;;
  *)
    print -u2 -- "Unknown family: $FAMILY (expected api2, api3 or tokenhub)"
    exit 2
    ;;
esac

if (( ${#ONLY_CAMPAIGNS[@]} > 0 )); then
  typeset -a SEL_C SEL_R SEL_E
  for (( i=1; i<=${#CAMPAIGNS[@]}; i++ )); do
    if (( ${ONLY_CAMPAIGNS[(Ie)${CAMPAIGNS[$i]}]} )); then
      SEL_C+=("${CAMPAIGNS[$i]}"); SEL_R+=("${ROUTES[$i]}"); SEL_E+=("${ENDPOINTS[$i]}")
    fi
  done
  (( ${#SEL_C[@]} == ${#ONLY_CAMPAIGNS[@]} )) || {
    print -u2 -- "--campaign must name campaigns of the $FAMILY family: ${CAMPAIGNS[*]}"
    exit 2
  }
  CAMPAIGNS=("${SEL_C[@]}"); ROUTES=("${SEL_R[@]}"); ENDPOINTS=("${SEL_E[@]}")
fi

cd "$REPO_ROOT"
[[ -x "$PYTHON_BIN" ]] || { print -u2 -- "Python runtime is unavailable: $PYTHON_BIN"; exit 2; }
[[ -f "$RESOURCE_BINDINGS" && ! -L "$RESOURCE_BINDINGS" ]] || {
  print -u2 -- "Retrieval resource bindings are unavailable: $RESOURCE_BINDINGS"
  exit 2
}
if [[ "$FAMILY" == "tokenhub" ]]; then
  [[ -x "$PROXY_LAUNCHER" ]] || { print -u2 -- "TokenHub proxy launcher is missing"; exit 2; }
  [[ "$PROXY_PORT" == <-> ]] && (( PROXY_PORT >= 1024 && PROXY_PORT <= 65535 )) || {
    print -u2 -- "--proxy-port must be an integer from 1024 through 65535"; exit 2
  }
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

print -r -- "Open-space +10 [$FAMILY]  mode=$MODE"
print -r -- "Repo: $REPO_ROOT"
print -r -- "Campaigns (brief_10..19, retries=3, run concurrently): ${CAMPAIGNS[*]}"

# ---- Stage 1: public contracts + resource gate (no credential, no network) ----
for c in "${CAMPAIGNS[@]}"; do
  "$PYTHON_BIN" -m benchmark.scene_generation check --campaign "$c" >/dev/null || {
    print -u2 -- "check failed: $c"; exit 2
  }
done
print -r -- "check: ${#CAMPAIGNS[@]} campaign(s) valid"
# The resource gate loads the encoder once per call; one campaign is enough
# because all plus10 campaigns share the same retrieval profile.
"$PYTHON_BIN" -m benchmark.scene_generation resource-gate \
  --campaign "${CAMPAIGNS[1]}" --resource-bindings "$RESOURCE_BINDINGS" >/dev/null || {
  print -u2 -- "resource-gate failed"; exit 2
}
print -r -- "resource-gate: ready"
if [[ "$MODE" == "check" ]]; then
  print -r -- "Check-only passed; no credential read, proxy started, or API request sent"
  exit 0
fi

# ---- Stage 2: credential + private route binding ----
if [[ "$MODE" == "run" ]]; then
  for c in "${CAMPAIGNS[@]}"; do
    [[ ! -e "$OUTPUT_BASE/$c" ]] || {
      print -u2 -- "Refusing to overwrite existing output: $OUTPUT_BASE/$c"
      print -u2 -- "The frozen single-room workflow never resumes; choose a new --output-base."
      exit 2
    }
  done
fi

if [[ -z "${(P)CREDENTIAL_ENV:-}" ]]; then
  read -r -s "REPLY?$CREDENTIAL_PROMPT: "
  print
  typeset -g "$CREDENTIAL_ENV"="$REPLY"
  unset REPLY
fi
[[ -n "${(P)CREDENTIAL_ENV}" ]] || { print -u2 -- "$CREDENTIAL_ENV is empty"; exit 2; }
if [[ "$FAMILY" == "api2" && "${${(P)CREDENTIAL_ENV}%%\?*}" != *:* ]]; then
  print -u2 -- "API2_APP_CREDENTIAL must have APP_ID:APP_KEY form"; exit 2
fi
export "$CREDENTIAL_ENV"

TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/layoutddd-plus10-${FAMILY}.XXXXXX")
GENERATION_BINDINGS="$TEMP_DIR/generation_bindings.json"
PROXY_PID=""
typeset -a CHILD_PIDS
CHILD_PIDS=()

cleanup() {
  local pid
  for pid in "${CHILD_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
  done
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  unset API2_APP_CREDENTIAL API3_API_KEY TOKENHUB_API_KEY TOKENHUB_PROXY_KEY LITELLM_MASTER_KEY LITELLM_HY4_TOKENHUB_PORT
  command rm -f -- "$GENERATION_BINDINGS" 2>/dev/null || true
  command rmdir -- "$TEMP_DIR" 2>/dev/null || true
}
interrupt_run() {
  print -u2 -- "Interrupted; a case in flight is not resent automatically. Inspect outputs before rerunning."
  exit 130
}
trap cleanup EXIT
trap interrupt_run INT TERM HUP

BINDING_CREDENTIAL_ENV="$CREDENTIAL_ENV"
if [[ "$FAMILY" == "tokenhub" ]]; then
  # The generation runner talks to an isolated loopback LiteLLM proxy whose
  # ephemeral master key is what the binding names; the upstream TokenHub key
  # is only ever read by the proxy process.
  LITELLM_MASTER_KEY="sk-layoutddd-$(openssl rand -hex 24)"
  TOKENHUB_PROXY_KEY="$LITELLM_MASTER_KEY"
  export LITELLM_MASTER_KEY TOKENHUB_PROXY_KEY
  export LITELLM_HY4_TOKENHUB_PORT="$PROXY_PORT"
  BINDING_CREDENTIAL_ENV="TOKENHUB_PROXY_KEY"
  "$PROXY_LAUNCHER" >/dev/null 2>&1 &
  PROXY_PID=$!
  PROXY_READY=0
  for _ in {1..60}; do
    kill -0 "$PROXY_PID" 2>/dev/null || break
    if curl -fsS --max-time 2 "http://127.0.0.1:${PROXY_PORT}/health/liveliness" >/dev/null 2>&1; then
      PROXY_READY=1; break
    fi
    sleep 1
  done
  (( PROXY_READY == 1 )) || { print -u2 -- "Local TokenHub proxy did not become ready on port $PROXY_PORT"; exit 2; }
  print -r -- "tokenhub proxy: ready on 127.0.0.1:$PROXY_PORT (proxy retries disabled)"
fi

# Write the private binding file: route id -> endpoint + credential env NAME.
BINDING_ROUTES="${(j:,:)ROUTES}" BINDING_ENDPOINTS="${(j:,:)ENDPOINTS}" \
BINDING_CREDENTIAL_ENV="$BINDING_CREDENTIAL_ENV" GENERATION_BINDINGS="$GENERATION_BINDINGS" \
  "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
routes = os.environ["BINDING_ROUTES"].split(",")
endpoints = os.environ["BINDING_ENDPOINTS"].split(",")
bindings = {}
for route, endpoint in zip(routes, endpoints):
    existing = bindings.get(route)
    if existing is not None and existing["endpoint"] != endpoint:
        raise SystemExit(f"route {route} bound to two endpoints")
    bindings[route] = {"endpoint": endpoint, "credential_env": os.environ["BINDING_CREDENTIAL_ENV"]}
Path(os.environ["GENERATION_BINDINGS"]).write_text(
    json.dumps({"schema_version": "generation_route_bindings_v2", "bindings": bindings}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
chmod 600 "$GENERATION_BINDINGS"

# ---- Stage 3: resolve + live preflight, one probe per campaign ----
for c in "${CAMPAIGNS[@]}"; do
  "$PYTHON_BIN" -m benchmark.scene_generation resolve --campaign "$c" \
    --generation-bindings "$GENERATION_BINDINGS" --resource-bindings "$RESOURCE_BINDINGS" >/dev/null || {
    print -u2 -- "resolve failed: $c"; exit 2
  }
done
print -r -- "resolve: bindings present for ${#CAMPAIGNS[@]} campaign(s)"

# Persist the sanitized preflight report per campaign so a later reader (or a
# background monitor) can see pass/fail without the terminal.  The CLI's last
# line is its public terminal surface: logical ids, http status and failure
# category only; never an endpoint, credential name, header or server body.
LOG_DIR="$OUTPUT_BASE/_launcher_logs"
mkdir -p -- "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PREFLIGHT_RECORD="$LOG_DIR/preflight.${FAMILY}.${STAMP}.jsonl"
: > "$PREFLIGHT_RECORD"

PREFLIGHT_FAILED=0
for c in "${CAMPAIGNS[@]}"; do
  set +e
  # `$status` is zsh's read-only alias of `$?`; use a distinct name.  With a
  # pipeline, capture the preflight command's own exit via pipestatus.
  report=$("$PYTHON_BIN" -m benchmark.scene_generation preflight --campaign "$c" \
    --generation-bindings "$GENERATION_BINDINGS" --resource-bindings "$RESOURCE_BINDINGS" 2>&1 | tail -1; print -r -- "__EXIT__=${pipestatus[1]}")
  preflight_exit="${report##*__EXIT__=}"
  report="${report%$'\n'__EXIT__=*}"
  set -e
  ok=$(print -r -- "$report" | jq -r '.ok // false' 2>/dev/null || print false)
  jq -c -n --arg campaign "$c" --arg mode "$MODE" --argjson exit "${preflight_exit:-255}" \
    --arg observed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg raw "$report" \
    '{observed_at: $observed_at, mode: $mode, campaign_id: $campaign, exit: $exit,
      report: (try ($raw | fromjson) catch {unparsed: $raw})}' >> "$PREFLIGHT_RECORD" 2>/dev/null || true
  if [[ "$ok" == "true" && "$preflight_exit" == "0" ]]; then
    print -r -- "preflight OK   : $c"
  else
    PREFLIGHT_FAILED=1
    detail=$(print -r -- "$report" | jq -r '"http=\(.http_status // "-") category=\(.failure_category // "-")"' 2>/dev/null || print "unparsed")
    print -u2 -- "preflight FAIL : $c  ($detail)"
  fi
done
print -r -- "preflight record: $PREFLIGHT_RECORD"
(( PREFLIGHT_FAILED == 0 )) || { print -u2 -- "Preflight failed for at least one campaign; nothing generated"; exit 2; }
if [[ "$MODE" == "preflight" ]]; then
  print -r -- "Preflight-only passed for [$FAMILY]; nothing generated"
  exit 0
fi

# ---- Stage 4: generate, all campaigns of the family concurrently ----
print -r -- "generation: launching ${#CAMPAIGNS[@]} process(es) -> $OUTPUT_BASE"
typeset -A PID_TO_CAMPAIGN
for c in "${CAMPAIGNS[@]}"; do
  log="$LOG_DIR/${c}.${STAMP}.log"
  "$PYTHON_BIN" -m benchmark.scene_generation run --campaign "$c" \
    --generation-bindings "$GENERATION_BINDINGS" --resource-bindings "$RESOURCE_BINDINGS" \
    --output-dir "$OUTPUT_BASE/$c" >"$log" 2>&1 &
  pid=$!
  CHILD_PIDS+=("$pid")
  PID_TO_CAMPAIGN[$pid]="$c"
  print -r -- "  started $c  pid=$pid  log=$log"
done

FAILED=0
for pid in "${CHILD_PIDS[@]}"; do
  set +e
  wait "$pid"
  rc=$?
  set -e
  c="${PID_TO_CAMPAIGN[$pid]}"
  summary="$OUTPUT_BASE/$c/summary.json"
  if [[ -f "$summary" ]]; then
    line=$(jq -r '"complete=\(.complete) failed=\(.failed) eligible=\(.eligible) stopped_early=\(.stopped_early)"' "$summary" 2>/dev/null || print "summary unreadable")
  else
    line="no summary.json (terminated before run_terminal)"
  fi
  if (( rc == 0 )); then
    print -r -- "  done  $c  exit=0  $line"
  else
    FAILED=1
    print -u2 -- "  done  $c  exit=$rc  $line"
  fi
done
CHILD_PIDS=()

if (( FAILED == 0 )); then
  print -r -- "Open-space +10 [$FAMILY] complete: $OUTPUT_BASE"
  exit 0
fi
print -u2 -- "Open-space +10 [$FAMILY] ended with at least one failed/partial campaign: $OUTPUT_BASE"
exit 2
