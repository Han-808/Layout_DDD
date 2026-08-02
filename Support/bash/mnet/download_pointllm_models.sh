#!/usr/bin/env bash
set -Eeuo pipefail

# STAGE 1 of 3 - checkpoint acquisition only.
#
# Downloads the PointLLM baseline (arXiv:2308.16911) and PointLLM-R
# (arXiv:2605.22013) checkpoints at pinned revisions into the persistent MNET
# model store, then verifies them byte-for-byte.
#
# This stage does not create a runtime environment (stage 2:
# setup_pointllm_env.sh) and does not run inference (stage 3:
# run_pointllm_inference_smoke.sh). A pass here only means the bytes are on
# disk and match the pinned revision.
#
# The script is idempotent: if verification already passes it downloads
# nothing. Model weights live outside the repository and are never part of a
# code sync bundle.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
MODELS_ROOT=${MODELS_ROOT:-/mnt/group/cmh/models}
REGISTRY=${REGISTRY:-${REPO_ROOT}/configs/models/pointllm_mnet_registry.json}

# The downloader deliberately does not use the PointLLM venv, so a broken
# stage-2 runtime can never look like a broken download. Any interpreter with a
# working huggingface_hub will do; set DOWNLOAD_PY_OVERRIDE to force one.
DOWNLOAD_PY=${DOWNLOAD_PY:-/mnt/group/cmh/envs/hf-download/bin/python}
BOOTSTRAP_PY=${BOOTSTRAP_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}

export HF_HOME=${HF_HOME:-/mnt/group/cmh/.cache/huggingface}
export HF_HUB_DISABLE_TELEMETRY=1
# Set HF_ENDPOINT=https://hf-mirror.com before invoking if huggingface.co is
# unreachable from the pod.

# Space-separated registry keys. Datasets are opt-in.
MODELS=${MODELS:-"PointLLM_7B_v1.2 PointLLM-R-7B"}
DATASETS=${DATASETS:-"modelnet40_test objaverse_val_gt"}
SHA256_MODE=${SHA256_MODE:-lfs}
FORCE=${FORCE:-0}
MIN_FREE_GB=${MIN_FREE_GB:-80}

# Stall watchdog. Eight parallel connections were observed to wedge on this
# link; four is slower to start but has not stalled.
FETCH_TIMEOUT_SECONDS=${FETCH_TIMEOUT_SECONDS:-1200}
FETCH_MAX_ATTEMPTS=${FETCH_MAX_ATTEMPTS:-60}
FETCH_RETRY_SLEEP=${FETCH_RETRY_SLEEP:-15}
FETCH_MAX_WORKERS=${FETCH_MAX_WORKERS:-4}
# The Rust transfer backend is the more common source of silent hangs.
HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}

LOG_ROOT=${LOG_ROOT:-/mnt/group/cmh/logs}
MANIFEST=${MANIFEST:-${LOG_ROOT}/pointllm_download_manifest_$(date '+%Y%m%d_%H%M%S').json}

log() {
  echo "==== $(date '+%F %T') $* ===="
}

fail() {
  echo "$*" >&2
  exit 1
}

[[ -f "$REGISTRY" ]] || fail "Missing registry: $REGISTRY (sync the repo first)"

# The verifier is pure standard library, so any working python3 can run it.
if [[ ! -x "$BOOTSTRAP_PY" ]]; then
  BOOTSTRAP_PY=$(command -v python3 || true)
  [[ -n "$BOOTSTRAP_PY" ]] || fail "No python3 available to run the checkpoint verifier"
  echo "note: falling back to $BOOTSTRAP_PY for verification"
fi

# Without this the stall watchdog silently degrades into 60 no-op retries.
command -v timeout >/dev/null 2>&1 \
  || fail "coreutils 'timeout' is required for the download stall watchdog"

case "$MODELS_ROOT" in
  "$REPO_ROOT"|"$REPO_ROOT"/*)
    fail "MODELS_ROOT=$MODELS_ROOT is inside the code repo. Weights must stay separate from synced code."
    ;;
esac

mkdir -p "$MODELS_ROOT" "$LOG_ROOT" "$HF_HOME"

log "stage 1 preflight"
echo "registry     : $REGISTRY"
echo "models root  : $MODELS_ROOT"
echo "HF_HOME      : $HF_HOME"
echo "HF_ENDPOINT  : ${HF_ENDPOINT:-https://huggingface.co (default)}"
echo "models       : $MODELS"
echo "datasets     : $DATASETS"
df -h "$MODELS_ROOT"

free_gb=$(df -BG --output=avail "$MODELS_ROOT" | tail -1 | tr -dc '0-9')
if (( free_gb < MIN_FREE_GB )); then
  fail "Only ${free_gb}GB free under $MODELS_ROOT; need at least ${MIN_FREE_GB}GB. Free space or set MIN_FREE_GB."
fi

log "stage 1: resolve an interpreter that can talk to the Hub"

# Prefer an environment that already works over building a new one. Creating an
# empty venv forces pip to resolve huggingface_hub's whole dependency tree
# against the pod's package index; if that index is missing even one transitive
# dependency such as filelock, the install fails and the download never starts.
# Any environment running transformers or torch already has both.
usable_python() {
  local candidate=$1
  [[ -x "$candidate" ]] || return 1
  "$candidate" -c 'import huggingface_hub' >/dev/null 2>&1
}

RESOLVED_PY=""
for candidate in \
  "${DOWNLOAD_PY_OVERRIDE:-}" \
  "$DOWNLOAD_PY" \
  "$BOOTSTRAP_PY" \
  /mnt/group/cmh/envs/sglang-qwen3vl/bin/python; do
  [[ -n "$candidate" ]] || continue
  if usable_python "$candidate"; then
    RESOLVED_PY="$candidate"
    break
  fi
done

if [[ -z "$RESOLVED_PY" ]]; then
  cat >&2 <<'EOF'
No available interpreter can import huggingface_hub, and building a dedicated
venv is not attempted automatically because it depends on the pod package index
being complete.

Diagnose the index first:
  pip config list
  echo "$PIP_INDEX_URL $PIP_EXTRA_INDEX_URL $PIP_CONSTRAINT"
  /mnt/group/cmh/.venvs/layoutddd_sys/bin/python -m pip download --no-deps -d /tmp/probe filelock

If filelock is genuinely absent from the index, install into an environment that
already has it (torch depends on filelock) using:
  <python> -m pip install --no-deps 'huggingface_hub>=0.23'

Then re-run with DOWNLOAD_PY_OVERRIDE=<python>.
EOF
  exit 1
fi

DOWNLOAD_PY="$RESOLVED_PY"
echo "downloader python: $DOWNLOAD_PY"
"$DOWNLOAD_PY" -c 'import huggingface_hub; print("huggingface_hub", huggingface_hub.__version__)'

# hf_transfer is a throughput optimisation, never a requirement, and it is off
# by default here because it hangs rather than erroring when a socket dies.
# Set HF_HUB_ENABLE_HF_TRANSFER=1 to opt back in on a healthy link.
if [[ "$HF_HUB_ENABLE_HF_TRANSFER" == "1" ]] \
   && ! "$DOWNLOAD_PY" -c 'import hf_transfer' >/dev/null 2>&1; then
  echo "hf_transfer requested but not installed; using the standard downloader"
  HF_HUB_ENABLE_HF_TRANSFER=0
fi
export HF_HUB_ENABLE_HF_TRANSFER
echo "hf_transfer enabled: ${HF_HUB_ENABLE_HF_TRANSFER}"

verify() {
  "$BOOTSTRAP_PY" "${REPO_ROOT}/scripts/verify_pointllm_checkpoint.py" \
    --registry "$REGISTRY" \
    --models-root "$MODELS_ROOT" \
    --sha256 "$SHA256_MODE" \
    "$@"
}

# A stalled transfer is the common failure on this link: sockets die without
# closing and the download hangs indefinitely rather than erroring. A retry loop
# alone cannot recover from that, so each attempt runs under a timeout. Partial
# files live in .cache/huggingface/download/*.incomplete and resume, so a killed
# attempt costs nothing already transferred.
fetch() {
  local kind=$1 key=$2
  local rc
  log "downloading $key"
  for attempt in $(seq 1 "$FETCH_MAX_ATTEMPTS"); do
    rc=0
    timeout --signal=TERM --kill-after=60 "$FETCH_TIMEOUT_SECONDS" \
      "$DOWNLOAD_PY" "${REPO_ROOT}/scripts/hf_snapshot_download.py" \
      --registry "$REGISTRY" \
      --kind "$kind" \
      --key "$key" \
      --models-root "$MODELS_ROOT" \
      --max-workers "$FETCH_MAX_WORKERS" || rc=$?
    if (( rc == 0 )); then
      return 0
    fi
    if (( rc == 124 || rc == 137 )); then
      echo "attempt ${attempt}/${FETCH_MAX_ATTEMPTS} for ${key} stalled past ${FETCH_TIMEOUT_SECONDS}s; resuming"
    else
      echo "attempt ${attempt}/${FETCH_MAX_ATTEMPTS} for ${key} exited rc=${rc}; resuming"
    fi
    sleep "$FETCH_RETRY_SLEEP"
  done
  fail "Download of ${key} did not complete after ${FETCH_MAX_ATTEMPTS} attempts"
}

log "stage 1a: probe whether anything actually needs downloading"
# This probe is not the stage-1 gate. A negative result here is the normal
# trigger to download, so its detailed report is kept out of the log to avoid
# reading like a failure. The real gate is stage 1b, below.
PROBE_LOG="${LOG_ROOT}/pointllm_probe_$$.log"
needs_download=0
if [[ "$FORCE" == "1" ]]; then
  echo "FORCE=1, re-running the download for every selected model"
  needs_download=1
elif verify $(for m in $MODELS; do printf -- '--model %s ' "$m"; done) >"$PROBE_LOG" 2>&1; then
  log "all selected checkpoints already verified; skipping download"
else
  echo "probe: one or more checkpoints are absent or incomplete, downloading"
  grep -E '^  (files|local_dir)' "$PROBE_LOG" || true
  needs_download=1
fi
rm -f "$PROBE_LOG"

if (( needs_download )); then
  for model in $MODELS; do
    fetch models "$model"
  done
fi

for dataset in $DATASETS; do
  fetch datasets "$dataset"
done

log "stage 1b: verify checkpoint completeness against pinned revisions"
verify $(for m in $MODELS; do printf -- '--model %s ' "$m"; done) --json-out "$MANIFEST"

log "stage 1 complete"
echo "manifest: $MANIFEST"
echo
echo "Stage 1 verified bytes on disk only."
echo "Next: Support/bash/mnet/setup_pointllm_env.sh   (stage 2, runtime environment)"
echo "Then: Support/bash/mnet/run_pointllm_inference_smoke.sh (stage 3, real point-cloud inference)"
