#!/usr/bin/env bash
set -Eeuo pipefail

# STAGE 2 of 3 - runtime environment only.
#
# Creates a dedicated PointLLM virtual environment and checks out the two code
# repositories at pinned commits. PointLLM needs the transformers 4.28.0.dev
# era; the benchmark venv (transformers 4.57) and the SGLang venv cannot host
# it. This script therefore never touches /mnt/group/cmh/.venvs/layoutddd_sys
# or /mnt/group/cmh/envs/sglang-qwen3vl.
#
# Both checkouts ship a `pointllm` package, so neither is pip-installed.
# Dependencies are installed once and the active code tree is selected at run
# time through PYTHONPATH. That keeps the baseline-versus-PointLLM-R comparison
# to a single controlled variable.
#
# This stage does not download weights (stage 1) and does not run inference
# (stage 3). Passing here means "imports resolve and CUDA is visible", nothing
# more.

REPO_ROOT=${REPO_ROOT:-/mnt/group/cmh/Layout_DDD}
TOOLS_ROOT=${TOOLS_ROOT:-/mnt/group/cmh/tools}
ENV_DIR=${ENV_DIR:-/mnt/group/cmh/envs/pointllm}
ENV_PY=${ENV_PY:-${ENV_DIR}/bin/python}
REQUIREMENTS=${REQUIREMENTS:-${REPO_ROOT}/Support/bash/mnet/pointllm_requirements.txt}

POINTLLM_REPO=${POINTLLM_REPO:-https://github.com/InternRobotics/PointLLM.git}
POINTLLM_COMMIT=${POINTLLM_COMMIT:-cb72f4e6ab625ddab92f84931127e12bc326b4be}
POINTLLM_DIR=${POINTLLM_DIR:-${TOOLS_ROOT}/PointLLM}

POINTLLM_R_REPO=${POINTLLM_R_REPO:-https://github.com/Xqle/PointLLM-R.git}
POINTLLM_R_COMMIT=${POINTLLM_R_COMMIT:-3bd1501a1d7a43a070ce66f8f1ad7a4a28514b8e}
POINTLLM_R_DIR=${POINTLLM_R_DIR:-${TOOLS_ROOT}/PointLLM-R}

# MNET ships Python 3.11 and no 3.10, uv, or conda. Upstream tested on 3.10.13,
# but the sole 3.10-only pin was tokenizers 0.12.1, and transformers 4.28 accepts
# tokenizers <0.14, so 0.13.3 clears the constraint with a cp311 wheel.
MIN_PYTHON=${MIN_PYTHON:-3.10}
BOOTSTRAP_PY=${BOOTSTRAP_PY:-/mnt/group/cmh/.venvs/layoutddd_sys/bin/python}
export HF_HOME=${HF_HOME:-/mnt/group/cmh/.cache/huggingface}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/group/cmh/.cache/pip}

log() {
  echo "==== $(date '+%F %T') $* ===="
}

fail() {
  echo "$*" >&2
  exit 1
}

[[ -f "$REQUIREMENTS" ]] || fail "Missing requirements file: $REQUIREMENTS (sync the repo first)"
mkdir -p "$TOOLS_ROOT" "$PIP_CACHE_DIR" "$HF_HOME"

log "stage 2a: pinned code checkouts"
checkout() {
  local url=$1 commit=$2 dir=$3
  if [[ -d "${dir}/.git" ]]; then
    git -C "$dir" fetch --depth 50 origin "$commit" 2>/dev/null || git -C "$dir" fetch origin
  else
    git clone "$url" "$dir"
  fi
  git -C "$dir" checkout --detach "$commit"
  echo "$dir -> $(git -C "$dir" rev-parse HEAD)"
}
checkout "$POINTLLM_REPO" "$POINTLLM_COMMIT" "$POINTLLM_DIR"
checkout "$POINTLLM_R_REPO" "$POINTLLM_R_COMMIT" "$POINTLLM_R_DIR"

for dir in "$POINTLLM_DIR" "$POINTLLM_R_DIR"; do
  [[ -d "${dir}/pointllm/model" ]] || fail "Checkout $dir does not contain pointllm/model"
done

log "stage 2b: dedicated python environment at ${ENV_DIR}"
BASE_PY=""
if [[ ! -x "$ENV_PY" ]]; then
  for candidate in "${BASE_PYTHON:-}" python3.11 python3 "$BOOTSTRAP_PY"; do
    [[ -n "$candidate" ]] || continue
    resolved=$(command -v "$candidate" 2>/dev/null || true)
    [[ -n "$resolved" ]] || continue
    if "$resolved" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= tuple(int(p) for p in '${MIN_PYTHON}'.split('.')) else 1)" 2>/dev/null; then
      BASE_PY="$resolved"
      break
    fi
  done
  [[ -n "$BASE_PY" ]] || fail "No python >= ${MIN_PYTHON} found to create ${ENV_DIR}"
  echo "base interpreter: $BASE_PY ($("$BASE_PY" --version 2>&1))"

  # This pod's system python ships venv but not ensurepip's bundled wheels, so
  # the default venv creation dies partway. A pip-less venv is fine here because
  # the install is driven by an external pip anyway.
  if ! "$BASE_PY" -m venv "$ENV_DIR"; then
    echo "venv with pip failed; retrying without pip"
    rm -rf "$ENV_DIR"
    "$BASE_PY" -m venv --without-pip "$ENV_DIR" \
      || fail "could not create $ENV_DIR even with --without-pip"
  fi
fi
[[ -x "$ENV_PY" ]] || fail "Environment creation did not produce $ENV_PY"
"$ENV_PY" -c 'import sys; print("python", sys.version)'

log "stage 2c: install pinned dependencies"
# The venv may have no pip at all, and a distro-provided one can be old enough
# to misresolve packages the index actually serves. An external pip with
# --python installs into the target interpreter without needing pip there.
# Note the argument order: --python must precede the subcommand.
#
# Capability is established by running the install, not by grepping `pip --help`
# for the flag. This environment lives on a network filesystem where loading
# pip's full command table took over thirty seconds, so a probe that parses help
# text hangs longer than the work it was guarding.
INSTALLER=("$BOOTSTRAP_PY" -m pip --python "$ENV_PY" install)
INSTALLER_LABEL="external pip targeting $ENV_PY"
if [[ ! -x "$BOOTSTRAP_PY" ]]; then
  "$ENV_PY" -m pip --version >/dev/null 2>&1 \
    || fail "no usable pip: $BOOTSTRAP_PY is missing and $ENV_PY has no pip module"
  INSTALLER=("$ENV_PY" -m pip install)
  INSTALLER_LABEL="venv-local pip"
fi
echo "installer: $INSTALLER_LABEL"

# A single index hiccup surfaces as "from versions: none", which reads like a
# missing package rather than a network fault. Retry before believing it.
pip_install() {
  local attempt
  for attempt in 1 2 3; do
    if "${INSTALLER[@]}" "$@"; then
      return 0
    fi
    if [[ $attempt -lt 3 ]]; then
      echo "pip attempt ${attempt} failed; retrying in 20s"
      sleep 20
    fi
  done
  return 1
}

# setuptools stays because a few of the pinned legacy packages still import
# pkg_resources at runtime. pip itself is deliberately not installed into the
# target; the external pip owns this environment.
pip_install --upgrade setuptools wheel \
  || fail "could not install setuptools/wheel into $ENV_PY after three attempts"
pip_install -r "$REQUIREMENTS" \
  || fail "dependency install failed after three attempts"

log "stage 2d: verify imports and CUDA visibility"
PYTHONPATH="$POINTLLM_DIR" "$ENV_PY" - <<'PY'
import torch
import tokenizers
import transformers

print("torch          :", torch.__version__)
print("transformers   :", transformers.__version__)
print("tokenizers     :", tokenizers.__version__)
print("cuda available :", torch.cuda.is_available())

# PointLLM subclasses the LLaMA implementation as it stood in 4.28. The cache
# and attention-mask refactors from 4.36 onward change that base class, so a
# silently newer transformers would fail at generate time rather than here.
if not transformers.__version__.startswith("4.28."):
    raise SystemExit(
        f"expected transformers 4.28.x, got {transformers.__version__}; "
        "the pinned git requirement did not take effect"
    )
if tuple(int(p) for p in tokenizers.__version__.split(".")[:2]) >= (0, 14):
    raise SystemExit(
        f"tokenizers {tokenizers.__version__} exceeds the <0.14 cap declared by transformers 4.28"
    )

print("device count   :", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    print(f"  gpu{index}: {name} sm_{major}{minor}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the PointLLM environment")

# Hopper (sm_90) needs a torch built against CUDA 11.8 or newer. Upstream
# PointLLM was tested on CUDA 11.7, which silently has no sm_90 kernels.
build_cuda = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
if any(torch.cuda.get_device_capability(i)[0] >= 9 for i in range(torch.cuda.device_count())):
    if build_cuda < (11, 8):
        raise SystemExit(
            f"Hopper GPU present but torch was built against CUDA {torch.version.cuda}; "
            "reinstall with a cu118 or newer wheel"
        )

from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.conversation import conv_templates
from pointllm.data import load_objaverse_point_cloud  # noqa: F401

assert "vicuna_v1_1" in conv_templates, "vicuna_v1_1 conversation template is missing"
print("PointLLMLlamaForCausalLM:", PointLLMLlamaForCausalLM.__module__)
print("pointllm import: OK (no weights loaded)")
PY

log "stage 2e: verify the PointLLM-R checkout imports the same way"
PYTHONPATH="$POINTLLM_R_DIR" "$ENV_PY" -c \
  'from pointllm.model import PointLLMLlamaForCausalLM; import pointllm; print("PointLLM-R tree:", pointllm.__file__)'

log "stage 2 complete"
cat <<EOF

Environment : $ENV_PY
Baseline code : $POINTLLM_DIR   ($POINTLLM_COMMIT)
PointLLM-R code: $POINTLLM_R_DIR ($POINTLLM_R_COMMIT)

Stage 2 verified imports and CUDA visibility only. No checkpoint was loaded and
no point cloud was processed.
Next: Support/bash/mnet/run_pointllm_inference_smoke.sh (stage 3)
EOF
