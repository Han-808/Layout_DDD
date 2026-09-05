#!/usr/bin/env bash
set -u
set -o pipefail

# Local, resumable calibration workflow:
#   1. source-valid cal_dataset1 FP rate (no rendering)
#   2. Support deterministic local TopK ablation over existing exp1_1 evidence
#
# The script reuses the already-running local LiteLLM proxy and the
# LITELLM_MASTER_KEY inherited from the terminal that launches it.  It never
# reads, prints or writes the upstream OPENAPI_GPT_KEY.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PHASE="${PHASE:-all}"
MAX_WORKERS="${MAX_WORKERS:-2}"
RESUME="${RESUME:-1}"
BLENDER_BIN="${BLENDER_BIN:-/Applications/Blender.app/Contents/MacOS/Blender}"

DATASET_ROOT="$REPO_ROOT/Support/datasets/cal_dataset1"
JUDGE_CONFIG="$REPO_ROOT/configs/models/gpt5_6_sol_litellm_local_fine_edge_judge.json"
VALID_OUT="$REPO_ROOT/Support/artifacts/outputs/exp2_valid_fp_gpt56"
TOPK_EVIDENCE_OUT="$REPO_ROOT/Support/artifacts/outputs/exp2_support_topk_evidence"
TOPK_JUDGE_OUT="$REPO_ROOT/Support/artifacts/outputs/exp2_support_topk_gpt56"
INVALID_EVIDENCE="$REPO_ROOT/Support/artifacts/outputs/exp1_1"
AMBIGUOUS_EVIDENCE="$REPO_ROOT/Support/artifacts/outputs/exp1_1_fine_edge"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$JUDGE_CONFIG" ]]; then
  echo "Judge config is missing: $JUDGE_CONFIG" >&2
  exit 2
fi
if [[ -z "${LITELLM_MASTER_KEY:-}" ]]; then
  echo "LITELLM_MASTER_KEY is not set in this terminal." >&2
  exit 2
fi
if [[ "$PHASE" != "all" && "$PHASE" != "valid" &&
      "$PHASE" != "topk_prepare" && "$PHASE" != "topk_judge" ]]; then
  echo "PHASE must be all, valid, topk_prepare, or topk_judge." >&2
  exit 2
fi

resume_flag="--no-resume"
if [[ "$RESUME" == "1" ]]; then
  resume_flag="--resume"
fi

failures=0

run_valid() {
  echo "==== source-valid FP experiment: 10 scenes / 200 valid events ===="
  local case_args=()
  local index
  for index in $(seq 1 10); do
    case_args+=(--case-id "$(printf 'source_valid_%03d' "$index")")
  done

  "$PYTHON" Support/legacy/scripts/run_cal_dataset1_experiment.py \
    --dataset-root "$DATASET_ROOT" \
    --out-root "$VALID_OUT" \
    --track deterministic \
    --arm mesh \
    --judge-config "$JUDGE_CONFIG" \
    "$resume_flag" \
    --continue-on-error \
    "${case_args[@]}"
  local experiment_rc=$?
  if [[ "$experiment_rc" -ne 0 ]]; then
    echo "source-valid evaluator completed with one or more case errors (rc=$experiment_rc)." >&2
    failures=$((failures + 1))
  fi

  "$PYTHON" scripts/summarize_cal_dataset1_valid_fp.py \
    --dataset-root "$DATASET_ROOT" \
    --run-root "$VALID_OUT" \
    --out-dir "$VALID_OUT/fp_analysis" \
    --allow-partial
  local summary_rc=$?
  if [[ "$summary_rc" -ne 0 ]]; then
    echo "source-valid FP summarization failed (rc=$summary_rc)." >&2
    failures=$((failures + 1))
  fi
}

run_topk_prepare() {
  echo "==== Support TopK evidence: reuse Top1/Top2, render missing Top3 only ===="
  if [[ ! -x "$BLENDER_BIN" ]]; then
    echo "Blender is not executable: $BLENDER_BIN" >&2
    failures=$((failures + 1))
    return
  fi
  "$PYTHON" scripts/prepare_cal_dataset1_support_topk.py \
    --source-root "$INVALID_EVIDENCE" \
    --source-root "$AMBIGUOUS_EVIDENCE" \
    --out-dir "$TOPK_EVIDENCE_OUT" \
    --blender-bin "$BLENDER_BIN" \
    "$resume_flag" \
    --continue-on-error
  local prepare_rc=$?
  if [[ "$prepare_rc" -ne 0 ]]; then
    echo "Support TopK preparation completed with errors (rc=$prepare_rc)." >&2
    failures=$((failures + 1))
  fi
}

run_topk_judge() {
  echo "==== Support TopK judgement: K1/K2/K3, same local API environment ===="
  "$PYTHON" scripts/judge_cal_dataset1_support_topk.py \
    --evidence-root "$TOPK_EVIDENCE_OUT" \
    --dataset-root "$DATASET_ROOT" \
    --judge-config "$JUDGE_CONFIG" \
    --out-dir "$TOPK_JUDGE_OUT" \
    --max-workers "$MAX_WORKERS" \
    "$resume_flag" \
    --continue-on-error
  local judge_rc=$?
  if [[ "$judge_rc" -ne 0 ]]; then
    echo "Support TopK judge completed with errors (rc=$judge_rc)." >&2
    failures=$((failures + 1))
  fi
}

case "$PHASE" in
  all)
    run_valid
    run_topk_prepare
    run_topk_judge
    ;;
  valid)
    run_valid
    ;;
  topk_prepare)
    run_topk_prepare
    ;;
  topk_judge)
    run_topk_judge
    ;;
esac

echo
echo "==== outputs ===="
echo "valid FP:       $VALID_OUT/fp_analysis"
echo "TopK evidence:  $TOPK_EVIDENCE_OUT"
echo "TopK judgement: $TOPK_JUDGE_OUT"

if [[ "$failures" -ne 0 ]]; then
  echo "Workflow finished all requested phases with $failures failed component(s)." >&2
  exit 1
fi
echo "Workflow completed successfully."
