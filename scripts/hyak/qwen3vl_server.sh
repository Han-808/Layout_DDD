#!/usr/bin/env bash
set -euo pipefail

# Slurm-backed Qwen3-VL SGLang server for Hyak.
#
# Run from the repo root on a Hyak login node:
#   bash scripts/hyak/qwen3vl_server.sh submit
#   bash scripts/hyak/qwen3vl_server.sh status
#   bash scripts/hyak/qwen3vl_server.sh tail
#   bash scripts/hyak/qwen3vl_server.sh stop
#
# The server writes a ready file after /v1/models is reachable:
#   logs/ready/qwen3vl-server.url

MODE="${1:-submit}"
MODEL_PROFILE="${MODEL_PROFILE:-32b}"
case "$MODEL_PROFILE" in
  2b)
    DEFAULT_JOB_NAME="qwen3vl-2b-server"
    DEFAULT_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct"
    DEFAULT_PARTITION="gpu-l40s"
    DEFAULT_CPUS_PER_TASK="8"
    DEFAULT_MEMORY="80G"
    DEFAULT_CONTEXT_LENGTH="8192"
    DEFAULT_MEM_FRACTION_STATIC=""
    ;;
  8b)
    DEFAULT_JOB_NAME="qwen3vl-8b-server"
    DEFAULT_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
    DEFAULT_PARTITION="gpu-a100"
    DEFAULT_CPUS_PER_TASK="12"
    DEFAULT_MEMORY="120G"
    DEFAULT_CONTEXT_LENGTH="8192"
    DEFAULT_MEM_FRACTION_STATIC=""
    ;;
  32b)
    DEFAULT_JOB_NAME="qwen3vl-32b-server"
    DEFAULT_MODEL_ID="Qwen/Qwen3-VL-32B-Instruct"
    DEFAULT_PARTITION="gpu-h200"
    DEFAULT_CPUS_PER_TASK="16"
    DEFAULT_MEMORY="240G"
    DEFAULT_CONTEXT_LENGTH="65536"
    DEFAULT_MEM_FRACTION_STATIC="0.90"
    ;;
  *)
    echo "ERROR: unsupported MODEL_PROFILE='$MODEL_PROFILE' (expected 2b, 8b, or 32b)" >&2
    exit 2
    ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
ACCOUNT="${ACCOUNT:-h2lab}"
PARTITION="${PARTITION:-$DEFAULT_PARTITION}"
GPU_REQUEST="${GPU_REQUEST:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$DEFAULT_CPUS_PER_TASK}"
MEMORY="${MEMORY:-$DEFAULT_MEMORY}"
WALLTIME="${WALLTIME:-24:00:00}"

MODEL_ID="${MODEL_ID:-$DEFAULT_MODEL_ID}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-$DEFAULT_CONTEXT_LENGTH}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$DEFAULT_MEM_FRACTION_STATIC}"
SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:-}"

SGLANG_ENV="${SGLANG_ENV:-/gscratch/stf/mohanc3/.conda/envs/sglang311}"
HF_HOME="${HF_HOME:-/gscratch/h2lab/mohanc3/hf-cache}"
WANDB_DIR="${WANDB_DIR:-/gscratch/h2lab/mohanc3/wandb}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/gscratch/h2lab/mohanc3/xdg-cache}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/gscratch/h2lab/mohanc3/torch-extensions}"
TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-/gscratch/h2lab/mohanc3/tvm-ffi-cache}"

LOG_DIR="${LOG_DIR:-logs}"
READY_DIR="${READY_DIR:-logs/ready}"
READY_FILE="${READY_FILE:-$READY_DIR/qwen3vl-server.url}"
STOP_EXISTING="${STOP_EXISTING:-1}"
JOB_IDS="${JOB_IDS:-}"

usage() {
  cat <<EOF
Usage: bash scripts/hyak/qwen3vl_server.sh <submit|server|status|tail|stop|dry-run>

Common overrides:
  MODEL_PROFILE=32b
  MODEL_ID=Qwen/Qwen3-VL-32B-Instruct
  PARTITION=gpu-h200
  GPU_REQUEST=gpu:1
  MEMORY=240G
  CONTEXT_LENGTH=65536
  MEM_FRACTION_STATIC=0.90
  WALLTIME=24:00:00
  PORT=8000
  SGLANG_EXTRA_ARGS="--arg value"
  JOB_ID=36252666

Examples:
  bash scripts/hyak/qwen3vl_server.sh submit
  MODEL_PROFILE=8b bash scripts/hyak/qwen3vl_server.sh submit
  MODEL_PROFILE=2b bash scripts/hyak/qwen3vl_server.sh submit
  MODEL_PROFILE=32b CONTEXT_LENGTH=65536 bash scripts/hyak/qwen3vl_server.sh submit
  JOB_IDS="36168488 36168454" bash scripts/hyak/qwen3vl_server.sh stop
  bash scripts/hyak/qwen3vl_server.sh stop
EOF
}

print_config() {
  cat <<EOF
job_name=$JOB_NAME
model_profile=$MODEL_PROFILE
account=$ACCOUNT
partition=$PARTITION
gpu_request=$GPU_REQUEST
cpus_per_task=$CPUS_PER_TASK
memory=$MEMORY
walltime=$WALLTIME
model_id=$MODEL_ID
served_model_name=$SERVED_MODEL_NAME
host=$HOST
port=$PORT
context_length=$CONTEXT_LENGTH
mem_fraction_static=$MEM_FRACTION_STATIC
sglang_env=$SGLANG_ENV
hf_home=$HF_HOME
ready_file=$READY_FILE
stop_existing=$STOP_EXISTING
sglang_extra_args=$SGLANG_EXTRA_ARGS
EOF
}

submit_job() {
  mkdir -p "$LOG_DIR" "$READY_DIR"
  rm -f "$READY_FILE"
  if [ "$STOP_EXISTING" = "1" ]; then
    stop_jobs
  fi
  echo "Submitting Qwen3-VL server job with config:"
  print_config
  job_id=$(
    sbatch --parsable \
      --account="$ACCOUNT" \
      --partition="$PARTITION" \
      --gres="$GPU_REQUEST" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEMORY" \
      --time="$WALLTIME" \
      --job-name="$JOB_NAME" \
      --output="$LOG_DIR/%x-%j.out" \
      --error="$LOG_DIR/%x-%j.err" \
      "$0" server
  )
  echo "Submitted job_id=$job_id"
  echo "Monitor:"
  echo "  squeue -j $job_id"
  echo "  tail -f $LOG_DIR/$JOB_NAME-$job_id.out"
  echo "Ready file:"
  echo "  $READY_FILE"
  echo "After ready, open local tunnel from Windows:"
  echo "  ssh -N -L $PORT:<compute-node>:$PORT mohanc3@klone.hyak.uw.edu"
}

stop_jobs() {
  if [ -n "$JOB_IDS" ]; then
    echo "Cancelling explicit job id(s): $JOB_IDS"
    # shellcheck disable=SC2086
    scancel $JOB_IDS || true
  fi
  echo "Cancelling running or pending jobs named $JOB_NAME for user $USER"
  ids=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" || true)
  if [ -z "$ids" ]; then
    echo "No matching jobs found."
    return 0
  fi
  echo "$ids" | xargs -r scancel
  echo "Cancelled job(s): $ids"
}

status_jobs() {
  squeue -u "$USER" -o "%.18i %.10P %.35j %.8T %.10M %.8C %.10m %.30R"
  if [ -s "$READY_FILE" ]; then
    echo "Ready URL: $(cat "$READY_FILE")"
  else
    echo "Ready URL: not ready yet ($READY_FILE missing or empty)"
  fi
}

tail_logs() {
  job_id="${JOB_ID:-}"
  if [ -z "$job_id" ]; then
    job_id=$(squeue -h -u "$USER" -n "$JOB_NAME" -o "%i" 2>/dev/null | tail -n 1 || true)
  fi

  if [ -n "$job_id" ]; then
    target="$LOG_DIR/$JOB_NAME-$job_id.out"
    echo "Current Slurm job for $JOB_NAME: $job_id"
    echo "Tailing $target"
    while [ ! -e "$target" ]; do
      if ! squeue -h -j "$job_id" >/dev/null 2>&1; then
        echo "Job $job_id is no longer in squeue and log was not created: $target" >&2
        return 1
      fi
      echo "Waiting for log file to be created. Job may still be pending..."
      sleep 10
    done
    tail -f "$target"
    return 0
  fi

  latest=$(ls -t "$LOG_DIR"/"$JOB_NAME"-*.out 2>/dev/null | head -n 1 || true)
  if [ -z "$latest" ]; then
    echo "No active Slurm job and no logs found for $JOB_NAME under $LOG_DIR"
    return 1
  fi
  echo "No active Slurm job found for $JOB_NAME; tailing latest historical log."
  echo "Tailing $latest"
  tail -f "$latest"
}

server_main() {
  module load cuda/12.4.1 >/dev/null 2>&1 || true
  module load gcc/13.2.0 >/dev/null 2>&1 || true

  export CUDA_HOME=/sw/cuda/12.4.1
  export CUDA_PATH=/sw/cuda/12.4.1
  export CUDACXX=/sw/cuda/12.4.1/bin/nvcc
  export NVCC=/sw/cuda/12.4.1/bin/nvcc
  export CMAKE_CUDA_COMPILER=/sw/cuda/12.4.1/bin/nvcc
  export PATH=/sw/gcc/13.2.0/bin:/sw/cuda/12.4.1/bin:"$SGLANG_ENV/bin":${PATH}
  export LD_LIBRARY_PATH=/sw/gcc/13.2.0/lib64:/sw/cuda/12.4.1/lib64:${LD_LIBRARY_PATH:-}

  export CC=/sw/gcc/13.2.0/bin/gcc
  export CXX=/sw/gcc/13.2.0/bin/g++
  export CUDAHOSTCXX=/sw/gcc/13.2.0/bin/g++

  export HF_HOME
  export TRANSFORMERS_CACHE="$HF_HOME"
  export HF_DATASETS_CACHE="$HF_HOME"
  export WANDB_DIR
  export XDG_CACHE_HOME
  export TORCH_EXTENSIONS_DIR
  export TVM_FFI_CACHE_DIR
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false

  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  compute_host="$(hostname -s)"
  compute_fqdn="$(hostname -f)"
  export NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1,$compute_host,$compute_fqdn"
  export no_proxy="$NO_PROXY"

  mkdir -p "$HF_HOME" "$WANDB_DIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TVM_FFI_CACHE_DIR" "$READY_DIR"

  py="$SGLANG_ENV/bin/python"
  if [ ! -x "$py" ]; then
    echo "ERROR: Python not found or not executable: $py" >&2
    exit 1
  fi

  echo "Server job started at $(date)"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "COMPUTE_HOST=$compute_host"
  echo "COMPUTE_FQDN=$compute_fqdn"
  print_config
  "$py" --version
  "$py" -c "import sglang; print('sglang ok', sglang.__file__)"
  which nvcc
  "$CUDA_HOME/bin/nvcc" --version
  nvidia-smi

  rm -f "$READY_FILE"
  extra_args=()
  if [ -n "$MEM_FRACTION_STATIC" ]; then
    extra_args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
  fi
  if [ -n "$SGLANG_EXTRA_ARGS" ]; then
    # shellcheck disable=SC2206
    extra_args+=($SGLANG_EXTRA_ARGS)
  fi

  "$py" -m sglang.launch_server \
    --model-path "$MODEL_ID" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --context-length "$CONTEXT_LENGTH" \
    "${extra_args[@]}" &

  server_pid=$!
  echo "SGLang pid=$server_pid"

  deadline=$((SECONDS + 3600))
  while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "ERROR: SGLang server exited before readiness." >&2
      wait "$server_pid"
      exit $?
    fi
    if curl --noproxy "*" -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "http://$compute_host:$PORT/v1" > "$READY_FILE"
      echo "READY $(cat "$READY_FILE")"
      break
    fi
    if [ "$SECONDS" -gt "$deadline" ]; then
      echo "ERROR: server was not ready within 3600s" >&2
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" || true
      exit 1
    fi
    sleep 10
  done

  wait "$server_pid"
}

case "$MODE" in
  submit)
    submit_job
    ;;
  server)
    server_main
    ;;
  stop)
    stop_jobs
    ;;
  status)
    status_jobs
    ;;
  tail)
    tail_logs
    ;;
  dry-run)
    print_config
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
