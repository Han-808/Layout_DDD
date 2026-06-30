# Experiment Config Notes

This file collects practical Hyak operating notes for running model servers and workflow jobs. Keep heavy compute off login nodes.

## Collaboration Boundary

GitHub repo:

```text
https://github.com/Han-808/ERL_repo
```

Branch used on Hyak:

```text
frozenlake_sokoban_eval
```

Local Windows workspace:

```text
C:\Users\32394\OneDrive\Desktop\3D benchmarking
```

Current local repo folder used by this project:

```text
C:\Users\32394\OneDrive\Desktop\3D benchmarking\3d_layout_benchmark
```

If `C:\Users\32394\OneDrive\Desktop\3D benchmarking` is not a git checkout, do not run `git add`, `git commit`, or `git push` there. Use a real checkout of the target GitHub repo, or upload a single file through the GitHub web UI when only one script needs to be added.

Hyak repo path:

```text
/gscratch/h2lab/mohanc3/projects/ERL_repo
```

Assumptions for this project:

- Codex should not assume it has SSH access to Hyak.
- The user handles GitHub staging, commits, pushes, pulls, and branch management.
- Codex may edit local repo files and provide exact commands to run on Hyak.
- The user pulls changes on Hyak, starts/stops Slurm jobs, and opens SSH tunnels.
- Do not rely on Codex to run `ssh`, `salloc`, `sbatch`, `scancel`, `git push`, or `git pull` for this workflow.

## Current Takeaway

The end-to-end smoke path is confirmed working:

```text
local Windows repo
  -> local pipeline script
  -> OpenAI-compatible model adapter
  -> localhost:8000
  -> SSH tunnel
  -> Hyak compute node SGLang server
  -> Qwen/Qwen3-VL-8B-Instruct or Qwen/Qwen3-VL-32B-Instruct
  -> local pipeline artifacts
  -> local web viewer
```

The successful server run used:

```text
job id: 36168924
compute node: g3083
ready URL: http://g3083:8000/v1
model: Qwen/Qwen3-VL-8B-Instruct
partition: gpu-a100
gpu: 1 x A100 80GB
context length: 8192
```

The result quality was not correct, but the infrastructure works: the local repo can call a real open-weights model on Hyak through the OpenAI-compatible endpoint, and the existing viewer can show the generated layout artifacts.

For the current Hyak checkout, the script was uploaded at repo root:

```text
/gscratch/h2lab/mohanc3/projects/ERL_repo/qwen3vl_server.sh
```

So the current Hyak commands are:

```bash
cd /gscratch/h2lab/mohanc3/projects/ERL_repo
bash qwen3vl_server.sh status
bash qwen3vl_server.sh tail
bash qwen3vl_server.sh stop
```

If the script is later moved into the intended path, use:

```bash
bash scripts/hyak/qwen3vl_server.sh status
bash scripts/hyak/qwen3vl_server.sh tail
bash scripts/hyak/qwen3vl_server.sh stop
```

To cancel the current server job immediately, prefer the script-managed stop command:

```bash
cd /gscratch/h2lab/mohanc3/projects/ERL_repo
bash qwen3vl_server.sh stop
```

Fallback if the script is unavailable:

```bash
scancel 36168924
```

## Access

Login:

```bash
ssh <netid>@klone.hyak.uw.edu
```

Common login nodes look like:

```text
klone-login01
klone-login02
klone-login03
```

Use login nodes for editing, git, packaging, monitoring, and submitting Slurm jobs only.

## Resource Checks

```bash
squeue -u <netid>
squeue -j <job_id>
sinfo -o "%.20P %.8a %.16F %.20P"
hyakalloc
scontrol show reservation
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,JobName%60,Partition,State,Elapsed,ExitCode,Reason%60
```

Common states/reasons:

- `RUNNING`: job is running.
- `PENDING`: job is queued.
- `Resources`: waiting for requested GPUs/CPUs/memory.
- `JobArrayTaskLimit`: array hit its concurrency cap, for example `%12`.
- `QOSGrpMemLimit`: account/group memory cap is currently reached.
- `ReqNodeNotAvail`: walltime may overlap maintenance or reserved nodes.
- `Dependency`: job waits for another job.
- `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`: inspect `.out`, `.err`, and `sacct`.

## Interactive GPU Allocation

Some Hyak Slurm setups do not support `salloc --pty`. Allocate first, then SSH to the assigned node:

```bash
salloc -A h2lab -p gpu-l40s --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=24:00:00
```

When Slurm prints a node, for example `Nodes g3115 are ready for job`, enter it:

```bash
ssh g3115
hostname -f
nvidia-smi
```

If the command running inside the allocated shell exits, the allocation may be relinquished. Avoid `set -e` in ad-hoc interactive paste blocks until the environment has been verified. A failed `source .../bin/activate` can terminate the shell and release the node.

Use `gpu-l40` instead of `gpu-l40s` if needed:

```bash
salloc -A h2lab -p gpu-l40 --gres=gpu:1 --cpus-per-task=8 --mem=80G --time=24:00:00
```

## Environment Setup

Load modules inside batch scripts, not only interactively:

```bash
module load cuda/<version> >/dev/null 2>&1 || true
module load gcc/<version> >/dev/null 2>&1 || true
```

Keep model caches and generated files off home:

```bash
export HF_HOME=/gscratch/<group>/<netid>/hf-cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export XDG_CACHE_HOME=/gscratch/<group>/<netid>/xdg-cache
export TORCH_EXTENSIONS_DIR=/gscratch/<group>/<netid>/torch-extensions
export WANDB_DIR=/gscratch/<group>/<netid>/wandb
```

Prefer absolute paths in Slurm jobs.

## Current Project Defaults

For the current 3D layout workflow experiments:

```text
login: mohanc3@klone.hyak.uw.edu
account: h2lab
preferred partitions: gpu-l40s, gpu-l40
repo on Hyak: /gscratch/h2lab/mohanc3/projects/ERL_repo
HF cache: /gscratch/h2lab/mohanc3/hf-cache
SGLang env: /gscratch/stf/mohanc3/.conda/envs/sglang311
Python in env: /gscratch/stf/mohanc3/.conda/envs/sglang311/bin/python
```

The SGLang env may not have a working `bin/activate` script. Use the environment's Python directly:

```bash
PY=/gscratch/stf/mohanc3/.conda/envs/sglang311/bin/python
$PY --version
$PY -c "import sglang; print('sglang ok', sglang.__file__)"
```

Do not assume `/mmfs1/gscratch/.../bin/activate` or `/gscratch/.../bin/activate` exists even if the conda env directory exists.

## Qwen3-VL SGLang Server

This is the current smoke-test path:

```text
small structured-relation JSON
  -> local workflow text prompt
  -> SSH tunnel to Hyak
  -> SGLang OpenAI-compatible server on one H200 for the default 32B profile
  -> Qwen3-VL layout JSON response
  -> local renderer creates global/group views
  -> same Qwen3-VL endpoint judges rendered PNGs
  -> local VLM-as-judge evaluation artifacts
```

Recommended path: submit a Slurm-backed server job from the repo root. This keeps the model server alive even if the SSH terminal disconnects.

```bash
bash qwen3vl_server.sh submit
bash qwen3vl_server.sh status
bash qwen3vl_server.sh tail
```

Default server config now targets the H200 32B profile:

```text
account: h2lab
partition: gpu-h200
gpu: 1
memory: 240G
time: 24:00:00
model_profile: 32b
model: Qwen/Qwen3-VL-32B-Instruct
context length: 65536
mem_fraction_static: 0.90
port: 8000
ready file: logs/ready/qwen3vl-server.url
```

Useful commands:

```bash
# Stop previous script-managed Qwen server jobs.
bash qwen3vl_server.sh stop

# Stop explicit old allocation/job ids if needed.
JOB_IDS="36168488 36168454" bash qwen3vl_server.sh stop

# Print the exact config without submitting.
bash qwen3vl_server.sh dry-run

# Submit the default 32B H200 server.
bash qwen3vl_server.sh submit

# Watch job and ready-file state.
bash qwen3vl_server.sh status
bash qwen3vl_server.sh tail
```

`tail` follows the current Slurm job for the selected profile. If that job is still pending and the `.out` file has not been created yet, it waits for the matching log instead of falling back to an older historical log. To force a specific job:

```bash
MODEL_PROFILE=32b JOB_ID=36252666 bash scripts/hyak/qwen3vl_server.sh tail
```

To smoke-test with 2B on L40S instead:

```bash
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
PARTITION=gpu-l40s \
MEMORY=80G \
bash qwen3vl_server.sh submit
```

To run the older 8B A100 profile instead:

```bash
MODEL_PROFILE=8b bash qwen3vl_server.sh submit
```

To reduce 32B context if 64K is too memory-heavy on one H200:

```bash
MODEL_PROFILE=32b CONTEXT_LENGTH=32768 MEM_FRACTION_STATIC=0.90 bash qwen3vl_server.sh submit
```

If SGLang loads all 32B shards and then fails at KV cache initialization with:

```text
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

the model weights loaded successfully, but the server did not have enough reserved static GPU memory for KV cache. Use the updated script default or launch explicitly with:

```bash
MODEL_PROFILE=32b MEM_FRACTION_STATIC=0.90 bash qwen3vl_server.sh submit
```

If that still fails on one H200, lower context:

```bash
MODEL_PROFILE=32b CONTEXT_LENGTH=32768 MEM_FRACTION_STATIC=0.90 bash qwen3vl_server.sh submit
```

First model load can spend time downloading weights to `HF_HOME`.

If SGLang warmup returns an HTML Squid error for `127.0.0.1:8000/model_info`, localhost requests are going through Hyak's proxy. Set `NO_PROXY/no_proxy` and unset `HTTP_PROXY`, `HTTPS_PROXY`, and `ALL_PROXY` before launching the server.

If FlashInfer fails with `/usr/local/cuda/bin/nvcc: No such file or directory`, set `CUDACXX`, `NVCC`, `CMAKE_CUDA_COMPILER`, `CUDA_HOME`, and `CUDA_PATH` to `/sw/cuda/12.4.1`. If a failed FlashInfer build left bad cached objects, remove that specific cache directory before retrying:

```bash
rm -rf /mmfs1/home/mohanc3/.cache/flashinfer/0.6.7.post3/89
```

After the ready file appears, check readiness on Hyak:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
curl --noproxy "*" http://127.0.0.1:8000/v1/models
```

The ready file contains the compute node URL:

```bash
cat logs/ready/qwen3vl-server.url
```

From local Windows, open the tunnel after the server is ready. Replace `<compute-node>` with the node printed in the log or ready URL:

```powershell
ssh -N -L 8000:<compute-node>:8000 mohanc3@klone.hyak.uw.edu
```

For example:

```powershell
ssh -N -L 8000:g3083:8000 mohanc3@klone.hyak.uw.edu
```

Then test locally:

```powershell
curl http://localhost:8000/v1/models
```

Run the local smoke workflow from the local repo:

```powershell
cd "C:\Users\32394\OneDrive\Desktop\3D benchmarking\3d_layout_benchmark"

py scripts/run_single_case.py `
  --experiment hssd_small_room_qwen3vl32b_local `
  --out outputs/hssd_small_room_vlm_judge_smoke `
  --serve `
  --port 8080
```

If the `python` command is not available on Windows, use `py` or the explicit Python 3.12 path.

The repo model config should point at:

```text
endpoint: http://localhost:8000/v1
model: Qwen/Qwen3-VL-8B-Instruct
```

For the default 32B Hyak server, use:

```text
endpoint: http://localhost:8000/v1
model: Qwen/Qwen3-VL-32B-Instruct
local model config key: qwen3vl_sglang_32b
```

The same endpoint is used twice in the local pipeline: first for layout generation from text/schema context, then for VLM-as-judge over rendered PNG evidence. The judge evidence package is one global top view plus each object group's `xy`, `yz`, and `xz` views.

The local generation prompt sends the full `bm_instance` JSON to the model, including HSSD raw metadata when present, while using a compact output contract instead of the full layout schema. With the current small full HSSD room:

```text
case: data/benchmark_cases/hssd_small_room_full/102344115_structured_basic.json
objects: 74
rough prompt size: about 20k tokens by chars/4
recommended server: qwen3vl_sglang_32b with 64K context
experiment model_overrides.max_tokens: 24000
```

The base 32B local server profile lives in `configs/model_config.yaml`; the HSSD case-specific output budget lives in `configs/experiment_config.yaml`. The case file remains full, but the model-facing prompt uses a compact adapter view with room, object IDs, categories, bbox sizes, floor positions, height positions, and source metadata summary. This keeps cases independent while avoiding raw HSSD metadata exhausting the 64K context window.

The larger `102343992` full scene is about 357 objects and roughly 97k prompt tokens by chars/4, so it should not be treated as a one-call 64K smoke case.

For temporary endpoint/model-id checks, use the dedicated server smoke script instead of adding API debug parameters to the main workflow:

```powershell
py scripts/check_model_endpoint.py `
  --model qwen3vl_sglang_32b `
  --model_endpoint http://localhost:8000/v1 `
  --model_id Qwen/Qwen3-VL-32B-Instruct `
  --timeout_seconds 300 `
  --response_format_json `
  --multimodal
```

If a parseable layout has schema, bbox, renderability, collision, or boundary problems, those issues are passed to Qwen3-VL as judge-facing flags. The local pipeline only skips VLM judging when the model output cannot be parsed into any layout scene.

## Model Server Pattern

For experiments that use an LLM/VLM server:

1. Start one persistent model server as a Slurm GPU job.
2. Make the server write its URL to a ready file when ready.
3. Start worker jobs only after the ready file exists.
4. Add a cleanup job that cancels the server after workers finish.

Ready-file wait logic:

```bash
READY_FILE="logs/ready/<run_tag>/server.url"
deadline=$((SECONDS + 3600))
while [ ! -s "$READY_FILE" ]; do
  if [ "$SECONDS" -gt "$deadline" ]; then
    echo "ERROR: server was not ready within 3600s" >&2
    exit 1
  fi
  sleep 10
done
SERVER_URL=$(cat "$READY_FILE")
```

If CPU workers cannot reach GPU-node servers, run workers on a GPU partition with no GPU request, for example `gpu-l40` with 1 CPU and 4G memory.

## Typical Slurm Commands

GPU server:

```bash
sbatch \
  --account=<account> \
  --partition=gpu-l40s \
  --gres=gpu:1 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=80G \
  --time=24:00:00 \
  --job-name="<run_tag>-server" \
  --output="logs/%x-%j.out" \
  --error="logs/%x-%j.err" \
  script.sh --server
```

Worker array:

```bash
sbatch \
  --account=<account> \
  --partition=gpu-l40 \
  --array=0-99%12 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=4G \
  --time=12:00:00 \
  --job-name="<run_tag>-workers" \
  --output="logs/%x-%A_%a.out" \
  --error="logs/%x-%A_%a.err" \
  script.sh --worker
```

Cleanup job:

```bash
sbatch \
  --account=<account> \
  --partition=gpu-l40 \
  --dependency=afterany:<worker_job_id> \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  --time=00:10:00 \
  --job-name="<run_tag>-cleanup" \
  --output="logs/%x-%j.out" \
  --error="logs/%x-%j.err" \
  --wrap="scancel <server_job_id> || true"
```

## Monitoring

```bash
watch -n 60 'squeue -u <netid>'
watch -n 60 'squeue -j <job_id_1>,<job_id_2> -o "%.18i %.10P %.45j %.8T %.10M %.8C %.10m %.30R"'
tail -n 100 logs/<job>.out
tail -n 100 logs/<job>.err
grep -H "ready\|ERROR\|Traceback\|OutOfMemory\|CUDA\|Killed\|Done" logs/<run_tag>*.out logs/<run_tag>*.err
```

GPU utilization from an allocated GPU job:

```bash
srun --jobid=<server_job_id> --overlap nvidia-smi \
  --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv
```

## Debug Checklist

1. Check queue: `squeue -u <netid>`
2. Check job reason: `scontrol show job <job_id>`
3. Check accounting: `sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode,Reason%60`
4. Check server logs for `ready`, `ERROR`, `Traceback`, `Killed`.
5. Check worker logs, especially array logs with `%A_%a`.
6. Check GPU utilization.
7. If workers timed out, verify ready file, server job status, server URL reachability, partition reachability, and timeout length.
8. If pending due to maintenance, shorten `--time` or wait.

## Data Handling

- Store large outputs on scratch/project storage.
- Do not store model caches in home.
- Do not commit generated results, logs, tarballs, model outputs, or caches.
- Package only needed files for download.
- Exclude huge traces unless needed.
- Use short output folder names before tar, especially when downloading to Windows.

Example packaging:

```bash
cd /gscratch/<group>/<netid>/<project>
OUT=results_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
rsync -a --exclude "*.tmp" --exclude "llm_calls_*.jsonl" <source_dir>/ "$OUT/"
tar -czf "${OUT}.tar.gz" "$OUT"
ls -lh "${OUT}.tar.gz"
echo "$PWD/${OUT}.tar.gz"
```

Download to Windows:

```powershell
cd C:\path\to\local\repo
$archive = "results_YYYYMMDD_HHMMSS.tar.gz"
New-Item -ItemType Directory -Force .\downloads | Out-Null
scp <netid>@klone.hyak.uw.edu:/path/on/hyak/$archive .\downloads\
tar -xzf ".\downloads\$archive" -C ".\downloads"
```

## Git Rules

- Keep one source of truth: local edit/push, then Hyak `git pull`, or edit on Hyak intentionally.
- Use `git status --short` before committing.
- Add specific files, not `git add .`.
- Do not commit logs, results, tarballs, caches, or large model outputs.
- Run new submit scripts with `--dry-run` when available.

## Operational Rules

- Always dry-run new submit scripts.
- Always print job IDs after submit.
- Always print monitor commands after submit.
- Always use cleanup jobs for persistent servers.
- Always inspect `.err` files when jobs fail or exit silently.
- Prefer conservative walltimes near maintenance windows.
- Use explicit array concurrency caps, for example `--array=0-35%12`.
- Keep monitors compact and readable.
