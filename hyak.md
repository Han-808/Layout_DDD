# Hyak Experiment Notes

This file collects practical Hyak operating notes for running model servers and workflow jobs. Keep heavy compute off login nodes.

## Collaboration Boundary

GitHub repo:

```text
https://github.com/Han-808/Layout_DDD
```

Local Windows workspace:

```text
C:\Users\32394\OneDrive\Desktop\3D benchmarking
```

Current local repo folder used by this project:

```text
C:\Users\32394\OneDrive\Desktop\3D benchmarking\3d_layout_benchmark
```

If `C:\Users\32394\OneDrive\Desktop\3D benchmarking` is not a git checkout, clone `Han-808/Layout_DDD` into a sibling checkout directory, sync the project contents into that checkout, then commit/push from the checkout. Do not run `git add`, `git commit`, or `git push` from a folder that has no `.git`.

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

## Qwen3-VL SGLang Text Server

This is the current smoke-test path:

```text
HSSD bm_instance JSON
  -> local workflow text prompt
  -> SSH tunnel to Hyak
  -> SGLang OpenAI-compatible server on one L40/L40S
  -> Qwen3-VL text response
  -> local layout pipeline artifacts
```

Recommended path: submit a Slurm-backed server job from the repo root. This keeps the model server alive even if the SSH terminal disconnects.

```bash
bash scripts/hyak/qwen3vl_server.sh submit
bash scripts/hyak/qwen3vl_server.sh status
bash scripts/hyak/qwen3vl_server.sh tail
```

Default server config:

```text
account: h2lab
partition: gpu-a100
gpu: 1
memory: 120G
time: 24:00:00
model: Qwen/Qwen3-VL-8B-Instruct
port: 8000
ready file: logs/ready/qwen3vl-server.url
```

Useful commands:

```bash
# Stop previous script-managed Qwen server jobs.
bash scripts/hyak/qwen3vl_server.sh stop

# Stop explicit old allocation/job ids if needed.
JOB_IDS="36168488 36168454" bash scripts/hyak/qwen3vl_server.sh stop

# Print the exact config without submitting.
bash scripts/hyak/qwen3vl_server.sh dry-run

# Submit the 8B A100 server.
bash scripts/hyak/qwen3vl_server.sh submit

# Watch job and ready-file state.
bash scripts/hyak/qwen3vl_server.sh status
bash scripts/hyak/qwen3vl_server.sh tail
```

To smoke-test with 2B on L40S instead:

```bash
MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen3-VL-2B-Instruct \
PARTITION=gpu-l40s \
MEMORY=80G \
bash scripts/hyak/qwen3vl_server.sh submit
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
ssh -N -L 8000:g3113:8000 mohanc3@klone.hyak.uw.edu
```

Then test locally:

```powershell
curl http://localhost:8000/v1/models
```

The repo model config should point at:

```text
endpoint: http://localhost:8000/v1
model: Qwen/Qwen3-VL-8B-Instruct
```

This path is currently text-only. It does not yet send rendered PNGs to Qwen3-VL as multimodal inputs.

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
