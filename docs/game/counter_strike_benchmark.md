# Counter-Strike Static 3D Benchmark

This track evaluates only the frozen static 3D environment of an
instrumentable Three.js Counter-Strike-like arena. It does not evaluate
gameplay, weapons, controls, HUD, bot behaviour, player skill, or match
outcomes.

The canonical implementation lives in:

- `configs/game/game_mode_canonical_v1.yaml`
- `configs/game/counter_strike/benchmark_v1.yaml`
- `src/benchmark/game_scene/`
- `src/benchmark/game_scene/counter_strike/`
- `scripts/game/counter_strike/run_benchmark_v1.sh`

The checked-in pipeline captures the original Three.js runtime once, exports
canonical geometry and collision meshes, renders controlled views through the
original renderer, builds a trusted case bundle, and evaluates the result
through the canonical L0--L4 submission path.

## Active metrics

The canonical game profile evaluates:

- Collision
- Navigability
- Style Consistency

The Counter-Strike spatial-design extension additionally reports:

- Zone Clarity
- Route Structure
- Spawn Balance
- Landmark Legibility
- Cover Diversity

The primary result is the metric vector. A composite is emitted only when all
active metrics resolve; missing or failed results are never silently
renormalized away.

## Corpus runner

The local LiteLLM proxy and `LITELLM_MASTER_KEY` must already be available in
the terminal. The key is read from the environment and is never printed or
written to artifacts.

```bash
cd /path/to/Layout_DDD

nohup env \
  PHASE=all \
  MAX_WORKERS=2 \
  RESUME=1 \
  bash scripts/game/counter_strike/run_benchmark_v1.sh \
  > /private/tmp/layoutddd-cs-benchmark.log 2>&1 < /dev/null &

echo $! > /private/tmp/layoutddd-cs-benchmark.pid
tail -F /private/tmp/layoutddd-cs-benchmark.log
```

The default corpus and output roots remain under ignored `Support/` data and
artifact directories. Override `CORPUS_ROOT` or `RUN_ROOT` through environment
variables when needed.

## One-case runner

The Python runner can capture, judge, or aggregate one frozen case:

```bash
PYTHONPATH="$PWD/src:$PWD" \
.venv/bin/python -m benchmark.game_scene.counter_strike.runner --help
```

The general canonical game route is also exposed as:

```bash
layout-ddd-evaluate-game --help
```

## Failure semantics

- Unsupported runtimes or custom render pipelines are `not_ingestable`.
- Missing probe/capture artifacts are failures.
- Insufficient visual evidence and judge repeat disagreement are `unresolved`.
- Algorithm failures are `metric_failed`.
- A computed design failure may validly score zero.

The evaluator never converts missing evidence, failed model calls, or excluded
arms into valid metric values.
