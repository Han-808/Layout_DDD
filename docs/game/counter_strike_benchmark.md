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

The three perceptual spatial-design boundaries are frozen as follows:

- **Zone Clarity is permissive.** Open functional regions count when opposing
  spawns, departure bands, route convergence, and peripheral bypass geometry
  make them repeatably locatable. Enclosure, signage, and doorways are useful
  evidence, not prerequisites. The five required roles must nevertheless
  identify non-overlapping spatial regions; one patch cannot satisfy several
  roles.
- **Landmark Legibility is strict.** A counted landmark needs a salient,
  appearance-based identity visible at global-view scale. A generic repeated
  box does not become a new landmark because it is elsewhere, larger, or
  rotated.
- **Cover Diversity is strict.** Fragments and repeated instances are collapsed
  into visual archetypes. Translation, rotation, color, or modest scale
  variation of one generic primitive does not create another cover form.

VLM metrics use three exact-input repeats. The discrete verdict is a strict
majority and the continuous score is the median. Any insufficient-evidence
repeat remains unresolved; a single valid sample never overrides the others.
Each metric prompt exposes its frozen `valid_threshold` and a response example
that satisfies the same threshold and finding-count constraints enforced by
the validator. One malformed model-authored response may be retried once for
the same repeat; evidence, transport, and selector-contract failures are not
silently retried.

Collision keeps `invalid_pair_count / canonical_object_count` as its frozen
penalty. Exact coplanar zero-penetration contact is directly valid when complete
mesh-in-OBB guards hold. The CS profile also has a narrow deterministic
certificate for numeric near-contact: an at-most-2 mm plane with at most 1 mm
overlap is directly valid only when the other OBB remains on one side of the
plane centre. A plane slicing through an object therefore remains VLM-routed;
this is not a generic relaxed penetration threshold. Intentional decorative
attachment or partial assembly embedding is allowed only after metric-scoped
VLM adjudication; a relationship claim alone is not an exemption.

## Visual evidence repair

The browser capture remains the source of truth. The evaluator does not
round-trip a Three.js scene through Blender merely to obtain another view.

- Collision now receives two complementary frozen local camera angles. Each
  angle contributes one original-runtime RGB image (or a deterministic
  brightness-repaired display copy when dark) and one same-pose projected OBB
  overlay. Together with the two global frames, the final P0b packet stays
  within the six-image budget.
- Zone Clarity, Landmark Legibility, and Cover Diversity start from the two
  frozen global views. A dark global frame is deterministically replaced by its
  bounded brightness-repaired copy before judging; this choice is not delegated
  to the VLM. If the repaired global packet is still unusable or the final judge
  reports insufficient evidence, a separate selector may choose at most two
  additional camera angles from the frozen regional bank.
- Brightness repair changes display luminance only. The source view ID/hash,
  gain, and before/after luminance statistics are retained in evidence
  metadata. It does not modify geometry or fabricate a new camera.
- The selector cannot return a metric verdict. The final judge receives the
  fixed repaired packet and remains responsible for the metric result. If no
  helpful observation exists, the metric is `unresolved`.

The primary result is the metric vector. A composite is emitted only when all
active metrics resolve; missing or failed results are never silently
renormalized away.

The frozen corpus runner currently includes six source implementations:
Claude Opus 4.7, Claude Opus 4.8, Hy3, Kimi K3, MiniMax M3, and
Qwen3.8-Max-Preview. Each implementation has a checked-in source/hash/spawn
contract under `configs/game/counter_strike/corpus/`.

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
- Safe judge error codes are retained for diagnosis; raw provider responses
  and exception bodies are not written to reports.
- A computed design failure may validly score zero.

The evaluator never converts missing evidence, failed model calls, or excluded
arms into valid metric values.
