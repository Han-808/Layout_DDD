# Thirty-scene blind grouping experiment (historical)

> Deprecated comparison fixture. The canonical evaluator and its launchers use
> VLM grouping only. This three-backend experiment remains available solely for
> historical replay; set `ALLOW_DEPRECATED_GROUPING_COMPARISON=1` explicitly
> when re-running its grouping phase.

The grouping decision is settled: do not use this fixture to select the active
backend. For the current VLM-only run, use
`configs/experiments/grouping_vlm20_visual_scope_v2.yaml` and
`scripts/run_grouping_vlm20.py`.

This experiment compares the three grouping implementations through one
shared interface:

- topology grouping;
- anchor-object grouping;
- VLM grouping through the local GPT-5.6-Sol LiteLLM endpoint.

Grouping is evaluated as an evidence-scope partition, not as a benchmark
metric or scene-validity verdict.

## Frozen dataset

The checked-in sample is
`configs/experiments/grouping_blind30_sample_v1.yaml`.

- Seed: `20260730`.
- Sampling policy: random within five object-count strata, with randomized
  scene-type round-robin selection inside each stratum.
- Six scenes are sampled from each of `5–10`, `11–20`, `21–30`, `31–45`,
  and `46+` renderable-object strata.
- The final sample contains 30 scenes, 19 scene types, and 5–59 objects per
  scene.
- Every source file is pinned by SHA-256. A changed or missing source fails
  closed.
- Dataset fingerprint:
  `0d0c191932d9dda2344a7d0b66c2b58413bb2e03fd3822db8112124f0d39fac3`.

Converted scenes do not contain stable object IDs. Preparation derives
`scene_object_NNN` from immutable source order, records the mapping, and
materializes a read-only render scene. The source scene is never modified.

## Shared inputs

All three grouping implementations receive the same:

- materialized scene and object catalog;
- original Workbench perspective render;
- original Workbench top render;
- neutral top-down object identity map;
- grouping contract and evidence-scope goal.

The identity map links visible `O001`, `O002`, … aliases to exact object IDs.
It is an observation aid, not a grouping result.

## Blindness

For every scene, the three implementations are independently shuffled into
Result A, B, and C. The mapping is stored only in:

```text
<output_root>/private/method_key.json
```

The public review payload and UI exclude backend, policy, model, endpoint,
anchor, provenance, and free-text implementation rationale. Do not inspect
the private key until the blind review is complete.

## Run

The primary config is:

```text
configs/experiments/grouping_blind30_gpt56_v1.yaml
```

After the local LiteLLM proxy and `LITELLM_MASTER_KEY` are available in the
same Terminal session:

```bash
cd /Users/han_mohan/Desktop/Layout_DDD

RUN_ROOT="$PWD/Support/artifacts/outputs/grouping_blind30_gpt56_20260730_r1"
mkdir -p "$RUN_ROOT"

nohup env \
  JUDGE_ENDPOINT=http://127.0.0.1:4010/v1 \
  JUDGE_MODEL=gpt-5.6-sol \
  JUDGE_API_KEY_ENV=LITELLM_MASTER_KEY \
  RESUME=1 \
  PHASE=all \
  Support/bash/local/run_grouping_blind30_gpt56.sh \
  >"$RUN_ROOT/launcher.log" 2>&1 < /dev/null &

echo $! >"$RUN_ROOT/run.pid"
echo "PID=$(cat "$RUN_ROOT/run.pid")"
echo "LOG=$RUN_ROOT/launcher.log"
```

The historical launcher:

1. verifies or recreates the frozen materialized inputs;
2. renders the shared scene evidence;
3. verifies the existing port-4010 listener is LiteLLM;
4. runs a real text and multimodal GPT-5.6-Sol preflight;
5. runs topology, anchor, and VLM grouping only for explicit historical
   comparison;
6. validates complete, non-overlapping object partitions;
7. builds the blind review UI.

The launcher never starts a duplicate proxy and never receives a credential
value through a CLI argument. `RESUME=1` reuses only artifacts whose input
fingerprints still match.

Monitor with:

```bash
tail -F "$RUN_ROOT/launcher.log"
```

## Review

After all 90 results are complete:

```bash
cd /Users/han_mohan/Desktop/Layout_DDD

.venv/bin/python scripts/serve_grouping_blind30_review.py \
  --output-root \
  Support/artifacts/outputs/grouping_blind30_gpt56_20260730_r1
```

Open:

```text
http://127.0.0.1:8765/index.html
```

The UI shows the three shared input images followed by anonymous A/B/C
partition overlays. It supports:

- per-result `correct`, `partially correct`, `incorrect`, or `unclear`;
- scene-level best-result selection, including tie and unclear;
- result-specific and scene-level notes;
- reviewed status and progress filters;
- JSON and TSV export.

When served locally, reviews are saved atomically to:

```text
<output_root>/human_reviews/blind_reviews.json
```

Browser local storage remains a fallback, but the local review server is the
authoritative reusable record.

## Scoring after the blind review

Do not run the scorer until the blind review is complete. It reads the private
method key only after the human labels are finished, then maps each result's
quality label to a numeric value:

- `correct` = `1.0`;
- `partially_correct` = `0.5`;
- `incorrect` = `0.0`.

`unclear` is reported as unscored rather than silently treated as incorrect.
The normalized score is `score_sum / scored_label_count`; coverage and
unscored counts are recorded separately. The scorer reports overall, backend,
and three object-count groups: `<11`, `11–30`, and `>30` objects.

```bash
cd /Users/han_mohan/Desktop/Layout_DDD

.venv/bin/python scripts/score_grouping_blind30_review.py \
  --output-root \
  Support/artifacts/outputs/grouping_blind30_gpt56_20260730_r1
```

The command requires all 30 scenes and all three result labels per scene to be
present and marked reviewed. For a provisional progress check, add
`--allow-incomplete`; the output is marked `complete: false` and must not be
used as the final experiment score.

The unblinded score file is written to:

```text
<output_root>/human_reviews/grouping_scores.json
```

## Output layout

```text
<output_root>/
  dataset_manifest.json
  experiment_manifest.json
  model_preflight.json
  run_summary.json
  cases/
    case_NNN/
      input/
      render/
      grouping/
  private/
    method_key.json
  blind_review/
    index.html
    review_data.json
    assets/
  human_reviews/
    blind_reviews.json
    grouping_scores.json
```

Failures remain explicit `failure.json` records. They are never converted to
a singleton partition, a valid result, or an empty score.
