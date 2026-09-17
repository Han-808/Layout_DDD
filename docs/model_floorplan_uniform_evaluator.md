# Uniform Model floorplan evaluator

Implemented on `codex/uniform-model-floorplan-evaluator`, 2026-09-17.
This is a new Model-only execution protocol, not a relabelling of historical
scores or a leaderboard publication. Live evaluation and performance acceptance
are separate steps owned by Eval_Final after the frozen handoff.

## Scope and ancestry

Model Open-space and rectangular Multi-room use one runner, one source tree,
one Judge/configuration and one evidence/score acceptance contract. A rectangular
room is already a native prepared scene case; `--mode` is provenance only and
never selects metric code, weights, fallback, or budgets. No cross-room metric is
introduced. Pipeline, Agent and nonrect runs are not migrated or restarted.

The inherited evaluation implementation is the R15 Open-space candidate from
`missing67_r15_acquisition_terminal_v2/release_manifest.json`:

- Parent manifest SHA256: `b1aad3d6cda533977ee251093f26e92fe9b9494d791bdc5a6e77ae8ac8d65de4`.
- Parent Open-space tree SHA256: `4bc2a4c7f3933c4eee9196e29e6cf98cf705ba8994bce9a650c004c9290b53d3`.
- The 88-case September campaign originally bound the September 9 evaluator;
  “88 cases” is a cohort, not proof of identical evaluator source.
- The branch began at `7e03e7de5df0560aa289fc42313464c3dc8018d5`. R15 evaluator,
  evidence acquisition, rendering and model dependencies were imported with
  pinned-file verification. Unrelated generation, materialization, browser and
  campaign launcher changes were removed before handoff.

Historical frozen checkouts, baseline registry records, reports and published
website scores remain unchanged. The only proposed **new Model execution path**
is `scripts/run_uniform_model_evaluation.py`; historical compatibility APIs
remain for the out-of-scope consumers. Do not launch new Model comparisons with
the old per-experiment runners or combine them as if source hashes matched.

## Fixed execution configuration

`configs/runners/model_floorplan_unified_v1.json` is the operator protocol.

- Core: `benchmark.api.evaluation.run_evaluate`.
- Judge: `gpt-5.6-sol`, 8192 output tokens, 120000 context characters.
- Evidence: `evidence_consistency_fallback_v2`.
- Terminal decision: `best_available_final_v1`, an actual model decision; not
  program-default-valid. Persistent response/transport/input faults stay faults.
- Canonical C4 / Support12 / OOB2 / L3v8 / prompt31 implementations inherited.
- Existing L1/L3 layer weights, five L3 weights, canonical N, burden coefficients,
  deduction multiplier 2, and typed/residual Placement scoring remain unchanged.
- Existing acquisition budgets and 768×768 / 256×256 render/preview settings.

The CLI admits v2 in the common orchestrator. Judge policy is injected through
the case dependency graph, including threaded execution; no process-global
Atelier monkeypatch is needed. Launching a fresh Python process uses the same
script/source binding. Mode, model-under-test and attempt number cannot select a
different policy through this runner.

## Large scene context repair

Atelier's 173-object context retained 19 groups whose routing audit alone was
about 131k characters. Full member IDs and structured scene facts were not the
reason this audit needed to accompany every judgement.

`judge_group_context_v1` omits **only** `formation_edges` and `edge_reasons` from
judge-facing group objects, recording the original group's SHA256. All other
fields, unknown future fields, exact membership, objects, geometry, ownership,
required checks and global inventory stay intact. The original grouping artifact
is never mutated. Projection is applied at request construction and outbound
serialization, so a caller cannot accidentally reinsert the full audit.

This is not arbitrary truncation or a promise every scene fits 120k characters.
Truly oversized mandatory facts still produce an explicit capacity failure.

## Judgement coverage and score contract

The uniform runner adds `model_judgement_coverage_v1` to **new** reports only.
Existing raw metric statuses, verdicts and scores remain available for audit.

For each metric, `c_m` is accepted judgement obligations / planned obligations.
The original plan includes missing obligations and downstream typed obligations.
Accepted best-available model inference counts even if visual grounding is low;
legitimate metric-owned deterministic certificates count separately. Program
defaults do not count as judgements.

Partial burden is reconstructed only from accepted L1 rows or a metric-owned,
post-exemption/ownership/deduplication L3 observation ledger. No raw-response
scan, detector-to-verdict conversion, unseen-valid certificate, altered canonical
N, or second multiplication by coverage is permitted. If a partial ledger cannot
be verified, projection fails explicitly instead of manufacturing a score.

`S = sum(w_m * c_m * s_m) / sum(w_m * c_m)`

`C = sum(w_m * c_m) / sum(w_m)`

The scene is score-eligible at `C >= 0.80`, with a verified observed score and no
infrastructure-failure metric. There is no per-metric 80% gate and no 8/8 gate.
Unknowns are omitted from S but retained in C's planned denominator. A conditional
observed score does not assert validity for unjudged objects or scopes.

Consumption rules:

- `evaluation_report.json`: `judgement_coverage_summary`, projected
  `benchmark_score`, `benchmark_score_100`, `benchmark_score_status` and
  `pre_uniform_score` for the predecessor aggregation values.
- Each metric view: `judgement_coverage_projection`, including planned/accepted
  counts, separate visual fraction, source report hash and observed burden ledger.
- Case manifest/outcome: `uniform_score_acceptance` is the new eligibility signal.
  Raw `final_decision_status` still describes all-scope resolution; do not require
  it to say `resolved` when the uniform score is legitimately partial.
- `case_scoring_summary` recognizes the projection and uses the same C/S contract.
  Historical callers/reports without this field keep the predecessor behavior.

This is evaluator scoring, **not** a change to the website's reporting transform,
weights, category presentation or cross-room/model aggregation. Publication must
explicitly consume the new eligibility/observed ledger after result acceptance.

## Input and release safety

The default invocation is read-only: native discovery, canonical/mesh ownership,
file/mesh hashes, original geometry limitations and source identity. It starts
neither a proxy nor a model call/render. Incomplete original geometry remains an
explicit limitation; it is not silently rebuilt or relabelled complete.

Live `--run` requires:

1. A sealed release manifest whose exact file inventory, tree hash and protocol
   match the imported source. Mixed checkout imports and source escape fail.
2. A reviewed input manifest produced by that sealed runner's read-only check.
   Exact case population, input/mesh hashes, mode and release identity must match.
3. A fresh output directory separate from input and source, fixed Judge binding,
   and an available Blender binary. There is no overwrite or automatic resume.

Credentials stay in the existing private environment; none belong in a manifest.
The runner does not create proxies, select queues, regenerate geometry, publish,
or promote the baseline registry. The sealed source-checkout runner is the
supported operator entrypoint; the wheel alone is not a portable experiment.

Operator command shape (replace placeholders; **not run during implementation**):

```bash
python -B RELEASE/scripts/run_uniform_model_evaluation.py \
  --mode open-space --dataset-root DATASET --case-id CASE \
  --release-manifest RELEASE/release_manifest.json

# Save the check JSON as INPUT_RECEIPT and review it before launch.
python -B RELEASE/scripts/run_uniform_model_evaluation.py \
  --mode open-space --dataset-root DATASET --case-id CASE \
  --release-manifest RELEASE/release_manifest.json \
  --input-manifest INPUT_RECEIPT --output-root NEW_OUTPUT --run
```

For Multi-room change only mode and input selection. Recheck a new receipt for
each selection. Supply no additional metric/policy/weight flags.

## Verification and remaining gates

The branch includes regressions for exact projection retention, real oversized
173-object Atelier contexts, public Style/global Placement valid and invalid
reports, whole-scene coverage boundaries, partial observed invalid burden,
unknown/default/failure separation, persisted report consumption, source/input
drift rejection, and actual native Open-space/Multi-room CLI → case → Judge
construction with socket/process guards and a stubbed scoring boundary.

Final command/counts, hashes and exclusions are recorded in the separate offline
acceptance artifact at release sealing. Default-suite results are not substituted
for targeted acceptance: the broader repository also has historical frozen-trace
expectation mismatches, missing local datasets, generation compatibility pins,
and environment-dependent Blender/loopback/process restrictions. Do not claim a
clean default suite or live performance validation from the targeted tests.

Eval_Final must complete the audited rerun population and prepared-input
readiness before starting. The requested population is Model Open-space100 and
Multi-room217, not automatically 317 ready cases; the current 216 scoreable
Multi-room rows are not authority to discard the 217th planned case. HY3/Opus4.8
legacy cohorts are excluded, and no Pipeline/Agent rerun is part of this change.
The latest status from Eval_Final says this readiness work is still pending.

Before broad dispatch, run and accept one real Open-space and one real Multi-room
case with the same frozen release; preserve genuine gaps/faults. Only verified
actual use permits the coordinated baseline-registry update. No old result is
retroactively relabelled, and neither rank order nor score equivalence is promised.
