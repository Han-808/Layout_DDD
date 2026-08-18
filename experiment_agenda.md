# Experiment Agenda

## Current Stage

The repository is in the final **method-validation and configuration-freeze**
stage, not yet the formal full-benchmark stage.

The current Functional workflow has recently gained typed per-check judging and
optional group-level evidence sharing. Historical runs were produced by
different code revisions, prompts, schemas, budgets, models, and API health
conditions, so they are useful for failure discovery but are not directly
comparable experimental arms. Before reporting benchmark scores, the latest
implementation must first demonstrate terminal completeness and repeatability.

The immediate sequence is:

```text
freeze implementation
-> latest-version acceptance pilot
-> full-pipeline stability
-> isolate Judge variance and establish human agreement
-> core Functional and evidence-acquisition ablations
-> freeze production configuration
-> formal full-dataset scoring
```

## Experimental Priorities

| Priority | Experiment | Scope | Code change |
|---|---|---|---|
| P0 | Latest-version acceptance pilot | 8 representative scenes, 1 run | None |
| P1 | Full-pipeline stability | Same scenes, 5 fresh repeats | Small analyzer extension if needed |
| P1 | Human/check-level annotation agreement | Same sentinel subset, second annotator | Small to medium data/UI work |
| P2 | Frozen-evidence Judge stability | Replay identical Judge requests 5 times | Medium |
| P3 | Functional scheduling F0/F1/F2 | 8-12 scenes, 3 arms, ideally 3 repeats | Arms exist; fair upstream freezing needs plumbing |
| P4 | Evidence and camera causal ablation | Fixed, deterministic, VLM, cascade, random, oracle | Small to medium |
| P5 | Formal N001-N030, then larger-set scoring | Frozen production configuration | None for evaluation |
| P6 | Grouping, Judge-model, router, and scoring studies | Secondary explanatory experiments | Experiment-dependent |

## P0: Latest-Version Acceptance Pilot

Use the current production default:

```text
group-local granularity: per_check
group-local evidence policy: isolated_episode
active evidence window: 6 images
```

Recommended sentinel scenes:

| Scene | Coverage purpose |
|---|---|
| N002 | Within-group chair/table correspondence and Placement |
| N007 | Functional-valid case for false-positive auditing |
| N011 | TV-sofa cross-group relation and directed bookshelf |
| N016 | Clearance and Placement behavior |
| N020 | Dense groups and required-check coverage |
| N021 | Difficult cross-group relation and prior false-positive pattern |
| N022 | Chair-table relation and refrigerator usability |
| N024 | Bathroom clearance and Placement |

Run all canonical metrics once so that workflow integration, metric ownership,
and total cost remain visible. This is an engineering acceptance run, not a
paper result.

Acceptance gates:

- zero infrastructure failures;
- zero unresolved metric/check outcomes;
- 100% required-check coverage;
- no group-wide loss caused by schema or response-repair failure;
- complete Functional ownership and Placement check ledgers;
- complete discovery -> evidence -> check -> decision audit chain;
- UI reconstruction shows every required check, evidence episode, and final
  decision.

If these gates fail, fix the workflow before starting repeated experiments.

## P1: Full-Pipeline Stability

After the acceptance pilot passes, run the same sentinel scenes five times with
fresh output roots. Freeze the following across repeats:

- source commit and dirty-worktree fingerprint;
- dataset fingerprint;
- model and endpoint route;
- grouping and evaluation configs;
- camera and evidence budgets;
- worker count;
- prompts and schemas;
- no resume from previous outputs.

Report at least:

- scene-level metric-verdict agreement;
- typed-check completion and decision agreement;
- affected/scoring-object agreement;
- defect-set Jaccard similarity;
- grouping partition and discovery-candidate stability;
- evidence-selection and stage-transition stability;
- forced-choice rate;
- API/render/schema failure rate;
- calls, renders, tokens, and latency.

Infrastructure failures must remain separate from scientific outcomes. If the
full pipeline is unstable, diagnose modules in this order:

```text
Final Judge
-> Grouping
-> Affordance Discovery
-> Relation Discovery
-> Usable-Side Detection
-> Camera Selection
```

Do not continue to expensive method ablations until the principal source of
variance is identified.

## Annotation Requirement

The current scene annotations provide metric labels and affected object IDs,
but they are not sufficient to evaluate all typed-check claims. Create a small
human-adjudicated check-level gold set for the sentinel scenes with:

```text
check_type
subject/target object IDs
relation pair when applicable
expected valid/invalid result
metric ownership
```

A second annotator should independently label this subset. Report binary and
severity agreement, affected-object agreement, relation/check agreement, and
Functional-versus-Placement ownership agreement. Without this subset,
F0/F1/F2 can be compared for cost and stability, but not rigorously for
check-level precision and recall.

## P2: Frozen-Evidence Judge Stability

Replay exactly the same immutable Judge requests and image hashes five times,
without rerunning grouping, discovery, camera selection, or rendering.

This separates final-model randomness from upstream evidence-selection
randomness. A reusable replay runner should:

- read completed audit records;
- verify request and image hashes;
- call only the selected Judge model;
- preserve raw and normalized responses;
- support repeated calls and model substitution.

If the Judge is unstable under identical input, small downstream differences
between camera or scheduling arms cannot be interpreted reliably.

## P3: Functional Scheduling Ablation

The current runner supports three legal group-local arms:

| Arm | Granularity | Evidence policy | Hypothesis |
|---|---|---|---|
| F0 | `batched` | `isolated_episode` | Lowest call count, but possible attention dilution and cross-check interference |
| F1 | `per_check` | `isolated_episode` | Cleanest attribution, potentially highest cost |
| F2 | `per_check` | `shared_group_bank` | Preserve atomic decisions while reducing repeated evidence acquisition |

`batched + shared_group_bank` is invalid by design.

For a fair comparison, all arms must reuse the same grouping, Functional
discovery, usable-side outputs, and proactive evidence. Otherwise upstream VLM
randomness is confounded with scheduling policy. The preferred experiment is
8-12 scenes x 3 arms x 3 repeats after the stability gate passes.

Primary outcomes:

- check-level precision/recall on the adjudicated subset;
- required-check completion;
- target attribution accuracy;
- duplicate-finding rate;
- valid-scene false-positive rate;
- Judge and CameraSelector calls;
- new renders and reused evidence;
- tokens, latency, and total cost.

Do not treat `Global only` as a fourth arm yet. Stage removal is not currently a
canonical config and requires an explicit ablation-only coverage contract so
that omitted stages are recorded as `ablated_by_experiment`, not as missing or
unresolved work.

## P4: Evidence-Acquisition Causal Ablation

After scheduling is stable, compare matched-budget evidence policies:

```text
fixed evidence
deterministic selection
VLM selection
deterministic -> VLM cascade
seeded random selection from the same candidate bank
human oracle selection from the same candidate bank
```

All arms must share the same initial camera, candidate bank, image count,
rounds, and action/selector budgets. The seeded random arm distinguishes active
selection from the benefit of merely receiving more images.

The oracle arm is central to the paper claim because it decomposes failures:

```text
oracle evidence still fails
-> Judge reasoning limitation

oracle succeeds but candidate bank lacks the view
-> candidate-generation limitation

candidate bank contains a successful view but selector misses it
-> camera-selection limitation

correct selected evidence still produces the wrong decision
-> Judge grounding/reasoning limitation
```

Prefer candidate-bank oracle selection before implementing expensive freeform
Blender-camera annotation.

## Formal Scoring

Run formal N001-N030 scoring only after:

- acceptance and stability gates pass;
- Frozen-Judge variance is understood;
- annotation agreement is known;
- the production Functional and camera policies are frozen;
- unresolved and infrastructure-failure handling is verified.

Then expand to the larger scene set. Always report raw check-level findings in
addition to aggregate scores:

- per-metric accuracy;
- valid-scene false positives;
- invalid-scene recall;
- object/target attribution;
- check coverage;
- forced-choice rate;
- infrastructure failures;
- API/render/token/latency cost.

Scoring-profile sensitivity should be computed post hoc from the same raw
findings; it must not rerun discovery, cameras, or the Judge.

## Deferred Experiments

Defer the following until the core evidence workflow is stable:

- full factorial combinations of every policy;
- Functional stage-removal profiles;
- full-stack model replacement as a proxy for Judge quality;
- large-scale grouping ablations;
- broad budget sweeps;
- formal full-dataset scoring before configuration freeze.

A direct environment-level model swap changes grouping, discovery,
usable-surface decoding, camera selection, and final judging simultaneously.
Judge-model comparison should therefore use frozen Judge-request replay rather
than full-stack replacement.

## Immediate Decision

The next concrete experiment is:

```text
freeze the current implementation
-> run the 8-scene acceptance pilot with F1
-> audit terminal completeness
-> if clean, run five full-pipeline repeats
```

Only after that should F0/F1/F2 and camera-policy ablations begin.
