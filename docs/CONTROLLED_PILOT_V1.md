# Controlled FrozenAssets Pilot v1

## Purpose

This pilot asks how generation workflows differ when the room, logical object
slots, exact asset IDs, asset meshes, local bbox, and native scale are frozen.
It is a five-case integration pilot, not a statistically powered benchmark.

The checked-in case definition is
`configs/generation_comparison/controlled_pilot_v1.json`. It covers bedroom,
living-room, dining-room, and office layouts with 4--6 objects, varied room
aspect ratios, and varied density. Each prepared case records object count,
room area, objects/m2, and `C(n, 2)` pairwise interaction proxy.

## Asset source and preflight

The pilot uses a 12-asset frozen subset of the local Imaginarium catalog. Runtime
preparation requires `--asset-root`; no absolute asset paths or meshes are
committed. Preparation checks exact CSV category/description, nonempty FBX
existence, positive transformed local bbox, finite bbox center, native scale
`[1, 1, 1]`, nonzero canonical front when supplied, and mesh/metadata SHA-256.
When a spec pins the source CSV/FBX/metadata hashes, preparation also compares
the live bytes to those pins and fails before generation on any drift.

Unavailable canonical fronts are recorded as `unavailable_not_invented`; the
pilot does not synthesize them.

## Commands

Prepare immutable cases and asset reports without calling a model:

```bash
layout-ddd-controlled-pilot prepare \
  --spec configs/generation_comparison/controlled_pilot_v1.json \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --out-dir outputs/controlled_pilot/frozen_assets_rect_furniture_pilot_v1 \
  --method-configs /PATH/TO/private_method_configs.json
```

Run the mandatory one-case dry run first:

```bash
layout-ddd-controlled-pilot run \
  --prepared-dir outputs/controlled_pilot/frozen_assets_rect_furniture_pilot_v1 \
  --method-configs /PATH/TO/private_method_configs.json \
  --dry-run-only
```

Use a fresh prepared output directory for the full pilot after dry-run success;
the runner refuses to overwrite any previous result or failed method/case
directory. A credential-free configuration template is provided at
`configs/generation_comparison/controlled_pilot_methods.example.json`.

This file is an execution-contract template, not a set of supplied upstream
bridges. In particular, its `integration_entrypoint.py` commands require operator
code that is not shipped here or by those upstream repositories. The example
`comparison_support` flags are declarations to implement and verify, not evidence
that an unmodified upstream method honors them. See the per-method gaps and
real-upstream acceptance gate in
[`COMPATIBILITY_STATUS.md`](../COMPATIBILITY_STATUS.md).

## Eligibility versus execution readiness

The pilot independently records semantic eligibility and execution readiness.
The former asks whether a method can faithfully accept FrozenAssets. The latter
requires its checkout, command, endpoint, and credential environment to exist.

`catalog_placement` is now a first-class FrozenAssets method. Its native prompt
receives the public comparison contract, must output `uniform_scale=1.0`, and
retains the true model response, validated native placement, request metadata,
and hashes. Reusing one exact asset across slots is supported without making the
frozen asset record slot-dependent.

The model-facing message contains logical asset IDs, descriptions, bbox, and
protocol hashes but strips host-local mesh/metadata/cache paths. Full locators
remain available only to the local converter, renderer, and audit manifest.

LayoutVLM is semantically eligible through its native scene asset table. Other
external methods remain fail-closed unless configured thin runners attest and
then demonstrate exact inventory, ID, and scale controls. SceneWeaver also needs
immutable bindings and inventory across every iteration.

## Evaluation and reporting

All valid runs call unchanged `benchmark.api.evaluation.run_evaluate`. The
prepared evaluator policy SHA-256 is copied into every case protocol and result.
Generator-specific evaluator options are not accepted.

Outputs include `protocol.json`, `catalog_manifest.json`,
`asset_preflight.json`, `evaluator_config.json`, `compatibility_report.json`,
per-case manifests/runs, `results.jsonl`, `results.csv`, `summary.json`, and a
runtime README. Failures distinguish method failures from infrastructure
failures and are never discarded or overwritten.

Each case has a requested seed, but it is labeled
`not_guaranteed_unless_runner_reports`; stochastic APIs are not described as
deterministic merely because the benchmark recorded a seed value.

SceneWeaver summaries derive initial/final score, delta, score regressions,
first success iteration, hard-failure fixes, and hard-failure regressions only
from existing iteration reports. Official evaluator feedback is never sent to
the native loop.

Without the required upstream checkouts/endpoints/evidence services,
preparation remains reproducible but no method-quality score is claimed.
Offline artifacts are labeled `offline_artifact`, never `real_generation`.

## S100--S109 Frozen Imaginarium track

The larger harness-comparison track is documented separately in
[`FROZEN_IMAGINARIUM_SCENE10.md`](FROZEN_IMAGINARIUM_SCENE10.md). It reuses the
existing public S100--S109 briefs together with ten hash-pinned, human-selected
SceneBoard baseline inventories. Exact removals, replacements, additions, and
support parents are materialized as 269 slots over 168 immutable Imaginarium
assets. It supports the Catalog Placement Stage-C baseline and ships thin
controlled bridges for LayoutGPT, DirectLayout, LayoutVLM, and conditional
SceneWeaver integration. The user approved the complete 269-slot asset contract
on 2026-09-05. Real generation still requires each method's independent runner,
execution-identity, credentials, and method-specific input prerequisites.
