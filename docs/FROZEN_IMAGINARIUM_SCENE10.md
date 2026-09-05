# Frozen Imaginarium Scene10 harness track

## Purpose

This track compares placement workflows while holding the main confounders
constant:

```text
same S100--S109 rectangular rooms
  + same public object plans and relations
  + same expanded object slots
  + same exact Imaginarium assets and physical dimensions
  + same backing model for the four scene-generation harnesses
  -> method-native pose generation
  -> preserved native artifact
  -> existing strict converter
  -> the same canonical evaluator
```

The four same-model **scene-generation harnesses** are `layout_gpt`,
`direct_layout`, `layout_vlm`, and `scene_weaver`. In this document, “harness”
never means a coding agent such as Codex or Claude. `catalog_placement` is a
separate Stage-C baseline and is deliberately excluded from the same-model
group. ReSpace is not part of this track.

No evaluator score, error report, hidden annotation, or private evidence is
visible to generation. SceneWeaver reflection remains native and its saved
iterations are evaluated only after the loop.

The initial track intentionally matches the current reference-free Scene10
evaluation profile: its public object plan constrains generation, but is not
promoted into a confirmed benchmark-owned `specification_contract`. Therefore
the track does not claim instance-level L2 relation scores. A future relation-GT
experiment must freeze its own reviewed contract and audited logical/native ID
projection; this implementation does not alter the evaluator's semantic object
mapping to manufacture one.

## Frozen case source

`configs/generation_comparison/frozen_imaginarium_scene10_v1.json` is generated
deterministically from the human-reviewed curation manifest
`configs/generation_comparison/frozen_imaginarium_scene10_curation_v1.json`.
Each room names one hash-pinned, high-scoring SceneBoard canonical scene as its
inventory/asset baseline, followed by exact remove, rebind, add, count, zone,
and support decisions. It reuses only:

- the public natural-language task and rectangular room dimensions;
- public task-slot category, role, and description fields;
- exact selected Imaginarium asset IDs.

It does not reuse generated positions, absolute coordinate hints, evaluator
reports, scores, or evaluator-private data. The visual-review `.blend` files
are provenance only and are never consumed as inventory, geometry, or pose
inputs. Every harness receives exactly the same 269 object slots. The current
materialized snapshot contains 168 unique assets.

Stage-A `estimated_size` values remain in the source spec for selection audit,
but the generator-visible FrozenAssets plan replaces them with each selected
asset's exact physical dimensions. A harness therefore never receives two
conflicting scale targets.

The user gave explicit final approval to all ten per-room selections on
2026-09-05. The checked-in snapshot therefore records
`asset_selection_status=human_approved`; preparation and controlled execution
may pass the asset-selection gate. Approval does not launch an experiment and
does not override separate prerequisites such as runner configuration,
execution identity, credentials, or the LayoutGPT ICL snapshot.

`audit_frozen_imaginarium_scene10_assets.py` is review support only: it never
replaces an asset. Its HIGH-priority flags include assets whose bbox is taller
than the case room, as well as footprint and combined semantic/scale outliers;
the review tables retain each room's dimensions so those decisions are auditable.

| Case | Baseline | Scene | Room metres | Curated slots |
| --- | --- | --- | ---: | ---: |
| S100 | GPT-5.6-Sol | open-plan living/dining/reading | 9.2 x 7.2 x 3.1 | 31 |
| S101 | Claude Opus 5 | kitchen/dining/utility | 10.0 x 5.6 x 3.0 | 32 |
| S102 | HY4-SFT0812 | shared office/library/meeting | 8.8 x 6.6 x 3.0 | 29 |
| S103 | Claude Opus 5 | bedroom/dressing/workspace | 8.2 x 6.4 x 3.0 | 23 |
| S104 | HY4-SFT0812 | media/music/game/recreation | 10.4 x 7.6 x 3.2 | 30 |
| S105 | HY4-0823dev | children's study/play/art | 9.0 x 6.8 x 3.0 | 25 |
| S106 | Claude Opus 5 | bathroom/laundry/utility | 8.4 x 5.8 x 3.0 | 26 |
| S107 | Claude Opus 5 | workshop/repair/storage | 9.2 x 6.6 x 3.1 | 25 |
| S108 | GLM-5.3 | cafe reading/coworking lounge | 8.6 x 6.4 x 3.0 | 23 |
| S109 | HY4-0823dev | fitness/yoga/recovery/hobby | 10.2 x 7.4 x 3.2 | 25 |

The architecture is still the common-denominator contract: one axis-aligned
rectangular room. The semantic briefs are more demanding, but they do not add
nonrectangular geometry, multiple rooms, generated openings, or topology.

## Asset geometry and GLB bundle

The local Imaginarium source is FBX plus source metadata. DirectLayout,
LayoutVLM, and the proposed SceneWeaver frozen initializer require exact GLBs.
Only the selected snapshot is converted; the full asset database is not copied.

`prepare_imaginarium_glb_bundle.py` creates a content-addressed plan and invokes
a Blender worker when a Blender executable is supplied. The worker does not
rescale, recenter, or repair geometry. It verifies:

- source FBX bytes against the planned hash;
- imported FBX bbox size/center against Imaginarium metadata;
- re-imported GLB bbox size/center against the FBX import;
- final GLB content hashes.

The plan is bound to the current source and bundle roots, and its geometry
tolerance is fixed at `1e-4` metres. Validation recomputes bbox comparisons from
the reported measurements instead of trusting a worker boolean. Every run also
re-hashes the case's selected local GLBs immediately before and after external
generation; a missing or changed byte sequence invalidates the run.

An invalid or incomplete bundle is rejected before generation. Catalog records
preserve the source FBX hash, selected mesh hash, local bbox, local bbox center,
native scale, physical dimensions, and canonical front when the source policy
provides one.

Plan without converting:

```bash
PYTHONPATH=src python scripts/prepare_imaginarium_glb_bundle.py \
  --spec configs/generation_comparison/frozen_imaginarium_scene10_v1.json \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --bundle-root /ABSOLUTE/PATH/TO/frozen_imaginarium_scene10_glb
```

Convert and verify after the asset snapshot is approved:

```bash
PYTHONPATH=src python scripts/prepare_imaginarium_glb_bundle.py \
  --spec configs/generation_comparison/frozen_imaginarium_scene10_v1.json \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --bundle-root /ABSOLUTE/PATH/TO/frozen_imaginarium_scene10_glb \
  --blender-executable /ABSOLUTE/PATH/TO/blender
```

## Harness routes

| Method | Controlled route | Current real-upstream status |
| --- | --- | --- |
| Catalog Placement | Existing exact asset selection is supplied directly to the existing Stage-C placement prompt; native `catalog_placement_v1` is then converted/evaluated normally. | Integration present; separate baseline model policy. |
| LayoutGPT | Public plan plus a frozen released-style ICL message set and fixed row order/dimensions -> CSS-style numerical layout -> existing LayoutGPT converter. Exact IDs are preserved in the runner sidecar because the released CSS rows do not encode mesh IDs. | Controlled ICL bridge implemented; a frozen ICL file is required; not real-smoke-tested here. |
| DirectLayout | Released two-list request plus per-slot exact GLB library -> released DirectLayout pipeline -> native numerical JSON -> existing converter. A path-only shim gives initial/refined artifacts one stable room token while preserving the full semantic prompt in every model call. Every native state is snapshotted and checked before rendering; terminal optimizer failure is no longer swallowed, and states from failed retries cannot become the selected result. | Thin released-pipeline bridge implemented; not real-smoke-tested here. |
| LayoutVLM | Released scene config with exact frozen asset table -> released one-shot solver/optimizer -> native layout JSON -> existing converter. The bridge reproduces the released X/Y processed-bbox and fixed mesh-frame transform, preserves the prepared scene config, replaces only the released two-decimal size literal with the exact frozen value, and rejects randomized/unplaced fallback or model-mutated size. Model-authored Python is restricted to the released pose/constraint DSL before the upstream executor sees it. | Thin released-solver bridge implemented; not real-smoke-tested here. |
| SceneWeaver | Exact initialized GLBs -> native reflect/modify loop -> every `layout_N.json` -> existing converter/evaluator per iteration. The plugin must report the exact public-plan hash and native room dimensions, then for each iteration report observed asset ID, mesh path/hash, full-precision canonical local bbox, catalog-bbox bottom-center rebase, and canonical-front-to-native-+X basis. A boolean attestation or benchmark-derived binding alone is insufficient. The converter audits the released rounded world AABB but reconstructs the canonical local OBB rather than rotating that AABB twice. | Conditional: the released initializer/exporter cannot enforce this contract. A versioned upstream-side frozen plugin is required to lock initialization/tools and emit the additional geometry/basis evidence without feeding benchmark scores into reflection. |

Frozen identity does not imply identical mesh observability inside each method.
LayoutGPT is a numerical bbox planner and receives exact IDs, descriptions, and
physical dimensions rather than rendered mesh pixels. DirectLayout renders the
verified GLBs. LayoutVLM binds the GLBs into its scene and may expose already
placed assets in iterative visual context, but its normal one-shot first group
does not necessarily show those meshes to the model. An eligible SceneWeaver
plugin operates on the exact GLBs. This difference is part of the harness and
is recorded rather than hidden by postprocessing.

Released SceneWeaver serializes `layout.size` as a two-decimal, post-rotation
world AABB and independently rounds its Euler pose to two decimals. It is not
the asset-local bbox. The required plugin keeps those released fields for
native-loop fidelity, separately observes the full-precision Euler pose and GLB
local bbox, and bakes a catalog-front-to-native-+X basis when a catalog front
exists. Native AABB verification uses those full-precision observations before
checking the released rounding; it does not recompute exact geometry from an
already rounded angle. The observed GLB bbox may differ from catalog metadata
only within the same frozen `1e-4` metre tolerance used by bundle preflight.
For assets without a validated front, the basis is explicitly unavailable and
is never invented. Because Imaginarium mesh origins are not guaranteed to match
SceneWeaver's bbox-bottom-center convention, the plugin must deterministically
rebase each catalog bbox and report that transform separately. It must also
prove that the native solver received the frozen room width, depth, height, and
metre unit. Host-local mesh paths are withheld from model messages; the plugin
resolves them from the separately supplied comparison catalog and must attest
`model_input_asset_locators_used=false` and the hash of the exact public object
plan delivered to the model.

The benchmark does not vendor any upstream repository. Example runner commands
are in
`configs/generation_comparison/frozen_imaginarium_scene10_methods.example.json`.
Every configured path, interpreter, endpoint, credential environment, and
timeout remains external configuration.

Publication runs pin each upstream Git commit, bridge-entrypoint SHA-256, and a
canonical bridge-bundle SHA-256 covering that entrypoint plus its sibling
`_common.py`.
Pinned upstream checkouts must be clean before and after execution; the runner
sets `PYTHONDONTWRITEBYTECODE=1` so normal imports do not dirty them. The four
configured endpoint strings may differ only by LayoutGPT's required
`/chat/completions` suffix and must match the protocol's normalized base hash.
API keys remain inherited secrets and are never stored in the checked config or
runner report.

LayoutVLM's released solver executes the model-authored constraint program as
Python. The controlled bridge validates the final post-rewrite program at the
last boundary before that execution. It admits only explicit finite
position/rotation assignments for every current-group instance and the five
constraint calls advertised by the pinned prompt; imports, arbitrary calls,
control flow, definitions, comprehensions, attribute introspection, and missing
pose/constraint coverage fail closed. The accepted source hash and policy
version are preserved without copying program text into the runner report (the
native upstream work directory already preserves the raw response). The bridge
also supplies credentials directly to the preconstructed model client and
removes them from the process environment before the constraint program runs.
The exact prepared scene-config bytes are separately preserved, and their
SHA-256 must match the solver-input hash reported by the bridge before that
sidecar is accepted by the converter.

LayoutGPT additionally requires a frozen JSON list of alternating
user/assistant ICL messages. It should be derived and versioned from the released
LayoutGPT training examples used for the experiment. The bridge refuses an
empty or malformed ICL file; silently degrading LayoutGPT into a zero-shot CSS
prompt would no longer represent the intended harness. It copies the exact ICL
bytes into the native run artifacts, records the SHA-256, and fails if the source
changes during generation. The expected ICL hash is required in runner
configuration, and roles must be complete alternating `user`/`assistant` pairs.

All four scene-generation harness bridges receive `LAYOUT_DDD_MODEL_PROVIDER`,
`LAYOUT_DDD_MODEL_ID`, deployment ID, and API-base fingerprint from the
versioned comparison protocol. The configured identity is checked before
launch, and a preserved runner report must attest the observed identity after
generation. The protocol also freezes a normalized API-base SHA-256; LayoutGPT's
full completion URL and the other base URLs must normalize to that same value.
A common non-secret deployment ID is recorded as operator-attested routing
metadata, while the API response model string is independently observed. Call
count, token usage, rendering, optimization, and iteration
budget remain method-native and are recorded rather than falsely equated.

Compute the two bridge source pins with the repository's canonical algorithm
before filling the private run configuration (the bundle hash is not a plain
`sha256sum`):

```bash
layout-ddd-controlled-pilot hash-bridge \
  --entrypoint scripts/external_harness_bridges/layout_gpt_frozen.py
```

Run it once for each of the four bridge entrypoints and copy the reported
`expected_entrypoint_sha256` and `expected_bridge_bundle_sha256` values into its
method configuration. Any subsequent bridge or `_common.py` edit intentionally
invalidates those pins.

The example SceneWeaver configuration explicitly selects the highest numbered
native iteration for the final-scene comparison. It also evaluates every
numbered state independently. The selected iteration and all available
iterations are persisted; no evaluator result is used to choose or modify a
state.

## Stage-C baseline

The frozen object plan and exact asset selection can bypass Stage A/B and enter
the existing Catalog Placement Stage C directly. This makes it a useful
baseline for the same rooms and assets without pretending it is another
same-model external harness. The usual controlled-run validation still checks
exact slots, IDs, dimensions, room identity, native-artifact preservation, and
converter exact-only behavior.

## Prepare and run

Regenerate the materialized curation from the checked-in candidate prompts,
hash-pinned SceneBoard canonical scenes, and exact curation manifest:

```bash
PYTHONPATH=src python scripts/build_frozen_imaginarium_scene10_spec.py \
  --base-spec configs/generation_comparison/frozen_imaginarium_scene10_v1.json \
  --curation configs/generation_comparison/frozen_imaginarium_scene10_curation_v1.json \
  --repo-root /ABSOLUTE/PATH/TO/Layout_DDD \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --output configs/generation_comparison/frozen_imaginarium_scene10_v1.json
```

For a reproducibility check, write to a temporary path and byte-compare or
canonical-JSON-compare it with the checked-in spec. The builder verifies the
catalog CSV hash, all source canonical-scene hashes, every exact asset's FBX
and metadata bytes, every removal/rebinding identity, room geometry, and all
explicit object-support parents. It does not read source pose fields.

Prepare immutable inputs and verify source/converted assets without launching a
model:

```bash
layout-ddd-controlled-pilot prepare \
  --spec configs/generation_comparison/frozen_imaginarium_scene10_v1.json \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --asset-bundle-root /ABSOLUTE/PATH/TO/frozen_imaginarium_scene10_glb \
  --method-configs /ABSOLUTE/PATH/TO/private_method_configs.json \
  --out-dir outputs/frozen_imaginarium_scene10_v1
```

After human approval, use a fresh prepared directory and run the first case as
the mandatory dry run:

```bash
layout-ddd-controlled-pilot run \
  --prepared-dir outputs/frozen_imaginarium_scene10_v1 \
  --method-configs /ABSOLUTE/PATH/TO/private_method_configs.json \
  --dry-run-only
```

Then prepare another fresh directory for the full ten-case run. Pilot outputs
and native method/case directories are never overwritten.

## What has and has not been executed

The integration boundary is covered with synthetic fixtures and existing
converter/evaluator regressions. Local Imaginarium source metadata and all 168
materialized asset files were preflighted. No real LayoutGPT, DirectLayout,
LayoutVLM, or SceneWeaver model run was executed as part of this implementation,
and no quality score is claimed.
