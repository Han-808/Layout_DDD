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
LayoutVLM, and the SceneWeaver frozen initializer require exact GLBs.
Only the selected snapshot is converted; the full asset database is not copied.

`prepare_imaginarium_glb_bundle.py` creates a content-addressed plan and invokes
a Blender worker when a Blender executable is supplied. The worker does not
rescale, recenter, or repair geometry. It verifies:

- source FBX bytes against the planned hash;
- imported FBX bbox size/center against Imaginarium metadata;
- re-imported GLB bbox size/center against the FBX import;
- final GLB content hashes.

The exporter explicitly retains native loose mesh vertices and edges, which
Blender's default glTF export otherwise drops. Near-square XY ordering is tested
at the same `1e-4` metre tolerance as individual axes; an ordering sign change
below that tolerance is not evidence of an axis swap. This is not a front-
direction inference: canonical-front metadata still comes only from the source.
The worker records its own hash, Blender version and export flags. Existing
plans/GLBs/reports are never overwritten; each conversion attempt needs a fresh
directory (including a plan-only attempt followed by a build).

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
| SceneWeaver | Exact initialized GLBs -> native reflect/modify loop -> every `layout_N.json` -> existing converter/evaluator per iteration. The plugin reports the exact public-plan hash and native room dimensions, then per-iteration observed asset ID, mesh path/hash, full-precision canonical local bbox, native Euler, exported bottom-center position, `obj.dimensions`, origin rebase, and front basis. The existing converter audits native rounded fields against those observations. | `scene_weaver_frozen_plugin.py` now implements the native driver/worker route, validated with mocked execution and real-input static preflight. **Not real-upstream-loop smoke tested**; Linux/native environment and production model qualification remain required. |

Frozen identity does not imply identical mesh observability inside each method.
LayoutGPT is a numerical bbox planner and receives exact IDs, descriptions, and
physical dimensions rather than rendered mesh pixels. DirectLayout renders the
verified GLBs. LayoutVLM binds the GLBs into its scene and may expose already
placed assets in iterative visual context, but its normal one-shot first group
does not necessarily show those meshes to the model. An eligible SceneWeaver
plugin operates on the exact GLBs. This difference is part of the harness and
is recorded rather than hidden by postprocessing.

No-model native mesh-assembly probes at the pinned DirectLayout/LayoutVLM
commits established the actual frozen mesh frames (bpy 4.3.0). DirectLayout
uses **clockwise** native degrees with canonical `Rz(180 - z_angle)`; LayoutVLM
uses canonical `Rz(native_yaw + 90)`. The DirectLayout converter and bridge
provenance now record that released convention. Canonical-front metadata is
preserved independently; its absence must not erase the fixed mesh rotation.
Captured asymmetric-vertex tests cover an offset local bbox, nonzero room origin
and 0/90/37-degree poses. Real approved GLB probes additionally include a dresser,
a near-square office chair and the refrigerator with loose geometry. These are
geometry diagnostics, **not** model/render/optimization E2E smoke results; see
`PIPELINE_COMPATIBILITY_RUNS_READINESS.md` for exact evidence and remaining gates.

For SceneWeaver, the user additionally approved **internal** fixed-scale and
no-deletion controls, not merely removal of named tools: the released position
update also rescales objects, and native physics cleanup deletes them. The
source-pinned `scene_weaver_frozen_mutations.py` worker component intercepts those
operations before they mutate frozen geometry/inventory; native pose processing,
physics measurements and reflection are retained. The deletion/re-optimize cycle
stops if no legal deletion exists, leaving collisions visible, not scoring them
as fixed. This must be reported as **SceneWeaver–FrozenAssets (restricted mutation
set)**. The plugin connects these controls to native initialization and child
workers; failed, unaccepted initial placement candidates may be rolled back so
the native second attempt can run. An accepted scene object cannot be deleted.
Mocked integration does not certify the real initializer/optimizer/renderer/loop;
the harness remains conditional until host and real-loop qualification finish.
Upstream files, frozen assets and benchmark scoring stay unchanged.

### SceneWeaver frozen plugin execution

The existing bridge's `--plugin-entrypoint` now points to
`scripts/external_harness_bridges/scene_weaver_frozen_plugin.py`. The example
method configuration includes its SHA-256. That pinned file also checks the
hashes of its four transitive helpers **before importing them**. All workers
verify the pinned, clean upstream checkout. Do not reuse an old prepared run
after changing either plugin or configuration identity.

The driver calls the released `SceneDesigner.run()`; it does not implement a
replacement reflection loop. The released three-stage GPT initializer receives
the exact public plan, slot counts, dimensions and fixed factory mapping. Its
native output must match those inputs; wrong mappings, counts, dimensions or
room size are rejected, not rewritten. The worker uses the released native
`generate_indoors.main()` and pose optimizer. Native additional-object stages
are disabled through configuration. A pinned import-time room-height binding
prevents modules from retaining a different randomly sampled height; a headless
UI-only overlay avoids requiring a Blender viewport. Neither changes room shape.

`LAYOUT_DDD_SCENEWEAVER_PYTHON` selects the native worker interpreter; it defaults
to the plugin interpreter. Before any model call, the driver starts a no-model
worker import/overlay preflight. Missing native dependencies stop execution
there. The worker receives no API-key/token/secret environment variables. It
stays in the outer runner's process group so its existing timeout/cancellation
cleanup covers the complete native process tree.

For a **static, no-upstream-import/no-API** check of already materialized public
inputs (all variables below are explicit operator paths):

```bash
python scripts/external_harness_bridges/scene_weaver_frozen_plugin.py --preflight-only --repo-path "$SCENEWEAVER_REPO" --request "$PUBLIC_REQUEST" --method-input "$PUBLIC_METHOD_INPUT" --comparison-input "$COMPARISON_CONTROL" --comparison-catalog "$METHOD_CATALOG" --output-root "$NEW_OUTPUT" --plugin-report "$NEW_PREFLIGHT_REPORT"
```

This reports `STATIC_PREFLIGHT_ONLY`, never production readiness. It does not
load Blender, optimize, render, or call a model. Actual execution uses the
existing controlled-pilot/bridge command after separate API authorization;
`controlled-pilot run --dry-run-only` is still a **real generation** command.

Artifacts beneath the new output root:

- `generation_asset_selection.json`: exact IDs saved before generation.
- `frozen_worker_input.json`: immutable worker bindings; not model-visible.
- `model_calls/call_*/`: effective request and original SDK response saved before
  native consumption, observed model identity, token usage and redacted failure.
  Native SDK/reasoning retry budgets remain native; no fallback model/endpoint.
- `worker_attempts/attempt_*/`: argv, cwd, native args, stdout/stderr, return code,
  runtime, source overlay audit and mutation journal.
- `sceneweaver_native/`: working native layouts, renders, state/blend/pickle
  files, native reflection records and full-precision observations.
- `native_archive/`: append-only snapshots with original relative names and
  content-addressed, hash-verified bytes. Backtracking cannot erase earlier
  versions. Blob names avoid duplicate `layout_N.json` discovery. To evaluate
  an archived attempt, restore its manifest's original names into a **new**
  directory, verify hashes, then use the existing offline converter/iteration API.
- `plugin_report.json`: produced only after successful native termination and
  validation of all selected working iterations through the existing bridge.
  A failed last action cannot promote an earlier layout into a successful run.

The plugin does not import or call the benchmark evaluator. Existing post-hoc
trajectory evaluation remains responsible for canonical scenes and reports.
Actual SDK call count and tokens are recorded; rendering-call totals remain
unknown where retries prevent a reliable count (not silently equated with the
number of selected states). The operator must still qualify the configured
model's real image/tool response behavior. Interface reference:
[OpenAI Chat Completions Python API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).

Released SceneWeaver serializes the solver placeholder's `obj.dimensions` into
`layout.size` to two decimals. These are scaled **local-axis object dimensions**,
not a post-rotation world AABB. An input basis baked into the mesh changes them;
runtime `rotation_euler` does not. This corrects the earlier world-AABB assumption
using the pinned native exporter on actual bpy objects (Blender 4.3 and 5.2).
The required plugin keeps native JSON unchanged and separately observes the
full-precision Euler, exported bottom-center position, object dimensions, and
canonical-frame local bbox for every iteration. The converter verifies all three
native rounded fields against their observed full-precision counterparts, then
uses the exact frozen asset dimensions and observed pose. The observed GLB bbox
and basis-baked object dimensions may differ from catalog-derived dimensions
only within the same frozen `1e-4` metre tolerance used by bundle preflight.
The frozen contract is `released_object_dimensions_rounded_2dp`. The erroneous
`released_world_aabb_rounded_2dp` contract is rejected, not silently reinterpreted.
Old prepared inputs/sidecars remain unchanged; prepare a new version for this
contract. Ordinary offline `scaled_object_local_bbox_dimensions` remains intact.
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

The controlled LayoutVLM bridge also requires the release's differentiable
`Rotated_IoU` backend before constructing model clients. Its CPU fallback
detaches corner coordinates to NumPy/Shapely and creates new leaf loss tensors;
a positive overlap loss consequently need not provide gradients to object pose.
Do not label that fallback as the same optimization treatment. The current
macOS preparation environment imports the release but lacks the CUDA extension;
full LayoutVLM execution remains blocked until a compatible backend is verified.
This gate does not replace or modify the upstream optimizer.

LayoutGPT additionally requires a frozen JSON list of alternating
user/assistant ICL messages. It should be derived and versioned from the released
LayoutGPT training examples used for the experiment. The bridge refuses an
empty or malformed ICL file; silently degrading LayoutGPT into a zero-shot CSS
prompt would no longer represent the intended harness. It copies the exact ICL
bytes into the native run artifacts, records the SHA-256, and fails if the source
changes during generation. The expected ICL hash is required in runner
configuration, and roles must be complete alternating `user`/`assistant` pairs.

An offline, reproducible recipe is now provided at
`configs/generation_comparison/layoutgpt_icl_recipe_v1.json`. It pins eight
public training examples (the first four in each frozen `rect_train` list,
bedroom then livingroom), the two dataset statistics files, upstream split
hashes, and the released formatter hash. The official source is the
[LayoutGPT preprocessing download](https://github.com/UCSB-AI/LayoutGPT#preparation-for-3d-indoor-scene-synthesis).
The source subset can be extracted from `data_output.zip` without models.

```bash
PYTHONPATH=src python scripts/prepare_layoutgpt_frozen_icl.py \
  --recipe configs/generation_comparison/layoutgpt_icl_recipe_v1.json \
  --repo-path /ABSOLUTE/PATH/TO/LayoutGPT \
  --training-root /ABSOLUTE/PATH/TO/data_output \
  --out-dir /ABSOLUTE/NEW/PATH/TO/layoutgpt_icl
```

The builder invokes only the pinned `load_room_boxes` function, with metre units
and normalization disabled. Its condition/layout text, two-decimal formatting,
and native angle fields are preserved; it does not import the whole upstream
CLI, run a tokenizer/model, use target scenes, or use released LLM predictions
as ground-truth examples. The manifest explicitly labels this **adapted frozen
ICL**, not native per-target k-similar retrieval. Training CSS does not supply
canonical asset fronts; the target Imaginarium frames remain those explicitly
given by the controlled bridge, not inferred from these demonstrations.
Raw training records and generated message snapshots belong outside Git. Set
the new prepared protocol's ICL identity to the audited snapshot hash; never
change an old prepared run to accept a different snapshot.

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
  --output /ABSOLUTE/PATH/TO/FRESH/curation_rebuild.json
```

For a reproducibility check, write to a temporary path and byte-compare or
canonical-JSON-compare it with the checked-in spec. The builder verifies the
catalog CSV hash, all source canonical-scene hashes, every exact asset's FBX
and metadata bytes, every removal/rebinding identity, room geometry, and all
explicit object-support parents. It does not read source pose fields.

That rebuild is source/curation verification, not the current launch spec.
For this experiment apply the approved public-brief v2 recipe below, then bind
the actual host/deployment/ICL with the operator configuration step below.
Use the resulting spec **and** methods **and** evaluator runtime together in
fresh prepares. The original example has a pending ICL and no injected evaluator
runtime; it cannot serve as the production launch configuration unchanged.

## Actual-asset public brief revision v2

The approved bindings contain legacy slot names and inherited functional briefs
that do not always describe the fixed catalog asset. For example, S109's
`treadmill_1` binds `a_Ladder`, and S106's `bath_mat_1` binds `e_SM_BATHUB_01`.
The user explicitly approved revising the public descriptions against the actual
frozen catalog **without changing any asset or slot**. Do not interpret this as
permission to retrieve replacements or rescale a mesh.

`frozen_imaginarium_scene10_public_brief_v2.json` is a source-hash-pinned input
revision recipe. It projects the approved slot category/description into the
public plan, synchronizes duplicate requested-text metadata, and explicitly
corrects obsolete counts, roles, zones and universal `-Y` front assumptions.
Source IDs, exact asset bindings, geometry, physical scale, support relations,
seeds, model/budget policy and evaluator configuration remain unchanged.

```bash
PYTHONPATH=src python -m benchmark.generation_comparison.public_brief \
  --spec configs/generation_comparison/frozen_imaginarium_scene10_v1.json \
  --recipe configs/generation_comparison/frozen_imaginarium_scene10_public_brief_v2.json \
  --out-dir /ABSOLUTE/PATH/TO/FRESH/public_brief_v2
```

This preparation-only command writes `spec.json` and `public_brief_audit.json`
with every before/after change. It refuses an existing output directory, does
not call models, and never rewrites the source spec. All five methods receive
the revised plan through the existing public projection; the unchanged evaluator
uses that same revised plan. The audit with obsolete text is **not** model input.
Catalog text is not a new visual certification, and difficult geometric/support
requirements are not automatically relaxed or repaired.

This is a **new input treatment**. In particular, S109 is now described as a
recreation/recovery/hobby room using the supplied objects. Do not describe results
as evaluations against byte-identical original SceneBoard prompts, or pool scores
across public-brief versions without identifying that difference.

## Predeclared smoke, dense pilot, and repetitions

`prepare --case-id` now selects existing cases from a fully validated source
spec. Repeat the flag for a shared subset. Source case order is retained; the
whole logical catalog is retained; the selected case definitions are unchanged.
The selected/source case IDs and full source-spec hash are recorded in the
byte-pinned root protocol. There is no run-time `--case-id` or implicit resume.

| Stage | Existing cases | Repetitions | Planned units with all five methods |
| --- | --- | --- | --- |
| Smoke | S100 | 1 | 5 |
| Dense pilot | S101 | 1 | 5 |
| Formal | S100--S109 | 3, separate fresh prepares | 150 |

S100 is the first source case, not a score-selected case. S101 has 32 slots in
56 m², the highest slot count and slot/area density in the approved ten-case
input. This is a pre-run complexity criterion, not a quality-score criterion.
Do not advance to formal generation until every included method passes its
real native-workflow, identity, frozen-input and full-evaluation qualification.
Blocked methods are not silently dropped; changing the comparison cohort needs
an explicit amended protocol. Catalog Placement remains separately reported.

For the smoke prepare, use the new public spec and include:

```bash
layout-ddd-controlled-pilot prepare \
  --spec /ABSOLUTE/PATH/TO/bound_operator_config/spec.json \
  --asset-root /ABSOLUTE/PATH/TO/imaginarium_assets \
  --asset-bundle-root /ABSOLUTE/PATH/TO/frozen_imaginarium_scene10_glb \
  --method-configs /ABSOLUTE/PATH/TO/private_method_configs.json \
  --evaluation-runtime-config /ABSOLUTE/PATH/TO/trusted_evaluation_runtime.json \
  --case-id S100 --out-dir /ABSOLUTE/PATH/TO/FRESH/smoke

layout-ddd-controlled-pilot preflight \
  --prepared-dir /ABSOLUTE/PATH/TO/FRESH/smoke \
  --method-configs /ABSOLUTE/PATH/TO/private_method_configs.json
```

Use `--case-id S101` and a different directory for the dense pilot. Omit
`--case-id` for each of the three formal prepares. The preflight command is
no-call; **`run --dry-run-only` is not**. The above remains blocked until audited
ICL/runtime/model bindings and all readiness requirements are resolved.
Repetitions mean separate native invocations, not best-of selection or reuse of
cached outputs. Native seed enforcement is not guaranteed unless reported by
the runner; record effective native parameters, calls, tokens and time rather
than claiming equal or randomized seed treatment. Evaluator failure does not
authorize regenerating; use append-only offline reevaluation when possible.

## Bind one host and deployment, without executing it

Use `configs/generation_comparison/pipeline_operator_bindings.example.json`
as the private binding template, and
`configs/generation_comparison/canonical_evaluation_runtime.example.json`
as the private evaluator-runtime template. Keep edited files outside the source
checkout and old prepared runs. Do not put API keys in either file.
The compiler is a checkout script, not an additional installed-wheel entrypoint.
Virtual-environment interpreter symlinks are preserved so the native dependency
environment is not silently replaced by its base Python installation.

Fill every placeholder explicitly. The binding compiler does not choose a
model, discover private credentials, install environments or authorize a run.
The example pins the approved public-brief v2 source and released-derived ICL
bytes; it does not substitute synthetic ICL or silently bless a changed file.

| Binding | Meaning and constraint |
| --- | --- |
| `benchmark_python` | Absolute interpreter for the pinned benchmark environment; generated commands use the current checkout's absolute `PYTHONPATH` |
| `upstreams.<method>` | Actual pinned clean upstream checkout and its own Python environment, not the benchmark interpreter by default |
| `shared_model` | One configured provider, exact expected response model ID, deployment ID and API **base** for the four harnesses; a deployment label alone is not production identity evidence |
| `layoutgpt_icl` | Approved messages path, exact byte hash, approval and source/selection provenance; snapshot is copied byte-for-byte into the new config directory |
| `asset_root`, `asset_bundle_root` | Original Imaginarium sources and verified 168-asset GLB bundle; no new asset selection or normalization |
| `runs_root` | Fresh root for smoke, dense pilot and three formal repetitions; separate from the config directory |
| `catalog_placement` | Separately reported Stage-C model/endpoint/key environment name; not implicitly included in the four-method same-model claim |
| `evaluation_runtime_config` | Trusted post-hoc Judge/camera/Blender configuration; never added to method input or generation environment |
| `evaluation_acceptance_policy` | Optional experiment gate, outside scoring kwargs. This experiment explicitly uses `frozen_assets_required_metrics_v1`; omitted bindings retain `complete_score_v1`. |

On the eventual execution host, from the pinned benchmark checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python \
  scripts/prepare_pipeline_operator_config.py \
  --spec /ABSOLUTE/PATH/TO/public_brief_v2/spec.json \
  --bindings /ABSOLUTE/PATH/TO/private_operator_bindings.json \
  --out-dir /ABSOLUTE/PATH/TO/FRESH/bound_operator_config
```

This writes `spec.json`, `methods.json`, `evaluation_runtime.json`, the exact
`layoutgpt_icl_messages.json`, `launch_plan.json` and `configuration_manifest.json`.
The only source-spec changes are the configured model identity/deployment/API
fingerprint, ICL identity/status/provenance, and explicitly selected experiment
acceptance policy. Cases, inventory, assets, native budgets and evaluator scoring
kwargs remain unchanged. Source and bound evaluator-policy identities are
recorded separately; opting in requires a new preparation, never editing an old
prepared run. Source, compiler, template and
output hashes are recorded. A bridge/plugin pin mismatch fails rather than
automatically accepting new code. Existing output directories are rejected.

`CONFIG_BOUND_NOT_PREFLIGHTED` means precisely that. In particular, an absolute
Linux Python path is a binding, not proof that its CUDA/Blender/native imports
work. The compiler constructs the existing evaluator client objects without
credentials or service calls, but does **not** run native environment probes,
generation, rendering, the evaluator, or the commands it emits.

### Credentials and launch sequence

The four external bridges consume `LAYOUT_DDD_API_KEY` from the authorized
process environment. Catalog Placement uses its configured `api_key_env` and
the evaluator uses its configured Judge/camera key environment names. Set values
privately; never put them in command arguments, bindings, commits or chat.
Do not export another experiment's complete environment or source upstream key
files. Model/deployment identity is injected by the existing comparison runner
from the bound protocol; endpoint conventions are derived consistently (full
`/chat/completions` for LayoutGPT, API base for the other three).

`launch_plan.json` contains argument arrays, not shell fragments. Use its `cwd`
and `environment`, and review stages sequentially:

1. Execute S100 `prepare_argv`, then `preflight_argv`. Both are no-call.
   Missing keys produce a blocked preflight without contacting the service.
2. Only with separate real-execution authorization and native prerequisites
   qualified, execute S100 `run_argv`. This spends generation **and** evaluator
   calls. The stage has one case, so the misleading `--dry-run-only` flag is
   unnecessary. API success alone is not full E2E success.
3. Require all five planned units' real workflow, identity, frozen inputs and
   experiment evaluation acceptance, then separately prepare/preflight/run S101.
4. After that acceptance, use `formal_r1`, `formal_r2`, `formal_r3`, in fresh
   directories, with the same bound config: 50 units each, 150 formal units.
   These are independent native invocations, not best-of/cached results.

Do not batch-execute all emitted commands as an unattended shell script. No
automatic stage advancement or production API probe is implemented by the
configuration compiler. Existing `preflight`/`run` return exit 2 when blocked;
no-call preflight cannot certify the actual provider or native loop.

### Approved FrozenAssets evaluation acceptance

The user approved **L3 coverage >= 0.8**, with only
`object_pairing_consistency` and `style_consistency` allowed to be inapplicable.
The experiment example explicitly opts into `frozen_assets_required_metrics_v1`.
This is an orchestration acceptance rule, not a new evaluator/scoring profile.

The gate requires factual benchmark-owned frozen asset identity, category,
appearance and scale, with generator-owned arrangement, in both the prepared
policy and actual report. It verifies the evaluator's own metric plan against
coverage, raw metric envelopes and reliability records. L0 must pass, every
other applicable metric (including active L2 claims) must be complete, and
missing API/render/evidence or unresolved claims cannot be exempted. A high
score or overall coverage above 0.8 cannot waive a missing required metric.
Quality scores themselves are not an engineering acceptance cutoff.

The existing evaluator can report L3 coverage 0.84 and raw
`benchmark_score_status=partial_coverage` when only these two metrics are
inapplicable. That raw status, raw score, weights, denominator and report bytes
are not modified. Each run separately records `evaluation_accepted`, the
versioned acceptance decision and report hash. Existing `evaluation_success`
continues to mean a complete raw benchmark score. Accepted partial reports can
advance this experiment, but are excluded from the existing complete-score
mean/median/paired aggregates and counted separately. There is no silent
publishability or complete-score claim.

SceneWeaver records the same gate for **every** preserved iteration, post-hoc.
A qualifying final report cannot mask an unaccepted iteration. Neither the
gate nor official evaluator reports are given to its native reflection loop.
Append-only final-state reevaluation uses the original prepared acceptance
policy and cannot retrofit an opt-in into a historical run.

Without explicit opt-in, the previous complete-score requirement and its
frozen-applicability preflight block remain unchanged. Native environment,
model route and real render/loop qualification are still independent gates.

### Costs, stopping, and recovery

Keep the configured native workflow budgets. Record effective model parameters,
observed response identity, tokens/calls, retries, tool/render calls where
available, runtime and iteration counts. Outer timeouts in the pinned methods
template are 900/3600/3600/14400 seconds for LayoutGPT/DirectLayout/LayoutVLM/
SceneWeaver; these are execution ceilings, not equal-cost claims. Any monetary
cap or per-call price remains an operator/provider input, not an inferred value.

Stop on identity/input/asset drift, unreaped processes or infrastructure failure.
Ctrl-C terminates the controlled process group, records cancellation and does not
launch the next unit. Failed/missing/unattempted units remain in the ledger;
they are not zero scores. Existing native retry behavior is retained and audited,
but this plan adds no benchmark-side generation retry or `--resume` mechanism.
Keep failed runs intact; regeneration needs a separately approved new attempt.

For a valid preserved generation whose evaluation failed, use the existing
append-only recovery, with a fresh directory **outside** the original run:

```bash
layout-ddd-reevaluate-controlled \
  --prepared-dir /ABSOLUTE/PATH/TO/ORIGINAL/prepared_run \
  --case-id S100 --method direct_layout \
  --out-dir /ABSOLUTE/PATH/TO/FRESH/reevaluation_attempt1
```

This verifies preserved native/canonical/input hashes and retains the original
frozen evaluator configuration. It spends only post-hoc render/Judge/camera work,
never regenerates or reconverts, and cannot be used to retrofit a different
evaluator policy into an old run. SceneWeaver here re-evaluates its originally
selected final state; its original pilot trajectory remains separately recorded.
Do not claim this recovery command re-evaluates the entire trajectory. No official
report is ever fed back into the native reflection loop.

## What has and has not been executed

The integration boundary is covered with synthetic fixtures and existing
converter/evaluator regressions. Local Imaginarium source metadata and all 168
materialized asset files were preflighted. No real LayoutGPT, DirectLayout,
LayoutVLM, or SceneWeaver model run was executed as part of this implementation,
and no quality score is claimed.

Linux environments were installed under the user-authorized
`/hcccao/pipeline_compatibility_runs/runtime_v1/`. Native imports and a CPU/mocked
compatibility suite passed; LayoutVLM's CUDA extension compiled, but no GPU
kernel, scene render or generation loop was run. The archived bpy wheel's
internal platform tag causes explicit dependency-check failures; successful
imports do not erase that caveat. See the current evidence section in
`PIPELINE_COMPATIBILITY_RUNS_READINESS.md`. That installation is pinned to
`d0b5e4f`; the later acceptance checkpoint is not implicitly deployed by these
local code/tests. Approved asset synchronization and production bindings remain
separate preparation steps.
