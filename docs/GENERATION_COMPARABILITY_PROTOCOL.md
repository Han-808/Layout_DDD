# Generation Comparability Protocol v1

## Scope and invariant

This layer controls generation inputs for cross-method experiments. It does not
change converter or evaluator behavior:

```text
controlled public input
  -> method-native generation workflow
  -> preserved native output
  -> existing strict converter
  -> canonical_scene_v1
  -> benchmark.api.evaluation.run_evaluate
```

Version 1 is intentionally limited to one furniture-focused, axis-aligned
rectangular room. It does not claim controlled comparability for nonrectangular
boundaries, multiple rooms, generated walls/openings, or room topology.

## The three protocols

| Mode | Held constant | Method-owned variables | Scientific question |
| --- | --- | --- | --- |
| `native` | Canonical evaluator and declared case room | Native asset source, retrieval, inventory, scale, and workflow | How do the released end-to-end systems compare? |
| `shared_db` | Exact immutable logical catalog and room; the first track also freezes object inventory | Method-specific queries, retrieval/selection, placement, and workflow; scale is explicit | How do retrieval plus workflow differ within the same asset universe? |
| `frozen_assets` | Room, slots, exact asset IDs, meshes, local bbox, canonical front when known, physical scale, and evaluator | Pose and method-native reasoning/workflow only | How do placement/workflow decisions differ after removing asset and inventory variation? |

Every input uses `protocol_id=generation_comparison_v1` and
`protocol_version=1`. The normalized contract records mode, architecture,
inventory policy, asset policy, scale policy, retrieval policy, generation
budget policy, and evaluator policy. Checked-in examples are under
`configs/generation_comparison/`.

## Common track

The first strict track is:

```text
single room
axis-aligned rectangle
furniture objects only
frozen logical object slots
frozen exact asset bindings
fixed_native_scale
same canonical evaluator
```

`fixed_native_scale` means the placed canonical `object.size` must equal the
catalog's physical dimensions. Those dimensions are explicit, or are computed
deterministically as `bbox_size_local * native_scale`. A method may not change
scale and rely on postprocessing to normalize it. Any difference makes the run
invalid. Scale-enabled experiments require a future, separately versioned
protocol.

## Canonical asset catalog

`canonical_asset_catalog_v1` is a database-agnostic immutable logical snapshot.
Each record contains:

- exact `asset_id`, `source_db`, category, and description;
- mesh URI/path when available;
- asset-local bbox size and center;
- canonical-front metadata only when supplied by the source;
- native scale and evaluated physical dimensions;
- optional metadata and content hashes.

Catalog physical dimensions use the required `linear_unit=meter`; materializers
perform only declared coordinate-axis representation changes. Version 1 rejects
records whose explicit physical dimensions disagree with
`bbox_size_local * native_scale` rather than conflating those concepts.

Assets are sorted by exact ID and hashed as canonical JSON. The identity tuple
is `catalog_id`, `catalog_version`, and `catalog_sha256`. Reordering source rows
does not change the digest; changing any logical record does. When mesh bytes
have a supplied/computed SHA-256, that content identity replaces a host-specific
local mesh locator in the logical hash; the serialized record still retains its
actual locator for materialization. The runner receives a materialized
read-only-by-contract snapshot, and its bytes are checked again after generation.
Controlled execution also verifies a declared mesh hash against local bytes when
that mesh path is present.

The original concrete backend is `load_3d_future_subset_catalog()`. It
normalizes an explicitly selected 3D-FUTURE subset and optionally hashes local
mesh bytes. The S100--S109 controlled track adds a preflighted Imaginarium
snapshot and a selected-only FBX-to-GLB materialization. Neither backend
searches, ranks, or retrieves records during conversion. They are pilot
backends, not database dependencies in the comparison architecture. See
[`FROZEN_IMAGINARIUM_SCENE10.md`](FROZEN_IMAGINARIUM_SCENE10.md).

## Generation-side materializers

The same logical catalog is exposed through thin method views:

| Method | Materialized generation input |
| --- | --- |
| LayoutGPT | Dataset asset index plus native category-counter object IDs and an exact binding map |
| DirectLayout | Asset-library manifest plus non-mutating mesh symlinks when local files exist |
| LayoutVLM | Candidate table; FrozenAssets directly populates the released scene-config `assets` table |
| ReSpace | Asset metadata/cache view plus frozen SSR seed-object records |
| SceneWeaver | Asset-source view plus exact per-object binding records |

These files are generation inputs. They never rewrite native outputs. For a
subprocess runner, paths are available as `{comparison_input}` and
`{comparison_catalog}` template variables and as
`LAYOUT_DDD_COMPARISON_INPUT` / `LAYOUT_DDD_COMPARISON_CATALOG`. Callback runners
receive the same public control in `method_input.json`. DirectLayout also gets
`{comparison_asset_root}` / `LAYOUT_DDD_COMPARISON_ASSET_ROOT` when local meshes
were materialized as non-mutating symlinks.

LayoutGPT uses deterministic native IDs such as `chair_1`; its materialization
records the bijection to logical protocol slots. Other v1 materializers preserve
the slot ID directly. The logical slot map is audited and reversed for output
validation; converter object IDs are not rewritten.

## Eligibility

Eligibility is checked and persisted before an upstream process starts. A
controlled runner that adds capabilities outside the released default declares
them under `adapter_config.comparison_support`. The relevant flags are:

```json
{
  "shared_catalog": true,
  "fixed_object_inventory": true,
  "exact_asset_ids": true,
  "fixed_native_scale": true,
  "frozen_iteration_bindings": true,
  "no_object_insertion_removal": true
}
```

This declaration is not accepted as proof of a valid result. Native IDs,
catalog membership, inventory, architecture, and physical scale are checked
again after generation. Missing declarations fail closed with structured reason
codes and the upstream command is not launched.

The eligibility report explicitly records `control_evidence.pre_run_basis` as
`capability_declarations` and `real_upstream_smoke_test_verified=false`.
`ELIGIBLE` is permission to attempt the configured protocol, not proof of
released-upstream controllability or publication readiness. Post-run validation
proves the observed run's identities, not that every future run will honor the
control flags. A publication baseline also needs the real-upstream gate in
[`COMPATIBILITY_STATUS.md`](../COMPATIBILITY_STATUS.md).

| Method | Native | SharedDB | FrozenAssets | Reason / limitation |
| --- | --- | --- | --- | --- |
| Current `catalog_placement` | YES | NOT_APPLICABLE | YES | It already receives exact selected assets and slots; controlled input fixes `uniform_scale=1.0` and validates native placement identity. |
| LayoutGPT | YES | CONDITIONAL | CONDITIONAL | Its released output is prompt-driven and needs a thin runner to consume the shared index; strict freezing additionally requires generation-time exact IDs and fixed inventory/scale. A post-hoc binding does not qualify. |
| DirectLayout | YES | CONDITIONAL | CONDITIONAL | The native two-list request and asset directory remain intact. The runner must consume the materialized library and demonstrate fixed slots/IDs for FrozenAssets. |
| LayoutVLM | YES | CONDITIONAL | YES | SharedDB retrieval still needs an upstream selection wrapper. FrozenAssets maps exact slots and UIDs into the native scene config; output scale is independently checked. |
| ReSpace | YES | CONDITIONAL | CONDITIONAL | The runner must constrain its native catalog/cache. FrozenAssets requires a genuine fixed SSR inventory with asset sampling disabled, not converter resampling. |
| SceneWeaver | YES | CONDITIONAL | CONDITIONAL | FrozenAssets is ineligible by default because native tools may insert/remove/replace objects. It becomes eligible only when the runner can lock inventory and bindings across every native iteration. |

Thus all five are not unconditionally eligible for the strict FrozenAssets
track. Synthetic CI runners explicitly declare and honor the controls so the
integration and validators can be exercised for all five without claiming that
their real upstream checkouts were smoke-tested.

## Identity and fairness gates

The comparison validator reports violations; it never repairs them. It checks:

- protocol catalog identity versus active and materialized catalog identity;
- comparison-case, method-native input, and canonical-output architecture hashes;
- exact logical object inventory and category identity when frozen;
- selected IDs from the preserved native artifact or generation-time sidecar;
- native-to-canonical asset identity preservation;
- selected-ID membership in SharedDB;
- exact `slot_id -> asset_id` equality in FrozenAssets;
- fixed physical dimensions under `fixed_native_scale`;
- optional strict local-mesh byte snapshots immediately before and after
  execution (`generation.require_local_asset_bytes=true`);
- exact catalog source/mesh/front and the separately retained local bbox,
  local-center, native-scale, and physical-dimension provenance;
- unexpected insertion/removal and SceneWeaver iteration drift;
- `asset_resolution_policy=exact_only` at conversion.

The native selection audit understands the five released artifact shapes.
LayoutGPT and SceneWeaver may use only sidecars already preserved by the external
execution boundary. The expected binding file generated by the benchmark is
never passed to the converter as a substitute for missing upstream identity.

## Run manifest and resource accounting

`comparison/run_manifest.json` records protocol/case identity, the three
architecture hashes, catalog identity, inventory and binding hashes, scale and
retrieval policy, eligibility, upstream repo/commit, preserved native artifact
and hash, native selections, canonical scene and hash, the unchanged evaluator
route/report, and available generation resource metadata.

The manifest retains the pre-run `control_evidence` separately from observed
post-run validation. `runner.source_provenance` records the actual shim/callback
file's pre/post execution hashes, discoverable Git commit, and tracked/modified
state. When a sibling `_common.py` exists, it also records a canonical bundle
hash over both files. This narrow bundle is not a general transitive dependency
lock or verification of the runner's capability attestation. Version and retain
the actual bridge and its dependencies for publication; a clean upstream commit
alone does not identify an untracked local wrapper.

Wall time is always recorded. Model, token, generation-call, tool-call,
retrieval-call, rendering-call, and iteration counts are recorded when reported
by the upstream runner. `method_native_recorded` deliberately does not claim
that structurally different workflows received identical token/tool budgets;
quality and cost remain separate outcomes.

When `generation.model_policy` is present, focused runs can require the same
configured provider/model, one operator-attested deployment ID, one normalized
API-base SHA-256, and API-response-observed model identity. The endpoint string
and credentials are not persisted. This prevents method-specific endpoint
configuration from silently passing merely because all responses use the same
model alias.

## SceneWeaver trajectories

The complete native SceneWeaver directory remains the source of truth. Every
`layout_0.json ... layout_K.json` is independently converted with the existing
SceneWeaver converter, evaluated with the same `run_evaluate()`, and checked
against the same comparison protocol. A FrozenAssets trajectory is valid only
when every iteration retains the same slots, exact asset IDs, and fixed physical
dimensions. The focused Imaginarium bridge additionally requires each iteration
to report the actually loaded mesh path and content hash; capability booleans or
a benchmark-authored binding sidecar alone do not prove asset identity. Its
upstream-side plugin must also prove the exact public object-plan hash, native
room width/depth/height/unit, catalog-bbox bottom-center rebase, full-precision
local bbox and Euler pose before released two-decimal serialization, and the
canonical-front basis used by every iteration.

The benchmark report is post-hoc. Neither scores nor evaluator error details are
placed in the method input or fed into SceneWeaver's native reflection loop.

## Invocation

### Prepared pilot safety and evaluator runtime

The prepared pilot verifies the actual bytes of **all** cases before starting
any method and rechecks before each unit. Pins cover public generation input,
evaluation object plan, protocol, catalog and local mesh contents, and evaluator
configuration. A drifted directory cannot be accepted by refreshing its hashes:
prepare a new directory. Independently archive
`prepared_manifest_identity_sha256`; local checks detect accidental drift, not
replacement of an entire manifest and all of its pins by a malicious operator.

Use `prepare --evaluation-runtime-config` with a private copy of
`configs/generation_comparison/canonical_evaluation_runtime.example.json`.
This constructs the existing Blender/Judge/grouping/camera runtime. Configuration
contains environment-variable names, never literal credentials. Runtime and
evaluator-private evidence remain outside method inputs. Every final scene and
SceneWeaver iteration gets fresh, scene-bound evidence through the same
`benchmark.api.evaluation.run_evaluate`; no metric or scoring branch is added.

```bash
layout-ddd-controlled-pilot preflight \
  --prepared-dir /ABSOLUTE/PATH/TO/prepared \
  --method-configs /ABSOLUTE/PATH/TO/private_methods.json
```

`preflight` is read-only and performs **no generation, rendering or model calls**.
It checks files, identities and credential presence, not service availability.
It exits 2 when not ready. In contrast, `run --dry-run-only` still executes the
first case and can spend generation and evaluation API calls; it is not an
offline preflight. Real generation is blocked if evaluator runtime or the
complete-score policy is not ready.

Known baseline limitation: honest FrozenAssets ownership makes Style and Pairing
`not_relevant`, but the current canonical evaluator retains their defaulted,
ungrounded terms in the denominator. A deterministic external-observation
fixture through the unchanged evaluator yields L3 coverage `0.84` and
`partial_coverage`. The preflight reports
`canonical_applicability_prevents_complete_coverage`. This implementation does
not falsify ownership, change weights, disable metrics or label partial scores
complete. Resolving that experiment-policy issue needs a separate user decision.

Pilot status distinguishes `blocked`, `failed`, `partial`, `cancelled` and
`completed`; every planned method/case gets a row, including skipped units with
reasons. Zero attempts never succeeds. Complete-score summaries and paired
deltas exclude incomplete reports, while raw numeric partial scores remain in
per-run rows. Cancellation is propagated after generation process-group cleanup
and log preservation; subsequent cohort units are not launched.

### Append-only evaluation recovery

Successful native generation and fairness validation are frozen in
`comparison/generation_manifest.json` **before** evaluation. If rendering or a
Judge fails, `run_manifest.json` records `EVALUATION_FAILED` (or cancellation),
without discarding the native/canonical outputs.

```bash
layout-ddd-reevaluate-controlled \
  --prepared-dir /ABSOLUTE/PATH/TO/original_prepared \
  --case-id S100 --method direct_layout \
  --out-dir /ABSOLUTE/PATH/TO/new_reevaluation_attempt
```

This verifies original native/canonical/input hashes, reuses the frozen evaluator
policy, and calls the same evaluator in a fresh directory outside the source
run. It does not rerun generation, retrieval or conversion. It **can** spend
evaluation API calls. Incomplete scores remain incomplete (exit 2). This is a
final-state recovery utility, not a generation `--resume` feature or a new
SceneWeaver loop; the existing iteration API remains available for trajectories.

Current preparation evidence, remaining blockers and real-vs-mocked status are
recorded in [the runs work ledger](PIPELINE_COMPATIBILITY_RUNS_READINESS.md).

### Single controlled unit

```bash
layout-ddd-compare-generation \
  --generation-input generation_input.json \
  --protocol configs/generation_comparison/frozen_rectangular_v1.example.json \
  --asset-catalog configs/generation_comparison/synthetic_pilot_catalog_v1.json \
  --adapter layout_vlm \
  --adapter-config layout_vlm_runner.json \
  --out-dir run/layout_vlm_frozen \
  --run-generation
```

Replace the synthetic mesh paths/hashes before a real run. No real upstream
repository or full asset database is required by CI, and no real-upstream smoke
test is claimed by this implementation.
