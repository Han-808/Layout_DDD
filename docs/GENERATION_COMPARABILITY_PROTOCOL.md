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

The first concrete backend is `load_3d_future_subset_catalog()`. It normalizes an
explicitly selected 3D-FUTURE subset and optionally hashes local mesh bytes. It
does not search, rank, or retrieve records. This is a pilot backend, not a
3D-FUTURE dependency in the comparison architecture. The synthetic checked-in
catalog is for CI and configuration examples only.

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

Wall time is always recorded. Model, token, generation-call, tool-call,
retrieval-call, rendering-call, and iteration counts are recorded when reported
by the upstream runner. `method_native_recorded` deliberately does not claim
that structurally different workflows received identical token/tool budgets;
quality and cost remain separate outcomes.

## SceneWeaver trajectories

The complete native SceneWeaver directory remains the source of truth. Every
`layout_0.json ... layout_K.json` is independently converted with the existing
SceneWeaver converter, evaluated with the same `run_evaluate()`, and checked
against the same comparison protocol. A FrozenAssets trajectory is valid only
when every iteration retains the same slots, exact asset IDs, and fixed physical
dimensions.

The benchmark report is post-hoc. Neither scores nor evaluator error details are
placed in the method input or fed into SceneWeaver's native reflection loop.

## Invocation

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
