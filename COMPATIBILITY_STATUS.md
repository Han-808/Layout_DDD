# External Harness Compatibility Status

## Compatibility invariant

An external harness is evaluator-compatible only through this path:

```text
native output -> deterministic converter -> canonical_scene_v1 -> run_evaluate()
```

The converter may change representation (coordinates, units, rotation encoding,
transform space, anchor convention, and schema fields). It must not move, rotate,
resize, replace, invent, or semantically reinterpret generated scene content.
Adapter provenance is audit metadata and is not an evaluator input signal.

## Fully exercised compatibility

The following adapters have end-to-end tests that materialize a native fixture,
validate the canonical scene, and run the unchanged canonical L0--L4 evaluator.

| Adapter | Supported scope | Asset identity source | Explicitly rejected |
| --- | --- | --- | --- |
| `direct_layout` | One axis-aligned rectangular room; native object OBB poses; optional exact mesh dereference | Native `asset_id`/`jid`, or persisted `new_object_id` binding | Multi-room output, nonrectangular request boundaries, architecture fields |
| `layout_vlm` | One axis-aligned rectangular room; optimized poses joined to its native asset table; OBB or exact mesh | `scene_config.assets[*].uid`/`asset_id` | Multi-room output, polygon boundaries, architecture fields, missing required asset geometry |
| `respace` | One axis-aligned rectangular room; Y-up SSR converted to canonical Z-up; native OBB | `sampled_jid`/`jid` | Multi-room output, polygon bounds, architecture fields |
| `scene_weaver` | One axis-aligned rectangular room; latest selected iteration; bottom-center converted to bbox-center | Native `asset_id`/`jid`, or explicit `asset_bindings` | Multi-room output, nonempty `structure`, walls/openings/topology, missing asset binding |

`LayoutGPT`, `Holodeck`, and `SceneSmith` also have converters, but they are not
part of this four-adapter full-compatibility claim.

## Asset-resolution policy

External canonicalization defaults to `asset_resolution_policy="exact_only"`.
In this mode, a provider may only resolve the persisted native/bound asset ID to
metadata or a mesh. It may not return a different ID, and `retrieve()` is never
called. A missing exact record can remain an OBB proxy only when the native
artifact already supplies all required geometry; otherwise conversion fails with
`ArtifactValidationError`.

`asset_resolution_policy="allow_retrieval"` is an explicit development/non-strict
option retained for workflows that intentionally perform semantic retrieval. It
is not the evaluation-compatible default. Production harness retrieval should
happen before conversion and persist the selected ID in the native artifact or a
binding artifact.

Asset databases connect through `asset_provider`, `asset_manifest`,
`asset_manifest_path`, or the existing `retrieval_runtime` bridge. Exact mode uses
only ID lookup (`resolve`).

## Semantic gates and current limits

Capabilities are checked as a subset relation between scene/evaluation
requirements and adapter support. The four adapters currently declare:

- room model: `single_room`
- boundary model: `axis_aligned_rectangle`
- architecture features: none
- geometry fidelity: `bbox` and optional exact mesh
- asset identity preservation: required

Point-cloud-only asset metadata is not promoted to evaluator geometry. A native
OBB remains the canonical proxy; without required OBB/mesh geometry, conversion
fails.

Multi-room and nonrectangular outputs are not flattened into this canonical
single-room path. The repository's separately selected nonrectangular evaluation
mode remains a distinct interface; these four adapters do not claim compatibility
with it. Walls, openings, and room topology are rejected when they cannot be
preserved rather than being stored only as metadata or silently discarded.

## Audit provenance

Canonical scenes retain adapter name, native schema, source artifact, coordinate
conversion, asset-resolution policy, native asset IDs where available, and each
object's geometry provenance. Tests also verify that changing adapter-name
provenance alone does not change the evaluator report.
