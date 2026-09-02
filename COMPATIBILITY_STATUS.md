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
Native representation assumptions were audited against the official upstream
repositories at DirectLayout `4430535`, LayoutVLM `85d06b4`, ReSpace `ce8e570`,
and SceneWeaver `7ae54b2`.

| Adapter | Native boundary | Native pose | Asset identity | Geometry source | Supported scope | Remaining caveats |
| --- | --- | --- | --- | --- | --- | --- |
| `direct_layout` | Benchmark input room; native output room geometry is rejected | Z-up bbox center and degree yaw | Native `asset_id`/`jid`, or persisted `new_object_id` binding | Native placed OBB after DirectLayout rescaling | One axis-aligned rectangular room | Mesh URI is reference-only; source does not persist a separate scale transform or unambiguous canonical front |
| `layout_vlm` | `scene_config.boundary`, required to match the benchmark room | Z-up bbox center and degree XYZ Euler/yaw | `scene_config.assets[*].uid`/`asset_id` | Asset-local bbox multiplied by explicit placement scale | One axis-aligned rectangular room | Polygon and architecture output are rejected; processed-asset +X front convention is recorded |
| `respace` | Y-up `bounds_bottom`/`bounds_top`, required to be planar, aligned, and benchmark-matching | Y-up bottom-center; released default is unitless XYZW quaternion; Euler/yaw require explicit encoding and unit | `sampled_asset_jid` first, then legacy `sampled_jid`/`jid` | SSR placed OBB; sampled asset bbox and native scale are separate audit fields | One axis-aligned rectangular room in the canonical evaluator path | Upstream rectilinear polygons are rejected because `canonical_scene_v1` remains rectangular; they are never flattened |
| `scene_weaver` | Native `roomsize`, required to match the benchmark room | World bbox bottom-center and Blender XYZ radians | Native `asset_id`/`jid`, or explicit `asset_bindings` | Native placed bbox dimensions | One axis-aligned rectangular room and one explicitly selected iteration | Directory conversion requires `selected_iteration`/`layout_path`; `latest` is explicit non-strict convenience only; nonempty structure is rejected |

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

A mesh URI alone also does not make evaluator geometry an `asset_mesh`. For the
four adapters above, canonical `object.size` and `asset_proxy.bbox_size` are the
audited evaluated OBB. When available, asset-local bbox, local bbox offset,
native scale, and resulting evaluated dimensions are stored separately under
`metadata.geometry_audit`.

For bottom-center formats (ReSpace and SceneWeaver), the local half-height
offset is transformed by the resolved native rotation matrix before producing
canonical `object.center`; it is not assumed to be a world-Z-only offset.

Multi-room and nonrectangular outputs are not flattened into this canonical
single-room path. The repository's separately selected nonrectangular evaluation
mode remains a distinct interface; these four adapters do not claim compatibility
with it. Walls, openings, and room topology are rejected when they cannot be
preserved rather than being stored only as metadata or silently discarded.

Native room geometry is compared with the benchmark room modulo deterministic
origin translation, cyclic vertex start, and winding. Dimension, height,
top/bottom footprint, multi-room, and unsupported topology conflicts fail before
canonical output is written.

SceneWeaver directory inputs are strict by default: select an iteration with
`selected_iteration: N` or provide `layout_path`. Every discovered
`layout_0 ... layout_K` can therefore be canonicalized and evaluated
independently. `iteration_selection_policy: latest` must be requested explicitly.

## Audit provenance

Canonical scenes retain adapter name, native schema, source artifact, coordinate
conversion, asset-resolution policy, native asset IDs where available, and each
object's geometry provenance. Tests also verify that changing adapter-name
provenance alone does not change the evaluator report.
