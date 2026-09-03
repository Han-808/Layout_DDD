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

## Fixture-tested evaluation compatibility

The following adapters have end-to-end tests that materialize a native fixture,
validate the canonical scene, and run the unchanged canonical L0--L4 evaluator.
Native representation assumptions were audited against the official upstream
repositories at DirectLayout `4430535`, LayoutVLM `85d06b4`, ReSpace `ce8e570`,
and SceneWeaver `7ae54b2`.

| Adapter | Native boundary | Native pose | Asset identity | Geometry source | Supported scope | Remaining caveats |
| --- | --- | --- | --- | --- | --- | --- |
| `direct_layout` | Benchmark input room; native output room geometry is rejected | Z-up bbox center and degree yaw | Native `asset_id`/`jid`, or persisted `new_object_id` binding | Native placed OBB after DirectLayout rescaling | One axis-aligned rectangular room | Mesh URI is reference-only; source does not persist a separate scale transform or unambiguous canonical front |
| `layout_gpt` | Benchmark input room | Released `left/top/depth` bbox center, unit normalization, degree yaw | Persisted generation-time `asset_ids` binding; exact lookup only | Released numerical placed bbox | One axis-aligned rectangular room and explicitly selected output record | Native asset binding and dataset/unit assumptions must be supplied; per-case upstream bridge is not shipped |
| `layout_vlm` | `scene_config.boundary`, required to match the benchmark room | Z-up bbox center and degree XYZ Euler/yaw | `scene_config.assets[*].uid`/`asset_id` | Asset-local bbox multiplied by explicit placement scale | One axis-aligned rectangular room | Polygon and architecture output are rejected; processed-asset +X front convention is recorded |
| `respace` | Y-up `bounds_bottom`/`bounds_top`, required to be planar, aligned, and benchmark-matching | Y-up bottom-center; released default is unitless XYZW quaternion; Euler/yaw require explicit encoding and unit | `sampled_asset_jid` first, then legacy `sampled_jid`/`jid` | SSR placed OBB; sampled asset bbox and native scale are separate audit fields | One axis-aligned rectangular room in the canonical evaluator path | Upstream rectilinear polygons are rejected because `canonical_scene_v1` remains rectangular; they are never flattened |
| `scene_weaver` | Native `roomsize`, required to match the benchmark room | World bbox bottom-center and Blender XYZ radians | Native `asset_id`/`jid`, or explicit `asset_bindings` | Native placed bbox dimensions | One axis-aligned rectangular room and one explicitly selected iteration | Directory conversion requires `selected_iteration`/`layout_path`; `latest` is explicit non-strict convenience only; nonempty structure is rejected |
| `holodeck` | ProcTHOR `floorPolygon` or selected SceneState floor; rectangle and benchmark-room match required | Unity XYZ converted to canonical Z-up, or SceneState transform | Native `assetId` or SceneState `modelId`; exact lookup only | Native/provider bbox; SceneState applies the native transform | One rectangular room, including an explicitly selected per-room projection | Compatibility only, not whole-house topology evaluation; raw Unity rotation assumptions remain; no real upstream smoke test |
| `scene_smith` | Native `room_geometry` or selected SceneState floor; rectangle and benchmark-room match required | WXYZ quaternion and object-frame origin plus rotated local bbox center, or SceneState transform | Native/generated asset ID or SceneState `modelId`; exact lookup only | Native local bbox and transform, or SceneState/provider bbox | One rectangular room, including an explicitly selected per-room projection | Compatibility only; missing geometry may use the public room contract and is labeled as such; no real upstream smoke test |

LayoutGPT also has mocked executable-to-evaluator coverage. Holodeck and
SceneSmith now have room-contract and full-evaluator tests on both their raw and
shared SceneState import routes in `tests/test_external_room_contracts.py`.
These fixtures certify only the declared import scope, not production readiness
of an upstream generator or every possible native geometry/transform variant.

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
requirements and adapter support. All seven external adapters currently declare:

- room model: `single_room`
- boundary model: `axis_aligned_rectangle`
- architecture features: none
- geometry fidelity: `bbox` and optional exact mesh
- asset identity preservation: required

Point-cloud-only asset metadata is not promoted to evaluator geometry. A native
OBB remains the canonical proxy; without required OBB/mesh geometry, conversion
fails.

A mesh URI alone also does not make evaluator geometry an `asset_mesh`. For
DirectLayout, LayoutVLM, ReSpace, and SceneWeaver, canonical `object.size` and
`asset_proxy.bbox_size` are the
audited evaluated OBB. When available, asset-local bbox, local bbox offset,
native scale, and resulting evaluated dimensions are stored separately under
`metadata.geometry_audit`.

For bottom-center formats (ReSpace and SceneWeaver), the local half-height
offset is transformed by the resolved native rotation matrix before producing
canonical `object.center`; it is not assumed to be a world-Z-only offset.

Multi-room and nonrectangular outputs are not flattened into this canonical
single-room path. The repository's separately selected nonrectangular evaluation
mode remains a distinct interface; these adapters do not claim compatibility
with it. Walls, openings, and room topology are rejected when they cannot be
preserved rather than being stored only as metadata or silently discarded.

Native room geometry is compared with the benchmark room modulo deterministic
origin translation, cyclic vertex start, and winding. Dimension, height,
top/bottom footprint, multi-room, and unsupported topology conflicts fail before
canonical output is written.

Holodeck, SceneSmith, and their shared SceneState path perform the boundary-model
and room-match checks before asset lookup. Invalid/partial native SceneSmith
dimensions and invalid heights fail closed; zero height does not mean "use the
benchmark height." Multiple floors remaining after `room_id` selection and
inconsistent wall heights are rejected instead of dropping geometry. Explicit
per-room projection does not claim multi-room/topology support. Provenance
records `source_boundary`, `source_height`, and the room-match check; if a native
format omits the room/height, the existing public-input fallback is explicitly
labeled `generation_input` rather than presented as observed native geometry.

SceneWeaver directory inputs are strict by default: select an iteration with
`selected_iteration: N` or provide `layout_path`. Every discovered
`layout_0 ... layout_K` can therefore be canonicalized and evaluated
independently. `iteration_selection_policy: latest` must be requested explicitly.

## Executable integration status

The status terms below are intentionally distinct. `YES` means the existing
strict converter is evaluation-compatible for its declared scope.
`IMPLEMENTED_NOT_REAL_SMOKE_TESTED` means the execution boundary is implemented
and exercised with fake upstream repositories in the test suite, but this
repository has not run the actual external project. **It does not mean a clean
upstream checkout is executable without additional bridge code.**
`COMPATIBILITY_ONLY` means Mode A conversion
is retained without a configured Mode B execution profile.

| Adapter | Evaluation compatibility | Executable integration | Real upstream smoke tested | Loop-state support | Asset protocol | Controlled-comparison readiness |
| --- | --- | --- | --- | --- | --- | --- |
| `layout_gpt` | YES | IMPLEMENTED_NOT_REAL_SMOKE_TESTED | NO | NOT_IMPLEMENTED | Native LayoutGPT layout plus persisted generation-time `asset_ids` binding | SHARED_DB/FROZEN_ASSETS CONDITIONAL on runner-declared controls and post-run gates |
| `direct_layout` | YES | IMPLEMENTED_NOT_REAL_SMOKE_TESTED | NO | NOT_IMPLEMENTED | Native asset-library IDs / `new_object_id` and DirectLayout asset-layout workflow | SHARED_DB/FROZEN_ASSETS CONDITIONAL on runner-declared controls and post-run gates |
| `layout_vlm` | YES | IMPLEMENTED_NOT_REAL_SMOKE_TESTED | NO | NOT_IMPLEMENTED | Native scene config, Objaverse asset table, and optimized layout | FROZEN_ASSETS implemented and mock-tested; SHARED_DB selection wrapper CONDITIONAL |
| `respace` | YES | IMPLEMENTED_NOT_REAL_SMOKE_TESTED | NO | NOT_IMPLEMENTED | Native ReSpace sampling; selected `sampled_asset_jid` remains in SSR | SHARED_DB/FROZEN_ASSETS CONDITIONAL on native cache/sampling controls |
| `scene_weaver` | YES | IMPLEMENTED_NOT_REAL_SMOKE_TESTED | NO | YES | Native/procedural assets plus persisted generation-time `asset_bindings` | SHARED_DB CONDITIONAL; FROZEN_ASSETS requires inventory/asset locks across all iterations |
| `holodeck` | YES | COMPATIBILITY_ONLY | NO | NOT_IMPLEMENTED | Existing ProcTHOR/Objathor or SceneState conversion | CONDITIONAL |
| `scene_smith` | YES | COMPATIBILITY_ONLY | NO | NOT_IMPLEMENTED | Existing generated/native asset state conversion | CONDITIONAL |

No row claims that the five executable methods are already comparable under a
real shared or frozen asset database without the listed eligibility conditions.
The versioned generation-side protocol, catalog materializers, validators, and
mocked five-method route are documented in
[`docs/GENERATION_COMPARABILITY_PROTOCOL.md`](docs/GENERATION_COMPARABILITY_PROTOCOL.md).
This does not claim a real-upstream controlled smoke test.

The executable-integration column above describes benchmark orchestration.
The actual release-to-request transport gaps are separate:

| Method | Released entrypoint/interface | Per-case bridge shipped here | Clean-checkout blocker |
| --- | --- | --- | --- |
| LayoutGPT | `run_layoutgpt_3d.py` dataset/batch workflow | NOT_IMPLEMENTED | Operator bridge must provide the benchmark case to the released workflow and persist true generation-time asset bindings |
| DirectLayout | `demo.py` numerical-layout batch CLI | Direct CLI configuration only | Upstream environment/assets/API configuration and a real smoke run; the FrozenAssets pilot still needs a verified control bridge |
| LayoutVLM | `main.py` scene-config CLI | Direct CLI configuration only | Processed assets, model/API environment, and a real smoke run |
| ReSpace | Released Python SSR generation/sampling API | NOT_IMPLEMENTED | Operator bridge, model/checkpoint/cache, and native sampled asset decisions |
| SceneWeaver | `Pipeline/main.py` native loop | NOT_IMPLEMENTED | Operator bridge, Blender/Infinigen/assets, complete trajectory capture, and exact native bindings |

No method in this table has met the real-upstream publication gate in this
repository. A fake subprocess or hand-authored native fixture is not that gate.

## Runner architecture and artifacts

Mode A remains available with `--method-output`. The supplied file or directory
is copied byte-for-byte into `generator/native_artifacts/primary` before the
existing converter reads it. Mode B uses `--run-generation` with exactly one of:

- `adapter_config.runner` for an injected plugin callback;
- `adapter_config.raw_output_path` for a configured existing native artifact;
- `adapter_config.execution.command` for a shell-free subprocess.

Subprocess commands use string-list templates and may reference
`{python_executable}`, `{repo_path}`, `{method_input}`, `{native_input}`, and
`{upstream_output_dir}`. Additional non-secret values belong in
`execution.template_variables`. A whole-token `{env:NAME}` reads an inherited or
overridden environment variable while persisting only a redacted marker.

Each run writes:

```text
generator/method_input.json
generator/native_input/<method-specific input>.json
generator/execution/runner_config.json
generator/execution/execution_result.json
generator/execution/stdout.txt
generator/execution/stderr.txt
generator/execution/native_artifact_manifest.json
generator/native_artifacts/primary/<unaltered native artifact>
generator/native_artifacts/auxiliary/<named sidecars, when configured>
generated_scene.json
adapter_metadata.json
evaluation_report.json                 # when run through the scene harness
```

The execution result records command, cwd, redacted environment overrides,
return code, timestamps/runtime, timeout state, upstream checkout and discoverable
Git commit, source and preserved native paths, SHA-256, auxiliary artifacts, and
the resulting canonical scene. The native artifact hash is checked once after
copying and again after conversion. A converter never receives the mutable
upstream output path.

`execution_result.json.runner_provenance` also fingerprints the actual Python
script or inspectable callback source **before** execution. It records the
entrypoint path, SHA-256, discoverable Git commit, whether the source file is
tracked/modified, and its hash again after execution. Relative script paths are
resolved against the configured upstream `cwd`, not the benchmark checkout.
This distinguishes an uncommitted `integration_entrypoint.py` from the released
upstream commit. Only the entrypoint file is fingerprinted, not its transitive
dependencies; this is identity evidence, never control-verification evidence.
Undiscoverable entrypoints (for example `python -m ...`) remain `NOT_DISCOVERED`;
offline artifacts are `NOT_EXECUTED`. Source content/secrets are not copied into
this provenance record.

Asset selection remains a generation responsibility. LayoutGPT may expose an
`asset_ids` sidecar and SceneWeaver an `asset_bindings` sidecar through
`execution.auxiliary_artifacts`; these sidecars are copied and hashed before the
strict converter loads them. LayoutVLM preserves its public native scene config
and asset table as the method-specific input. ReSpace persists upstream
`sampled_asset_jid` choices directly in the SSR.

## Real-upstream configuration examples

Execution-contract templates live under
`configs/external_harness_execution/`. All paths are placeholders and must be
set explicitly; no upstream checkout is vendored or assumed.
Where an official project lacks a per-case file CLI, `integration_entrypoint.py`
means an operator-maintained shim that **is not included here or in the released
upstream checkout**. Replacing placeholder paths does not implement it. It may translate
the request-file transport into official API calls and persist their true return
artifact; it must not canonicalize, repair, resample, or fabricate benchmark
output.

| Adapter | Official checkout / prerequisites | Native artifact consumed here |
| --- | --- | --- |
| `layout_gpt` | [UCSB-AI/LayoutGPT](https://github.com/UCSB-AI/LayoutGPT); its documented Python/ATISS environment, prepared 3D-FRONT/3D-FUTURE data, and upstream model credentials | Released 3D output JSON record/list; exact selected-asset sidecar for strict conversion |
| `direct_layout` | `https://github.com/rxjfighting/DirectLayout`; Python 3.10 environment, configured reasoning API, and its scene asset library | One `output_layout/<room>.json` numerical layout |
| `layout_vlm` | `https://github.com/sunfanyunn/LayoutVLM`; processed Objaverse assets and `OPENAI_API_KEY` | `layout.json`, paired with the exact native scene-config/asset-table input |
| `respace` | `https://github.com/GradientSpaces/respace`; documented Python/CUDA environment, checkpoints/HF access, 3D-FUTURE assets and its preprocessing cache | `scene.json` SSR after native asset sampling, including `sampled_asset_jid` |
| `scene_weaver` | `https://github.com/Scene-Weaver/SceneWeaver`; SceneWeaver and Infinigen/Blender environments plus its configured LLM and asset backends | Complete scene output root with `record_scene/layout_N.json`, emitted render/state artifacts, and exact bindings |

- **LayoutGPT:** official `run_layoutgpt_3d.py` is batch/dataset-oriented. For an
  arbitrary benchmark case, configure an upstream-local thin entrypoint that
  consumes `layoutgpt_request.json`, calls the official LayoutGPT workflow, and
  writes the released JSON record/list plus an `asset_ids.json` sidecar. Expected
  primary artifact: released LayoutGPT JSON accepted by the existing converter.
  The upstream batch reference command is `python run_layoutgpt_3d.py
  --dataset_dir ... --icl_type k-similar --K 8 --room bedroom --llm_type gpt4
  --unit px --normalize --regular_floor_plan`; it is not silently substituted for
  the per-case wrapper.
- **DirectLayout:** the generated native input is the released two-list batch
  input. The example invokes `demo.py --input ... --output-dir ... --assets-dir
  ... --render-dir ...`. Expected artifact: exactly one numerical layout JSON.
- **LayoutVLM:** the generated native input is its scene config, including the
  room boundary and asset table. The example invokes `main.py
  --scene_json_file ... --save_dir ... --model ... --openai_api_key ...
  --asset_dir ...`. Expected artifact: `layout.json`; the preserved scene config
  remains the converter's asset table.
- **ReSpace:** the official interface is a Python API rather than a frozen
  per-case CLI. Configure an upstream-local entrypoint that reads
  `respace_request.json`, invokes the official ReSpace generation and asset
  sampling path, and persists `scene.json` with every `sampled_asset_jid`.
- **SceneWeaver:** configure an upstream-local entrypoint around the official
  native loop. It must preserve the complete output root, including every
  `record_scene/layout_N.json` and emitted render/state artifact, plus an exact
  asset-binding sidecar when native layout objects lack IDs. The upstream
  reference invocation is `cd Pipeline && python main.py --prompt ... --cnt 1
  --basedir ...`; the configured wrapper only adapts the public request file and
  artifact paths, not the reflection policy.

Example benchmark invocation:

```bash
layout-ddd-generate \
  --generation-input generation_input.json \
  --adapter direct_layout \
  --adapter-config configs/external_harness_execution/direct_layout.example.json \
  --out-dir run/direct_layout \
  --run-generation

layout-ddd-evaluate \
  --scene run/direct_layout/generated_scene.json \
  --out run/direct_layout/evaluation_report.json
```

These examples exercise/configure the implemented benchmark execution contract;
they are not complete per-case upstream bridges or records of real smoke tests.

Before claiming a paper baseline is executable, preserve evidence for:

```text
fresh pinned upstream checkout + versioned bridge (if needed)
  + documented dependencies/assets/model configuration
  + one real public benchmark case
  -> unmodified native artifact and generation-time asset decisions
  -> existing strict converter
  -> unchanged full evaluator report (including coverage/completion status)
```

Retain the upstream and bridge commits/hashes, installed dependency versions,
public input, execution result/logs, native artifact hashes, converter metadata,
and evaluator configuration/report. Successful conversion with an incomplete
evaluator report is not a successfully scored benchmark. SceneWeaver additionally
needs every native iteration, evaluated post-hoc with no official feedback.

## SceneWeaver native trajectory evaluation

The executable runner preserves the entire native trajectory directory and
inventories every discovered `layout_N.json` with its own hash and related
artifacts. Evaluate it offline with:

```bash
layout-ddd-evaluate-sceneweaver-iterations \
  --native-output run/scene_weaver/generator/native_artifacts/primary/SCENE_ROOT \
  --generation-input generation_input.json \
  --adapter-config sceneweaver_conversion_config.json \
  --out-dir run/scene_weaver/iteration_evaluation
```

Each layout is passed independently through the existing `scene_weaver`
converter and unchanged canonical `run_evaluate()` route. The resulting
`iteration_summary.json` references each native layout and hash, canonical scene,
converter metadata, evaluation report, canonical benchmark score, and adjacent
score delta. The utility hashes the whole native trajectory before and after
evaluation.

Official benchmark reports are never fed into the SceneWeaver process. The
adapter rejects benchmark self-reflection input; this API evaluates only states
already emitted by SceneWeaver's own reason-act-reflect loop.

## Audit provenance

Canonical scenes retain adapter name, native schema, source artifact, coordinate
conversion, asset-resolution policy, native asset IDs where available, and each
object's geometry provenance. Tests also verify that changing adapter-name
provenance alone does not change the evaluator report.

## Automated regression evidence

`.github/workflows/external-harness-compatibility.yml` defines a credential-free
Python 3.11/3.13 gate for pull requests and the pipeline-refinement branch.
It installs only `.[dev]`, runs the converter/IO/execution/comparison/pilot suites
plus the relevant generation/evaluation harness tests, and uploads JUnit results,
the exact checked-out commit, and installed dependency versions. Upstream repos,
large datasets, Blender, and model services are not CI prerequisites. GitHub
Actions are pinned to verified commit SHAs and the workflow has read-only
repository permissions.

The reviewed baseline `89440dfac38d44a53671e3ecf4fd31746bfa46da` had no attached
combined status or PR workflow runs when checked. Adding this workflow is not
retroactive CI evidence: require a successful run for the exact eventual PR/head
commit before merge. This focused gate is not a claim that every unrelated test
in the repository passes.

PR #4 also includes earlier evaluator/generation work beyond these adapters.
Updating its description or splitting/rebasing it requires a separate scope
decision; this refinement does not rewrite branch history or change scoring.
