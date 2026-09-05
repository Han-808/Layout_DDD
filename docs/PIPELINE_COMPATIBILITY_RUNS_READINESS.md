# Pipeline compatibility runs: API-ready work ledger

## Fixed experiment boundary

Reviewed baseline: `dfaba4e5cd5c88757df54686f1cc8f1bce3e95ba` (verified against
GitHub main on 2026-09-05). Development is isolated on
`codex/pipeline-compatibility-runs`; the shared desktop checkout is not modified.

The user approved the ten preparation decisions on 2026-09-05. This is **fixed
asset input**, not a retrieval experiment: S100--S109, ten rectangular single
rooms, 269 expanded slots, 168 approved Imaginarium assets. Exact slot bindings,
meshes, local bbox, native scale, and available canonical front remain frozen.
No replacement, selection, insertion/removal, rescaling, or converter repair.

Methods: LayoutGPT (adapted frozen-ICL workflow), DirectLayout, LayoutVLM,
SceneWeaver, plus an independently reported Catalog Placement Stage-C baseline.
The four external harnesses must share the actual backing model/deployment.
Native reasoning, optimization, and reflection budgets remain method-native;
record cost rather than claiming equal cost. SceneWeaver's final complete native
state is selected, never its highest evaluator score; all states are evaluated
post-hoc without official feedback to generation.

Formal plan: 3 independent repetitions per method/case (150 planned generation
units if all five pass qualification), after one-case smoke and a denser case
check. Pro's single-repetition 40+10 checklist is expanded to the user's approved
three repetitions. Pilot artifacts have separate identities from formal results.

## Goal and approval boundary

Prepare the complete runtime, immutable inputs/assets, thin upstream interfaces,
offline gates and API configuration templates. No paid API calls or formal
generation in this preparation goal. `run --dry-run-only` **does execute real
generation**; it is not a credential-free preflight. Actual service identity and
full real upstream smoke remain explicit post-credential acceptance gates.

Pro feedback is source review, not dynamic evidence. Its pipeline findings
F2/F3/F4/F6 are in scope. F1/F5/F7 belong to Complicated_Generation and are not
implemented here. Evaluator metrics, scoring weights, hidden reference/evidence
policies and converters are not redesigned.

## Required evidence (unchecked means not yet established)

- [x] F2 wiring: production evaluator runtime constructed before generation
      readiness; per-scene render/Judge/grouping/camera wiring uses unchanged
      run_evaluate. Tested with the actual evaluator and mocked external I/O.
- [ ] F2 production acceptance: a complete real evaluator report. Honest frozen
      ownership currently makes pairing/style not relevant, while the unchanged
      scoring denominator retains those terms: observed L3 coverage 0.84 and
      `partial_coverage`. Preflight blocks paid generation on this known issue.
      No ownership relabeling, metric removal, or score override was introduced.
- [x] F3: prepared input, object plan, protocol, catalog and evaluator hashes
      revalidated before any generation, including later cases.
- [x] F4: cancellation propagates, child process group reaped, logs retained,
      no subsequent scheduled task launched.
- [x] F6: every planned unit has a terminal status; blocked/partial/cancelled
      are distinct from complete; zero attempts cannot exit successfully.
- [x] Current focused compatibility/execution and new regression tests run.
      Latest exporter-focused: 64 passed; input-factory contract: 15 passed.
      Extended: 541 passed, one baseline-existing failure;
      this is not an all-green full-suite certification (see evidence below).
- [x] Fresh wheel installation: 11 entrypoints imported/`--help` passed, 12
      selected schemas/prompts/render workers checked, new ICL module imported,
      and mesh-collision dependencies imported outside the source checkout.
- [x] Approved spec/inventory/source audit: ten cases, 269 slots, 168 assets;
      source CSV, FBX and catalog metadata hashes verified. No selection changes.
- [x] Public brief versus approved inventory conflict audit recorded. The user
      additionally approved actual-asset functional-description corrections,
      not just counts. Revision v2 retains all geometry, slots and asset bindings;
      it is explicitly a new public-input treatment (details below).
- [x] 168-asset GLB bundle built and source/hash/bbox/center/scale verified.
- [x] Four upstream checkouts pinned/clean; bridge bundle hashes checked.
- [ ] All upstream production dependencies qualified. LayoutGPT and DirectLayout
      local imports work; LayoutVLM imports but its CUDA overlap extension is
      unavailable locally. SceneWeaver's Linux runtime/plugin are not qualified.
- [x] Released-derived LayoutGPT ICL snapshot with source/selection/leakage audit;
      eight training-only examples, byte-identical independent message rebuild.
- [ ] Actual SceneWeaver frozen initializer/export plugin preserving native loop,
      or evidence-backed ineligibility (never substitute a fake implementation).
- [ ] Method-native geometry invariants and real asymmetric/near-symmetric pose
      round-trips. DirectLayout/LayoutVLM native mesh assembly: 24/24 captures
      now match conversion, including asymmetric/offset/near-square meshes.
      SceneWeaver exact exporter diagnostics now match all 16 captured bpy
      states; frozen initialization and actual renderer/optimizer/loop E2E
      acceptance remain separate, unresolved gates.
- [ ] Private API templates, no-call readiness command, smoke/full launch plans,
      append-only offline reevaluation path and budget/failure policy documented.
- [ ] Final requirement-by-requirement evidence audit; per-method status accurate.

## Evidence log

- 2026-09-05: read both Pro attachments and shared implementation guide in full.
  Verified remote main SHA and created isolated worktree from the reviewed SHA.
  Inspected existing pilot/execution/converter tests and evaluator runtime
  factories. No real upstream/API run performed.

## Current evidence and remaining runtime limits

The implementation and diagnostics in this checkpoint are **not** a claim that
production generation is ready. No production model service or generation
workflow has been executed. Real upstream code executed locally was limited to
imports, LayoutGPT's released formatter/parser preparation, a small LayoutVLM
overlap-gradient diagnostic, DirectLayout/LayoutVLM native mesh assembly stopped
before pixel rendering, and exact SceneWeaver exporter functions on synthetic
bpy objects. No benchmark score was fed to a generator.

| Method | Pinned upstream commit | Local prerequisite evidence | Remaining gate |
| --- | --- | --- | --- |
| LayoutGPT | `fc31954962553e5b65bf267a904a6930d50b1f5e` | Eight public `rect_train` demonstrations prepared using the release's metric formatter; source hashes and disjoint train/test/val selection checked | Configure audited ICL identity in a new prepared protocol; production response identity and native-parser smoke |
| DirectLayout | `4430535304124dbc8d48f6bb0ea5891e92267986` | Native pipeline imports; 12 actual mesh assemblies match the corrected clockwise-yaw converter, in CPython 3.11.15 / bpy 4.3.0 / NumPy 1.26.4 | Real rendering/refinement and model/API smoke |
| LayoutVLM | `85d06b4cd2478551188a0b4a47cd658c85c41315` | Native solver imports; 12 actual mesh assemblies confirm the existing +90-degree basis; CPU fallback probe gives positive loss but both corner gradients are null | Build/verify native differentiable Rotated_IoU on compatible CUDA host; no silent CPU-fallback ablation |
| SceneWeaver | `7ae54b2ec3fc66147704faa7daf7b017ba8b1bd9` | Native loop/factories inspected; exact exporter AST executed on real bpy objects in Blender 4.3/5.2; all 16 captured states match the corrected frozen conversion | Full frozen initialization/export plugin and Linux runtime; update_layout/update_rotation both invoke update_graph, which rescales and deletes, so disabling named resize/remove tools alone is insufficient |
| Catalog Placement | This benchmark checkout | Existing fixed-selection Stage-C baseline and offline same-evaluator tests retained | Production Stage-C/API and full evaluation smoke, separately reported |

All four upstream checkouts were clean after these inspections/imports. Source
archives, environments, downloaded NPZs, GLBs and reports stay under the ignored
`Support/pipeline_compatibility_runs/api_ready_v1/` prefix in the desktop data
checkout, never in the code commit.

### Asset and ICL outputs

- GLB `glb_bundle_attempt3`: Blender 5.2.0 LTS, 168/168 passed. The refrigerator
  source includes 48 loose vertices; `use_mesh_edges/use_mesh_vertices` preserves
  them. The near-square office chair uses the existing 1e-4 metre tolerance
  for axis-order classification. No source mesh, asset choice or scale changed.
  Failed attempt directories/logs remain intact.
- `verified_glb_preflight`: catalog
  `7716a5e313f98708a9c169cd70bc742f40641fee6be5400de992576ff177ead8`;
  prepared identity
  `a9dc1eb153aa6392c9a486a1e90a5df060d71866fe708f3d22266449a5532070`.
  This is preparation evidence, not a generated experiment. It has no production
  evaluator runtime and is not retroactively modified to add one.
- `layoutgpt_icl_snapshot_v1/messages.json`: SHA-256
  `1209a0b3e5adfeaeb627cbd605d6d2c0d5e8c907d9ba9282fe3bb42e85787386`;
  `layoutgpt_icl_snapshot_rebuild_v1/messages.json` is byte-identical. Official
  ZIP size was 1,156,127,563 bytes; only index/selected member byte ranges were
  read. No whole-archive SHA is claimed. Individual NPZ/statistics hashes are
  pinned by `layoutgpt_icl_recipe_v1.json`.

### Current test and installed-artifact evidence

Commands below ran in the isolated code checkout using Python 3.13.14. Each
pytest invocation used its own `--basetemp` and JUnit file under
`/private/tmp/pipeline-runs-final-regression.zxKf1s`; the other task uses different
temporary paths.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -o addopts='' -q \
  tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py \
  tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py \
  tests/test_frozen_imaginarium_scene10.py
# 215 passed (pro_focused_tests.xml)

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -o addopts='' -q \
  tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py \
  tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py \
  tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py \
  tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py \
  tests/test_scene_harness.py tests/test_catalog_placement_adapter.py \
  tests/test_external_converter_correctness.py tests/test_external_room_contracts.py \
  tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py \
  tests/test_evaluation_mode_interface.py tests/test_submission_api.py
# 463 passed, 1 failed (checkpoint3_tests.xml)
```

The failure is
`test_canonical_l3_group_judge_repair_reaches_vlm_and_renders`: expected one VLM
selector request, observed zero. It was independently reproduced in an untouched
`dfaba4e` worktree; evidence is in
`/private/tmp/pipeline-runs-regression.FiYPaQ/baseline.xml`. It was not weakened,
suppressed, or fixed by changing evaluator behavior in this task.

Clean wheel checkpoint3 SHA-256:
`c1e0862624129a7c7e18ee21df23cfb29ada3b4758d52528f4dd2f507fce6a58`.
It was installed with `mesh-collision` extras into a new isolated environment;
verification ran from `/private/tmp` without checkout `PYTHONPATH`. The report
is `wheel3_verification.json` in the current evidence directory. Upstream bridge
scripts, operator configs and raw assets remain explicit checkout/external
inputs, not implicitly promised package resources.

Recovery is append-only: `layout-ddd-reevaluate-controlled` verifies preserved
generation/native/canonical/evaluator identities, creates a new output location,
and does not regenerate or reconvert. It can spend Judge/camera calls and is not
a no-call preflight. `layout-ddd-controlled-pilot preflight` is the no-call gate.

Next: qualify SceneWeaver's thin frozen boundary and its native geometry export,
finish private runtime bindings and versioned smoke/three-repeat operator plans;
resolve complete-score applicability without silently altering the evaluator.
After combining with Complicated_Generation, certify a new shared commit/release
identity rather than treating this checkpoint's wheel/tests as that merged build.

### Public brief v2 and case-subset preparation checkpoint

After being shown concrete catalog/brief mismatches (S109 treadmill→ladder,
yoga blocks→inflatable chairs; S106 bath mat→bathtub), the user explicitly
approved updating public descriptions to the actual frozen assets while keeping
assets and slots unchanged. This is not a new asset-selection decision.

- Added an exact source-hash-pinned revision recipe and preparation-only
  `benchmark.generation_comparison.public_brief` command. It produces a fresh
  spec and complete before/after audit, including duplicate metadata text.
  All 269 slots / 168 assets / ten room geometries and the evaluator policy
  remain identical. No hidden annotations, layout answers, retrieval or model
  calls are used. The approved catalog descriptions are not newly certified
  visual truth; unchanged geometric/support feasibility is still evaluated.
- Revised spec canonical SHA-256:
  `2d0328d8ef016c430e8655cda183898b578a6faf3fbde72a5072a678f52c9e2b`.
  Files: ignored `api_ready_v1/public_brief_v2/{spec,public_brief_audit}.json`.
  Old specs and prepared directories are unchanged. Do not claim identical
  original SceneBoard prompts, especially for the revised S109 functional brief.
- Added `prepare --case-id` to select an existing case/subset without modifying
  the full source spec or its asset catalog. Selection is source-ordered and
  byte-pinned before launch. The first selected case remains the full-evaluation
  gate for subsequent selected cases; existing run/cancellation behavior stays.
- Actual no-call preparations: `public_brief_v2_smoke_preflight` (S100, 31 slots)
  and `public_brief_v2_dense_preflight` (S101, 32 slots), each retaining the same
  168-asset catalog and all five method entries. S100 no-call preflight returned
  exit 2 / ready=false as expected with example bindings and no runtime. Neither
  directory is a real smoke result or a final published runtime release.
- Predeclared schedule documented: S100 smoke, S101 dense pilot, then three
  independent fresh ten-case formal prepares (150 formal units if qualified).
  S101 selection uses slot count/density, not score. No automated promotion,
  formal execution, seed-equivalence claim or resume workaround was introduced.
- SceneWeaver audit additionally located automatic rescaling during
  `populate_state_placeholders_mid`, collision-based deletion and unsupported
  object deletion inside `compose_indoors`, beyond the named update tools.
  No native physics/reflection code was removed to conceal those conflicts;
  the full frozen plugin and Linux runtime remain unresolved prerequisites.

Tests in `/private/tmp/pipeline-runs-brief-plan.A5m3SZ`, same source Python and
command prefix as above, with unique `--basetemp` and JUnit paths:

- `tests/test_controlled_generation_pilot.py tests/test_pipeline_evaluation_runtime.py`:
  **50 passed** (`subset_tests.xml`).
- Above plus `tests/test_frozen_public_brief.py`: **62 passed** (`brief_tests.xml`).
- Pro's five focused files listed above: **224 passed** (`focused_tests.xml`).
- The previous sixteen-file extended command plus `tests/test_frozen_public_brief.py`:
  **484 passed, 1 failed** (`extended_tests.xml`), exactly the same baseline-existing
  camera-selector failure. No test/evaluator weakening to make it disappear.

Fresh wheel SHA-256 for this newer source:
`7bba00911784d4d3c6dce393ceee199df18b721fa55f7c9a14e6f579cb6a6645`.
Built/installed offline into a new `benchmark-wheel-brief-v2` environment.
All 11 entrypoints and 12 selected resources passed the existing installed-wheel
check from `/private/tmp` without checkout `PYTHONPATH`. The installed
`prepare --help` includes `--case-id`; the installed public-brief module produced
a byte-identical revised spec. Evidence: `wheel_verification.json`,
`installed_public_brief_report.json` and `installed_public_brief/` under the
current evidence directory. This is still no-call preparation certification,
not a real upstream or production evaluator smoke result.

### Actual native mesh-assembly checkpoint

Starting commit: `39ff0dff8bfdbd8609e89cb4ab66a889a4f70fac`. No upstream
checkout or evaluator source was edited. The concrete converter bug below was
found by executable upstream geometry, not inferred from a formula/comment.

- Actual `DirectLayout/utils/assemble.py::build_and_render_scene` and
  `LayoutVLM/utils/blender_render.py::render_existing_scene` assembled three
  approved GLBs (dresser `0_Chest_of_drawers`, near-square office chair
  `e_conference_1_27`, refrigerator `a_SM_Kitchen_Refrigerator`) plus a separate
  asymmetric diagnostic mesh with local bbox center `[0.31,-0.27,0.44]`.
  All ran at native yaw 0, 90 and 37 degrees. No benchmark asset was edited.
- DirectLayout was stopped when entering room construction, after furniture
  transforms; LayoutVLM was stopped at rendering-settings setup and omitted its
  hardcoded floor-material lookup. Asset import, centering, rotation, scaling
  and placement operations were the actual unmodified upstream functions.
  These hooks did not generate pixels, invoke a model, or run optimization.
- Before the fix: 16/24 vertex-set comparisons passed; all eight failures were
  DirectLayout at nonzero yaw. Its released operator yields `Rz(180 - yaw)`, not
  the previous `Rz(180 + yaw)`. The fixed geometric basis is independent of
  semantic front, so missing/changed front metadata must not alter mesh pose.
- After the narrow converter correction: **24/24 passed**, maximum symmetric
  vertex-set distance **7.397566738550413e-7 m** at the existing `1e-4 m`
  geometry tolerance. Canonical origin shift was also applied. All raw GLBs and
  native layouts remained unchanged. LayoutVLM conversion needed no change.
- Added 22 CI tests, including a small synthetic vertex capture tied to the
  exact upstream source hashes. No real asset/model dependency was added to CI.
  The prior DirectLayout 20-degree expectation was demonstrably wrong and is
  corrected from 200 to 160 degrees; strict asset/evaluator tests were not weakened.
- Bridge input text/report records clockwise degrees. New DirectLayout entry
  SHA `92825f6fa803a3773e45bede1748709fb3e797cb2d8505ae5484d019a0f168dc`,
  bundle SHA `33628c733e7c137fb50e8a6890a246afaa98a2744f9d7132aafd2894d8ca7730`.
  Example pins were updated. Old prepared artifacts retain their original pins.

Evidence directory: `/private/tmp/pipeline-native-geometry.5syUFN`; diagnostic
scripts, captured vertices/native layouts, before/after comparisons, logs and
JUnit XML were also copied without modification to ignored
`Support/pipeline_compatibility_runs/api_ready_v1/native_geometry_probe_v1/`.
This archive is not committed. Small synthetic CI captures are code fixtures,
not formal experimental data. Exact commands and converter hashes are in the
archive's `audit_manifest.json`; abbreviated probe commands:

```bash
"$DIRECT_PYTHON" native_geometry_probe.py --method direct_layout --repo "$DIRECT_REPO" --catalog "$CATALOG" --out "$FRESH_DIRECT_DIAGNOSTIC"
"$LAYOUTVLM_PYTHON" native_geometry_probe.py --method layout_vlm --repo "$LAYOUTVLM_REPO" --catalog "$CATALOG" --out "$FRESH_LAYOUTVLM_DIAGNOSTIC"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python compare_canonical.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python compare_canonical_after_fix.py
```

The two comparison scripts preserve before/after vertex comparisons; the audit
manifest pins converter identities and native reports pin upstream identities.
They are one-off diagnostics, not generation launchers, and have fixed paths;
the archive preserves those references rather than rewriting history.

Latest exact source-test command uses source Python 3.13.14:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -o addopts='' -q tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py --basetemp /private/tmp/pipeline-native-geometry.5syUFN/extended_tests --junitxml /private/tmp/pipeline-native-geometry.5syUFN/extended_tests.xml --tb=short
```

The Pro five-file focused command was also rerun independently with this source:
**224 passed**, 2.18s (`pro_focused_tests.xml`); it uses the first five test paths
of the extended command above and its own basetemp/XML.

Extended result: **506 passed, 1 failed**, 10.29s; the only failure remains the separately
reproduced baseline-existing camera-selector test. The focused command with
`tests/test_native_mesh_frame_roundtrip.py tests/test_frozen_imaginarium_scene10.py`,
`focused_tests_v2` basetemp/XML, passed **66 tests**. An earlier new-test attempt
had 9 failures from a wrong test-only provenance lookup; correcting the lookup
to existing `metadata.harness_compatibility.coordinate_conversion` resolved
them. The initial XML remains archived; no implementation was changed for that
test typo.

This representation correction changes canonical DirectLayout poses. Historical
native artifacts must not be overwritten or retrospectively assigned the new
converter identity. Any authorized reconversion uses the existing offline
`--method-output` route and a fresh output directory. The append-only
reevaluation-only command does not reconvert scenes and cannot apply this fix.

New installed wheel SHA-256:
`5e9c783afe070c7b2aa8c00a68729a222037771ca8328659eb271a122e0a1533`.
Built offline and installed into a new `benchmark-wheel-native-geometry-v1`
environment. From `/private/tmp` with `PYTHONPATH` unset, all 11 entrypoints and
12 selected resources passed the existing verification; runtime construction
made zero service calls. The installed DirectLayout converter file hash equals
the audited fixed source hash. `wheel_build.log`, `wheel_install.log` and
`wheel_verification.json` are retained in the evidence directory. This package
check does not close CUDA, SceneWeaver-plugin, production API or full evaluator
acceptance gates.

## SceneWeaver native exporter correction (2026-09-05)

Starting code checkpoint: `b29b129c8433002a56eed93f6bfee7c85e2d0176`.
The pinned `infinigen_examples/steps/tools.py` source hash is
`3cba7aed0f2e634663669e29bad8cd5918539b333aa62a50278acd848f0fd5ac`.
Its exact `export_layout` and `calc_position_bias` function ASTs were executed
without modification on real synthetic bpy placeholders. This avoids unrelated
Linux/LLM imports: it is **not** a complete upstream import, GLB initializer,
render, solver, or reflection-loop qualification.

Native `size` comes from `obj.dimensions`, not a post-rotation world AABB.
Blender 4.3.0 and 5.2.0 LTS each produced eight states: baked input basis 0/90
degrees crossed with zero, 0.65-radian yaw, tilted XYZ, and 90-degree yaw.
In each version all eight rounded sizes matched `obj.dimensions`, while only
two matched the rounded world AABB. The earlier frozen world-AABB contract,
including the corresponding statement in Pro's checklist, was incorrect.

The narrow fix leaves native JSON and ordinary offline conversion unchanged.
The frozen contract is now `released_object_dimensions_rounded_2dp` and requires
per-iteration observed full-precision Euler, exported bottom-center position,
object dimensions, and canonical-frame local bbox. It verifies native rounding
against those observations; canonical rotation still composes the explicit
input basis using matrices. Runtime rotation does not enter the local-dimension
check. Missing observations, mismatched scale/position/rotation/dimensions, or
the retired world-AABB contract fail closed. No retrieval, placement repair,
asset selection, or evaluator change is introduced.

Read-only conversion of the 16 **original native files** against captured bpy
world vertices gave:

- Before: 4 conversions accepted, 12 rejected; none met the `1e-6` metre vertex
  threshold (even accepted states lost native position precision).
- After: 16 accepted and 16 matched; maximum vertex distance
  `1.46971082678919e-7` metres. All native file hashes remained unchanged.

CI captures one runtime's eight small synthetic states in
`tests/fixtures/external_harnesses/sceneweaver_native_export_v1.json` (SHA-256
`567b069a45674ac9c815519d6a99ea5d0e8b847473122b6671044a57a83bd779`).
Twenty new tests verify actual captured pose/vertices, missing and conflicting
observations, retired-contract rejection, absent-front preservation, and bridge
observation -> existing converter for each explicit iteration. They require no
upstream checkout, real asset dataset, Blender, or API. Existing trajectory
tests continue exercising the same canonical evaluator.

Evidence root: `/private/tmp/pipeline-sceneweaver-export.vEWGAL`; selected files
are archived append-only under the owned ignored prefix
`Support/pipeline_compatibility_runs/api_ready_v1/sceneweaver_export_probe_v1/`.
Scripts, original reports/layouts, before/after comparisons, test XML/logs,
package verification, and an audit manifest are retained. No experimental data
or wheel is committed to Git.

Exact native diagnostics (no model or pixel-render calls):

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/environments/direct-layout-cp311/bin/python /private/tmp/pipeline-sceneweaver-export.vEWGAL/probe_native_export.py --repo /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/upstream/SceneWeaver --out /private/tmp/pipeline-sceneweaver-export.vEWGAL/bpy43
/Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup --python /private/tmp/pipeline-sceneweaver-export.vEWGAL/probe_native_export.py -- --repo /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/upstream/SceneWeaver --out /private/tmp/pipeline-sceneweaver-export.vEWGAL/blender52
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python /private/tmp/pipeline-sceneweaver-export.vEWGAL/compare_native_exports.py --checkout /private/tmp/layout-ddd-pipeline-compatibility-runs --evidence /private/tmp/pipeline-sceneweaver-export.vEWGAL --out /private/tmp/pipeline-sceneweaver-export.vEWGAL/converted_v1
```

All three exited 0. From the isolated source checkout, exact final regression
commands (source Python 3.13.14):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_sceneweaver_native_export.py tests/test_frozen_imaginarium_scene10.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-export.vEWGAL/pytest_focused --junitxml=/private/tmp/pipeline-sceneweaver-export.vEWGAL/focused_tests.xml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-export.vEWGAL/pytest_pro --junitxml=/private/tmp/pipeline-sceneweaver-export.vEWGAL/pro_tests.xml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_sceneweaver_native_export.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-export.vEWGAL/pytest_extended_v2 --junitxml=/private/tmp/pipeline-sceneweaver-export.vEWGAL/extended_tests_v2.xml
```

Results: **64 passed** (0.75s), **224 passed** (2.30s), and **526 passed / 1
failed** (11.31s), respectively. The remaining failure is the independently
baseline-reproduced `test_canonical_l3_group_judge_repair_reaches_vlm_and_renders`
at line 374. It is not suppressed or weakened. The initial extended run also
encountered a sandbox denial of `ps` in the cancellation test and an accidentally
changed legacy provenance label; local process inspection was permitted for
the rerun and the ordinary offline label was restored. Initial new-test setup
errors (conflicting selectors, missing required tolerance, wrong expected error
text) were fixed in tests, not by weakening conversion. Earlier XML remains.

Fresh installed wheel SHA-256:
`0dc85bf5ae04f66a0b2a3cee16bb58771d20235d969e45cfeed455b41bb1aa0d`.
Offline build/install used a new `benchmark-wheel-sceneweaver-export-v1`
environment. All 11 entrypoints and 12 selected package resources verified from
`/private/tmp` with `PYTHONPATH` unset; runtime construction made zero service
calls. Installed and source SceneWeaver converter hashes both equal
`895d54fc8065d099ec91bc20954e5b487694ddcaae82bc49af5c5b889bcd8c8b`.

Still open: actual frozen GLB factory/initializer and every native loop operation,
Linux/CUDA host qualification, production model identity/render/optimizer smoke,
and the evaluator applicability decision. Export-only evidence does not certify
any of those. Old prepared inputs and sidecars are not upgraded in place.

## Exact-GLB factory component (not a complete plugin)

`scripts/external_harness_bridges/scene_weaver_frozen_assets.py` now implements
the generation-input portion of the pending plugin. It verifies exact bytes,
local bbox/center, native physical scale and available cardinal front; imports
every static mesh/instance with hierarchy transforms; preserves loose geometry
and materials; and bakes only the declared scale, basis and bottom-center rebase.
No mesh is fitted to a target bbox. Its factory builder accepts the released
AssetFactory base and overrides only `create_asset` / `create_placeholder`,
leaving native spawn/pose bookkeeping to that base. Different slots can bind the
same exact mesh independently.

This component is **not wired into a complete executable plugin**. Readiness
remains blocked on that plugin. Native factory registration, slot mapping,
initialization prompts, mutation guards, observations, model routing and full
loop qualification are still required. The future plugin must pin this helper
and its imported bridge/common helpers; an entrypoint-only hash cannot cover
that bundle. These checks are not reported as real upstream generation.

No-call bpy 4.3 evidence: `/private/tmp/pipeline-sceneweaver-factory.a6VsAm/`.
Attempt 1 tested three approved GLBs (dresser, near-square office chair,
refrigerator with loose geometry), all passed. Attempt 2 added a synthetic
two-instance hierarchy with nonzero center, loose geometry and non-unit declared
scale: **4/4 passed**. Maximum symmetric vertex distance was
`2.235174177411814e-7` metres. Vertex counts and source hashes remained unchanged;
placeholders matched baked local dimensions. Wrong bbox-center input failed
without deleting pre-existing scene objects. The diagnostic invokes only the
new creation hooks, not native AssetFactory spawn or solver registration. A
direct native-base import in the candidate environment fails on missing `gin`;
no native runtime certification is inferred from these checks.

Exact successful component command (exit 0):

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/environments/layout-vlm-cp311/bin/python /private/tmp/pipeline-sceneweaver-factory.a6VsAm/probe_factory.py --checkout /private/tmp/layout-ddd-pipeline-compatibility-runs --catalog /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/verified_glb_preflight/catalog_manifest.json --out /private/tmp/pipeline-sceneweaver-factory.a6VsAm/attempt2
```

Component SHA-256:
`891d79755a132a84df62c5b648360eb340ae9d3f6971f54f85327a9b49612d3e`.
Both diagnostic script versions are retained and match report hashes. Selected
evidence is archived in the owned ignored `api_ready_v1/sceneweaver_factory_probe_v1/`
prefix, never Git.

New CI contract tests: **15 passed**, no bpy/upstream/dataset/API dependency:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_sceneweaver_frozen_assets.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-export.vEWGAL/pytest_factory_contract --junitxml=/private/tmp/pipeline-sceneweaver-export.vEWGAL/factory_contract_tests.xml
```

The preceding 19-file extended command was rerun with
`tests/test_sceneweaver_frozen_assets.py` immediately after the native-export
test, basetemp `/private/tmp/pipeline-sceneweaver-factory.a6VsAm/pytest_extended`,
and JUnit `/private/tmp/pipeline-sceneweaver-factory.a6VsAm/extended_tests.xml`:
**541 passed / 1 failed**, 11.89s. Only the same baseline-existing camera test
failed. Core package source is unchanged from the verified exporter wheel;
this helper is an external source-checkout script, not a new packaged entrypoint.
