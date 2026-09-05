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
- [x] User-approved experiment acceptance gate: L3 coverage >= 0.8, only
      pairing/style may be inapplicable; all other applicable metrics required.
      Opt-in is versioned and hashed outside evaluator kwargs. The default
      complete-score gate remains unchanged. See the current checkpoint below.
- [ ] F2 production acceptance: a real evaluator report passing that gate.
      Raw `partial_coverage`, scores and weights remain unchanged; no real
      production evaluator report has been obtained by this preparation work.
- [x] F3: prepared input, object plan, protocol, catalog and evaluator hashes
      revalidated before any generation, including later cases.
- [x] F4: cancellation propagates, child process group reaped, logs retained,
      no subsequent scheduled task launched.
- [x] F6: every planned unit has a terminal status; blocked/partial/cancelled
      are distinct from complete; zero attempts cannot exit successfully.
- [x] Current focused compatibility/execution and new regression tests run.
      Latest acceptance/operator/runtime/pilot/protocol-focused: 165 passed.
      Latest extended regression: 687 passed, one baseline-existing failure;
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
- [x] Linux installation/import checkpoint: five isolated Python environments,
      native LayoutGPT/DirectLayout/LayoutVLM/SceneWeaver imports, LayoutVLM
      extension compilation and 176 CPU/mocked tests at `d0b5e4f`. No GPU kernel.
- [ ] All upstream production dependencies qualified. bpy archive platform-tag
      conflicts remain explicit; actual ABI/render/solver/loop qualification,
      an allocated GPU and latest code deployment are not established.
- [x] Released-derived LayoutGPT ICL snapshot with source/selection/leakage audit;
      eight training-only examples, byte-identical independent message rebuild.
- [x] SceneWeaver frozen plugin implementation: native SceneDesigner entry,
      fixed initializer mapping, child-worker wiring and iteration observation.
      Mocked driver/worker integration reaches the existing strict converter.
- [ ] Qualify that plugin in the actual native environment and loop. Static
      source checks and mocked execution are not full-loop certification.
- [ ] Method-native geometry invariants and real asymmetric/near-symmetric pose
      round-trips. DirectLayout/LayoutVLM native mesh assembly: 24/24 captures
      now match conversion, including asymmetric/offset/near-square meshes.
      SceneWeaver exact exporter diagnostics now match all 16 captured bpy
      states; frozen initialization and actual renderer/optimizer/loop E2E
      acceptance remain separate, unresolved gates.
- [x] Private binding/evaluator templates, no-call readiness command, explicit
      smoke/dense/three-repeat launch plan, append-only offline reevaluation and
      budget/failure policy documented. A no-call compiler synchronizes existing
      spec/method/model/ICL contracts; no second runner was added.
- [ ] Actual production host/deployment bindings and native dependencies
      qualified. Template/compiler validation is not environment/API acceptance.
- [x] Requirement-by-requirement evidence audit recorded below, including
      incomplete requirements. This is not a completed production acceptance.
- [ ] All required production acceptance evidence obtained; goal completion
      remains unproven and full generation is not authorized by this checklist.

## Pro requirement closure audit at code checkpoint `9cd034a`

This is a historical checkpoint. The current acceptance-policy and Linux
installation evidence below supersede its applicability/installation status;
its recorded test counts and missing real-production evidence are not rewritten.

This maps every Pipeline checklist item in Pro's two attachments and the shared
implementation guide. `PASS_LOCAL` is limited to the named local evidence;
`PARTIAL` is not acceptance, and `NOT_RUN` is not a successful empty run.
F1/F5/F7 and the other task's production API runs are outside this branch's scope.
No other task's evidence is used to certify this experiment.

Source audit: isolated `codex/pipeline-compatibility-runs`, full code commit
`9cd034aa2eb2d283011772b654083d6b97f83c36`, clean before this documentation edit.
No `src/` changes exist after the installed-wheel checkpoint `8bd0261`.
Relative to Pro's `dfaba4e`, the evaluator, visual-judge and evaluation-profile
source/config paths have no changes. Only demonstrated converter representation
blockers were corrected in the earlier DirectLayout/SceneWeaver checkpoints.

### Pipeline checklist

| Pro item | Current result | Inspected evidence and remaining acceptance |
| --- | --- | --- |
| I.1 / F2: trusted complete evaluator runtime | PARTIAL | `evaluation_runtime.py`, `pilot._evaluation_readiness/_run_case`, `execution.runtime_evaluation_options`; actual `run_evaluate` in `test_frozen_runtime_exposes_existing_applicability_coverage_gap`, with mocked external render/model observations. Judge/grouping/camera wiring is exercised, but the report is `partial_coverage`, not a complete real-production report. |
| I.2 / F3: actual prepared hashes before calls | PASS_LOCAL | `prepared.verify_prepared_artifacts` checks byte and semantic identities; `test_prepared_artifact_drift_rejects_before_any_generation` covers six artifacts, including the four requested classes and a later case. Calls remain zero on drift. |
| I.3 / F4/F6: cancellation, cleanup, zero-unit status | PASS_LOCAL | Actual local SIGINT process-tree test preserves stdout/stderr and verifies workers terminal; pilot cancellation propagates and next-unit count stays zero. All 15 blocked fixture units are recorded and CLI exits 2. Tests and cleanup bodies inspected, not merely searched by name. |
| II.1: current focused/new tests with exact commands | PASS_LOCAL_WITH_KNOWN_FAILURE | Archived final JUnit: 80/0 focused; 628/1 extended; exact commands above. The one L3 camera test failure was independently reproduced at untouched `dfaba4e`, not suppressed. This is not an all-green evaluator certification. |
| II.2: approved source and immutable GLB bundle | PASS_LOCAL | Fresh read-only `_preflight_imaginarium_catalog` at `9cd034a`: 10 cases, 269 slots, 168 assets; source CSV/FBX/metadata and GLB bundle content/geometry validation passed. Catalog remains `7716a5e3...ead8`; source brief v2 remains `2d0328d8...c9e2b`. No conversion/rebuild or asset change performed in this audit. |
| II.3: upstream/bridge/environment pins and actual paths | PARTIAL | All four local upstream HEADs re-read and clean, matching the table below. Bridge/plugin source pins remain unchanged. Config binding compiler refuses source-pin drift and placeholders, but actual production host bindings and qualified native dependency environments are still absent. |
| II.4: approved ICL and substantive frozen plugin | PARTIAL | ICL snapshot/rebuild byte SHA rechecked: `1209a0b3...87386`, eight released training examples, no held-out/target/evaluator data per retained selection audit. SceneWeaver plugin is implemented with 16 pinned mutation targets and all-ten static inputs; real initializer/optimizer/render/reflection loop is NOT run. |
| II.5: faithful geometry/identity round-trips | PARTIAL | Preserved DirectLayout/LayoutVLM native mesh captures: 24/24; SceneWeaver exporter on real synthetic bpy states: 16/16; exact GLB factory component: four cases, max vertex difference 2.2352e-7 m, source meshes unchanged. These are component/representation observations, not all-method native loop certification. SceneWeaver released dimensions are local object dimensions, not world AABBs; the initial Pro wording was corrected from actual source evidence. Thin-rug native anchor behavior and complete native state still require real-loop qualification. |
| III.1: same production model/deployment/message/image route | NOT_RUN | Configured identity, deployment and endpoint hash gates exist; no response from a production model was obtained in this preparation task. Do not infer observed same-model identity from configuration. Requires supplied production bindings and separate execution authorization. |
| IV.1: real S100 native -> canonical -> full evaluator smoke | NOT_RUN | Five-unit S100 command plan exists. No paid generation, real native loop or complete production evaluator report is available; native output existence/API success alone cannot pass this requirement. |
| IV.2: native workflow authenticity and no evaluator feedback | PARTIAL | Bridge source, private-input regression, native artifact/iteration preservation and mocked worker/converter routing inspected. SceneWeaver uses native reflection, with the explicitly approved restricted mutation set. Real workflow/iteration evidence remains missing for all four harnesses. |
| V.1: shared small pilot, engineering-based progression | CONFIGURED_NOT_RUN | User-approved S100 smoke + S101 dense pilot use existing cases, same catalog/inventory for all five. `prepare --case-id` implemented/tested; no runtime subset/resume invention. No score-based method/case selection. Pilot acceptance not run. |
| VI.1: independent formal cohort | CONFIGURED_NOT_RUN | Three user-approved independent repeats, each 10 cases x 5 methods, 150 formal planned units; separate from smoke/pilot. This explicitly supersedes Pro's one-repeat 40+10 example. No formal unit executed or omitted as a passing shortcut. |
| VII.1: monitoring, stopping, budgets and costs | PARTIAL | Timeout/cancellation/identity-drift fixture tests and native cost journals exist; no benchmark-side automatic generation retry. No production resource/cost/iteration observations or monitored native process trajectory exists yet. |
| VII.2: safe append-only recovery, no fake resume | PASS_LOCAL / NOT_RUN_REAL | `reevaluation.py` verifies frozen native/canonical/evaluator inputs, rejects old output paths and never calls generator/converter. Failure/recovery tests inspect byte-unchanged original runs. The CLI preserves original final SceneWeaver selection; it is not whole-trajectory recovery. No production recovery was performed. |
| VIII.1: coverage, publishability and failure separation | PARTIAL | Pilot/result tests distinguish complete/partial/blocked/cancelled/failed and reject complete-score claims for the known applicability gap. There are no publishable production results or production cost comparisons. |
| VIII.2: immutable provenance/archive | PARTIAL | Source/asset/ICL/geometry/plugin/test artifacts are retained separately under the owned ignored data prefix. Latest archived JUnit hashes match originals. Runner/native/model/canonical/evaluation archival mechanisms are tested, but no production native/API/evaluation artifact chain exists to certify. |

### Shared implementation guide

| Requirement | Current result / evidence |
| --- | --- |
| Fixed complete code identity; separate tasks | PASS_LOCAL preparation: separate worktree, reviewed baseline and pushed commit recorded; no cross-task merge. Production must use the pinned checkout/environment, not the dirty shared desktop source. |
| Installed-artifact checks | PASS_LOCAL selected scope: wheel `0dc85bf5...aa0d`, 11 entrypoints and 12 selected resources checked outside checkout; current `src/` matches that checkpoint. Checkout-only operator/bridge scripts are documented as such. Linux native environments are a separate unpassed gate. |
| Separate test suites with actual outcomes | PASS_LOCAL_WITH_KNOWN_FAILURE: own compatibility/execution/evaluator results retained. Complicated_Generation and its arena acceptance are not inferred or reported as this task's results. |
| Faithful native -> canonical mapping | PARTIAL at production scope: exact identities/slots/scale and frame/pose invariants tested; component geometry evidence above. Real complete upstream trajectory still missing. |
| Fixed evaluation semantics | PASS_LOCAL source identity: no metric/profile/weight/hidden-reference/evidence semantics changes from reviewed baseline in the named evaluator paths; no ownership relabeling to hide frozen-asset inapplicability. |
| Generator/evaluator isolation | PASS_LOCAL source/tests: public projection, no evaluator runtime in method config/payload, post-hoc per-state runtime. Production native execution evidence not yet available. |
| Distinct terminal/result states | PASS_LOCAL F6/tests; production readiness, generated/canonicalized/evaluated/score-available/publishable are not interchangeable. |
| Append-only archival/recovery | PASS_LOCAL mechanisms and preparation archive; no deletion/overwrite of old experiment outputs, no shared mutable release. Production full-chain archive remains NOT_RUN. |

### Remaining external conditions (not implementation-completion claims)

1. Supply the execution host/path for qualified LayoutVLM CUDA and SceneWeaver
   native Linux environments. A Mac import or static patch compilation is not
   that evidence; do not silently use a CPU solver ablation or skip a method.
2. Decide how the experiment should report the unchanged evaluator's frozen-asset
   applicability gap. The current complete-score gate correctly blocks it.
   This task has no authority to change scoring/weights or falsely assign asset
   selection/style ownership to the generator.
3. Supply production model/evaluator bindings and separately authorize real
   API/upstream smoke. This preparation task has not authorized paid calls.

The goal is **NOT COMPLETE**. No final-scene/trajectory comparison, production
score, native E2E acceptance, or universal FrozenAssets comparability is claimed.
Further acceptance requires these external conditions; repeating component tests
or adding another compatibility layer would not establish the missing evidence.

## Operator configuration checkpoint (2026-09-05)

Starting code commit: `f571cd32c069df611547391f7efa69376aa7804b` on isolated
`codex/pipeline-compatibility-runs`. This checkpoint changes only the operator
compiler, one binding example, its tests/default-test registration and these
existing docs. No converter/evaluator or upstream bridge source changes.

`scripts/prepare_pipeline_operator_config.py` reads the approved revised source
and explicit non-secret host bindings. It freezes the same model/route in all
four harness configs and their source protocol, copies exact approved ICL bytes,
retains case/catalog/scoring/native budgets, verifies bridge/plugin pins and
writes new immutable config files plus a command plan. It never executes that
plan. The compiler refuses placeholders, literal credentials, altered source/ICL
hashes, a changed method cohort and reused config/run roots.

The command plan uses existing `benchmark.generation_comparison.pilot` CLI
prepare/preflight/run: S100 smoke, S101 dense pilot, and three full repetitions
(50 units each). Existing append-only reevaluation is explicitly separate, may
spend Judge calls, and does not regenerate/reconvert. SceneWeaver recovery here
selects the original final state, not a fresh trajectory. See the operator
section of `FROZEN_IMAGINARIUM_SCENE10.md` for exact semantics and commands.

Actual validation commands, in the isolated checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_pipeline_operator_config.py tests/test_pipeline_evaluation_runtime.py tests/test_frozen_public_brief.py tests/test_controlled_generation_pilot.py --tb=short --basetemp=/private/tmp/pipeline-operator-config.z3ijj7/final_focused --junitxml=/private/tmp/pipeline-operator-config.z3ijj7/final_focused.xml
# 80 passed, 0 failed, 0 skipped, 2.15 s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_sceneweaver_native_export.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_frozen_mutations.py tests/test_sceneweaver_frozen_plugin.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py tests/test_pipeline_operator_config.py --tb=short --basetemp=/private/tmp/pipeline-operator-config.z3ijj7/final_extended --junitxml=/private/tmp/pipeline-operator-config.z3ijj7/final_extended.xml
# 628 passed, 1 failed, 0 skipped, 11.11 s
```

The single failure is the same independently baseline-reproduced
`test_canonical_l3_group_judge_repair_reaches_vlm_and_renders` described below;
it was not weakened or suppressed. The 18 new compiler tests forbid network,
subprocess, generation and rendering during binding; verify unchanged source
files and exact ICL; exercise the real CLI argument parsers; and reject malformed
bindings without output writes. Native repos/models are not run by these tests.

Remaining acceptance gates are unchanged: real Linux/CUDA/SceneWeaver runtime,
production route binding and authorized API smoke, full native loop/geometry
qualification, and complete evaluator applicability. No real service call or
production operator binding is claimed by the mocked tests. The goal is not
complete, and the existing applicability block remains active.

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
| SceneWeaver | `7ae54b2ec3fc66147704faa7daf7b017ba8b1bd9` | Frozen plugin now wires the native driver and worker; mocked two-state execution reaches the existing converter; all S100–S109 public-input static preflights pass (269 slots), with 16 pinned mutation targets. Earlier real-bpy exporter/input diagnostics remain separate evidence | Linux runtime, real initialization/optimizer/render/reflection loop and production model route remain unqualified; implemented integration is not real-loop smoke |
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

Next: qualify the implemented SceneWeaver frozen boundary in its native Linux
environment, bind the actual production host/deployment, and resolve complete-score
applicability without silently altering the evaluator. The versioned no-call
configuration compiler and smoke/three-repeat operator plan are now implemented.
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

## Approved internal FrozenAssets mutation controls

The user explicitly approved narrowly changing SceneWeaver's internal update and
population paths to prevent scaling and object deletion, preserving native
planning, optimization and reflection. This experiment must be labeled
**SceneWeaver–FrozenAssets (restricted mutation set)**, not the unmodified release.
The scope and 269 approved slot/asset bindings are unchanged. The other task's
independent `complicated_agent_run_v6` package is untouched; no cross-task merge
or API launch is authorized by this approval.

The original `Solver.update_graph` rescales even position-only updates and
clamps height to at least 10 mm. A no-call bpy diagnostic of its exact size block
confirmed that approved `45_Capet04` is enlarged **131.579x** in height and
`36_ToolSet_11` **1.329x**, even when exact dimensions are supplied. A normal
dresser also changes size when the released two-decimal dimensions are echoed.
Disabling named resize/remove tools alone cannot enforce frozen scale/inventory.

`scene_weaver_frozen_mutations.py` now provides an opt-in worker component:

- Verifies six upstream source files against the pinned commit, checks loaded
  callable identity, and compiles 14 enumerated function overlays. All checks
  precede activation. Existing function references/gin wrappers remain intact;
  upstream files are never edited. The full native/patched AST, hashes and each
  intercepted operation are written to a fresh, non-overwritten worker journal.
- Accepts an exact dimension echo or the release's rounded two-decimal echo,
  including zero for very thin axes, **without applying it**. Different requested
  dimensions, existing scale changes, swapped assets/slots or changed physical
  dimensions fail; nothing is repaired. The asset and placeholder must already
  match the exact frozen identity and physical size before population.
- Validates the complete layout proposal before applying any pose. Position,
  rotation and native relation processing remain in the original update body.
  Explicit resize/removal and unexpected insertion/resampling paths fail before
  writes. Native pose-move weights are retained; inventory-changing moves are
  removed from the native move schedule.
- Intercepts automatic no-relation/support/collision deletion and retains the
  unresolved condition and objects. With no legal deletion available, the
  deletion/re-optimize cleanup cycle terminates instead of looping indefinitely.
  This is not a zero-collision verdict or termination of native reflection.
  Native physics measurements and the benchmark evaluator are unchanged.

The patched `infinigen_examples/steps/evaluate.py` function belongs to the
**upstream generation system**, not the benchmark evaluator. Its measurements
remain intact; only deletion/cleanup termination is constrained. This component
does not yet install itself in the complete SceneWeaver launcher. Initial slot
registration/planning, exact-only initialization binding (including disabling
the release's unconditional retrieval subprocess), model routing, child-worker
setup and iteration observations still need integration. No full-loop
eligibility or API readiness is claimed.

Evidence root: `/private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/`.
`guarded_v3` exercised the source-pinned function overlays on real bpy 4.3 objects
from the dresser, tool set and thin rug GLBs: **3/3 passed**. Local mesh vertices,
asset source hashes, proposal JSON hash and scale `[1,1,1]` stayed unchanged.
Maximum dimension float error after rotation was `2.384185791015625e-7` metres;
pose-anchor error was at most `1.0013580320489268e-7` metres. All three native
relation calls remained. Native pose-helper bodies were executed, but trimesh
sync/relation implementations were instrumented callbacks. Neither native module
imports, full initialization/optimizer, nor the reflection loop were qualified.

The initial unguarded diagnostic selected the wrong AST loop; the first guarded
attempt used the wrong catalog hash field, and the second demanded bit-exact
`obj.dimensions` after rotation. These diagnostic setup errors were corrected
without weakening the product contract: the final probe requires unchanged local
vertices and scale, and measures float error against 1 micrometre. Prior attempt
directories/logs remain. The source-pinned guard SHA-256 is
`ad0b471b25d5f43c061abcb9f382b84e8b9232b2f6595cb203e0ff7e2aea10eb`.

Exact successful native diagnostics (exit 0, no API or pixel-render calls):

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/environments/layout-vlm-cp311/bin/python /private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/probe_resize_block.py --repo /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/upstream/SceneWeaver --catalog /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/verified_glb_preflight/catalog_manifest.json --out /private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/observed_v2
PYTHONDONTWRITEBYTECODE=1 /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/environments/layout-vlm-cp311/bin/python /private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/probe_guarded_paths.py --checkout /private/tmp/layout-ddd-pipeline-compatibility-runs --repo /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/upstream/SceneWeaver --catalog /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/verified_glb_preflight/catalog_manifest.json --out /private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/guarded_v3
```

Exact CI commands (source Python 3.13.14, no real upstream/dataset/API dependency):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_sceneweaver_frozen_mutations.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_native_export.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/pytest_focused --junitxml=/private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/focused_tests.xml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_sceneweaver_native_export.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_frozen_mutations.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/pytest_extended --junitxml=/private/tmp/pipeline-sceneweaver-native-mutation.OVZNK7/extended_tests.xml
```

Results: **71 passed** (0.35s) and **577 passed / 1 failed** (11.03s).
The sole failure remains the independently baseline-reproduced camera test above;
no legitimate strict test or evaluator code was weakened. This checkpoint adds
only external source-checkout helper/tests/docs, so the previously verified core
wheel remains unchanged. No experimental artifacts are committed to Git.

## SceneWeaver driver/worker checkpoint (2026-09-05)

Implementation now includes `scene_weaver_frozen_plugin.py` (entry SHA-256
`7ca59a152d7001a0bf4627c7a565497bf5110278701f56d8246cd844b92e0da9`),
connected by the existing bridge/example config. It invokes released
`SceneDesigner.run()` and child `generate_indoors.main()`, not a replacement
generator. The initializer keeps native model calls and placement/relationship
logic while requiring exact supplied slot/factory bindings. Sixteen pinned
internal mutation targets guard the approved no-resize/no-delete policy;
initial unaccepted candidate rollback remains legal. Extra-object finalization
stages are disabled through native configuration. Native files are not edited.

The complete code route now includes public-input preparation, exact GLB factory
registration, source pins, no-model worker import/overlay preflight, configured
same-model SDK routing, native loop invocation, worker logs and return codes,
append-only native snapshots, observed per-iteration geometry, and validation
through the existing strict bridge/converter. The SDK interface follows the
official Chat Completions tool/image message contract; original native call and
retry budgets are retained. No credential file from the upstream checkout is
opened. A failed final action cannot relabel an older layout as a successful run.

Two implementation details were verified against pinned source: native layout
and rotation tools both send backend action `update`; native modules capture
`WALL_HEIGHT` during import, so public height is bound before that capture. The
observer additionally checks the actual native room contour instead of trusting
only the exported room-size header. Native inventory/scale failures are rejected,
not repaired by conversion. No `src/benchmark` or evaluator change is in this
checkpoint.

**Evidence boundary:** mocked native driver/Blender tests execute the real plugin
orchestration, preserve two native JSON states, pass the existing bridge, then
independently invoke the existing strict converter for iterations 0 and 1 with
an exact provider whose retrieval method raises. These are not real upstream
reasoning/optimizer/render/loop tests. Actual public-input static checks use
approved GLB paths/hashes and the unchanged frozen catalog, but import neither
Blender nor native modules and make zero model calls. They do not establish
production generation eligibility. The Linux runtime, full native mesh/anchor
round-trip (including ultra-thin assets), real model route and same-evaluator E2E
smoke still require qualification. The unrelated evaluator applicability gate
also remains unresolved; no score/ownership/weight override was introduced.

Exact final test commands, from the isolated worktree (Python 3.13.14):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_sceneweaver_frozen_plugin.py tests/test_sceneweaver_frozen_mutations.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_native_export.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-plugin.hfjeP4/pytest_checkpoint_focused --junitxml=/private/tmp/pipeline-sceneweaver-plugin.hfjeP4/checkpoint_focused.xml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_sceneweaver_native_export.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_frozen_mutations.py tests/test_sceneweaver_frozen_plugin.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py --tb=short --basetemp=/private/tmp/pipeline-sceneweaver-plugin.hfjeP4/pytest_checkpoint_extended --junitxml=/private/tmp/pipeline-sceneweaver-plugin.hfjeP4/checkpoint_extended.xml
```

Results: **104 passed / 0 failed** (0.54s); **610 passed / 1 failed** (11.04s).
The sole extended failure is the previously independently baseline-reproduced
`test_canonical_l3_group_judge_repair_reaches_vlm_and_renders` (expected one VLM
request, observed zero). No tests were weakened or skipped to hide it. Earlier
diagnostic/test attempts remain under `/private/tmp/pipeline-sceneweaver-plugin.hfjeP4/`.
The core wheel contents remain unchanged; plugin scripts execute from the pinned
source checkout rather than introducing an alternate installed evaluator.

All ten approved public inputs were prepared again in a **new diagnostic**
directory using public-brief-v2 spec SHA
`2d0328d8ef016c430e8655cda183898b578a6faf3fbde72a5072a678f52c9e2b`
and the unchanged verified 168-asset GLB bundle. Static plugin checks returned
exit 0 for S100–S109, respectively 31/32/29/23/30/25/26/25/23/25 slots (269 total).
The reports record the actual plugin/input file hashes. Exact commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -c 'from benchmark.generation_comparison.pilot import main; main()' prepare --spec /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/public_brief_v2/spec.json --asset-root /Users/han_mohan/Desktop/Layout_DDD/Support/Assets/imaginarium_assets --asset-bundle-root /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1/glb_bundle_attempt3 --method-configs configs/generation_comparison/frozen_imaginarium_scene10_methods.example.json --out-dir /private/tmp/pipeline-sceneweaver-plugin.hfjeP4/all_cases_prepared
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python /private/tmp/pipeline-sceneweaver-plugin.hfjeP4/preflight_public_cases.py --checkout /private/tmp/layout-ddd-pipeline-compatibility-runs --data /Users/han_mohan/Desktop/Layout_DDD/Support/pipeline_compatibility_runs/api_ready_v1 --prepared-root /private/tmp/pipeline-sceneweaver-plugin.hfjeP4/all_cases_prepared --out /private/tmp/pipeline-sceneweaver-plugin.hfjeP4/all_cases_static
```

The preparation intentionally retains placeholder production paths/API settings
and is **not** a production-ready or scored experiment. No native generation,
real rendering, optimizer, model call or evaluator call ran in these checks.
Fresh preparation with fully qualified runtime identities remains mandatory
before any separately authorized real run. Diagnostic JSON/JUnit/source script
evidence is stored outside Git under the owned ignored data prefix.

## Current checkpoint: FrozenAssets acceptance and Linux installation (2026-09-05)

Starting branch commit: `d0b5e4fd1e87a7c10fb5ad03e0af4aeba1c5ab85`, isolated
`codex/pipeline-compatibility-runs`. The subsequent code changes are confined to
generation-comparison acceptance, existing operator config compilation and
tests/docs. No evaluator, visual-judge, converter or native bridge source changed.

The user resolved the applicability decision: **minimum L3 coverage 0.8, and
only pairing/style may be inapplicable**. The opt-in
`frozen_assets_required_metrics_v1` validates the unchanged canonical report,
factual frozen ownership, evaluator plan/metric inventories, complete required
metric coverage, L0, reliability and absence of infrastructure/unresolved claims.
It does not drop weights, repair reports or improve scores. The sample binding
opts in outside `static_kwargs`; old bindings/default callers remain strict.
The new policy is included in prepared hashes and cannot be retrofitted by
editing historical artifacts. Source/bound evaluator identities are recorded.

Pilot results and append-only reevaluation distinguish `evaluation_accepted`
from raw complete-score `evaluation_success`. A qualifying partial report can
advance this experiment without being included in old complete-score aggregates.
SceneWeaver gets an independent, report-hashed acceptance decision for every
state; one rejected state prevents final-only acceptance. All evaluation remains
post-hoc, absent from generator inputs and native reflection.

Current tests used the real canonical evaluator with synthetic native artifacts
and mocks only for external observations/rendering. Two successive one-object
cases advance under opt-in with raw `partial_coverage`; strict default still
rejects it. Tests also cover renderer failure/recovery without regeneration or
source mutation, metric/evidence omission despite sufficient coverage, active
L2 obligations, prepared-policy drift and offline SceneWeaver iterations.

Exact local commands, from the isolated checkout (Python 3.13.14):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_pipeline_evaluation_acceptance.py tests/test_pipeline_evaluation_runtime.py tests/test_pipeline_operator_config.py tests/test_controlled_generation_pilot.py tests/test_generation_comparison_protocol.py --tb=short --basetemp=/private/tmp/pipeline-acceptance-current.Sq0Qei/focused_final --junitxml=/private/tmp/pipeline-acceptance-current.Sq0Qei/focused_final.xml
# 165 passed, 0 failed, 0 skipped; 4.68 s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py tests/test_adapter_io_contracts.py tests/test_generation_comparison_protocol.py tests/test_controlled_generation_pilot.py tests/test_frozen_imaginarium_scene10.py tests/test_external_harness_execution.py tests/test_pipeline_evaluation_runtime.py tests/test_pipeline_evaluation_acceptance.py tests/test_layoutgpt_frozen_icl.py tests/test_frozen_public_brief.py tests/test_native_mesh_frame_roundtrip.py tests/test_sceneweaver_native_export.py tests/test_sceneweaver_frozen_assets.py tests/test_sceneweaver_frozen_mutations.py tests/test_sceneweaver_frozen_plugin.py tests/test_scene_harness.py tests/test_catalog_placement_adapter.py tests/test_external_converter_correctness.py tests/test_external_room_contracts.py tests/test_canonical_metric_scoring.py tests/test_canonical_l3_camera_integration.py tests/test_evaluation_mode_interface.py tests/test_submission_api.py tests/test_pipeline_operator_config.py --tb=short --basetemp=/private/tmp/pipeline-acceptance-current.Sq0Qei/extended_verified --junitxml=/private/tmp/pipeline-acceptance-current.Sq0Qei/extended_verified.xml
# 687 passed, 1 failed, 0 skipped; 14.23 s

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_external_harness_adapters.py --basetemp=/private/tmp/pipeline-acceptance-current.Sq0Qei/adapters --junitxml=/private/tmp/pipeline-acceptance-current.Sq0Qei/adapters.xml
# 68 passed, 0 failed; 0.34 s
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/han_mohan/Desktop/Layout_DDD/.venv/bin/python -m pytest -q -o addopts='' tests/test_adapter_io_contracts.py --basetemp=/private/tmp/pipeline-acceptance-current.Sq0Qei/io --junitxml=/private/tmp/pipeline-acceptance-current.Sq0Qei/io.xml
# 34 passed, 0 failed; 0.26 s
```

The extended suite required reviewed permission for `ps` to inspect its own
cancelled test children. Its first sandboxed attempt was **686 passed / 2 failed**:
one `ps` permission error and the same historical L3 camera failure. With that
permission the cancellation test passed; the sole remaining failure is
`test_canonical_l3_group_judge_repair_reaches_vlm_and_renders`, line 374, expected
one VLM request and observed zero, already independently reproduced at the
untouched Pro baseline (`/private/tmp/pipeline-runs-regression.FiYPaQ/baseline.xml`).
No test/evaluator change or skip hides this failure. Earlier diagnostic attempts
are retained, including an initially underspecified two-object external fixture;
that fixture was restricted to the existing one-object observation contract,
not fixed by weakening the acceptance gate.

New wheel built with `python -m pip wheel . --no-deps --no-build-isolation`:
SHA-256 `91407968695a38461cba6171c4a993f2465f05dcb7d9803f1972beb257d2c290`.
Installed into a fresh target with `--no-index --no-deps`. Outside the checkout,
Python isolated mode imported the acceptance module, existing evaluator, pilot
and reevaluation; every loaded `benchmark` module resolved inside that target.
This reused the dependency environment and is a wheel-content/import check,
not a new native dependency or production qualification.

### Separate Linux installation evidence

User-authorized root: `/hcccao/pipeline_compatibility_runs/runtime_v1/`
(`/hcccao` resolves to `/dockerdata/hcccao_tf3`). Five isolated Python 3.10.12
environments and four pinned upstreams were installed there, approximately 18 GB
including caches. System Python, driver and existing Blender were not modified.
The benchmark installation is **still `d0b5e4f`**, not this later local acceptance
checkpoint; no implicit deployment claim is made.

Actual evidence: 176/0 CPU/mocked compatibility tests; released LayoutGPT parser;
DirectLayout native pipeline/assembly imports; SceneWeaver executor and planner
imports through the existing configuration overlay; LayoutVLM native overlap
extension compiled for H20 `sm_90` and imported with GPUs hidden. Native source
commits remained unchanged. No API request, GPU kernel, renderer smoke or native
generation/reflection loop was executed. All eight observed GPUs were busy;
no GPU allocation was claimed.

Important caveat: the official bpy 3.6 archive wheel has a cp310 filename but a
cp39 internal WHEEL tag. Both `pip check` and `uv pip check` return failure for
the three bpy environments, despite successful Python 3.10 imports. The original
bytes/tags were not relabeled. Full ABI/render qualification remains pending.
LayoutVLM alone uses NumPy 1.23.5 for released `np.int` code and a `.pth` for native
absolute imports; no upstream algorithm was edited to install it.

Installation logs, exact five dependency freezes, repo pins, warning/failure
records and test XML are retained remotely in `tools/INSTALLATION.md` and
`logs/runtime_installation_manifest.json`, with a local backup under
`Support/pipeline_compatibility_runs/api_ready_v1/linux_runtime_v1_installation/`.
Those are environment evidence, not generated scene experiments, and are not
pushed as code.

Remaining: deploy the tested acceptance checkpoint with a new provenance record;
synchronize/verify the approved assets and public inputs; bind production model
and evaluator routes; qualify designated GPU/native rendering and loops; then
obtain separate authorization for staged API smoke/pilot/formal runs. No asset
reselection, scoring change or real experiment is part of this checkpoint.
