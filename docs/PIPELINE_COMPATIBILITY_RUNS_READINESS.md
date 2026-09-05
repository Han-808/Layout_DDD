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
      Focused: 215 passed. Extended: 463 passed, one baseline-existing failure;
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
      round-trips. Catalog/GLB fidelity and converter golden tests pass; actual
      method renderer/optimizer E2E acceptance remains separate.
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
imports, LayoutGPT's released formatter/parser preparation, and a small
LayoutVLM overlap-gradient diagnostic. No benchmark score was fed to a generator.

| Method | Pinned upstream commit | Local prerequisite evidence | Remaining gate |
| --- | --- | --- | --- |
| LayoutGPT | `fc31954962553e5b65bf267a904a6930d50b1f5e` | Eight public `rect_train` demonstrations prepared using the release's metric formatter; source hashes and disjoint train/test/val selection checked | Configure audited ICL identity in a new prepared protocol; production response identity and native-parser smoke |
| DirectLayout | `4430535304124dbc8d48f6bb0ea5891e92267986` | Native pipeline imports in isolated CPython 3.11.15 / bpy 4.3.0 / NumPy 1.26.4 | Real rendering/refinement and model/API smoke |
| LayoutVLM | `85d06b4cd2478551188a0b4a47cd658c85c41315` | Native solver imports; CPU fallback probe gives positive loss but both corner gradients are null | Build/verify native differentiable Rotated_IoU on compatible CUDA host; no silent CPU-fallback ablation |
| SceneWeaver | `7ae54b2ec3fc66147704faa7daf7b017ba8b1bd9` | Native loop, factories, initializer/update/export paths inspected; clean sparse source checkout | Frozen initialization/export/plugin and Linux runtime; update_layout/update_rotation both invoke update_graph, which rescales and deletes, so disabling named resize/remove tools alone is insufficient |
| Catalog Placement | This benchmark checkout | Existing fixed-selection Stage-C baseline and offline same-evaluator tests retained | Production Stage-C/API and full evaluation smoke, separately reported |

All four upstream checkouts were clean after these inspections/imports. Source
archives, environments, downloaded NPZs, GLBs and reports stay under the ignored
`Support/pipeline_compatibility_runs/api_ready_v1/` prefix in the desktop data
checkout, never in the code commit.

### Asset and ICL outputs

- GLB `glb_bundle_attempt3`: Blender 5.2.0 LTS, 168/168 passed. The refrigerator
  source includes 48 loose vertices; `use_mesh_edges/use_mesh_vertices` preserves
  them. The near-square conference table uses the existing 1e-4 metre tolerance
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

Next: qualify SceneWeaver's thin frozen boundary, finish native geometry probes,
public-brief conflict audit and versioned smoke/three-repeat operator plans;
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
