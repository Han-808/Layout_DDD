# Current floor-plan evaluators on main — 2026-09-11

The goal is for main to contain usable latest-designated evaluator sources,
not only a registry pointing at an operator's disk. Different modes currently
have different actual implementations. Publication must not upgrade one mode
by accidentally replacing another mode's shared implementation.

## Authoritative repository entrypoint

```bash
python3 scripts/run_floorplan_evaluator.py --mode single_room
python3 scripts/run_floorplan_evaluator.py --mode non_rectangular_multi_room
python3 scripts/run_floorplan_evaluator.py --mode multi_room
```

These commands **verify and describe only**. They read the registry's `current`
mapping; `--run -- <runner arguments>` is required for execution. The Single-room
source is executable in a separate interpreter. Missing dependencies fail
explicitly; the selector never substitutes another mode or the working tree.

| Mode | Published implementation | Execution boundary |
|---|---|---|
| Single-room / Open-space | [Versioned source](../evaluator_snapshots/single_room_sceneweaver_20260909_v1/README.md), v4/v12/v2/v8/v31 | Hash-verified isolated entrypoint; supply prepared input, assets, Blender and private Judge configuration |
| Nonrect | Existing `src/benchmark`, core matching `3186983`, plus execution hardening v4, merged in [PR #8](https://github.com/Han-808/Layout_DDD/pull/8) | The exact combined142 recipe additionally requires locally sealed campaign releases; publication does not remove that dependency |
| Rectangular Multi-room | Historical baseline/provenance only | Exact historical source remains unavailable (`replay_ready=false`); execution through the selector refuses rather than inventing a source release |

Thus two verified evaluator source implementations are now available in the
repository. **Three current baseline records are not three independently
replayable environments.** No existing result is relabeled and no current
mapping is promoted by this change.

Existing low-level Python APIs, console commands and scripts retain their
compatibility behavior; they do not implicitly consult this selector. For the
latest-designated mode, use the entrypoint above. This is deliberately separate
from changing legacy defaults or restarting existing campaigns.

## Evaluation results to scores

For Single-room, the implementation to inspect is the published snapshot's
`src/benchmark/evaluator/scoring.py`, `scoring_profiles.py`, evaluator `profile.py`,
`api/evaluation.py`, and `camera_cal_scene_level/persisted_scoring.py`.
For Nonrect, inspect the corresponding root-package files plus
`src/benchmark/non_rectangular/projection.py` and `report.py`.
The same displayed metric version or numeric score alone does not establish
the same scoring/evidence contract. The snapshot preserves ambiguity flags,
infrastructure failures, null scores and coverage limitations as they were.

This publication does not upload generated experimental outputs or recompute
scores. It establishes the source baseline needed for the forthcoming scoring
discussion; result publication and any scoring-policy change are separate work.

## Safety and validation

Work is isolated from the operator's dirty checkout. No running source, plan,
environment, credentials, outputs, locks or worker settings are changed. The
original Single-room 860-file tree hash is verified before and after publication;
all 484 exported files match their original hashes. Root Nonrect core files and
execution modules remain unchanged. Source-integrity and isolated-import tests
are added to CI for Python 3.11 and 3.13.

The snapshot is intentionally outside setuptools' root `src` package discovery.
It does not replace installed modules or change the wheel's evaluator defaults.
Its manifest differentiates the complete historical hash inventory from the
published evaluator-source subset and lists the excluded scope.

Local validation used the same Python 3.13 environment in two independent
checkouts. The default test selection excluded `requires_blender`,
`requires_local_data`, `requires_loopback` and `requires_git_history` so no
running campaign, renderer, network service or historical data was exercised.

- Unchanged main `999e46d`: **2352 passed, 28 failed, 1 skipped, 86 deselected**.
- Publication branch: **2372 passed, 28 failed, 1 skipped, 86 deselected**.
- JUnit comparison: **zero newly failing test IDs**; all 28 failures are also
  present on unchanged main. The complete default suite is not green.
- The 20 new publication tests all pass, including exact source hashes,
  unchanged Nonrect core, rejection of changed/unlisted source and unsafe paths,
  missing Multi-room source, fresh-process isolation, clean-CWD runner help,
  actual v4/v12/v8/v31 imports, scoring/coverage boundaries and resource parity.
- Wheel build passes; inspection confirms the isolated snapshot is not included
  and the root evaluator version is unchanged.
- No credential-like literals or recognized credential-token patterns were
  found in the exported files. Only source/configuration text is published.

GitHub CI must pass for the exact PR head before merge; local no-new-regression
results are not a substitute for that gate.
