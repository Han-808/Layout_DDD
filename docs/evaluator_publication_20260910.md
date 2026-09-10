# Latest-used evaluator integration — 2026-09-10

## Scope and identity

The latest verified actual use is the baseline for each floor-plan mode. See
[the baseline registry](../configs/runners/floorplan_evaluator_baselines_v1.json)
and [status explanation](evaluator_baselines.md). An active baseline is not a
claim that its campaign has completed or its scores are publication-ready.

This integration starts from GitHub main `dfaba4e5cd5c88757df54686f1cc8f1bce3e95ba`.
It selectively imports these five evaluator commits, retaining their source IDs
in the cherry-pick messages:

| Source | Change |
|---|---|
| `fc0b656` | Sanitize task-slot audit aliases and preserve retry identity |
| `f19227f` | Match nonrect wall vertices with absolute tolerance |
| `7fb8f91` | Retain placement schema-repair parse failures |
| `f32031c` | Acquire only consumed Collision final evidence |
| `3186983` | Bundle final Collision evidence in one scene load |

The preceding generation/comparison work is not included. All 183 Python files
under `non_rectangular`, `evaluator`, `rendering`, `models`, and `visual_judge`
match the clean running `3186983` source byte-for-byte. The
[core file manifest](../configs/runners/nonrect_3186983_core_manifest_v1.json)
records those hashes. This is core parity, not whole-repository equivalence.
Cherry-picking changes Git commit IDs; the original runtime identity is not
replaced by the integration commit ID in historical reports.

Single-room's latest-used frozen tree and rectangular Multi-room's historical
reference are documented, not reconstructed from this Nonrect source. The latter
still has `replay_ready=false`; equal metric version labels do not prove equal
code. No old result is relabeled or recomputed.

## Execution layer and portability

The opt-in execution modules and Python runner are copied from the active
combined142 execution-hardening v4 source without changing their behavior.
The public shell wrapper alone is made repository-relative and fails early when
the private/local sealed release is absent. The running wrapper and its frozen
plan have NOT been modified. Accordingly, the wrapper hash in the historical
run registry describes that historical file, not this portability wrapper.

The combined142 recipe requires the original locally sealed releases, matching
input/reuse plans, assets and private runtime credentials. Those are intentionally
not included in this public source change. The wheel packages the core benchmark;
repository-local execution scripts are not a self-contained wheel entrypoint.
Do not copy or relax a frozen plan's hashes to reuse an existing campaign with a
different execution wrapper. A changed runtime needs a fresh execution identity.

No credentials, raw API exchanges, scene assets, generated evaluations or private
release directories belong in this change. Local artifact paths in the registry
are provenance references, not downloadable resources.

## Known limitations retained

- Four room slots are acquired by non-FIFO lock competition. A one-hour admission
  deadline can expire while other rooms are healthy. This integration does not fix
  queue fairness or change the active run's policy.
- Renderer observation itself does not retry. Room retries require typed transient
  evidence; a camera/renderer category alone does not qualify. GLM
  `scene_011687/room_000` has a `failed_nonretryable` core summary after one attempt
  and no room retries. It is not a valid zero score or automatically a hard skip.
- At 14:40 CST the campaign had 17 complete selected reports (15 reused, 2 new),
  one such nonretryable summary and one pending Sol admission timeout. The three
  remaining29 lanes were running; plus20 had not started. This is a dated snapshot,
  not final coverage.

## Validation and merge gate

Focused offline regressions pass: **416 passed, 2 skipped**. Skips require local
historical campaign artifacts or external Blender. Synthetic coordinator/retry
tests now also run against the checked-in source on a clean checkout, without
requiring the operator's frozen directory. One inherited test assertion was
corrected to match the existing policy de-duplication contract; the same failure
was reproduced on unchanged main before correction. No judge code was changed
to make that test pass.

Wheel build and shell syntax checks pass. The shell wrapper refuses a clean
checkout without its sealed release before starting evaluation. Core file parity
was verified independently of test results. The new evaluator CI runs on Python
3.11 and 3.13, with fake transports, synthetic catalogs, and no API keys or Blender.

The broad default suite also has failures outside this focused regression scope.
Its main-versus-integration comparison is recorded in the PR; do not describe the
entire default suite as green. Merge requires completed CI and no unaccounted
new regression. Active evaluation code, outputs, processes and locks are left
untouched throughout publication.
