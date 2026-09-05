# Pre-campaign generation replay snapshot

This directory preserves the exact Python source bytes of the tracked Kimi-K3,
GLM-5.3, Opus 4.8 High, selected-recovery, and fresh-retry generation commands
as they existed at the commit recorded in `replay_manifest.json`.

It is a source snapshot, not an active entrypoint. The historical modules
derive repository-relative paths from their original `tools/...` locations, so
executing files in this nested directory is intentionally unsupported. Replay
requires restoring the recorded commit or restoring the recorded files to the
original tree layout, verifying all SHA-256 identities, and supplying the
historical private deployment resources separately.

The three model transport files are not duplicated here because they contain
deployment bindings. Their Git blob, byte length, and SHA-256 identities are
recorded as external dependencies without exposing their contents.

No current legacy command forwards yet. The shared D1 planner is fail-closed:
it can characterize common full10 argv and private-binding parity, but refuses
execution until terminal JSON and artifact parity are proven. Selected-brief,
fresh-chance, Opus required-reasoning-signal, and historical-failure-set retry
surfaces remain explicit compatibility blockers.
