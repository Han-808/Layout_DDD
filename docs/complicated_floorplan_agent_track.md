# Complicated FloorPlan Agent track

This is an additive Agent-only generation track for **general-purpose coding
Agents**, such as Codex- and Claude Code-class systems. The contestants are not
specialized scene-generation pipelines such as LayoutGPT, LayoutVLM, or
SceneWeaver. It does not replace or modify the frozen Stage A → Top-1
retrieval → Stage C model workflow.

## Fixed experiment contract

- Participant class: a general-purpose coding Agent that can inspect a task
  workspace, iteratively invoke the benchmark-owned tools, revise artifacts,
  and seal a final submission.
- Unit of comparison: the complete general-purpose Agent system, including
  the Agent runtime/version, underlying model/version, and registered launch
  policy. A model running through two different Agent runtimes is two systems.
- Task suite: the approved ten customized SpatialLM FloorPlans (42 rooms, 314
  wall segments), with benchmark-owned architecture and room programs.
- Asset access: all Agents query the same content-addressed Imaginarium shared
  database. Assets are not preselected per scene.
- Output: one Agent-selected object plan, exact shared-DB bindings, and one
  shared-global placement, normalized into the existing non-rectangular
  canonical scene contract.
- Evaluation: the existing room-projected evaluator. Cross-room functionality
  remains outside its scoring scope.

The current shared database contract freezes the 2,043-row metadata index,
embedding matrix, encoder snapshot, golden suite, stable ordering, and query
policy. Every run fails closed if those identities drift.

## Agent tools

Each episode receives an isolated writable workspace and a bounded local tool
service. The workspace command `./layout-ddd-agent-tool` supports:

- `get-task`
- `search-assets`
- `inspect-asset`
- `validate-submission`
- `finalize-submission`

Search is deterministic and returns up to the common `top_k` bound. The Agent
may choose among returned assets and revise its draft repeatedly. The tool
service never returns benchmark scores, VLM Judge decisions, or hidden defect
labels.

## Backend boundary

An Agent backend is a thin adapter around a general-purpose coding-agent CLI
(for example, Codex or Claude Code) that consumes the task prompt on stdin. It
must pin and record the exact Agent implementation/version and model/version,
enforce task-workspace-only isolation, and use exit code 75 only for an
unambiguous retryable infrastructure failure. HTTP 429/500/502/503/504 handling
belongs in that thin backend wrapper. The cohort runner waits 30 seconds and
retries only that declared exit. A timeout is ambiguous and is not blindly
retried.

The adapter must not give the Agent repository access, arbitrary internet
access, a second asset source, benchmark scores, evaluator outputs, or hidden
labels. Both Codex- and Claude Code-class entrants receive the same task files,
shared database snapshot, tool surface, tool-call budget, and wall-clock
budget; only the Agent system differs.

The backend must call `finalize-submission submission.json`; a zero process
exit without a sealed submission is a failed case.

## Commands

Offline contract check:

```bash
layout-ddd-generate-nonrect-agent check \
  --profile configs/generation_extensions/non_rectangular_agent_v1/agent_track.example.json
```

The example profile is intentionally non-runnable until exact Agent commands,
versions, models, and credential environment names are registered. Resource
gate and generation additionally require the existing local retrieval resource
binding file.
