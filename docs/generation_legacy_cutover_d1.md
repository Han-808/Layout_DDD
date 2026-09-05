# Generation legacy cutover D1

## Decision

The model-agnostic generation campaign, invoked canonically as
`python -m benchmark.scene_generation`, is the owner for new API2/API3 runs.
The existing Kimi-K3, GLM-5.3, and Opus 4.8 commands remain compatibility
entrypoints for now. D1 does not rename them to “shims” and does not silently
change their output or artifact contracts.

`benchmark.scene_generation.legacy_cutover.forwarding` provides a shared,
fail-closed migration planner. It proves only facts it can check without a
network call or credential read:

- exact translation of the common full10 `check`, `preflight`, and `run
  --output-dir` argv surfaces;
- model/campaign/route mapping;
- equality of the legacy endpoint and credential-environment binding with a
  selected private campaign binding;
- absence of credential-value access during that comparison.

The planner refuses to execute a translated command while either terminal JSON
or written-artifact parity is unproven. Current contracts therefore report
`cutover_ready=false`.

## Explicit blockers

The following interfaces have no equivalent declarative campaign contract and
must remain compatibility code:

- API2 selected-brief timeout recovery;
- API2 independent fresh semantic/schema retry chances;
- Opus preflight with required visible reasoning signal;
- Opus retry selection from a historical schema-invalid failure set.

In addition, full10 commands currently write historical summary/provenance
schemas while campaign v2 writes the redacted artifact-v3 contract. Matching
requests and retry policies is insufficient to claim command compatibility.

## Replay preservation

`Support/legacy/frozen_replay/generation_pre_campaign_v1` contains exact source
snapshots and a hash manifest tied to a Git commit. Deployment-bound model
transport files are referenced only by Git blob, byte length, and SHA-256; they
are not duplicated. The snapshot is non-executable in place and exists only
for reviewed historical restoration.

## Future cutover gate

An entrypoint may become a thin forwarder only after tests prove all of:

1. legacy argv/help/exit behavior;
2. endpoint and credential-environment binding identity without reading the
   credential during static resolution;
3. request, retry, and preflight semantics;
4. terminal stdout/stderr JSON and redaction behavior;
5. complete output-tree filenames, schemas, provenance, and resume behavior.

Until those gates pass, preserving the compatibility entrypoint is safer than
forwarding it under an inaccurate compatibility claim.
