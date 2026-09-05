# Evaluation campaign orchestration

Status: active Phase C contract. The campaign controller wraps the current
scene-level evaluator and first-publishable selector as frozen subprocesses.
It does not import or modify evaluator logic.

This is an operational source-checkout controller. The library wheel contains
the Python package but deliberately does not duplicate the frozen scripts,
Blender inputs, evaluator datasets, or local deployment bindings. Outside a
Layout_DDD checkout it fails closed instead of guessing those resources.

## Boundary

The controller owns only:

- portable identity of an already-prepared evaluation dataset;
- public Judge protocol/model profiles;
- resolution of a private local deployment binding;
- endpoint readiness and the existing multimodal smoke command;
- immutable attempt-round scheduling;
- campaign-level resume and retry selection;
- invocation and provenance of the existing first-publishable finalizer.

The following remain authoritative and unchanged:

- `scripts/run_camera_cal_scene_level.py`;
- `scripts/select_first_publishable_scene_evaluations.py`;
- `src/benchmark/api/evaluation.py`;
- evaluator metrics, prompts, camera selection/rendering, scoring, and call
  order.

The evaluation campaign package does not import the scene-generation provider
package. It consumes only the prepared evaluator dataset. Source generation
configuration, model effort, and retrieval configuration do not participate in
evaluation dataset identity. Neutral canonical-JSON hashing, manifest writing,
and secret-field guards are the only appropriate cross-domain utilities.

## Public profile versus local deployment binding

`configs/evaluation/campaigns/judge_profiles_v1.json` is safe to commit. A
public profile contains only:

- a profile ID and logical binding ID;
- adapter and request-protocol identifiers;
- model alias/profile;
- non-sensitive, wire-affecting policy.

The `model_profile` block is deliberately limited to constants that the frozen
runner really executes (`send_temperature=false`,
`response_format_json=false`, and `max_tokens`). Provider-side reasoning is
not asserted there. For API2, `adapter_attestation` declares the expected
LiteLLM model entry; the controller parses the hash-pinned concrete adapter
YAML and verifies alias, provider model, and reasoning effort before launch.
The selected profile hash, normalized adapter attestation hash, and route hash
all enter the protocol/resume fingerprint.

The current frozen selector records `gpt-5.6-sol` as its evaluator model.
Accordingly, `check` and `run` reject a different alias or a mixture of aliases
across prior and current attempt roots before any evaluation work begins.

It must not contain a deployed endpoint, private URL, credential value,
credential environment selection, local port, PID, log path, or executable
path.

Real deployment information belongs only in the ignored file:

```text
configs/evaluation/campaigns/evaluation_bindings.local.json
```

Copy the redacted shape from
`evaluation_bindings.local.example.json`, then replace its placeholders only
in the ignored local copy. Never add the local copy to Git.

A managed deployment must use the reviewed
`src/benchmark/evaluation_campaign/owned_proxy_launcher.py`. It accepts and
consumes `--config`, `--host`, `--port`, and a controller ownership token. The package passes
the hash-verified adapter profile explicitly; merely naming an unused profile
for provenance is not accepted. Existing one-off local launchers are not
silently treated as this contract and are left untouched.

For a direct binding, the local file resolves the endpoint and credential
environment. For a managed-proxy binding, it resolves a hash-pinned launcher,
hash-pinned adapter profile, upstream environment mapping, and loopback port.
The managed session:

- refuses non-loopback evaluator exposure;
- refuses an occupied port;
- starts one owned child process;
- records a private durable ownership lease with PID, token, launcher hash,
  and argv identity, preventing PID-reuse/orphan false adoption;
- forwards SIGTERM/SIGINT and escalates child cleanup when necessary;
- never stops an unrelated PID;
- gives upstream secrets only to the proxy child;
- gives the evaluator only the ephemeral local proxy credential;
- records only a binding digest and configured/not-configured flags.

## Dataset identity

`EvaluationDatasetIdentity` hashes the exact evaluator inputs for every case:

- canonical scene and annotation;
- prepared Blend file;
- standardized perspective, top, and identity images;
- the path-independent identity-legend projection;
- the path-independent collision-manifest projection;
- every referenced collision-geometry payload.

Historical absolute source paths are excluded from the portable fingerprint.
The raw dataset-manifest hash is retained separately for audit only. Therefore
moving the same prepared dataset does not change its identity, while changing
any evaluator-consumed bytes does.

The frozen loader is not changed. Before a run, the controller makes a private
relocatable dataset projection under the attempt parent. Large immutable
artifacts use hard links when possible (copy fallback); only the private case
and collision manifests are normalized to relative paths. The projection is
re-inspected and must have the identical portable fingerprint. The original
dataset bytes are never rewritten. Smoke evidence is resolved through the
case manifest instead of a hard-coded source path. Scene-type identity also
participates in each case fingerprint.

Treat the prepared dataset as immutable for the lifetime of a campaign. Large
projection files may be hard-linked to their source, so in-place source
mutation during a running process is outside the contract; restart and resume
will revalidate the pinned identity and fail closed on drift.

To print and pin an identity:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m \
  benchmark.evaluation_campaign dataset-identity \
  --dataset-root Support/datasets/<dataset> \
  --case-id S100 --case-id S101
```

## Static check

Static checking reads public configuration and tracked runtime paths only. It
does not load the local binding, read credentials, start a proxy, or use the
network. A local dataset may be absent in a clean checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m \
  benchmark.evaluation_campaign check \
  --config configs/evaluation/campaigns/glm53_api1_full10_v1.json
```

`dataset_status=missing_allowed_for_static_check` means the configuration is
valid but not locally runnable. The check still walks and hashes the frozen
runner/selector dependency closure, the campaign package, packaging contract,
and evaluation/grouping YAML resources; missing or symlink-substituted sources
fail closed.

The first-publishable selector imports its persisted-score projection from
`benchmark.camera_cal_scene_level.persisted_scoring`, so that projection is
part of the protocol source identity. The read-only HTML evidence viewer is a
separate presentation consumer: changing viewer markup or CSS cannot affect
selection and therefore does not invalidate a campaign protocol fingerprint.

## Run and resume

Set the credential variables named by the ignored local binding in the current
terminal, then run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m \
  benchmark.evaluation_campaign run \
  --config configs/evaluation/campaigns/glm53_api1_full10_v1.json \
  --bindings configs/evaluation/campaigns/evaluation_bindings.local.json
```

The same command resumes. Resume is at campaign level:

- each evaluator round still receives `--no-resume`;
- a prior round is never overwritten;
- an interrupted round is retained and a later round receives only pending
  cases;
- a live recorded child blocks another controller;
- config, dataset, and evaluator-protocol drift fail closed;
- execution policy, selected Judge profile, adapter attestation, campaign
  package, selector dependency, and packaged YAML drift fail closed;
- prior roots are adopted through an immutable private manifest containing the
  complete ordered case inventory and source report hashes;
- first-publishable selection remains chronological and never score-seeking.

The controller holds a non-blocking file lock and atomically writes
`campaign_manifest.json` plus one `campaign_round.json` per attempt root. Final
selection remains owned by the frozen selector. An additive
`campaign_selection_provenance.json` maps every selected case to its attempt,
Judge profile, and source hashes. Public campaign provenance contains only
portable attempt indices and repository-relative identifiers; endpoints,
credential environment names, absolute paths, PIDs, and dirty path lists are
kept out. Existing final selections are adopted only after strict validation
of selection/run/summary schemas, case order and uniqueness, publishability,
source relations and hashes, case identity, and the self-contained snapshot
inventory. Final cases are regular copied directories with a complete tree
hash, so deleting attempt roots after successful finalization does not break
the selected collection.

Campaign configs reject overlapping dataset/attempt/final/prior roots.
`l3_only=true` is intentionally rejected by this layer: L3 recovery is a
separate explicit workflow because silently skipping L1 would change report
composition semantics.

## Legacy launchers and version cleanup

Phase C does not delete a launcher. A script may be quarantined only after its
full responsibility has a config-driven replacement and exact argv/env parity
has passed.

Pure evaluation or supplemental-retry launchers can migrate first. Combined
scripts that also validate generation collections, convert canonical scenes,
render, or build datasets are only partially replaced; their preparation
portion remains outside this controller. L3-only recovery mergers also remain
separate because they encode a distinct report-composition policy.

Tracked versions continue to follow
`docs/repository_version_lifecycle.md`: migrate, quarantine, observe, mark as a
deletion candidate, obtain explicit approval, then retain a deleted tombstone.
Untracked local launchers should not be added to Git merely to delete them;
preserve an exact mapping/hash, quarantine locally after parity, and request a
separate deletion approval.
