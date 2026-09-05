# Additive multi-room generation v1

`multi_room_with_architecture_v1` is a generation-side compatibility mode. It
does not refactor or replace the frozen single-room workflow, and it does not
change evaluation. It composes the existing campaign route/binding/preflight,
provider codec and gateway, deterministic Top-1 retrieval runtime, retry
classification, and frozen model-call primitive around a new room-wise
contract.

## Canonical lifecycle

The mode is selected only by an explicitly registered campaign ID. Supplying a
floor plan cannot turn a single-room campaign into multi-room mode, and omitting
the floor plan from a multi-room campaign fails before credentials or network.

```bash
python -m benchmark.scene_generation check \
  --campaign api2-kimi-k3-multi-room-v1 \
  --floor-plan /path/to/floor_plan.json

python -m benchmark.scene_generation resolve \
  --campaign api2-kimi-k3-multi-room-v1 \
  --floor-plan /path/to/floor_plan.json

python -m benchmark.scene_generation resource-gate \
  --campaign api2-kimi-k3-multi-room-v1 \
  --floor-plan /path/to/floor_plan.json

python -m benchmark.scene_generation preflight \
  --campaign api2-kimi-k3-multi-room-v1 \
  --floor-plan /path/to/floor_plan.json

python -m benchmark.scene_generation run \
  --campaign api2-kimi-k3-multi-room-v1 \
  --floor-plan /path/to/floor_plan.json \
  --output-dir /new/write-once/output
```

For terminal use, the local launcher wraps this exact lifecycle and retains the
same credential/retry behavior as the earlier generation runners:

```bash
cd /Users/han_mohan/Desktop/Layout_DDD

./Support/bash/local/run_multi_room_generation_v1.sh \
  --campaign api2-kimi-k3-multi-room-v1 \
  --input output/multi_room_generation_handoff_v1 \
  --output Support/artifacts/outputs/e2e_multi_room/api2_kimi_k3_v1
```

The checked-in sample root contains ten layouts, 31 rooms, and therefore 62
model-owned emissions (one Stage A and one Stage C per room). The launcher uses
the same existing API2/API3 endpoints and prompts for `API2_APP_CREDENTIAL` or
`API3_API_KEY` with hidden input. `--api-endpoint` is an explicit override;
credentials never enter the binding file, command line, or output artifacts.

The launcher validates every floor plan before reading a credential, runs the
retrieval resource/hash/golden gate, refuses low disk space and existing fresh
outputs, and executes layouts sequentially. Provider calls use only the
campaign's finite infrastructure retry policy: API2 permits three retries and
API3 permits two for `transport_failure` and HTTP
408/409/425/429/500/502/503/504. Ambiguous timeouts, model-content failures,
and schema failures are not resent. A failed room is terminal and later rooms
continue; a failed layout is reported and later layouts continue. No outer
fresh-generation retry or best-of selection is performed.

Use `--preflight-only` for a live route check without creating generation
output. Use `--resume` only with the same campaign, floor plans, and output
root; terminal summaries are skipped and incomplete runs pass the canonical
hash/identity gates before any new provider call.

Do not intentionally interrupt an active room request. A process stop after a
room reaches its terminal `room_result.json` is resumable, but an interruption
during the request leaves a deliberately nonterminal room directory that the
system will not resend automatically; this prevents an ambiguous duplicate
generation.

The current API3 three-model sequence is:

```bash
./Support/bash/local/run_api3_opus5_sonnet5_fable5_multi_room.sh
```

It prompts once for `API3_API_KEY`, then runs Opus 5, Sonnet 5, and Fable 5 in
that exact order, with a 60-second inter-model cooldown. These three campaigns
retain the same provider-default reasoning behavior as their prior generation
runner; the launcher does not claim or inject an explicit Xhigh override. Each
model writes to its own child of the output base, and a partial/failed model is
reported without preventing the later models from running.

`--resume` is available only for this additive mode. It skips a room only when
its terminal result and every declared source artifact still hash-match. An
existing nonterminal room is never resent, a completed run is never resumed,
and a partially published assembly is never overwritten.

The Python API exposes the same five stages:

```python
from benchmark.scene_generation import (
    prepare_generation_campaign,
    resolve_generation_campaign,
    resource_gate_generation_campaign,
    preflight_generation_campaign,
    run_generation_campaign,
)

prepared = prepare_generation_campaign(
    "api2-kimi-k3-multi-room-v1",
    floor_plan_path="/path/to/floor_plan.json",
)
```

The checked-in additive campaigns cover every current reviewed model profile:
Kimi-K3, GLM-5.3, Opus 4.8 High, Opus 4.8, Sonnet 5, Opus 5, and Fable 5.
They reference the existing model and route profiles; no endpoint, credential,
or model-specific runner is duplicated.

## Input and isolation

The authoritative input schema is `multi_room_floor_plan_v1`. Production room
and shared-wall IDs accept any non-empty decimal suffix (`room_1`, `room_100`,
`shared_wall_1`, and so on). The two-digit sample naming convention is not a
room-count limit. `generation_order`, not an ID suffix or array scan, is the
only execution order.

Static validation fails closed on duplicate/unknown JSON keys, unsupported
versions, non-finite or negative shared coordinates, non-rectangular rooms,
dimension/offset mismatch, positive-area room overlap, incorrect count tiers,
invalid wall-attachment ranges, invalid shared edges, inactive or non-opposing
shared walls, incomplete adjacency declarations, disconnected room graphs, and
an incorrect global envelope. No sample-derived 2–4 room cap or 1–2 attachment
cap exists.

Rooms execute sequentially. Each model call receives only one compiled room
brief, one room's frozen object plan/retrieval results, and its active-wall
contract. Every room has one semantic Stage A emission, one deterministic
retrieval invocation per public slot, and one Stage C placement emission.
Schema or semantic failure terminalizes that room without a model retry; later
rooms continue without seeing the failed room's content.

## Assembly and evaluator boundary

Assembly performs exactly two allowed operations:

1. prefix a local instance identity with its trusted `room_id`;
2. add the declared local-to-global translation to its centre.

It never clamps, packs, nudges, rotates, rescales, replaces, deletes, repairs,
or invents an object. Asset identity, slot binding, scale, Euler rotation,
support, facing, object count, and plan semantics are preserved.

The output tree is write-once:

```text
<output>/
  run_manifest.json
  execution_policy.json
  summary.json
  <layout_id>/
    floor_plan.json
    floor_plan_validation.json
    rooms/room_000/.../room_result.json
    compiled_architecture.json
    assembled_multi_room_scene.json
    evaluation_rooms/room_000/
      canonical_scene.json
      scene_request.json
      object_plan.json
      asset_selection.json
      generation_input.json
      architecture_contract.json
    room_evaluation_index.json
    assembly_manifest.json
```

`assembled_multi_room_scene.json` is `multi_room_scene_v1`; it is deliberately
not `canonical_scene_v1`. `compiled_architecture.json` deduplicates declared
shared walls and emits exterior subsets as active wall intervals minus shared
segments. `active_walls` is the room-local logical support topology; the
deduplicated physical inventory is exactly `exterior_walls + shared_walls`.
Successful rooms receive local `canonical_scene_v1` projections with
globally stable object IDs, unchanged local transforms, reversible global
centres, an existing validated single-room architecture contract, and source
hashes. Each projection is persisted with five matching evaluator inputs under
one request identity. The evaluator-owned object-plan companion uses
`room_evaluation_object_plan_v1` and records both the exact source artifact hash
and its normalized canonical-plan hash; the Stage A model-owned
`multi_room_object_plan_v1` artifact is never rewritten. The room-evaluation
index covers every expected room, requires the full companion set for every
successful room, forbids it for failed rooms, uses only layout-relative paths,
and verifies every indexed file hash before publication.

Current evaluators may later consume the indexed room projections without
rerunning the generator or retriever. Multi-room score aggregation, global
architecture scoring, cross-room collision, and cross-room functionality are
explicitly deferred. No evaluator, Judge prompt, camera path, metric, weight,
or scoring policy is changed by this mode.

`aggregate_shape` is retained as the caller-declared collection label. It is
not geometry authority and is never used to infer rooms or walls; validated
rectangles, exact local/global projections, adjacency, shared segments, and the
global envelope are authoritative.

## Selective failed-room recovery

The current Opus 5, Sonnet 5, and Fable 5 scene10 run has an explicit local
failed-room retry configuration at
`Support/configs/multi_room/failed_room_retry_opus5_sonnet5_fable5_r2.json`.
Launch it with:

```bash
./Support/bash/local/run_api3_multi_room_failed12_retry_r2.sh
```

The launcher validates the 12 declared source failures before reading an API
key.  For each affected layout it creates a new isolated resume root, copies
and hash-verifies only successful terminal room checkpoints, and deliberately
omits the failed rooms.  The canonical runtime therefore skips the successful
rooms and calls the model only for unresolved targets.  A target receives at
most three isolated fresh chances; later chances include the chronologically
first complete earlier checkpoint, never select by score, and continue across
room, layout, and model failures.  The original full run is never modified.
`retry_summary.json` records the selected chance and room-result hash for every
target, or the remaining unresolved set after the fixed ceiling.

## Versioning and trust

Existing base registry files remain byte-identical. The canonical loader reads
an explicitly listed, hash-pinned additive fragment only when resolving a
registered additive campaign. Duplicate fragment, workflow, artifact, or
campaign IDs fail closed. Provenance separately records static source/config
hashes and dynamic binding/preflight diagnostics; the caller floor plan is
part of the run fingerprint and resume identity.

Resume gates run before credentials or network access. They verify the floor
plan, public route-binding identity, exact terminal-room identity, terminal
artifact set, and hashes. If every room is already terminal, deterministic
assembly finalization resumes without constructing retrieval resources,
reading a credential, or issuing a model preflight. Existing final artifacts
are reused only when recomputed bytes and hashes match exactly; no file is
overwritten.

The mode's Python sources, independent Stage A/C prompts, floor-plan/object-plan
schemas, evaluator-companion/global/architecture/index/manifest schemas,
additive registry, shared
campaign runtime, frozen core, retrieval runtime, and retrieval catalog are all
covered by checked-in trust/resource inventories. Tests use only synthetic
fixtures and injected fake provider/retriever behavior; they do not call real
generation or Judge APIs.
