# Object grouping

## Current decision (2026-07-31)

The canonical grouping algorithm is now **VLM-only**. `VLMGroupingAlgorithm`
with policy `vlm_visual_evidence_scope_v2` and prompt version
`vlm_grouping_prompt_v2` is the only backend used by the canonical evaluator,
metric routing, and downstream camera/evidence acquisition.

`topology` and `anchor` remain importable only for explicit historical replay
and regression comparison. They are not defaults, are not fallback paths, and
are not part of the active experiment launcher. If the VLM is unavailable, the
workflow reports an explicit unavailable/unresolved state rather than silently
switching algorithms.

The active 20-scene replay is configured in
`configs/experiments/grouping_vlm20_visual_scope_v2.yaml`. It uses four seeded
scenes from each object-count stratum and re-runs the current VLM contract on
the shared frozen renders. Its generated gallery is written under
`Support/artifacts/outputs/grouping_vlm20_visual_scope_v2_20260731_r1/review/`.

Grouping defines which objects share a local visual-evidence scope. It is a
complete object partition, not a benchmark metric:

- every renderable object appears in exactly one group;
- grouping never emits a validity verdict or score;
- grouping does not judge position, orientation, accessibility, topology
  quality, functional arrangement, style, or prompt fidelity;
- a metric-specific Judge remains authoritative after grouping.

This boundary matters for `object_pairing_consistency`: the grouping backend
localizes the comparison, while the Judge decides whether the supplied group
members have compatible categories and roles. A grouping backend must not hide
an odd local object merely to make a group look plausible.

## Shared interface

All experimental backends implement:

```python
GroupingAlgorithm.group(GroupingRequest) -> GroupingResult
```

The stable request contains the scene, optional case metadata, optional visual
evidence, backend configuration, and non-authoritative context. The result
contains `object_groups`, backend and policy IDs, resolved configuration,
object-ID normalization metadata, and backend provenance.

```python
from benchmark.grouping import group_scene

result = group_scene(
    scene,
    config={"grouping": {"backend": "vlm"}},
    model=chat_model,
)
report = result.to_dict()
```

`report["object_groups"]` is directly compatible with existing
`object_grouping_report` consumers.

## Downstream evidence wiring

The canonical L3 evaluator now treats each selected group as an immutable
camera target rather than unioning members from multiple groups. It derives
that group's exact member IDs, axis-aligned target bounds, focus center, and
extent, then sends one request through the existing deterministic local camera
selector. If the Judge requests a refinement, the Controller keeps the same
group membership; a request that crosses into another group fails closed.

- Object Pairing stays limited to group-member category and semantic-role
  compatibility.
- Scale is evaluated per group, with the cached global scene image available
  as optional room context.
- Style runs once on global evidence and renders local evidence only for
  implicated groups.
- Functional Consistency is marked `experimental_non_scoring`: it is
  implemented, disabled by default, enabled only through explicit config, and
  excluded from the canonical L3 aggregate and weights. It evaluates generic
  real-world usability; L2 `functional_semantic_fidelity` remains the owner of
  prompt-conditioned functional requirements.

The global scene packet is reused across group requests. Local evidence cannot
replace a required global anchor, and a global image alone cannot satisfy a
group-local evidence request.

Canonical L3 semantic repair is Judge-driven:
`Judge.need_more_evidence → Camera DSL planner → deterministic selector →
render → integrity-only EvidenceGate → Judge`, with VLM camera selection
available only after a normal deterministic no-feasible/conflict/exhaustion
outcome. The older deterministic visual-sufficiency trigger remains only in
the explicitly labeled P0b compatibility provider.

## Backends

### Deprecated replay backends: `topology` and `anchor`

`TopologyGroupingAlgorithm` is the deterministic graph baseline refined from
the existing `deterministic_metadata_geometry` implementation. It uses:

1. supplied semantic regions when sufficiently complete;
2. must-link support and attachment edges;
3. explicit relation and derived-support edges;
4. metadata-local and scale-aware proximity edges;
5. existing group diameter and object-count limits.

`AnchorGroupingAlgorithm` first selects stable objects such as beds, sofas,
desks, dining tables, or other large/supporting objects. It then assigns
objects using, in order:

1. explicit support-parent links;
2. derived support contact;
3. floor distance and region consistency;
4. a small semantic-family tie-break.

Semantic affinity is intentionally weak. Spatially local but semantically odd
objects stay in the local scope so a downstream pairing Judge can still detect
them. If no assignment is defensible, the object remains a singleton.

Both backends remain importable only for explicit historical replay. Neither is
selected by the canonical evaluator, the factory default, or as a fallback
when VLM grouping is unavailable.

### `vlm`

`VLMGroupingAlgorithm` receives the complete normalized object catalog,
scene type and boundary, explicit known-ID relations, optional safe context,
and up to the configured number of images. Identity overlays or legends should
be supplied when the images need to map visible instances to object IDs.

The response is fail-closed:

```json
{
  "object_groups": [
    {
      "object_ids": ["known_object_id"],
      "label": "short local-scope label",
      "anchor_object_id": null,
      "reason": "grouping cues only"
    }
  ],
  "reason": "complete partition summary"
}
```

Unknown IDs, missing IDs, duplicate assignments, overlapping groups, anchors
outside their group, unsupported fields, metric verdicts, and scores are
rejected.

## Converted-scene normalization

`Support/Scenes/converted_scenes` objects do not contain `id` or `object_id`;
they contain asset `jid` values, which may repeat. The grouping interface does
not use `jid` as identity. It derives stable `scene_object_NNN` IDs from source
order without mutating the scene and records that policy in result provenance.

## Configuration

Reference configurations:

- active: `configs/grouping/vlm_visual_evidence_scope_v2.yaml`;
- deprecated replay only: `configs/grouping/topology_metadata_geometry_v2.yaml`;
- deprecated replay only: `configs/grouping/anchor_object_v1.yaml`.

The factory defaults to `vlm`, which requires an injected model with
`chat_messages()`. Missing or failed VLM grouping is an explicit unavailable
state in the canonical evaluator; it never silently falls back to topology or
anchor. No backend is allowed to mutate the scene.
