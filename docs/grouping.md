# Object grouping

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
    config={"grouping": {"backend": "anchor"}},
)
report = result.to_dict()
```

`report["object_groups"]` is directly compatible with existing
`object_grouping_report` consumers.

## Backends

### `topology`

`TopologyGroupingAlgorithm` is the deterministic graph baseline refined from
the existing `deterministic_metadata_geometry` implementation. It uses:

1. supplied semantic regions when sufficiently complete;
2. must-link support and attachment edges;
3. explicit relation and derived-support edges;
4. metadata-local and scale-aware proximity edges;
5. existing group diameter and object-count limits.

The existing production `deterministic_metadata_geometry` path remains
unchanged. The new backend is an explicit experimental interface over that
implementation.

### `anchor`

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

- `configs/grouping/topology_metadata_geometry_v2.yaml`
- `configs/grouping/anchor_object_v1.yaml`
- `configs/grouping/vlm_semantic_partition_v1.yaml`

The factory defaults to `topology`. Selecting `vlm` requires an injected model
with `chat_messages()`. No backend is allowed to mutate the scene.
