# Generic Validity Evaluator v0

This evaluator checks generated 3D scenes with deterministic geometry only. It is separate from OOR, OAR, VLM judge, and semantic plausibility scoring.

`generic_validity_v0` is real-asset-grounded but not mesh-aware: generated scene objects should reference real assets through `jid` and `asset_ref`, while geometric checks use bbox/OBB proxies derived from object pose and resolved asset metadata such as `transformed_size` or CSV `bbx`.

## Scene Objects

Preferred generated scene objects are flat real asset instances:

```json
{
  "id": "obj_000",
  "jid": "0_alarm_clock_01_2k_packed",
  "category": "alarm_clock",
  "retrieval_category": "alarm_clock",
  "desc": "A mint green vintage alarm clock with white metal bells and Roman numeral face",
  "short_desc": "mint green vintage alarm clock",
  "size": [0.13165, 0.066748, 0.174156],
  "center": [1.0, 1.0, 0.087078],
  "rotation": [0, 0, 0],
  "asset_ref": {
    "source_db": "imaginarium",
    "asset_key": "0_alarm_clock_01_2k_packed",
    "mesh_uri": "0_alarm_clock_01_2k_packed/0_alarm_clock_01_2k_packed.fbx",
    "pointcloud_uri": "0_alarm_clock_01_2k_packed/0_alarm_clock_01_2k_packed.ply",
    "metadata_uri": "0_alarm_clock_01_2k_packed/0_alarm_clock_01_2k_packed_metadata.json"
  },
  "asset_proxy": {
    "type": "obb_from_metadata",
    "bbox_center_local": [0.0, 0.0, 0.0],
    "bbox_size": [0.13165, 0.066748, 0.174156],
    "point_count": 8192,
    "has_color": true,
    "has_normal": true
  },
  "metadata": {
    "interactive": false,
    "is_centered": true,
    "is_normalized": false,
    "is_coordinate_transformed": true
  }
}
```

Mesh and point cloud files are preserved as reference URIs only. The evaluator does not open or parse `.fbx` or `.ply` files.

## Metrics

- `collision`: severe OBB/bbox overlap between object pairs.
- `oob`: footprint, floor, and optional scene-height out-of-bound checks.
- `navigability`: top-down connected free-space ratio.
- `accessibility`: manually annotated interactive objects reachable from the largest free-space component.
- `support`: floor/object/wall-proxy support from bottom-face samples.

## Scoring

```text
collision_score = 1 - min(collision_count / num_objects, 1)
oob_score = 1 - oob_count / num_objects
navigability_score = largest_connected_free_area / total_free_area
accessibility_score = accessible_interactive_objects / interactive_objects
support_score = supported_objects / total_objects
overall_score = unweighted average of active metric scores
```

Metrics with `status="not_applicable"` do not reduce the overall score. Metrics with `status="checked"` or `status="invalid_input"` are active.

## Asset Enrichment

The CLI can enrich scene objects from an asset CSV and asset root:

```bash
python scripts/evaluate_generic_validity.py   --scene outputs/generated_scene.json   --asset-csv data/imaginarium_asset_info.csv   --asset-root data/imaginarium_assets   --enrich-assets   --write-enriched-scene outputs/enriched_scene.json   --out outputs/generic_validity_report.json
```

Resolution priority for geometry size is:

```text
object.size -> object.asset_proxy.bbox_size -> metadata transformed_size -> asset_info.csv bbx
```

`asset_info.csv name_en` is treated as the asset key / `jid`; the CSV `id` field is only a row id.

## Assumptions

- Generated scenes contain real asset references.
- Geometry uses bbox/OBB proxy derived from asset metadata and object pose.
- Room boundary is a 2D floor polygon in XY.
- Floor is `z = 0`.
- `scene_height` is optional and used for height OOB and weak support evidence.
- Objects above `clearance_height` do not block navigation.
- Very low objects under `step_over_height` do not block navigation.
- Interactive accessibility uses only manual flags such as `metadata.interactive`; category is not used to infer interactivity.
- Support uses downward ray / bbox support proxy sampling.

## Limitations

- No mesh loading.
- No point-cloud loading.
- No mesh-level collision or support surfaces.
- No point-cloud-level collision or support surfaces.
- No physics engine.
- No VLM/LLM.
- No doors or windows.
- No semantic plausibility.
- No game-specific advanced metrics yet.

## CLI

```bash
python scripts/evaluate_generic_validity.py   --scene outputs/generated_scene.json   --out outputs/generic_validity_report.json
```

With config override:

```bash
python scripts/evaluate_generic_validity.py   --scene outputs/generated_scene.json   --config configs/generic_validity.json   --out outputs/generic_validity_report.json
```
