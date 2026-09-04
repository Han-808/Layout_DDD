# Frozen Imaginarium SceneBoard asset review

Status: `in_progress`

This is the human-decision ledger for the SceneBoard single-room asset review.
It is deliberately separate from the executable FrozenAssets specification
until every room has been reviewed and the missing slot, support, zone, count,
and relation semantics have been confirmed. Nothing in this file changes
`asset_selection_status`, approves the complete catalog, or launches a harness.

## Frozen sources

- SceneBoard: `http://29.98.161.115:8081/scene_liveboard.html`
- SceneBoard status snapshot generated at: `2026-09-01T07:32:39+00:00`
- Status snapshot SHA-256:
  `b1a392062e4ba9863227b1d68aa8467244049f14336f07f36e78088099b49ef8`
- Asset catalog CSV:
  `Support/Assets/imaginarium_assets/imaginarium_asset_info.csv`
- Asset catalog CSV SHA-256:
  `506ef357d2c1d36b145eb8f61743b6ac4d7b4864a9d7842ffab39268052fba85`
- Baseline selection policy: prefer a high-scoring SceneBoard scene with an
  existing readable `.blend`; the numerical score is a preference, not a hard
  requirement.

## Decision-state vocabulary

- `user_selected_pending_materialization`: the user selected the exact CSV
  asset, but it has not yet been added to the executable catalog/case.
- `materialized_pending_final_approval`: exact catalog and case records exist,
  but the complete ten-room FrozenAssets set is not approved.
- `human_approved`: reserved for the user's explicit final approval after all
  rooms and generated contracts have been reviewed.

## S100 — open-plan living, dining, and reading room

Review status: `user_selected_pending_materialization`

Recorded: `2026-09-05` (`Asia/Shanghai`)

### Baseline scene

- Model: `GPT-5.6-Sol`
- SceneBoard dataset key: `gpt56sol`
- Displayed liveboard score: `93.25`
- Source score: `94.96`
- Coverage: `100.0`
- Existing object count: `22`
- User removal decision: **remove nothing**
- Canonical scene:
  `Support/datasets/api1_gpt56sol_generations_v1/gpt-5.6-sol/S100/scene/canonical_scene.json`
- Canonical scene SHA-256:
  `00e2f644ea1317b2b1491078f439cfbbd273b2eba7d5755db721685618b58a26`
- Blender scene:
  `Support/datasets/api1_gpt56sol_generations_v1/gpt-5.6-sol/S100/prepared/evaluation.blend`
- Blender scene SHA-256:
  `5e705250f51b479561b622ae6d8f510a7f3b6594cbb499c3edd0412e729b5ca0`

The existing object named `reading_bookcase` resolves to asset
`a_SM_Speaker_03`, whose catalog description is a wooden bookshelf speaker.
The user explicitly chose not to remove it. The new true-bookcase selections
below must therefore be additions, not replacements.

### User-selected additions

The user selected candidate numbers `1 3 6 12 13 14 15 17 18`. Each row is
recorded as one requested addition. Do not collapse the two bookcases or infer
that either is an alternative unless the user later changes this decision.

| Original candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | CSV description | Intended support | Provisional role |
| ---: | ---: | --- | --- | --- | --- | --- | --- |
| 1 | 1271 | `a_SM_BookShelf_01a` | `1.000 × 0.400 × 2.000` | `Tall_bookshelf` / `bookcase` | Tall dark-brown wooden bookshelf with five open shelves | floor | Primary true bookcase in the reading zone |
| 3 | 968 | `5_SM_Shelf_A_B` | `0.932 × 0.294 × 1.954` | `Wine_cabinet` / `bookcase` | Tall beige bookshelf cabinet with open shelves and lower closed doors | floor | Secondary book/display storage; exact zone still pending |
| 6 | 452 | `17_SM_Plant_2` | `0.571 × 0.532 × 1.132` | `Large_potted_plant` / `pot` | Fiddle-leaf fig tree in a terracotta ribbed pot | floor | Soft boundary/accent near the reading or living zone |
| 12 | 1714 | `a_SM_TableLamp_01a` | `0.333 × 0.333 × 0.646` | `Desktop_table_lamp` / `lamp` | Classic table lamp with gray tapered shade and natural-wood base | on object | Reading task light; exact supporting table still pending |
| 13 | 448 | `17_SM_Magazine_2` | `0.250 × 0.324 × 0.039` | `magazine` / `stack` | Stack of white glossy interior-design magazines | on object | Living coffee-table reading detail |
| 14 | 450 | `17_SM_Opened_Book` | `0.388 × 0.239 × 0.052` | `Book` / `magazine` | Open fashion magazine with photography spreads | on object | Active-use reading detail; exact supporting surface pending |
| 15 | 1466 | `a_SM_Globe` | `0.265 × 0.279 × 0.412` | `Globe` / `globe` | Vintage decorative globe with sepia map and black stand | on object | Reading-zone bookshelf or side-table accent |
| 17 | 1418 | `a_SM_Decor_9` | `0.196 × 0.058 × 0.188` | `Desktop_ornament` / `statue` | Matte-black minimalist bird sculpture | on object | Media-console accent |
| 18 | 164 | `0_SM_Deco_thr008` | `0.184 × 0.181 × 0.432` | `Small_potted_plant` / `vase` | Cream calla lilies in a clear glass cylinder vase | on object | Dining-table centerpiece |

All nine asset directories exist in the recorded Imaginarium asset root. At
the time of this review, none of these nine exact IDs is present in
`configs/generation_comparison/frozen_imaginarium_scene10_v1.json`'s catalog.

### Materialization still required

Before these selections enter the executable FrozenAssets case:

1. Add exact immutable catalog records and verify source mesh/metadata hashes.
2. Add nine unique S100 object slots without modifying the existing 22 slots.
3. Confirm the final slot IDs and the exact support parent for each of the six
   on-object additions.
4. Confirm zones and only defensible public placement relations.
5. Re-run local asset and room-height preflight.
6. Re-run all harness eligibility checks; tabletop support must not be silently
   flattened to floor placement.
7. Keep the global asset set pending until the user explicitly approves the
   completed S100--S109 inventory.
