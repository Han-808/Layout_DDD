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

## S101 — combined kitchen, dining, and utility room

Review status: `user_selected_pending_materialization`

Recorded: `2026-09-05` (`Asia/Shanghai`)

### Baseline scene

- Model: `Claude Opus 5`
- SceneBoard dataset key: `anthropic_opus5`
- Displayed liveboard score: `85.32`
- Source score: `90.40`
- Coverage: `98.37`
- Existing object count: `25`
- Canonical scene:
  `Support/datasets/api3_anthropic_generations_v1/claude-opus-5-aihub/S101/scene/canonical_scene.json`
- Canonical scene SHA-256:
  `d8ff20328e481d79840a3f4bb859bf480fa3199c5975a79c765206c9cbd48a60`
- Blender scene:
  `Support/datasets/api3_anthropic_generations_v1/claude-opus-5-aihub/S101/prepared/evaluation.blend`
- Blender scene SHA-256:
  `559cb5759c3ceebbd1bf48db3a5ddb0b4a180f8b03d69f7010ec4a719d0fd015`

### User removal decision

Remove exactly one object: user-visible Blender object `Asset_range_cooker`,
recorded canonically as object `range_cooker`. Its current exact Imaginarium
asset is `0_SM_Deco021_02` (CSV row `176`), a `0.415 x 0.301 x 0.247 m`
stainless-steel **countertop oven**. This confirms that the current asset is not
a faithful freestanding range cooker. Preserve the source Blender/native scene
unchanged; apply the removal only when the approved FrozenAssets case is
materialized.

### Candidate shortlist

The following twenty exact CSV assets were checked against the local catalog.
Every listed asset directory exists. Candidate `1` is the functional-priority
replacement for the removed mislabeled range cooker. The remaining candidates
add kitchen storage, a utility work surface, appliances, supplies, and restrained
decor. These rows are the reviewed shortlist. Only the rows repeated in the
user-selected table below are approved for later materialization.

| Candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | Suggested placement / role |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 50 | `0_electric_stove_2k_packed` | `0.503 × 0.648 × 0.859` | `Kitchen_stove` / `stove` | Floor; proper four-burner freestanding cooker replacing the removed false range |
| 2 | 2029 | `e_kitchen_08` | `1.200 × 0.316 × 0.764` | `Kitchen_cabinet` / `cabinet` | Wall; dark-gray four-door upper cabinet above a clear counter run |
| 3 | 206 | `0_SM_Kitchen_top_shelf_50cm` | `0.600 × 0.320 × 0.750` | `Display_stand` / `shelf` | Wall; compact three-tier open kitchen shelf |
| 4 | 47 | `0_drawer_cabinet_2k_packed` | `1.141 × 0.488 × 1.881` | `Storage_rack` / `rack` | Floor; light-wood/black rack with shelves and lower drawer storage |
| 5 | 703 | `34_RollingShelf_01` | `1.142 × 0.642 × 1.700` | `Storage_rack` / `cart` | Floor; four-tier rolling utility cart with boxes |
| 6 | 279 | `0_wooden_display_shelves_01_2k_packed` | `1.078 × 0.372 × 1.556` | `Display_cabinet` / `cabinet` | Floor; light-wood cubby storage for pantry or utility organization |
| 7 | 2043 | `e_workbench_02` | `1.086 × 0.481 × 0.609` | `Workbench` / `workbench` | Floor; narrow utility folding/work table with lower shelf |
| 8 | 173 | `0_SM_Deco019` | `0.170 × 0.405 × 0.300` | `Coffee_machine` / `machine` | On counter; sage-green coffee machine |
| 9 | 170 | `0_SM_Deco017_02` | `0.231 × 0.140 × 0.162` | `Kettle` / `kettle` | On counter; white electric kettle with wood accents |
| 10 | 830 | `44_sk53_Toaster01` | `0.356 × 0.185 × 0.233` | `Toaster` / `toaster` | On counter; white two-slice toaster |
| 11 | 801 | `44_sk27_KitchenTool01` | `0.464 × 0.063 × 0.351` | `Kitchen_utensil` / `rack` | Wall; utensil rail with wood-handled stainless tools |
| 12 | 757 | `43_FruitBasket` | `0.241 × 0.229 × 0.130` | `Fruit_basket` / `bowl` | On dining table or island; white fruit bowl with assorted fruit |
| 13 | 453 | `17_SM_Plates` | `0.237 × 0.237 × 0.032` | `Plate` / `plate` | On shelf or counter; nested matte-black plates |
| 14 | 1504 | `a_SM_kitchen_Canisters01_C` | `0.053 × 0.053 × 0.080` | `Seasoning_bottle_or_jar` / `canister` | On shelf or counter; sage-green tea canister |
| 15 | 1508 | `a_SM_Kitchen_Decor_11_Black` | `0.077 × 0.077 × 0.211` | `Seasoning_bottle_or_jar` / `mill` | On counter or dining table; matte-black pepper mill |
| 16 | 739 | `4_SM_WashingPowder1_08` | `0.272 × 0.173 × 0.413` | `Cleaning_bottle` / `bottle` | On utility shelf; large laundry detergent bottle |
| 17 | 1438 | `a_SM_dishwasher_product` | `0.117 × 0.058 × 0.324` | `Cleaning_bottle` / `bottle` | On utility shelf or beside sink; cleaning spray |
| 18 | 1832 | `a_TrashCan3` | `0.258 × 0.258 × 0.502` | `Outdoor_trash_can` / `trashcan` | Floor; compact gray cylindrical utility/kitchen waste bin |
| 19 | 1513 | `a_SM_Kitchen_Decor_19` | `0.229 × 0.177 × 0.224` | `Small_potted_plant` / `pot` | On sideboard, shelf, or island; compact green plant in white pot |
| 20 | 1766 | `a_SM_Wall_Picture_2` | `0.950 × 0.020 × 1.400` | `Wall_mounted_picture_frame` / `art` | Wall; dark-blue geometric art for the dining zone |

### User-selected additions

The user selected candidate numbers `1, 2, 3, 4, 9, 12, 13, 18`. Each row is
one requested addition. The new stove replaces the removed false range cooker;
the other seven rows are additions to the retained baseline inventory.

| Original candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | Intended support | Provisional role |
| ---: | ---: | --- | --- | --- | --- | --- |
| 1 | 50 | `0_electric_stove_2k_packed` | `0.503 × 0.648 × 0.859` | `Kitchen_stove` / `stove` | floor | Exact functional replacement for `range_cooker` |
| 2 | 2029 | `e_kitchen_08` | `1.200 × 0.316 × 0.764` | `Kitchen_cabinet` / `cabinet` | wall | Four-door upper storage above a clear kitchen counter run |
| 3 | 206 | `0_SM_Kitchen_top_shelf_50cm` | `0.600 × 0.320 × 0.750` | `Display_stand` / `shelf` | wall | Open kitchen display/storage shelf |
| 4 | 47 | `0_drawer_cabinet_2k_packed` | `1.141 × 0.488 × 1.881` | `Storage_rack` / `rack` | floor | Secondary kitchen/utility storage with a distinct drawer role |
| 9 | 170 | `0_SM_Deco017_02` | `0.231 × 0.140 × 0.162` | `Kettle` / `kettle` | on object | Small appliance on a kitchen counter |
| 12 | 757 | `43_FruitBasket` | `0.241 × 0.229 × 0.130` | `Fruit_basket` / `bowl` | on object | Dining-table or island food centerpiece |
| 13 | 453 | `17_SM_Plates` | `0.237 × 0.237 × 0.032` | `Plate` / `plate` | on object | Tableware on the open shelf or kitchen counter |
| 18 | 1832 | `a_TrashCan3` | `0.258 × 0.258 × 0.502` | `Outdoor_trash_can` / `trashcan` | floor | Compact waste bin in the kitchen/utility zone |

### Candidate-selection constraints

- Candidate `1` is the only proposed functional replacement for the deleted
  range cooker; do not retain its old `0_SM_Deco021_02` binding.
- Candidates `2`--`6` are relatively large. Final selection must respect the
  available wall/floor footprint and should not duplicate the existing pantry,
  sideboard, and utility shelf without a distinct role.
- Candidates `8`--`17` require an explicit supporting counter, table, shelf, or
  sink-adjacent surface when materialized; do not flatten them to floor objects.
- Candidate `20` is wall-mounted and must retain that support semantics.
- All eight selected asset directories exist. None is materialized into the
  executable FrozenAssets case by this review-only update.

### Materialization still required

1. Remove the canonical `range_cooker` slot and its old
   `0_SM_Deco021_02` binding; add a replacement slot bound exactly to
   `0_electric_stove_2k_packed`.
2. Add seven additional unique object slots for candidates `2, 3, 4, 9, 12,
   13, 18` without changing other retained S101 slots.
3. Confirm the exact wall locations for candidates `2` and `3` and preserve
   their wall-support semantics.
4. Confirm the supporting counter/table/shelf parents for candidates `9`, `12`,
   and `13`; do not flatten them to floor placement.
5. Confirm that candidate `4` has a non-duplicative storage role and a collision-
   free footprint relative to the existing pantry, sideboard, and utility shelf.
6. Add immutable catalog records and source hashes, then re-run room-height,
   local-asset, frozen-binding, architecture, and harness-eligibility preflight.
