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

## S102 — shared office, library, and meeting room

Review status: `user_selected_pending_materialization`

Recorded: `2026-09-05` (`Asia/Shanghai`)

### Baseline scene

- Model: `HY4-SFT0812`
- SceneBoard dataset key: `hy4_sft0812`
- Displayed liveboard score: `99.10`
- Source score: `99.44`
- Coverage: `100.0`
- Existing object count: `22`
- Canonical scene:
  `Support/datasets/hy_server_generations_v1/hy4-sft0812/S102/scene/canonical_scene.json`
- Canonical scene SHA-256:
  `e965dddfd5e639c7daa70af44242eca2c12004c431f39181a22c102d27908635`
- Blender scene:
  `Support/datasets/hy_server_generations_v1/hy4-sft0812/S102/prepared/evaluation.blend`
- Blender scene SHA-256:
  `181c43fe50f8f74d813e876140ce9ba92c8599940f6fe8a748c980ee6525e312`

### User removal decision

Remove exactly two objects: user-visible Blender objects `Asset_monitor_a` and
`Asset_monitor_b`, recorded canonically as `monitor_a` and `monitor_b`. Both are
bound to exact Imaginarium asset `5_SM_PC_B_Monitor` (CSV row `964`). Preserve
the source Blender/native scene unchanged; apply removals only when the approved
FrozenAssets case is materialized.

### Local retirement-policy audit

The user's expectation that this asset is locally blocked is supported but not
fully enforced:

- `Support/scripts/migrate_multi_room_monitor_asset.py` explicitly labels
  `5_SM_PC_B_Monitor` as retired and orientation-incompatible.
- That migration defines `41_ComputerSet_03` as the reviewed fixed replacement,
  with canonical facing `local_neg_y`.
- The catalog CSV row for `5_SM_PC_B_Monitor` still has an empty `state` field.
- The current `frozen_imaginarium_scene10_v1.json` catalog and S102 case still
  reference `5_SM_PC_B_Monitor`.

Therefore the asset is retired by a specific local migration policy, but it is
not globally fail-closed in the catalog/retrieval boundary. This review records
the discrepancy without broadening scope into a block-policy refactor.

### Candidate shortlist

All twenty exact CSV assets and their local directories were verified. Candidate
`1` is the reviewed replacement monitor and is proposed with count two, one per
desk. The remaining candidates add workstation, library, meeting, storage, and
decorative detail. Only the rows repeated in the user-selected table below are
approved for later materialization.

| Candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | Suggested count / placement / role |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 742 | `41_ComputerSet_03` | `0.480 × 0.198 × 0.445` | `Computer_monitor` / `monitor` | 2; one on each desk; reviewed replacement for both retired monitors |
| 2 | 197 | `0_SM_Keyboard` | `0.326 × 0.137 × 0.010` | `Keyboard` / `keyboard` | 2; one in front of each replacement monitor |
| 3 | 207 | `0_SM_Laptop` | `0.383 × 0.282 × 0.257` | `Laptop_computer` / `laptop` | 1; secondary mobile workstation or meeting-table device |
| 4 | 304 | `11_SM_Books_03` | `0.140 × 0.035 × 0.210` | `Book` / `book` | 1; upright teal book on a bookcase shelf |
| 5 | 306 | `11_SM_Books_05` | `0.140 × 0.036 × 0.210` | `Book` / `book` | 1; upright parchment-covered reference book |
| 6 | 307 | `11_SM_Books_06` | `0.139 × 0.032 × 0.210` | `Book` / `book` | 1; upright dark-navy book on a bookcase shelf |
| 7 | 118 | `0_SM_002_Book001` | `0.196 × 0.298 × 0.026` | `Book` / `bookcase` | 1; flat finance book on desk, meeting table, or shelf |
| 8 | 119 | `0_SM_002_Book002` | `0.210 × 0.301 × 0.025` | `Book` / `bookcase` | 1; flat architecture book in the library/meeting zone |
| 9 | 448 | `17_SM_Magazine_2` | `0.250 × 0.324 × 0.039` | `magazine` / `stack` | 1; magazine stack on the reading side table |
| 10 | 450 | `17_SM_Opened_Book` | `0.388 × 0.239 × 0.052` | `Book` / `magazine` | 1; open book on the reading side table or desk |
| 11 | 1466 | `a_SM_Globe` | `0.265 × 0.279 × 0.412` | `Globe` / `globe` | 1; library shelf or cabinet display object |
| 12 | 157 | `0_SM_Coffee_cup_2` | `0.081 × 0.107 × 0.098` | `Water_cup` / `mug` | 2; one mug per workstation or meeting participants' area |
| 13 | 1715 | `a_SM_Tablet` | `0.463 × 0.293 × 0.014` | `Tablet_computer` / `tablet` | 1; flat collaboration device on meeting table |
| 14 | 1418 | `a_SM_Decor_9` | `0.196 × 0.058 × 0.188` | `Desktop_ornament` / `statue` | 1; restrained black sculpture on cabinet or shelf |
| 15 | 1766 | `a_SM_Wall_Picture_2` | `0.950 × 0.020 × 1.400` | `Wall_mounted_picture_frame` / `art` | 1; geometric wall art in reading or meeting zone |
| 16 | 1622 | `a_SM_Plant_01a` | `0.960 × 0.739 × 1.380` | `Large_potted_plant` / `vase` | 1; floor plant at an unoccupied room corner |
| 17 | 1623 | `a_SM_Plant_02a` | `0.924 × 0.576 × 0.549` | `Small_potted_plant` / `planter` | 1; floor or sufficiently large low cabinet; rectangular fern planter |
| 18 | 1832 | `a_TrashCan3` | `0.258 × 0.258 × 0.502` | `Outdoor_trash_can` / `trashcan` | 1; compact waste bin near the printer or desks |
| 19 | 279 | `0_wooden_display_shelves_01_2k_packed` | `1.078 × 0.372 × 1.556` | `Display_cabinet` / `cabinet` | 1; library storage with cubbies, distinct from the two tall bookcases |
| 20 | 1871 | `b_18` | `1.800 × 0.401 × 0.501` | `TV_cabinet` / `cabinet` | 1; low shared print/storage station if sufficient wall space remains |

### User-selected additions

The user selected candidates `1, 2, 6, 7, 12, 20` with explicit multiplicities.
This represents nine instances across six exact assets. The two replacement
monitors replace the two removed instances one-for-one; the remaining seven
instances are additions to the retained baseline inventory.

| Original candidate | Count | CSV id | Exact asset ID | CSV bbox W×D×H (m) | Intended support | Provisional role |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 2 | 742 | `41_ComputerSet_03` | `0.480 × 0.198 × 0.445` | on object | One reviewed flat-panel monitor on each desk, replacing `monitor_a` and `monitor_b` |
| 2 | 2 | 197 | `0_SM_Keyboard` | `0.326 × 0.137 × 0.010` | on object | One keyboard in front of each replacement monitor |
| 6 | 1 | 307 | `11_SM_Books_06` | `0.139 × 0.032 × 0.210` | on object | Upright dark-navy book on a bookcase shelf |
| 7 | 1 | 118 | `0_SM_002_Book001` | `0.196 × 0.298 × 0.026` | on object | Flat finance book on a desk, meeting table, or shelf |
| 12 | 2 | 157 | `0_SM_Coffee_cup_2` | `0.081 × 0.107 × 0.098` | on object | One coffee mug at each workstation |
| 20 | 1 | 1871 | `b_18` | `1.800 × 0.401 × 0.501` | floor | Low shared printer/storage station against a suitable wall |

### Candidate-selection constraints

- Candidate `1`, if selected, means two instances of the same exact reviewed
  asset, replacing `monitor_a` and `monitor_b` one-for-one.
- Candidate `2`, if selected, likewise means two instances, one per desk.
- Candidates `4`--`14` require explicit desk, table, cabinet, side-table, or
  bookcase support; do not flatten them to the floor.
- Candidate `15` requires wall support.
- Candidates `16`, `17`, `19`, and `20` have material footprints; select only if
  they retain circulation and distinct functional roles.
- All six selected asset directories exist. None is materialized into the
  executable FrozenAssets case by this review-only update.

### Materialization still required

1. Remove `monitor_a` and `monitor_b` and their retired
   `5_SM_PC_B_Monitor` bindings; recreate two monitor slots bound exactly to
   `41_ComputerSet_03` while retaining the workstation identities.
2. Add two keyboard slots and two mug slots, paired one-to-one with the two
   desks/workstations.
3. Add one upright-book slot, one flat-book slot, and one low print-station
   cabinet slot with unique identities.
4. Confirm exact support parents for all eight tabletop/shelf instances; do not
   flatten them to the floor.
5. Confirm the `b_18` cabinet has a collision-free wall footprint and place the
   existing office printer on it if that support relationship is selected.
6. Add immutable catalog records and source hashes, then re-run local-asset,
   frozen-binding, architecture, scale, support, and harness-eligibility
   preflight.

## S103 — bedroom, dressing, and workspace

Review status: `user_selected_pending_materialization`

Recorded: `2026-09-05` (`Asia/Shanghai`)

### Baseline scene

- Model: `Claude Opus 5`
- SceneBoard dataset key: `anthropic_opus5`
- Displayed liveboard score: `88.75`
- Source score: `91.34`
- Coverage: `97.34`
- Existing object count: `21`
- Canonical scene:
  `Support/datasets/api3_anthropic_generations_v1/claude-opus-5-aihub/S103/scene/canonical_scene.json`
- Canonical scene SHA-256:
  `b4b0a44f93d59fbffa670032cc2c7945d8d41cbeb8f67f4826db06e0f7930155`
- Blender scene:
  `Support/datasets/api3_anthropic_generations_v1/claude-opus-5-aihub/S103/prepared/evaluation.blend`
- Blender scene SHA-256:
  `d4cb07402755112cc7f29f027ed75b22062edeee7982043c49f007d2654f172c`

### User removal decision

Remove exactly two objects:

1. `computer_monitor`, bound to retired, orientation-incompatible exact asset
   `5_SM_PC_B_Monitor` (CSV row `964`).
2. `area_rug`, bound to exact asset `22_SM_Rug_Stacked_01g` (CSV row `589`).
   The catalog class is `Folded_fabric`; its description and dimensions
   (`0.572 x 0.542 x 0.063 m`) confirm that it is a folded fabric stack, not a
   floor rug.

Preserve the source Blender/native scene unchanged. Apply both removals only
when the approved FrozenAssets case is materialized.

### Candidate shortlist

All twenty exact CSV assets and local directories were verified. Candidates
`1` and `2` are the faithful replacements for the removed monitor and false
rug. The remaining candidates add workstation, bedside, dressing, textile, and
decorative detail. Only the rows repeated in the user-selected table below are
approved for later materialization.

| Candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | Suggested placement / role |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 742 | `41_ComputerSet_03` | `0.480 × 0.198 × 0.445` | `Computer_monitor` / `monitor` | On desk; reviewed replacement for the retired monitor |
| 2 | 891 | `45_Capet05` | `2.400 × 1.700 × 0.010` | `Carpet` / `rug` | Floor; actual muted gray/teal geometric area rug replacing the folded fabric |
| 3 | 197 | `0_SM_Keyboard` | `0.326 × 0.137 × 0.010` | `Keyboard` / `keyboard` | On desk; in front of the replacement monitor |
| 4 | 207 | `0_SM_Laptop` | `0.383 × 0.282 × 0.257` | `Laptop_computer` / `laptop` | On desk; secondary mobile work device |
| 5 | 434 | `17_SM_Clock` | `0.120 × 0.120 × 0.116` | `Alarm_clock` / `clock` | On a nightstand; modern gray alarm clock |
| 6 | 122 | `0_SM_003_Deco001` | `0.118 × 0.067 × 0.125` | `Jewelry_box` / `shelf` | On vanity; tiered jewelry organizer |
| 7 | 161 | `0_SM_Deco_Sec004` | `0.023 × 0.023 × 0.086` | `Nail_polish` / `nail` | On vanity; small nail-polish bottle |
| 8 | 440 | `17_SM_Decor_5` | `0.119 × 0.119 × 0.157` | `Skincare_product` / `bottle` | On vanity; frosted skincare bottle |
| 9 | 842 | `44_sk64_HairDryer01` | `0.339 × 0.147 × 0.066` | `Hair_dryer` / `hairdryer` | On vanity; black hair dryer with pink accents |
| 10 | 486 | `19_SM_Cloth_3` | `0.609 × 0.160 × 0.839` | `Hanging_cloth` / `hanger` | Hanging at wardrobe/dressing zone; rust-brown shirt on hanger |
| 11 | 1409 | `a_SM_Curtain_Open_01` | `1.891 × 0.227 × 2.413` | `Curtain` / `curtain` | Window/wall; tied-back dark-gray curtain panels if architecture has a matching window |
| 12 | 241 | `0_SM_Stool_Fir001` | `0.789 × 0.334 × 0.495` | `Low_stool` / `bench` | Floor; teal upholstered dressing or foot-of-bed bench |
| 13 | 1984 | `d_1000003759813` | `0.633 × 0.210 × 0.338` | `Pillow` / `pillow` | On bed or armchair; teal lumbar pillow |
| 14 | 590 | `22_SM_Rug_Stacked_01h` | `0.508 × 0.491 × 0.077` | `Folded_fabric` / `blanket` | On armchair, bench, or dresser; deliberately used as folded blanket, not rug |
| 15 | 480 | `19_SM_Book_7` | `0.025 × 0.235 × 0.312` | `Book` / `desk` | On bookshelf or desk; freelancer guide book |
| 16 | 1433 | `a_SM_desk_props_notebook` | `0.013 × 0.110 × 0.155` | `Notebook_stationery` / `notebook` | On desk; spiral notebook with sketches |
| 17 | 157 | `0_SM_Coffee_cup_2` | `0.081 × 0.107 × 0.098` | `Water_cup` / `mug` | On desk; dark-gray coffee mug |
| 18 | 549 | `21_SM_Picture_Frames_01b` | `0.614 × 0.027 × 0.448` | `Wall_mounted_picture_frame` / `frame` | Wall; black-and-white coastal photograph |
| 19 | 1622 | `a_SM_Plant_01a` | `0.960 × 0.739 × 1.380` | `Large_potted_plant` / `vase` | Floor; tall Monstera in an unoccupied corner |
| 20 | 1413 | `a_SM_Decor_3` | `0.065 × 0.065 × 0.058` | `Desktop_ornament` / `vase` | On nightstand or vanity; small white candle holder |

### User-selected additions

The user selected candidate numbers `1, 3, 5, 13`, each with count one. The
new monitor replaces the deleted retired monitor. The keyboard, alarm clock,
and lumbar pillow are additions. Candidate `2` was not selected, so the deleted
false rug has no replacement in the frozen inventory.

| Original candidate | Count | CSV id | Exact asset ID | CSV bbox W×D×H (m) | Intended support | Provisional role |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 742 | `41_ComputerSet_03` | `0.480 × 0.198 × 0.445` | on object | Reviewed monitor on the desk, replacing `computer_monitor` |
| 3 | 1 | 197 | `0_SM_Keyboard` | `0.326 × 0.137 × 0.010` | on object | Keyboard in front of the replacement monitor |
| 5 | 1 | 434 | `17_SM_Clock` | `0.120 × 0.120 × 0.116` | on object | Modern alarm clock on one nightstand |
| 13 | 1 | 1984 | `d_1000003759813` | `0.633 × 0.210 × 0.338` | on object | Teal lumbar pillow on the bed or reading armchair |

### Candidate-selection constraints

- Candidate `1` is the only proposed replacement for the deleted monitor and
  must not retain the old `5_SM_PC_B_Monitor` binding.
- Candidate `2` is the only proposed replacement for the false rug; unlike the
  removed object, it is an actual near-planar floor carpet.
- Candidates `3`--`9` and `13`--`17`, `20` require explicit supporting objects;
  do not flatten them to floor placement.
- Candidate `10` requires a defensible hanging/wardrobe support convention.
- Candidate `11` is eligible only if a matching native window/wall opening is
  present; it must not invent architecture.
- Candidate `18` requires wall support. Candidate `19` requires sufficient
  floor clearance.
- All four selected asset directories exist. None is materialized into the
  executable FrozenAssets case by this review-only update.

### Materialization still required

1. Remove `computer_monitor` and its retired `5_SM_PC_B_Monitor` binding;
   recreate the workstation monitor slot bound exactly to
   `41_ComputerSet_03`.
2. Remove `area_rug` and its false-rug `22_SM_Rug_Stacked_01g` binding without
   creating a replacement rug slot.
3. Add unique keyboard, alarm-clock, and lumbar-pillow slots without changing
   other retained S103 identities.
4. Bind the monitor and keyboard to the desk, the clock to one selected
   nightstand, and the pillow to either the bed or armchair; preserve these
   support relationships explicitly.
5. Add immutable catalog records and source hashes, then re-run local-asset,
   frozen-binding, architecture, scale, support, and harness-eligibility
   preflight.

## S104 — media, music, and game recreation room

Review status: `user_selected_pending_materialization`

Recorded: `2026-09-05` (`Asia/Shanghai`)

### Baseline scene

- Model: `HY4-SFT0812`
- SceneBoard dataset key: `hy4_sft0812`
- Displayed liveboard score: `89.13`
- Source score: `92.45`
- Coverage: `100.0`
- Existing object count: `27`
- Canonical scene:
  `Support/datasets/hy_server_generations_v1/hy4-sft0812/S104/scene/canonical_scene.json`
- Canonical scene SHA-256:
  `7b690fd0b9047c353db1430c64af82946221e1f36142d22cc776f352ae8cc744`
- Blender scene:
  `Support/datasets/hy_server_generations_v1/hy4-sft0812/S104/prepared/evaluation.blend`
- Blender scene SHA-256:
  `ec3af01aba4dc54150c1c474768d8f47f88265e99642e26d7a77642b6d416096`

### User removal decision

Remove exactly one object: `area_rug`, bound to exact asset
`22_SM_Rug_Stacked_01g` (CSV row `589`). Its `Folded_fabric` class,
description, and `0.572 x 0.542 x 0.063 m` dimensions confirm it is folded
fabric rather than a floor rug. Preserve the source Blender/native scene
unchanged; apply the removal only when the approved FrozenAssets case is
materialized.

### Candidate shortlist

All twenty exact CSV assets and local directories were verified. Candidate `1`
is the faithful floor-rug replacement. The remaining candidates add game,
music, media-control, reading, comfort, lighting, and decorative detail. Only
the rows repeated in the user-selected table below are approved for later
materialization.

| Candidate | CSV id | Exact asset ID | CSV bbox W×D×H (m) | CSV class/category | Suggested placement / role |
| ---: | ---: | --- | --- | --- | --- |
| 1 | 891 | `45_Capet05` | `2.400 × 1.700 × 0.010` | `Carpet` / `rug` | Floor; actual muted gray/teal geometric rug replacing the folded fabric |
| 2 | 59 | `0_gamepad_2k_packed` | `0.404 × 0.435 × 0.019` | `Game_console` / `controller` | On coffee or side table; wired vintage controller |
| 3 | 2034 | `e_ps5_03_set` | `0.160 × 0.171 × 0.139` | `Game_controller` / `controller` | On media console; dual controllers on charging dock |
| 4 | 441 | `17_SM_Headphones` | `0.207 × 0.245 × 0.094` | `Gaming_headset` / `headphones` | On media console, side table, or music station |
| 5 | 521 | `20_SM_Remote` | `0.066 × 0.265 × 0.025` | `TV_remote_control` / `remote` | On coffee or side table; TV remote control |
| 6 | 29 | `0_chess_set_2k_packed` | `0.553 × 0.553 × 0.112` | `Chessboard` / `board` | On board-game table; complete chess set |
| 7 | 478 | `19_SM_Book_5` | `0.033 × 0.248 × 0.290` | `Book` / `book` | On bookcase; music-themed hardcover book |
| 8 | 438 | `17_SM_Decor_10` | `0.128 × 0.255 × 0.180` | `Book` / `bookcase` | On bookcase; metal bookend with books/magazines |
| 9 | 448 | `17_SM_Magazine_2` | `0.250 × 0.324 × 0.039` | `magazine` / `stack` | On coffee table; magazine stack |
| 10 | 1569 | `a_SM_Opened_Book` | `0.052 × 0.388 × 0.239` | `Book` / `book` | On coffee/side table; open art book |
| 11 | 263 | `0_ukulele_01_2k_packed` | `0.527 × 0.178 × 0.051` | `Guitar_bass` / `ukulele` | On a suitable instrument stand or wall mount; small second instrument |
| 12 | 1646 | `a_SM_radio_scanner_mic` | `0.105 × 0.161 × 0.249` | `Desktop_microphone` / `microphone` | On media/music surface; desktop microphone |
| 13 | 1698 | `a_SM_Speaker_01` | `0.289 × 0.361 × 1.128` | `Speaker` / `speaker` | Floor; tall walnut speaker, preferably selected as a stereo pair |
| 14 | 1686 | `a_SM_SmartTV_B` | `0.101 × 0.101 × 0.037` | `Small_desktop_electronic_device` / `speaker` | On media console; compact smart speaker |
| 15 | 509 | `20_SM_Bowl_with_Oranges` | `0.254 × 0.253 × 0.146` | `Snack` / `bowl` | On board-game or coffee table; snack/fruit bowl |
| 16 | 157 | `0_SM_Coffee_cup_2` | `0.081 × 0.107 × 0.098` | `Water_cup` / `mug` | On side table; dark-gray coffee mug |
| 17 | 1984 | `d_1000003759813` | `0.633 × 0.210 × 0.338` | `Pillow` / `pillow` | On sofa or armchair; teal lumbar pillow |
| 18 | 1766 | `a_SM_Wall_Picture_2` | `0.950 × 0.020 × 1.400` | `Wall_mounted_picture_frame` / `art` | Wall; dark-blue geometric art |
| 19 | 1622 | `a_SM_Plant_01a` | `0.960 × 0.739 × 1.380` | `Large_potted_plant` / `vase` | Floor; tall Monstera in an unoccupied corner |
| 20 | 411 | `16_SM_Ceiling_Lamp` | `1.101 × 0.060 × 1.218` | `Wall_mounted_lamp_holder` / `lamp` | Ceiling; slim black linear pendant over the board-game table |

### User-selected additions

The user selected candidate numbers `1, 2, 14, 15`, each with count one. The
new floor carpet replaces the deleted false rug. The game controller, compact
smart speaker, and fruit/snack bowl are additions.

| Original candidate | Count | CSV id | Exact asset ID | CSV bbox W×D×H (m) | Intended support | Provisional role |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 891 | `45_Capet05` | `2.400 × 1.700 × 0.010` | floor | Actual geometric area rug replacing `area_rug` |
| 2 | 1 | 59 | `0_gamepad_2k_packed` | `0.404 × 0.435 × 0.019` | on object | Game controller on the coffee table or media console |
| 14 | 1 | 1686 | `a_SM_SmartTV_B` | `0.101 × 0.101 × 0.037` | on object | Compact smart speaker on the media console |
| 15 | 1 | 509 | `20_SM_Bowl_with_Oranges` | `0.254 × 0.253 × 0.146` | on object | Fruit/snack bowl on the board-game or coffee table |

### Candidate-selection constraints

- Candidate `1` is the only proposed floor-rug replacement and must not reuse
  the removed `22_SM_Rug_Stacked_01g` binding.
- Candidates `2`--`10`, `12`, and `14`--`17` require explicit supporting
  tables, shelves, consoles, seating, or cabinets; do not flatten them to the
  floor.
- Candidate `11` requires an instrument stand or wall-mount convention rather
  than unsupported placement.
- Candidate `13` is proposed as a stereo pair if selected with count two; a
  single instance is also valid only if the user explicitly chooses count one.
- Candidate `18` requires wall support, and candidate `20` requires ceiling
  support. Candidate `19` requires sufficient floor clearance.
- All four selected asset directories exist. None is materialized into the
  executable FrozenAssets case by this review-only update.

### Materialization still required

1. Remove `area_rug` and its false-rug `22_SM_Rug_Stacked_01g` binding; recreate
   the rug slot bound exactly to `45_Capet05` at fixed native scale.
2. Add unique game-controller, compact-smart-speaker, and fruit-bowl slots
   without changing other retained S104 identities.
3. Bind the game controller and smart speaker to a media surface and the bowl
   to either the board-game or coffee table; preserve support explicitly.
4. Confirm the `2.400 x 1.700 m` carpet footprint retains circulation and does
   not intersect game/music furniture.
5. Add immutable catalog records and source hashes, then re-run local-asset,
   frozen-binding, architecture, scale, support, and harness-eligibility
   preflight.
