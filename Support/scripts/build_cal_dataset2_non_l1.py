#!/usr/bin/env python3
"""Build the human-review draft for non-L1 VLM visual-evidence calibration.

The dataset is deliberately model-free and local-only.  It creates controlled
counterfactual scenes for all currently planned VLM-involved L2/L3 metrics, but
it does *not* promote assistant-authored labels to benchmark ground truth.

Dataset lifecycle:

1. this builder creates ``pending_human`` draft fixtures;
2. a human audits prompts, object identities, scene deltas, and proposed labels;
3. a separate approval step may freeze contracts/GT;
4. only then may render/judge experiments claim accuracy.

No API, renderer, VLM, or remote environment is invoked here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.evaluator.scene_quality.authorized_deviations import (  # noqa: E402
    validate_authorized_deviations,
)
from benchmark.evaluator.scene_quality.interfaces import (  # noqa: E402
    SCENE_QUALITY_INTERFACE_METRICS,
    normalize_metric_name,
)
from benchmark.evaluator.specification_fidelity.contract import (  # noqa: E402
    ACCEPTED_SPECIFICATION_CLAIM_FAMILIES,
    validate_specification_contract,
)
from benchmark.legend.workflow.grouping import build_object_grouping_report  # noqa: E402
from benchmark.reference_annotation import validate_reference_annotation  # noqa: E402
from benchmark.scene_io.validate import (  # noqa: E402
    validate_asset_selection,
    validate_generated_scene,
    validate_object_plan,
    validate_scene_request,
)


DATASET_ID = "cal_dataset2_non_l1_evidence"
DATASET_SCHEMA_VERSION = "non_l1_visual_evidence_calibration_v1"
CASE_SCHEMA_VERSION = "non_l1_calibration_case_v1"
BUILD_DATE = "2026-07-25"
CASE_COUNT = 108
BASE_CASES_PER_METRIC = 12
# Historical case-slice labels are retained so the original 108-case design is
# reproducible. ``canonical_metric`` is emitted separately and follows the
# simplified ownership contract.
METRICS = (
    "oor",
    "oar",
    "room_scene_type",
    "broad_semantic_intent",
    "required_functional_areas",
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
L2_METRICS = METRICS[:5]
L3_METRICS = METRICS[5:]
PROPOSED_LABELS = ("valid", "invalid", "ambiguous")
PUBLIC_CASE_ID_PREFIX = "n2"
COORDINATE_FRAME = {
    "origin": "room_min_corner_floor",
    "axes": "x_width_y_depth_z_up",
    "unit": "meter",
    "rotation_unit": "degree",
}
ASSET_POLICY = {
    "mode": "fixed_catalog_selection",
    "identity_owner": "benchmark",
    "category_selection_owner": "generator",
    "scale_owner": "generator",
    "appearance_owner": "generator",
    "arrangement_owner": "generator",
}
GROUPING_POLICY = {
    "policy_id": "deterministic_metadata_geometry",
    "implementation": "src/benchmark/legend/workflow/grouping.py",
    "config": "configs/grouping/deterministic_metadata_geometry.yaml",
    "role": "evidence_partition_not_metric_verdict",
}
PRIVATE_FILENAMES = {
    "metric_gt.json",
    "evidence_expectations.json",
    "construction_manifest.json",
    "provenance.json",
    "review.json",
}
FORBIDDEN_PUBLIC_LABEL_TOKENS = (
    "valid",
    "invalid",
    "wrong",
    "obvious",
    "subtle",
    "giant",
    "outlier",
)


@dataclass(frozen=True)
class AssetRecord:
    jid: str
    category: str
    short_desc: str
    desc: str
    size: tuple[float, float, float]
    scaling_strategy: str | None
    inner_placement: bool
    align_to_wall_normal: bool


@dataclass
class CaseDraft:
    metric: str
    scenario_id: str
    prompt: str
    prompt_granularity: str
    scene: dict[str, Any]
    proposed_label: str
    target_ids: list[str]
    event_relation: str | None
    claims: dict[str, list[dict[str, Any]]]
    reference_oor: list[dict[str, Any]]
    reference_oar: list[dict[str, Any]]
    review_question: str
    gt_basis: str
    required_visible_facts: list[str]
    declared_delta: dict[str, Any]
    difficulty: str
    asset_policy: dict[str, Any]
    authorized_deviations: list[dict[str, Any]]
    source_scene_ids: list[str]
    design_role: str = "base_metric_case"
    counterfactual_group_id: str | None = None
    prompt_authorization: str | None = None
    scene_deviation: str | None = None
    visual_source_case_id: str | None = None
    raw_coherence_label: str | None = None
    authorization_applied: bool = False
    l2_anomaly_label: str | None = None
    resolution_reason: str | None = None
    expected_cross_metric_effects: list[str] | None = None


class DatasetBuildError(RuntimeError):
    """Raised when a draft cannot satisfy construction-time invariants."""


class Builder:
    def __init__(self, out_root: Path) -> None:
        self.out_root = out_root
        self.catalog_path = (
            REPO_ROOT
            / "Support"
            / "Assets"
            / "imaginarium_assets"
            / "imaginarium_asset_info.csv"
        )
        self.asset_root = self.catalog_path.parent
        self.catalog = _read_catalog(self.catalog_path)
        grouping_path = REPO_ROOT / GROUPING_POLICY["config"]
        self.grouping_config = yaml.safe_load(grouping_path.read_text(encoding="utf-8"))
        self.case_counter = 0
        self.case_records: list[dict[str, Any]] = []
        self.review_rows: list[dict[str, Any]] = []
        self.prompt_rows: list[dict[str, Any]] = []
        self.counterfactual_rows: list[dict[str, Any]] = []
        self.geometry_to_first_case: dict[str, str] = {}
        self.group_id_map: dict[str, str] = {}

    def build(self) -> None:
        if self.out_root.exists() and any(self.out_root.iterdir()):
            raise DatasetBuildError(
                f"refusing to overwrite non-empty dataset directory: {self.out_root}"
            )
        self.out_root.mkdir(parents=True, exist_ok=True)
        for dirname in ("fixtures", "configs", "review/previews", "validation"):
            (self.out_root / dirname).mkdir(parents=True, exist_ok=True)

        drafts: list[CaseDraft] = []
        drafts.extend(self._build_oor_drafts())
        drafts.extend(self._build_oar_drafts())
        drafts.extend(self._build_room_type_drafts())
        drafts.extend(self._build_broad_intent_drafts())
        drafts.extend(self._build_required_area_drafts())
        drafts.extend(self._build_scale_drafts())
        drafts.extend(self._build_pairing_drafts())
        drafts.extend(self._build_style_drafts())
        drafts.extend(self._build_authorization_drafts())

        if len(drafts) != CASE_COUNT:
            raise DatasetBuildError(
                f"internal case-plan mismatch: expected {CASE_COUNT}, got {len(drafts)}"
            )
        for draft in drafts:
            self._write_case(draft)

        self._write_dataset_files()

    # ------------------------------------------------------------------
    # Scene/asset helpers
    # ------------------------------------------------------------------
    def object(
        self,
        jid: str,
        x: float,
        y: float,
        *,
        base_z: float = 0.0,
        rotation_z: float = 0.0,
        scale: float | tuple[float, float, float] = 1.0,
        category: str | None = None,
        description: str | None = None,
        object_id: str | None = None,
        source_region_id: str | None = None,
        support_parent: str | None = None,
    ) -> dict[str, Any]:
        asset = self._asset(jid)
        factors = (
            (float(scale), float(scale), float(scale))
            if isinstance(scale, (int, float))
            else tuple(float(value) for value in scale)
        )
        if len(factors) != 3 or any(value <= 0 for value in factors):
            raise DatasetBuildError(f"invalid scale for {jid}: {scale!r}")
        size = [round(asset.size[i] * factors[i], 6) for i in range(3)]
        result: dict[str, Any] = {
            "id": object_id or "",
            "category": category or asset.category,
            "description": description or asset.short_desc,
            "desc": description or asset.short_desc,
            "short_desc": description or asset.short_desc,
            "jid": jid,
            "size": size,
            "center": [round(float(x), 6), round(float(y), 6), round(base_z + size[2] / 2.0, 6)],
            "rotation": [0.0, 0.0, round(float(rotation_z), 6)],
            "geometry_provenance": "asset_mesh",
            "asset_ref": {
                "source_db": "imaginarium",
                "asset_key": jid,
                "mesh_uri": None,
                "pointcloud_uri": None,
                "metadata_uri": None,
            },
            "asset_proxy": {
                "type": "canonical_obb",
                "bbox_center_local": [0.0, 0.0, 0.0],
                "bbox_size": size,
            },
            "metadata": {
                "interactive": False,
                "plan_object_id": object_id or "",
                "catalog_scale_factors": list(factors),
            },
        }
        if source_region_id:
            result["source_region_id"] = source_region_id
            result["metadata"]["source_region_id"] = source_region_id
        if support_parent:
            result["support_parent"] = support_parent
        return result

    def scene(
        self,
        objects: Iterable[dict[str, Any]],
        *,
        width: float = 8.0,
        depth: float = 6.0,
        height: float = 3.2,
        scene_type: str = "room",
        semantic_regions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        normalized: list[dict[str, Any]] = []
        for index, raw in enumerate(objects):
            obj = deepcopy(raw)
            object_id = f"obj_{index:03d}"
            obj["id"] = object_id
            obj.setdefault("metadata", {})["plan_object_id"] = object_id
            normalized.append(obj)
        result: dict[str, Any] = {
            "schema_version": "canonical_scene_v1",
            "scene_id": "pending",
            "request_id": "pending",
            "scene_type": scene_type,
            "boundary": [[0.0, 0.0], [float(width), 0.0], [float(width), float(depth)], [0.0, float(depth)]],
            "scene_height": float(height),
            "objects": normalized,
            "metadata": {
                "coordinate_frame": deepcopy(COORDINATE_FRAME),
                "asset_policy": deepcopy(ASSET_POLICY),
                "semantic_ground_truth_visibility": "benchmark_private",
            },
        }
        if semantic_regions:
            result["metadata"]["semantic_regions"] = deepcopy(semantic_regions)
        return result

    def clone_scene(self, scene: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(scene)

    def _asset(self, jid: str) -> AssetRecord:
        try:
            return self.catalog[jid]
        except KeyError as exc:
            raise DatasetBuildError(f"asset {jid!r} is not in fixed catalog") from exc

    @staticmethod
    def _find(scene: dict[str, Any], object_id: str) -> dict[str, Any]:
        for obj in scene["objects"]:
            if obj["id"] == object_id:
                return obj
        raise DatasetBuildError(f"scene has no object {object_id!r}")

    @staticmethod
    def _set_xy(scene: dict[str, Any], object_id: str, x: float, y: float) -> None:
        obj = Builder._find(scene, object_id)
        obj["center"][0] = round(float(x), 6)
        obj["center"][1] = round(float(y), 6)

    @staticmethod
    def _set_yaw(scene: dict[str, Any], object_id: str, yaw: float) -> None:
        Builder._find(scene, object_id)["rotation"][2] = round(float(yaw), 6)

    @staticmethod
    def _set_base_z(scene: dict[str, Any], object_id: str, base_z: float) -> None:
        obj = Builder._find(scene, object_id)
        obj["center"][2] = round(float(base_z) + float(obj["size"][2]) / 2.0, 6)

    @staticmethod
    def _scale_about_base(
        scene: dict[str, Any],
        object_id: str,
        factor: float | tuple[float, float, float],
    ) -> None:
        obj = Builder._find(scene, object_id)
        factors = (
            (float(factor), float(factor), float(factor))
            if isinstance(factor, (int, float))
            else tuple(float(value) for value in factor)
        )
        base = float(obj["center"][2]) - float(obj["size"][2]) / 2.0
        obj["size"] = [round(float(obj["size"][i]) * factors[i], 6) for i in range(3)]
        obj["asset_proxy"]["bbox_size"] = list(obj["size"])
        old = obj["metadata"].get("catalog_scale_factors") or [1.0, 1.0, 1.0]
        obj["metadata"]["catalog_scale_factors"] = [
            round(float(old[i]) * factors[i], 6) for i in range(3)
        ]
        obj["center"][2] = round(base + float(obj["size"][2]) / 2.0, 6)

    @staticmethod
    def _place_on_top(
        scene: dict[str, Any],
        child_id: str,
        support_id: str,
    ) -> None:
        """Seat ``child_id`` exactly on the canonical top of ``support_id``."""

        child = Builder._find(scene, child_id)
        support = Builder._find(scene, support_id)
        support_top = float(support["center"][2]) + float(support["size"][2]) / 2.0
        child["center"][2] = round(
            support_top + float(child["size"][2]) / 2.0,
            6,
        )
        child["support_parent"] = support_id

    def _replace_asset(
        self,
        scene: dict[str, Any],
        object_id: str,
        jid: str,
        *,
        preserve_base: bool = True,
        scale: float = 1.0,
        category: str | None = None,
    ) -> None:
        obj = self._find(scene, object_id)
        asset = self._asset(jid)
        base = float(obj["center"][2]) - float(obj["size"][2]) / 2.0
        new_size = [round(value * scale, 6) for value in asset.size]
        obj.update(
            {
                "jid": jid,
                "category": category or asset.category,
                "description": asset.short_desc,
                "desc": asset.short_desc,
                "short_desc": asset.short_desc,
                "size": new_size,
                "asset_ref": {
                    "source_db": "imaginarium",
                    "asset_key": jid,
                    "mesh_uri": None,
                    "pointcloud_uri": None,
                    "metadata_uri": None,
                },
                "asset_proxy": {
                    "type": "canonical_obb",
                    "bbox_center_local": [0.0, 0.0, 0.0],
                    "bbox_size": new_size,
                },
            }
        )
        obj["metadata"]["catalog_scale_factors"] = [scale, scale, scale]
        if preserve_base:
            obj["center"][2] = round(base + new_size[2] / 2.0, 6)

    def _replace_asset_and_reseat(
        self,
        scene: dict[str, Any],
        object_id: str,
        jid: str,
        *,
        supported_child_id: str | None = None,
    ) -> None:
        self._replace_asset(scene, object_id, jid)
        if supported_child_id is not None:
            self._place_on_top(scene, supported_child_id, object_id)

    # ------------------------------------------------------------------
    # Reusable semantic layouts
    # ------------------------------------------------------------------
    def bedroom_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("0_SM_Bed", 3.0, 4.3, rotation_z=180.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 1.65, 4.35),
                self.object("0_painted_wooden_nightstand_2k_packed", 4.35, 4.35),
                self.object(
                    "0_SM_Deco029",
                    1.65,
                    4.35,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
                self.object(
                    "0_SM_Deco029",
                    4.35,
                    4.35,
                    base_z=0.616,
                    support_parent="obj_002",
                ),
            ],
            width=6.0,
            depth=6.0,
            scene_type="room",
        )

    def office_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("a_SM_desk_compiled", 4.0, 4.5, rotation_z=0.0, scale=0.85),
                self.object("b_8", 4.0, 3.4, rotation_z=0.0),
                self.object(
                    "44_sk20_MR01",
                    4.0,
                    4.45,
                    base_z=0.71825,
                    rotation_z=0.0,
                    scale=0.8,
                    support_parent="obj_000",
                ),
                self.object("0_steel_frame_shelves_03_2k_packed", 1.25, 4.9, rotation_z=0.0, scale=0.75),
                self.object("0_SM_Deco029", 6.5, 4.8, rotation_z=0.0),
            ],
            width=8.0,
            depth=6.5,
            scene_type="room",
        )

    def dining_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("b_47", 4.0, 3.0, rotation_z=0.0),
                self.object("d_1000004220137", 2.75, 3.0, rotation_z=-90.0),
                self.object("d_1000004220137", 5.25, 3.0, rotation_z=90.0),
                self.object("d_1000004220137", 4.0, 1.9, rotation_z=0.0),
                self.object("d_1000004220137", 4.0, 4.1, rotation_z=180.0),
            ],
            width=8.0,
            depth=6.0,
            scene_type="room",
        )

    def media_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("a_LeatherSofa", 2.2, 3.0, rotation_z=90.0),
                self.object("0_modern_wooden_cabinet_2k_packed", 6.3, 3.0, rotation_z=-90.0),
                self.object("15_SM_Floor_Lamp", 2.0, 1.2),
                self.object("b_85", 4.1, 3.0, rotation_z=90.0),
                self.object("18_SM_Carpet", 3.8, 3.0, scale=0.9),
                self.object(
                    "a_SM_TV_02",
                    6.3,
                    3.0,
                    base_z=0.68,
                    rotation_z=90.0,
                    support_parent="obj_001",
                ),
            ],
            width=8.5,
            depth=6.0,
            scene_type="room",
        )

    def bathroom_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("0_SM_Toilet", 1.2, 2.0, rotation_z=90.0),
                self.object("17_SM_Shower_Cabin", 1.15, 4.7, rotation_z=0.0),
                self.object("19_SM_Sink", 4.8, 4.7, rotation_z=180.0, scale=0.8),
                self.object("18_SM_Wall_Clock", 4.8, 5.9665, base_z=1.35, rotation_z=0.0),
            ],
            width=6.0,
            depth=6.0,
            scene_type="room",
        )

    def reading_layout(self) -> dict[str, Any]:
        return self.scene(
            [
                self.object("0_modern_arm_chair_01_2k_packed", 2.2, 3.2, rotation_z=-30.0),
                self.object("0_modern_arm_chair_01_2k_packed", 5.8, 3.2, rotation_z=30.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 4.0, 3.35),
                self.object("15_SM_Floor_Lamp", 4.0, 4.8),
                self.object(
                    "0_SM_Book001_02",
                    4.0,
                    3.35,
                    base_z=0.616,
                    support_parent="obj_002",
                ),
            ],
            width=8.0,
            depth=6.5,
            scene_type="room",
        )

    # ------------------------------------------------------------------
    # Draft constructors
    # ------------------------------------------------------------------
    def _triplet(
        self,
        *,
        metric: str,
        scenario_id: str,
        prompt: str,
        prompt_granularity: str,
        base_scene: dict[str, Any],
        target_ids: list[str],
        event_relation: str | None,
        claims_factory: Callable[[str], dict[str, list[dict[str, Any]]]],
        reference_factory: Callable[[str], tuple[list[dict[str, Any]], list[dict[str, Any]]]],
        mutators: dict[str, Callable[[dict[str, Any]], None]],
        review_question: str,
        required_visible_facts: list[str],
        delta_descriptions: dict[str, dict[str, Any]],
        difficulties: dict[str, str] | None = None,
        source_scene_ids: list[str] | None = None,
        expected_cross_metric_effects: list[str] | None = None,
    ) -> list[CaseDraft]:
        result: list[CaseDraft] = []
        for label in PROPOSED_LABELS:
            scene = self.clone_scene(base_scene)
            mutators[label](scene)
            claims = claims_factory(prompt)
            oor, oar = reference_factory(prompt)
            result.append(
                CaseDraft(
                    metric=metric,
                    scenario_id=scenario_id,
                    prompt=prompt,
                    prompt_granularity=prompt_granularity,
                    scene=scene,
                    proposed_label=label,
                    target_ids=list(target_ids),
                    event_relation=event_relation,
                    claims=claims,
                    reference_oor=oor,
                    reference_oar=oar,
                    review_question=review_question,
                    gt_basis=(
                        "controlled_counterfactual"
                        if label != "ambiguous"
                        else "controlled_boundary_case_pending_human"
                    ),
                    required_visible_facts=list(required_visible_facts),
                    declared_delta=deepcopy(delta_descriptions[label]),
                    difficulty=(difficulties or {}).get(
                        label, "clear" if label != "ambiguous" else "boundary"
                    ),
                    asset_policy=deepcopy(ASSET_POLICY),
                    authorized_deviations=[],
                    source_scene_ids=list(source_scene_ids or []),
                    expected_cross_metric_effects=list(expected_cross_metric_effects or []),
                )
            )
        return result

    @staticmethod
    def _empty_claims() -> dict[str, list[dict[str, Any]]]:
        return {
            family: [] for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES
        }

    def _merge_claim_sets(
        self, *claim_sets: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        merged = self._empty_claims()
        for claim_set in claim_sets:
            for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES:
                merged[family].extend(deepcopy(claim_set.get(family) or []))
        return merged

    @staticmethod
    def _span(prompt: str, text: str) -> dict[str, Any]:
        start = prompt.find(text)
        if start < 0:
            raise DatasetBuildError(f"claim span {text!r} is not in prompt {prompt!r}")
        return {
            "source_span": text,
            "source_span_start": start,
            "source_span_end": start + len(text),
        }

    def _oor_claims(
        self,
        prompt: str,
        *,
        claim_id: str,
        relation_type: str,
        target_ids: list[str],
        source_span: str,
    ) -> dict[str, list[dict[str, Any]]]:
        claims = self._empty_claims()
        claims["oor"].append(
            {
                "claim_id": claim_id,
                "claim_family": "oor",
                "required": True,
                "relation_id": claim_id,
                "relation_type": relation_type,
                "target_ids": list(target_ids),
                "expected": {"satisfied": True},
                **self._span(prompt, source_span),
            }
        )
        return claims

    def _oar_claims(
        self,
        prompt: str,
        *,
        claim_id: str,
        relation_type: str,
        subject_id: str,
        architecture: str,
        source_span: str,
    ) -> dict[str, list[dict[str, Any]]]:
        claims = self._empty_claims()
        claims["oar"].append(
            {
                "claim_id": claim_id,
                "claim_family": "oar",
                "required": True,
                "relation_id": claim_id,
                "relation_type": relation_type,
                "subject_id": subject_id,
                "architectural_element": architecture,
                "target_ids": [subject_id],
                "expected": {"satisfied": True},
                **self._span(prompt, source_span),
            }
        )
        return claims

    def _high_level_claims(
        self,
        prompt: str,
        *,
        family: str,
        records: list[tuple[str, str, dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        claims = self._empty_claims()
        for claim_id, source_span, expected in records:
            claims[family].append(
                {
                    "claim_id": claim_id,
                    "claim_family": family,
                    "required": True,
                    "expected": deepcopy(expected),
                    **self._span(prompt, source_span),
                }
            )
        return claims

    @staticmethod
    def _reference_pair(
        relation_id: str,
        relation_type: str,
        subject_id: str,
        object_id: str,
        raw_relation: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [
                {
                    "relation_id": relation_id,
                    "type": relation_type,
                    "raw_relation": raw_relation,
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "claim_state": "confirmed",
                }
            ],
            [],
        )

    @staticmethod
    def _reference_group(
        relation_id: str,
        relation_type: str,
        target_ids: list[str],
        raw_relation: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [
                {
                    "relation_id": relation_id,
                    "type": relation_type,
                    "raw_relation": raw_relation,
                    "subject_ids": list(target_ids),
                    "claim_state": "confirmed",
                }
            ],
            [],
        )

    @staticmethod
    def _reference_oar(
        relation_id: str,
        relation_type: str,
        subject_id: str,
        architecture: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return (
            [],
            [
                {
                    "relation_id": relation_id,
                    "type": relation_type,
                    "subject_id": subject_id,
                    "architectural_element": architecture,
                    "claim_state": "confirmed",
                }
            ],
        )

    # ------------------------------------------------------------------
    # L2 OOR: four three-way counterfactual families
    # ------------------------------------------------------------------
    def _build_oor_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []

        # OOR-1: bed centered between two nightstands.  Unknown predicate ->
        # mandatory VLM fallback rather than deterministic "between".
        base = self.bedroom_layout()
        prompt = (
            "Create a calm bedroom with a bed and two matching nightstands. "
            "Visually center the bed between the two nightstands."
        )
        span = "Visually center the bed between the two nightstands"
        cid = "oor::bed_visually_centered_between_nightstands"
        drafts += self._triplet(
            metric="oor",
            scenario_id="oor_centering_bed",
            prompt=prompt,
            prompt_granularity="fine_grained",
            base_scene=base,
            target_ids=["obj_000", "obj_001", "obj_002"],
            event_relation="visually_centered_between",
            claims_factory=lambda p: self._oor_claims(
                p,
                claim_id=cid,
                relation_type="visually_centered_between",
                target_ids=["obj_000", "obj_001", "obj_002"],
                source_span=span,
            ),
            reference_factory=lambda p: (
                [
                    {
                        "relation_id": cid,
                        "type": "visually_centered_between",
                        "raw_relation": span,
                        "subject_id": "obj_000",
                        "object_ids": ["obj_001", "obj_002"],
                        "claim_state": "confirmed",
                    }
                ],
                [],
            ),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._set_xy(s, "obj_001", 0.35, 4.35),
                    self._set_xy(s, "obj_003", 0.35, 4.35),
                ),
                "ambiguous": lambda s: (
                    self._set_xy(s, "obj_001", 1.05, 4.35),
                    self._set_xy(s, "obj_003", 1.05, 4.35),
                ),
            },
            review_question="Is the bed visually centered between the two matching nightstands?",
            required_visible_facts=[
                "bed silhouette and center",
                "both nightstand silhouettes and centers",
                "relative left-right spacing among all three targets",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {
                    "operation": "translate_supported_pair",
                    "target_ids": ["obj_001", "obj_003"],
                    "delta_m": [-1.3, 0.0, 0.0],
                },
                "ambiguous": {
                    "operation": "translate_supported_pair",
                    "target_ids": ["obj_001", "obj_003"],
                    "delta_m": [-0.6, 0.0, 0.0],
                },
            },
        )

        # OOR-2: symmetric flanking around a dining table.
        base = self.dining_layout()
        prompt = (
            "Create a dining setting with a rectangular table and four matching chairs. "
            "Arrange the two side chairs so they symmetrically flank the table."
        )
        span = "Arrange the two side chairs so they symmetrically flank the table"
        cid = "oor::side_chairs_symmetrically_flank_table"
        drafts += self._triplet(
            metric="oor",
            scenario_id="oor_symmetric_flanking",
            prompt=prompt,
            prompt_granularity="fine_grained",
            base_scene=base,
            target_ids=["obj_001", "obj_002", "obj_000"],
            event_relation="symmetrically_flank",
            claims_factory=lambda p: self._oor_claims(
                p,
                claim_id=cid,
                relation_type="symmetrically_flank",
                target_ids=["obj_001", "obj_002", "obj_000"],
                source_span=span,
            ),
            reference_factory=lambda p: (
                [
                    {
                        "relation_id": cid,
                        "type": "symmetrically_flank",
                        "raw_relation": span,
                        "subject_ids": ["obj_001", "obj_002"],
                        "object_ids": ["obj_000"],
                        "claim_state": "confirmed",
                    }
                ],
                [],
            ),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._set_xy(s, "obj_001", 2.65, 3.0),
                    self._set_xy(s, "obj_002", 2.65, 4.05),
                ),
                "ambiguous": lambda s: self._set_xy(s, "obj_002", 5.15, 3.55),
            },
            review_question="Do the two designated side chairs symmetrically flank the dining table?",
            required_visible_facts=[
                "table footprint",
                "both designated side-chair footprints",
                "bilateral spacing on both sides of the table",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {
                    "operation": "translate_pair",
                    "target_ids": ["obj_001", "obj_002"],
                    "purpose": "place both designated chairs on the same side",
                },
                "ambiguous": {
                    "operation": "translate",
                    "target_ids": ["obj_002"],
                    "purpose": "introduce moderate asymmetric offset",
                },
            },
        )

        # OOR-3: facing a monitor; deliberately uses an asset with a visible back.
        base = self.office_layout()
        prompt = (
            "Create a compact workstation with a desk, a monitor, and a high-backed chair. "
            "Make the chair face toward the monitor."
        )
        span = "Make the chair face toward the monitor"
        cid = "oor::chair_faces_toward_monitor"
        drafts += self._triplet(
            metric="oor",
            scenario_id="oor_chair_facing",
            prompt=prompt,
            prompt_granularity="fine_grained",
            base_scene=base,
            target_ids=["obj_001", "obj_002"],
            event_relation="faces_toward",
            claims_factory=lambda p: self._oor_claims(
                p,
                claim_id=cid,
                relation_type="faces_toward",
                target_ids=["obj_001", "obj_002"],
                source_span=span,
            ),
            reference_factory=lambda p: self._reference_pair(
                cid, "faces_toward", "obj_001", "obj_002", span
            ),
            mutators={
                "valid": lambda s: self._set_yaw(s, "obj_001", 0.0),
                "invalid": lambda s: self._set_yaw(s, "obj_001", 180.0),
                "ambiguous": lambda s: self._set_yaw(s, "obj_001", 72.0),
            },
            review_question=(
                "Does the high-backed chair face toward the monitor? "
                "First verify the chair asset's visible front/back convention."
            ),
            required_visible_facts=[
                "chair front/back orientation cues",
                "monitor screen/back orientation cues",
                "chair-to-monitor direction in the room plane",
            ],
            delta_descriptions={
                "valid": {"operation": "set_yaw", "target_ids": ["obj_001"], "yaw_deg": 0.0},
                "invalid": {"operation": "set_yaw", "target_ids": ["obj_001"], "yaw_deg": 180.0},
                "ambiguous": {"operation": "set_yaw", "target_ids": ["obj_001"], "yaw_deg": 72.0},
            },
        )

        # OOR-4: even row of three chairs.  The relation is holistic rather than
        # the deterministic "ordered" predicate.
        base = self.scene(
            [
                self.object("d_1000004220137", 2.0, 3.0, rotation_z=0.0),
                self.object("d_1000004220137", 4.0, 3.0, rotation_z=0.0),
                self.object("d_1000004220137", 6.0, 3.0, rotation_z=0.0),
                self.object("0_SM_Conference_table", 4.0, 5.0, scale=0.8),
            ],
            width=8.0,
            depth=6.5,
        )
        prompt = (
            "Create a meeting-room waiting area with three matching chairs. "
            "Arrange the three chairs as one visually even row."
        )
        span = "Arrange the three chairs as one visually even row"
        cid = "oor::chairs_form_visually_even_row"
        target_ids = ["obj_000", "obj_001", "obj_002"]
        drafts += self._triplet(
            metric="oor",
            scenario_id="oor_even_row",
            prompt=prompt,
            prompt_granularity="fine_grained",
            base_scene=base,
            target_ids=target_ids,
            event_relation="forms_visually_even_row",
            claims_factory=lambda p: self._oor_claims(
                p,
                claim_id=cid,
                relation_type="forms_visually_even_row",
                target_ids=target_ids,
                source_span=span,
            ),
            reference_factory=lambda p: self._reference_group(
                cid, "forms_visually_even_row", target_ids, span
            ),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: self._set_xy(s, "obj_001", 4.0, 4.15),
                "ambiguous": lambda s: self._set_xy(s, "obj_001", 4.0, 3.32),
            },
            review_question="Do the three designated chairs form one visually even row?",
            required_visible_facts=[
                "all three chair centers",
                "row-axis alignment",
                "adjacent spacing between chair centers",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {"operation": "translate", "target_ids": ["obj_001"], "delta_m": [0.0, 1.15, 0.0]},
                "ambiguous": {"operation": "translate", "target_ids": ["obj_001"], "delta_m": [0.0, 0.32, 0.0]},
            },
        )
        return drafts

    # ------------------------------------------------------------------
    # L2 OAR: four mandatory-VLM attachment families
    # ------------------------------------------------------------------
    def _wall_attachment_scene(
        self,
        jid: str,
        *,
        wall: str,
        gap: float,
        width: float = 7.0,
        depth: float = 6.0,
        center_z: float = 1.7,
    ) -> dict[str, Any]:
        asset = self._asset(jid)
        if wall == "east_wall":
            x, y, yaw = width - asset.size[1] / 2.0 - gap, depth / 2.0, 90.0
        elif wall == "west_wall":
            x, y, yaw = asset.size[1] / 2.0 + gap, depth / 2.0, 90.0
        elif wall == "north_wall":
            x, y, yaw = width / 2.0, depth - asset.size[1] / 2.0 - gap, 0.0
        elif wall == "south_wall":
            x, y, yaw = width / 2.0, asset.size[1] / 2.0 + gap, 0.0
        else:
            raise DatasetBuildError(f"unknown wall {wall!r}")
        obj = self.object(jid, x, y, base_z=center_z - asset.size[2] / 2.0, rotation_z=yaw)
        return self.scene(
            [
                obj,
                self.object("0_SM_Sofa_2", width / 2.0, 1.7, rotation_z=0.0, scale=0.75),
                self.object("0_SM_Coffee_table_1", width / 2.0, 3.1, rotation_z=0.0, scale=0.8),
            ],
            width=width,
            depth=depth,
            height=3.2,
        )

    def _build_oar_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []
        wall_specs = [
            (
                "oar_east_frame",
                "21_SM_Picture_Frames_01b",
                "east_wall",
                "Mount the black wooden picture frame flat on the east wall.",
                "Mount the black wooden picture frame flat on the east wall",
                "framed display",
            ),
            (
                "oar_north_clock",
                "18_SM_Wall_Clock",
                "north_wall",
                "Create a lounge and mount the round wall clock flat on the north wall.",
                "mount the round wall clock flat on the north wall",
                "wall clock",
            ),
            (
                "oar_west_art",
                "a_SM_Frame_01",
                "west_wall",
                "Create a seating area and mount the framed artwork flush with the west wall.",
                "mount the framed artwork flush with the west wall",
                "framed artwork",
            ),
        ]
        for scenario_id, jid, wall, prompt, span, noun in wall_specs:
            cid = f"oar::{scenario_id}"
            base = self._wall_attachment_scene(jid, wall=wall, gap=0.0)

            def mutate_gap(scene: dict[str, Any], gap: float, wall_name: str = wall) -> None:
                obj = self._find(scene, "obj_000")
                if wall_name == "east_wall":
                    obj["center"][0] -= gap
                elif wall_name == "west_wall":
                    obj["center"][0] += gap
                elif wall_name == "north_wall":
                    obj["center"][1] -= gap
                else:
                    obj["center"][1] += gap
                obj["center"] = [round(float(value), 6) for value in obj["center"]]

            drafts += self._triplet(
                metric="oar",
                scenario_id=scenario_id,
                prompt=prompt,
                prompt_granularity="fine_grained",
                base_scene=base,
                target_ids=["obj_000"],
                event_relation="mounted_on_wall",
                claims_factory=lambda p, c=cid, w=wall, sp=span: self._oar_claims(
                    p,
                    claim_id=c,
                    relation_type="mounted_on_wall",
                    subject_id="obj_000",
                    architecture=w,
                    source_span=sp,
                ),
                reference_factory=lambda p, c=cid, w=wall: self._reference_oar(
                    c, "mounted_on_wall", "obj_000", w
                ),
                mutators={
                    "valid": lambda s: None,
                    "invalid": lambda s, fn=mutate_gap: fn(s, 0.38),
                    "ambiguous": lambda s, fn=mutate_gap: fn(s, 0.065),
                },
                review_question=f"Is the designated {noun} mounted flat on the requested {wall.replace('_', ' ')}?",
                required_visible_facts=[
                    f"designated {noun} silhouette",
                    f"local patch of the {wall.replace('_', ' ')}",
                    f"room-axis legend identifying the {wall.replace('_', ' ')}",
                    "visible object-to-wall contact or separation",
                ],
                delta_descriptions={
                    "valid": {"operation": "identity_control", "nominal_gap_m": 0.0},
                    "invalid": {"operation": "translate_inward_from_wall", "target_ids": ["obj_000"], "gap_m": 0.38},
                    "ambiguous": {"operation": "translate_inward_from_wall", "target_ids": ["obj_000"], "gap_m": 0.065},
                },
            )

        # Ceiling pendant. The top of the object is the attachment witness.
        jid = "a_SM_Ceiling_Lamp_1"
        asset = self._asset(jid)
        height = 3.4
        attached_base = height - asset.size[2]
        base = self.scene(
            [
                self.object(jid, 4.0, 3.0, base_z=attached_base),
                self.object("b_47", 4.0, 3.0),
                self.object("d_1000004220137", 2.8, 3.0, rotation_z=-90.0),
                self.object("d_1000004220137", 5.2, 3.0, rotation_z=90.0),
            ],
            width=8.0,
            depth=6.0,
            height=height,
        )
        prompt = "Create a dining area and attach the industrial pendant lamp directly to the ceiling."
        span = "attach the industrial pendant lamp directly to the ceiling"
        cid = "oar::pendant_attached_to_ceiling"
        drafts += self._triplet(
            metric="oar",
            scenario_id="oar_ceiling_pendant",
            prompt=prompt,
            prompt_granularity="fine_grained",
            base_scene=base,
            target_ids=["obj_000"],
            event_relation="attached_to_ceiling",
            claims_factory=lambda p: self._oar_claims(
                p,
                claim_id=cid,
                relation_type="attached_to_ceiling",
                subject_id="obj_000",
                architecture="ceiling",
                source_span=span,
            ),
            reference_factory=lambda p: self._reference_oar(
                cid, "attached_to_ceiling", "obj_000", "ceiling"
            ),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: self._set_base_z(s, "obj_000", attached_base - 0.48),
                "ambiguous": lambda s: self._set_base_z(s, "obj_000", attached_base - 0.06),
            },
            review_question="Is the industrial pendant lamp directly attached to the ceiling?",
            required_visible_facts=[
                "pendant top attachment point",
                "local ceiling patch",
                "visible top-to-ceiling contact or separation",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control", "top_gap_m": 0.0},
                "invalid": {"operation": "translate_down", "target_ids": ["obj_000"], "top_gap_m": 0.48},
                "ambiguous": {"operation": "translate_down", "target_ids": ["obj_000"], "top_gap_m": 0.06},
            },
        )
        return drafts

    # ------------------------------------------------------------------
    # L2 high-level semantic claims
    # ------------------------------------------------------------------
    def _compose_scene_for_type(self, kind: str, *, mixed_with: str | None = None) -> dict[str, Any]:
        """Return a controlled semantic inventory in one frozen 9 m x 7 m shell."""

        layouts: dict[str, list[dict[str, Any]]] = {
            "bedroom": [
                self.object("0_SM_Bed", 2.5, 4.8, rotation_z=180.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 4.0, 5.0),
                self.object(
                    "0_SM_Deco029",
                    4.0,
                    5.0,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
                self.object("45_Capet05", 2.5, 2.7, scale=0.8),
            ],
            "home_office": [
                self.object("a_SM_desk_compiled", 2.6, 5.0, scale=0.8),
                self.object("b_8", 2.6, 3.9, rotation_z=0.0),
                self.object(
                    "44_sk20_MR01",
                    2.6,
                    5.0,
                    base_z=0.676,
                    scale=0.75,
                    support_parent="obj_000",
                ),
                self.object("0_steel_frame_shelves_03_2k_packed", 5.8, 5.4, scale=0.65),
            ],
            "dining_room": [
                self.object("b_47", 3.0, 3.5),
                self.object("d_1000004220137", 1.75, 3.5, rotation_z=-90.0),
                self.object("d_1000004220137", 4.25, 3.5, rotation_z=90.0),
                self.object("d_1000004220137", 3.0, 2.35, rotation_z=0.0),
                self.object("d_1000004220137", 3.0, 4.65, rotation_z=180.0),
            ],
            "media_room": [
                self.object("a_LeatherSofa", 2.2, 3.5, rotation_z=90.0),
                self.object("0_modern_wooden_cabinet_2k_packed", 6.2, 3.5, rotation_z=-90.0),
                self.object("b_85", 4.2, 3.5, rotation_z=90.0),
                self.object("15_SM_Floor_Lamp", 1.5, 1.5),
            ],
            "bathroom": [
                self.object("0_SM_Toilet", 1.4, 2.0, rotation_z=90.0),
                self.object("17_SM_Shower_Cabin", 1.3, 5.5),
                self.object("19_SM_Sink", 5.8, 5.5, rotation_z=180.0, scale=0.8),
                self.object("18_SM_Wall_Clock", 5.8, 6.9665, base_z=1.4),
            ],
            "laundry_room": [
                self.object("4_SM_Washer1_03", 1.5, 5.5),
                self.object("4_SM_Washer1_03", 2.5, 5.5),
                self.object("0_steel_frame_shelves_03_2k_packed", 5.8, 5.4, scale=0.7),
                self.object("0_SM_Plant_08", 7.5, 5.4, scale=0.5),
            ],
        }
        if kind not in layouts:
            raise DatasetBuildError(f"no semantic composition for {kind}")
        objects = deepcopy(layouts[kind])
        if mixed_with:
            # A 2+2 composition is intentionally balanced.  Keeping all four
            # requested-type objects and merely adding two donor objects made
            # the requested type remain dominant rather than borderline.
            objects = deepcopy(layouts[kind][:2])
            donor = deepcopy(layouts[mixed_with][:2])

            def move_pair_to(
                pair: list[dict[str, Any]], target_x: float, target_y: float
            ) -> None:
                mean_x = sum(float(obj["center"][0]) for obj in pair) / len(pair)
                mean_y = sum(float(obj["center"][1]) for obj in pair) / len(pair)
                dx, dy = target_x - mean_x, target_y - mean_y
                for obj in pair:
                    obj["center"][0] = round(float(obj["center"][0]) + dx, 6)
                    obj["center"][1] = round(float(obj["center"][1]) + dy, 6)

            move_pair_to(objects, 2.55, 4.65)
            move_pair_to(donor, 6.35, 2.15)
            objects.extend(donor)
        return self.scene(objects, width=9.0, depth=7.0, scene_type="room")

    def _build_room_type_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []
        specs = [
            ("room_type_bedroom", "bedroom", "home_office", "home_office"),
            ("room_type_dining", "dining room", "media_room", "media_room"),
            ("room_type_bathroom", "bathroom", "laundry_room", "laundry_room"),
            ("room_type_office", "home office", "bedroom", "bedroom"),
        ]
        kind_map = {
            "bedroom": "bedroom",
            "dining room": "dining_room",
            "bathroom": "bathroom",
            "home office": "home_office",
        }
        for scenario_id, requested, contrast, mixture in specs:
            prompt = f"Create a room whose primary scene type is a {requested}."
            source_span = f"primary scene type is a {requested}"
            claim_id = f"room_scene_type::{requested.replace(' ', '_')}"
            valid_scene = self._compose_scene_for_type(kind_map[requested])
            invalid_scene = self._compose_scene_for_type(contrast)
            ambiguous_scene = self._compose_scene_for_type(kind_map[requested], mixed_with=mixture)
            scenes = {
                "valid": valid_scene,
                "invalid": invalid_scene,
                "ambiguous": ambiguous_scene,
            }
            for label in PROPOSED_LABELS:
                scene = scenes[label]
                # The public generated-scene metadata stays neutral.  The
                # requested type exists only in the prompt/contract; neither a
                # matching nor a counterfactual type is leaked by metadata.
                scene["scene_type"] = "room"
                claims = self._high_level_claims(
                    prompt,
                    family="room_scene_type",
                    records=[
                        (
                            claim_id,
                            source_span,
                            {
                                "value": requested.replace(" ", "_"),
                                "dominance_required": True,
                            },
                        )
                    ],
                )
                drafts.append(
                    CaseDraft(
                        metric="room_scene_type",
                        scenario_id=scenario_id,
                        prompt=prompt,
                        prompt_granularity="coarse_grained",
                        scene=scene,
                        proposed_label=label,
                        target_ids=[obj["id"] for obj in scene["objects"]],
                        event_relation=None,
                        claims=claims,
                        reference_oor=[],
                        reference_oar=[],
                        review_question=(
                            f"Does the dominant visible composition function as a {requested}, "
                            "without relying on scene_type metadata?"
                        ),
                        gt_basis=(
                            "controlled_inventory_counterfactual"
                            if label != "ambiguous"
                            else "controlled_multipurpose_boundary_pending_human"
                        ),
                        required_visible_facts=[
                            "dominant furniture inventory",
                            "room-wide arrangement and hallmark functional groups",
                            "absence or presence of a competing room identity",
                        ],
                        declared_delta={
                            "operation": (
                                "requested_inventory"
                                if label == "valid"
                                else "contrast_inventory"
                                if label == "invalid"
                                else "mixed_inventory"
                            ),
                            "requested_scene_type": requested,
                        },
                        difficulty="clear" if label != "ambiguous" else "multipurpose_boundary",
                        asset_policy=deepcopy(ASSET_POLICY),
                        authorized_deviations=[],
                        source_scene_ids=[],
                    )
                )
        return drafts

    def _build_broad_intent_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []

        # Intent-1: a genuinely coarse semantic-intent family.  The prompt
        # does not prescribe a direct orientation relation.  The clear
        # counterfactual contrasts two complete per-seat reading stations
        # against a shared social conversation composition; the boundary
        # scene deliberately contains one of each.
        valid_scene = self.scene(
            [
                self.object("0_modern_arm_chair_01_2k_packed", 1.55, 2.6, rotation_z=0.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 2.75, 2.6),
                self.object(
                    "0_SM_Book001_02",
                    2.75,
                    2.6,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
                self.object("15_SM_Floor_Lamp", 1.35, 4.4),
                self.object("0_modern_arm_chair_01_2k_packed", 6.45, 2.6, rotation_z=0.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 5.25, 2.6),
                self.object(
                    "0_SM_Book001_02",
                    5.25,
                    2.6,
                    base_z=0.616,
                    support_parent="obj_005",
                ),
                self.object("15_SM_Floor_Lamp", 6.65, 4.4),
            ],
            width=8.0,
            depth=6.5,
        )
        invalid_scene = self.scene(
            [
                self.object("0_modern_arm_chair_01_2k_packed", 2.7, 3.0, rotation_z=90.0),
                self.object("b_85", 4.0, 3.0),
                self.object("0_modern_arm_chair_01_2k_packed", 5.3, 3.0, rotation_z=-90.0),
                self.object("15_SM_Floor_Lamp", 4.0, 5.0),
            ],
            width=8.0,
            depth=6.5,
        )
        ambiguous_scene = self.scene(
            [
                self.object("0_modern_arm_chair_01_2k_packed", 1.3, 2.6, rotation_z=0.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 2.5, 2.6),
                self.object(
                    "0_SM_Book001_02",
                    2.5,
                    2.6,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
                self.object("15_SM_Floor_Lamp", 1.3, 4.4),
                self.object("0_modern_arm_chair_01_2k_packed", 4.5, 2.6, rotation_z=90.0),
                self.object("b_85", 5.65, 2.6),
                self.object("0_modern_arm_chair_01_2k_packed", 6.8, 2.6, rotation_z=-90.0),
            ],
            width=8.0,
            depth=6.5,
        )
        prompt = "Create a quiet lounge intended for separate individual reading."
        span = "quiet lounge intended for separate individual reading"
        claims = self._high_level_claims(
            prompt,
            family="broad_semantic_intent",
            records=[
                (
                    "broad_intent::quiet_reading",
                    span,
                    {"value": "quiet_individual_reading_and_relaxation"},
                )
            ],
        )
        scenes = {
            "valid": valid_scene,
            "invalid": invalid_scene,
            "ambiguous": ambiguous_scene,
        }
        deltas = {
            "valid": {"operation": "two_complete_individual_reading_stations"},
            "invalid": {
                "operation": "shared_social_conversation_composition",
                "per_seat_reading_cues": "absent",
            },
            "ambiguous": {
                "operation": "mixed_reading_and_conversation_composition",
                "complete_reading_station_count": 1,
            },
        }
        for label in PROPOSED_LABELS:
            scene = scenes[label]
            drafts.append(
                CaseDraft(
                    metric="broad_semantic_intent",
                    scenario_id="intent_quiet_reading",
                    prompt=prompt,
                    prompt_granularity="coarse_grained",
                    scene=scene,
                    proposed_label=label,
                    target_ids=[obj["id"] for obj in scene["objects"]],
                    event_relation=None,
                    claims=deepcopy(claims),
                    reference_oor=[],
                    reference_oar=[],
                    review_question=(
                        "Does the dominant composition support separate individual "
                        "reading rather than shared social conversation?"
                    ),
                    gt_basis=(
                        "controlled_semantic_composition_counterfactual"
                        if label != "ambiguous"
                        else "controlled_mixed_intent_boundary_pending_human"
                    ),
                    required_visible_facts=[
                        "room-wide seating composition",
                        "per-seat book, side-table, and reading-light cues",
                        "whether furniture is organized as individual stations or one shared social group",
                    ],
                    declared_delta=deepcopy(deltas[label]),
                    difficulty=(
                        "clear"
                        if label != "ambiguous"
                        else "mixed_semantic_intent_boundary"
                    ),
                    asset_policy=deepcopy(ASSET_POLICY),
                    authorized_deviations=[],
                    source_scene_ids=[],
                    expected_cross_metric_effects=["object_pairing_consistency"],
                )
            )

        # Intent-2: focused workstation.
        base = self.office_layout()
        prompt = "Create a focused single-person workstation for sustained computer work."
        span = "focused single-person workstation for sustained computer work"
        drafts += self._triplet(
            metric="broad_semantic_intent",
            scenario_id="intent_focused_workstation",
            prompt=prompt,
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=["obj_000", "obj_001", "obj_002"],
            event_relation=None,
            claims_factory=lambda p: self._high_level_claims(
                p,
                family="broad_semantic_intent",
                records=[("broad_intent::focused_workstation", span, {"value": "focused_single_person_computer_work"})],
            ),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._set_xy(s, "obj_001", 6.5, 2.0),
                    self._set_yaw(s, "obj_001", 180.0),
                    self._set_xy(s, "obj_002", 1.2, 1.2),
                    self._set_base_z(s, "obj_002", 0.0),
                ),
                "ambiguous": lambda s: (
                    self._set_xy(s, "obj_001", 5.35, 3.45),
                    self._set_yaw(s, "obj_001", 70.0),
                ),
            },
            review_question="Does the visible arrangement support sustained single-person computer work?",
            required_visible_facts=[
                "desk, monitor, and chair as one functional group",
                "chair access and orientation to the monitor",
                "room-wide dominance of the workstation",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {"operation": "disperse_workstation_components"},
                "ambiguous": {"operation": "partially_displace_chair"},
            },
            expected_cross_metric_effects=["object_pairing_consistency"],
        )

        # Intent-3: TV viewing.
        base = self.media_layout()
        prompt = "Create a space primarily intended for watching television from the sofa."
        span = "watching television from the sofa"
        drafts += self._triplet(
            metric="broad_semantic_intent",
            scenario_id="intent_tv_viewing",
            prompt=prompt,
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=["obj_000", "obj_001", "obj_005"],
            event_relation=None,
            claims_factory=lambda p: self._high_level_claims(
                p,
                family="broad_semantic_intent",
                records=[("broad_intent::tv_viewing", span, {"value": "sofa_based_television_viewing"})],
            ),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: self._set_yaw(s, "obj_000", 90.0),
                "invalid": lambda s: self._set_yaw(s, "obj_000", -90.0),
                "ambiguous": lambda s: self._set_yaw(s, "obj_000", 25.0),
            },
            review_question="Does the dominant arrangement support watching the television from the sofa?",
            required_visible_facts=[
                "sofa front/back orientation",
                "visible television and television-cabinet viewing side",
                "unobstructed sofa-to-screen relation",
            ],
            delta_descriptions={
                "valid": {"operation": "set_sofa_yaw", "yaw_deg": 90.0},
                "invalid": {"operation": "set_sofa_yaw", "yaw_deg": -90.0},
                "ambiguous": {"operation": "set_sofa_yaw", "yaw_deg": 25.0},
            },
            expected_cross_metric_effects=["object_pairing_consistency"],
        )

        # Intent-4: communal dining.
        base = self.dining_layout()
        prompt = "Create a space intended for a small group to share meals together."
        span = "a small group to share meals together"
        drafts += self._triplet(
            metric="broad_semantic_intent",
            scenario_id="intent_communal_dining",
            prompt=prompt,
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=[f"obj_{i:03d}" for i in range(5)],
            event_relation=None,
            claims_factory=lambda p: self._high_level_claims(
                p,
                family="broad_semantic_intent",
                records=[("broad_intent::communal_dining", span, {"value": "small_group_shared_meals"})],
            ),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._set_xy(s, "obj_001", 1.2, 1.0),
                    self._set_xy(s, "obj_002", 2.2, 1.0),
                    self._set_xy(s, "obj_003", 3.2, 1.0),
                    self._set_xy(s, "obj_004", 4.2, 1.0),
                ),
                "ambiguous": lambda s: (
                    self._set_xy(s, "obj_003", 6.5, 1.1),
                    self._set_xy(s, "obj_004", 6.5, 2.2),
                ),
            },
            review_question="Does the composition support a small group sharing meals together?",
            required_visible_facts=[
                "table as a shared focal surface",
                "all chair positions and access directions",
                "how many seats visibly participate in the dining group",
            ],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {"operation": "move_all_chairs_into_detached_row"},
                "ambiguous": {"operation": "detach_two_of_four_chairs"},
            },
            expected_cross_metric_effects=["object_pairing_consistency"],
        )
        return drafts

    def _two_zone_scene(
        self,
        first: str,
        second: str,
        *,
        second_state: str,
    ) -> dict[str, Any]:
        """Create two spatially separated regions with complete/absent/partial zone 2."""

        zone_objects: dict[str, list[dict[str, Any]]] = {
            "sleep": [
                self.object("0_SM_Bed", 2.3, 5.2, rotation_z=180.0, source_region_id="zone_a"),
                self.object("0_painted_wooden_nightstand_2k_packed", 3.8, 5.2, source_region_id="zone_a"),
            ],
            "work": [
                self.object("0_SM_Modern_Table", 7.4, 5.2, source_region_id="zone_b"),
                self.object("b_8", 7.4, 4.0, rotation_z=0.0, source_region_id="zone_b"),
                self.object("41_ComputerSet_03", 7.4, 5.2, base_z=0.78, source_region_id="zone_b"),
            ],
            "living": [
                self.object("0_SM_Sofa_2", 2.2, 2.2, rotation_z=90.0, scale=0.75, source_region_id="zone_a"),
                self.object("b_85", 4.0, 2.2, rotation_z=90.0, source_region_id="zone_a"),
            ],
            "dining": [
                self.object("b_47", 8.0, 2.2, source_region_id="zone_b"),
                self.object("d_1000004220137", 6.8, 2.2, rotation_z=-90.0, source_region_id="zone_b"),
                self.object("d_1000004220137", 9.2, 2.2, rotation_z=90.0, source_region_id="zone_b"),
            ],
            "cooking": [
                self.object("19_SM_Fridge", 1.2, 5.4, source_region_id="zone_a"),
                self.object("e_kitchen_10", 2.6, 5.4, source_region_id="zone_a"),
                self.object(
                    "11_SM_Microwawe",
                    2.6,
                    5.4,
                    base_z=0.921,
                    source_region_id="zone_a",
                    support_parent="obj_001",
                ),
            ],
            "reading": [
                self.object("0_modern_arm_chair_01_2k_packed", 2.2, 2.0, rotation_z=-20.0, source_region_id="zone_a"),
                self.object("15_SM_Floor_Lamp", 3.2, 2.8, source_region_id="zone_a"),
                self.object("0_painted_wooden_nightstand_2k_packed", 3.3, 1.8, source_region_id="zone_a"),
            ],
            "conversation": [
                self.object("a_SM_Sofa_01b", 7.0, 2.0, rotation_z=-90.0, source_region_id="zone_b"),
                self.object("a_SM_Sofa_01b", 9.3, 2.0, rotation_z=90.0, source_region_id="zone_b"),
                self.object("b_85", 8.15, 2.0, source_region_id="zone_b"),
            ],
        }
        first_objects = deepcopy(zone_objects[first])
        second_objects = deepcopy(zone_objects[second])
        if second_state == "absent":
            second_objects = []
        elif second_state == "partial":
            second_objects = second_objects[:1]
        elif second_state != "complete":
            raise DatasetBuildError(f"unknown second-zone state {second_state}")
        regions = [
            {"id": "zone_a", "label": first, "boundary": [[0.4, 0.4], [5.4, 0.4], [5.4, 6.6], [0.4, 6.6]]},
            {"id": "zone_b", "label": second, "boundary": [[5.6, 0.4], [10.6, 0.4], [10.6, 6.6], [5.6, 6.6]]},
        ]
        result = self.scene(
            first_objects + second_objects,
            width=11.0,
            depth=7.0,
            scene_type="room",
            semantic_regions=regions,
        )
        # Resolve support identities after the two region-local object lists
        # have been concatenated and canonical IDs assigned.
        for child in result["objects"]:
            if child["jid"] not in {"41_ComputerSet_03", "11_SM_Microwawe"}:
                continue
            candidates = [
                parent
                for parent in result["objects"]
                if parent["id"] != child["id"]
                and parent.get("source_region_id") == child.get("source_region_id")
                and (
                    parent["category"] in {"desk", "table", "cabinet"}
                    or parent["jid"] in {"0_SM_Modern_Table", "e_kitchen_10"}
                )
            ]
            if len(candidates) != 1:
                raise DatasetBuildError(
                    f"cannot resolve unique support for {child['jid']} in {child.get('source_region_id')}"
                )
            child["support_parent"] = candidates[0]["id"]
            self._place_on_top(result, child["id"], candidates[0]["id"])
        return result

    def _build_required_area_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []
        specs = [
            ("areas_sleep_work", "sleeping", "work", "sleep", "work"),
            ("areas_living_dining", "living", "dining", "living", "dining"),
            ("areas_cooking_dining", "cooking", "dining", "cooking", "dining"),
            ("areas_reading_conversation", "reading", "conversation", "reading", "conversation"),
        ]
        for scenario_id, first_text, second_text, first_key, second_key in specs:
            prompt = (
                f"Include two distinct functional areas: a {first_text} area and a {second_text} area."
            )
            first_span = f"a {first_text} area"
            second_span = f"a {second_text} area"
            claims = self._high_level_claims(
                prompt,
                family="required_functional_areas",
                records=[
                    (
                        f"functional_area::{first_text}",
                        first_span,
                        {"area_id": f"{first_text}_area", "present": True, "spatially_distinct": True},
                    ),
                    (
                        f"functional_area::{second_text}",
                        second_span,
                        {"area_id": f"{second_text}_area", "present": True, "spatially_distinct": True},
                    ),
                ],
            )
            scenes = {
                "valid": self._two_zone_scene(first_key, second_key, second_state="complete"),
                "invalid": self._two_zone_scene(first_key, second_key, second_state="absent"),
                "ambiguous": self._two_zone_scene(first_key, second_key, second_state="partial"),
            }
            for label in PROPOSED_LABELS:
                scene = scenes[label]
                drafts.append(
                    CaseDraft(
                        metric="required_functional_areas",
                        scenario_id=scenario_id,
                        prompt=prompt,
                        prompt_granularity="coarse_grained",
                        scene=scene,
                        proposed_label=label,
                        target_ids=[obj["id"] for obj in scene["objects"]],
                        event_relation=None,
                        claims=deepcopy(claims),
                        reference_oor=[],
                        reference_oar=[],
                        review_question=(
                            f"Are both a distinct {first_text} area and a distinct {second_text} area visibly present?"
                        ),
                        gt_basis=(
                            "controlled_zone_presence_counterfactual"
                            if label != "ambiguous"
                            else "controlled_partial_zone_pending_human"
                        ),
                        required_visible_facts=[
                            f"objects and spatial extent of the {first_text} area",
                            f"objects and spatial extent of the {second_text} area",
                            "separation or overlap between the two functional groups",
                        ],
                        declared_delta={
                            "operation": f"second_zone_{'complete' if label == 'valid' else 'absent' if label == 'invalid' else 'partial'}",
                            "first_area": first_text,
                            "second_area": second_text,
                        },
                        difficulty="clear" if label != "ambiguous" else "partial_function_boundary",
                        asset_policy=deepcopy(ASSET_POLICY),
                        authorized_deviations=[],
                        source_scene_ids=[],
                        expected_cross_metric_effects=["broad_semantic_intent"],
                    )
                )
        return drafts

    # ------------------------------------------------------------------
    # L3 generic Scene Quality
    # ------------------------------------------------------------------
    def _l3_empty_claims(self) -> dict[str, list[dict[str, Any]]]:
        return self._empty_claims()

    def _build_scale_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []
        scenario_specs: list[dict[str, Any]] = []

        # Scale-1: chair/table.
        scene = self.dining_layout()
        self._set_xy(scene, "obj_001", 2.3846, 3.0)
        self._set_xy(scene, "obj_002", 5.6154, 3.0)
        scenario_specs.append(
            {
                "scenario_id": "scale_chair_table",
                "prompt": "Create a dining area with a table and four chairs.",
                "scene": scene,
                "targets": ["obj_001", "obj_000"],
                "valid": lambda s: None,
                "invalid": lambda s: self._scale_about_base(s, "obj_001", 2.8),
                "ambiguous": lambda s: self._scale_about_base(s, "obj_001", 1.45),
                "changed_target": "obj_001",
                "question": "Is the designated dining chair at a visually coherent scale relative to the table?",
                "facts": ["full chair silhouette", "full table silhouette", "shared floor plane and comparative dimensions"],
            }
        )

        # Scale-2: sofa / media cabinet.
        scene = self.media_layout()
        scenario_specs.append(
            {
                "scenario_id": "scale_sofa_media_console",
                "prompt": "Create a television-viewing area with a sofa and a media console.",
                "scene": scene,
                "targets": ["obj_000", "obj_001"],
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._scale_about_base(s, "obj_001", 2.25),
                    self._place_on_top(s, "obj_005", "obj_001"),
                ),
                "ambiguous": lambda s: (
                    self._scale_about_base(s, "obj_001", 1.42),
                    self._place_on_top(s, "obj_005", "obj_001"),
                ),
                "changed_target": "obj_001",
                "question": "Is the media console at a visually coherent scale relative to the sofa?",
                "facts": ["sofa width and height", "media-console width and height", "same room/floor scale reference"],
            }
        )

        # Scale-3: bedside lamp / nightstand / bed.
        scene = self.bedroom_layout()
        self._set_xy(scene, "obj_001", 1.4304, 4.35)
        self._set_xy(scene, "obj_003", 1.4304, 4.35)
        self._set_xy(scene, "obj_002", 4.5696, 4.35)
        self._set_xy(scene, "obj_004", 4.5696, 4.35)
        scenario_specs.append(
            {
                "scenario_id": "scale_bedside_lamp",
                "prompt": "Create a bedroom with a bed, two nightstands, and matching bedside lamps.",
                "scene": scene,
                "targets": ["obj_003", "obj_001", "obj_000"],
                "valid": lambda s: None,
                "invalid": lambda s: self._scale_about_base(s, "obj_003", 3.6),
                "ambiguous": lambda s: self._scale_about_base(s, "obj_003", 1.75),
                "changed_target": "obj_003",
                "question": "Is the designated bedside lamp coherently scaled relative to its nightstand and the bed?",
                "facts": ["lamp silhouette", "supporting nightstand", "bed as a room-scale reference"],
            }
        )

        # Scale-4: toilet / shower.
        scene = self.bathroom_layout()
        scenario_specs.append(
            {
                "scenario_id": "scale_bathroom_fixtures",
                "prompt": "Create a bathroom with a toilet, a shower enclosure, and a sink.",
                "scene": scene,
                "targets": ["obj_000", "obj_001"],
                "valid": lambda s: None,
                "invalid": lambda s: self._scale_about_base(s, "obj_000", 0.38),
                "ambiguous": lambda s: self._scale_about_base(s, "obj_000", 0.72),
                "changed_target": "obj_000",
                "question": "Is the toilet coherently scaled relative to the shower enclosure and the room?",
                "facts": ["toilet silhouette", "shower enclosure silhouette", "shared floor and wall scale cues"],
            }
        )

        for spec in scenario_specs:
            drafts += self._triplet(
                metric="scale_consistency",
                scenario_id=spec["scenario_id"],
                prompt=spec["prompt"],
                prompt_granularity="coarse_grained",
                base_scene=spec["scene"],
                target_ids=spec["targets"],
                event_relation="relative_physical_scale",
                claims_factory=lambda p: self._l3_empty_claims(),
                reference_factory=lambda p: ([], []),
                mutators={
                    "valid": spec["valid"],
                    "invalid": spec["invalid"],
                    "ambiguous": spec["ambiguous"],
                },
                review_question=spec["question"],
                required_visible_facts=spec["facts"],
                delta_descriptions={
                    "valid": {"operation": "catalog_scale_control"},
                    "invalid": {
                        "operation": "large_controlled_scale_change",
                        "target_ids": [spec["changed_target"]],
                    },
                    "ambiguous": {
                        "operation": "moderate_controlled_scale_change",
                        "target_ids": [spec["changed_target"]],
                    },
                },
                expected_cross_metric_effects=[],
            )
            for draft in drafts[-3:]:
                draft.raw_coherence_label = draft.proposed_label
        return drafts

    def _build_pairing_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []

        # Pairing-1: chair/desk orientation.
        base = self.office_layout()
        drafts += self._triplet(
            metric="object_pairing_consistency",
            scenario_id="pairing_chair_desk_orientation",
            prompt="Create a conventional single-person desk workstation.",
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=["obj_001", "obj_000"],
            event_relation="functional_orientation",
            claims_factory=lambda p: self._l3_empty_claims(),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: self._set_yaw(s, "obj_001", 0.0),
                "invalid": lambda s: self._set_yaw(s, "obj_001", 180.0),
                "ambiguous": lambda s: self._set_yaw(s, "obj_001", 72.0),
            },
            review_question="Does the chair orientation support conventional use of the desk and monitor?",
            required_visible_facts=[
                "chair front/back orientation",
                "desk working side and monitor-facing side",
                "chair access position relative to the desk",
            ],
            delta_descriptions={
                "valid": {"operation": "set_chair_yaw", "yaw_deg": 0.0},
                "invalid": {"operation": "set_chair_yaw", "yaw_deg": 180.0},
                "ambiguous": {"operation": "set_chair_yaw", "yaw_deg": 72.0},
            },
        )

        # Pairing-2: sofa/media orientation.
        base = self.media_layout()
        drafts += self._triplet(
            metric="object_pairing_consistency",
            scenario_id="pairing_sofa_media_orientation",
            prompt="Create a conventional sofa and television-console grouping.",
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=["obj_000", "obj_001", "obj_005"],
            event_relation="functional_orientation",
            claims_factory=lambda p: self._l3_empty_claims(),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: self._set_yaw(s, "obj_000", 90.0),
                "invalid": lambda s: self._set_yaw(s, "obj_000", -90.0),
                "ambiguous": lambda s: self._set_yaw(s, "obj_000", 25.0),
            },
            review_question="Does the sofa orientation support conventional viewing of the television console?",
            required_visible_facts=[
                "sofa facing direction",
                "visible television and console viewing side",
                "line of sight between the sofa and screen",
            ],
            delta_descriptions={
                "valid": {"operation": "set_sofa_yaw", "yaw_deg": 90.0},
                "invalid": {"operation": "set_sofa_yaw", "yaw_deg": -90.0},
                "ambiguous": {"operation": "set_sofa_yaw", "yaw_deg": 25.0},
            },
        )

        # Pairing-3: all dining chairs participate in the table group.
        base = self.dining_layout()
        drafts += self._triplet(
            metric="object_pairing_consistency",
            scenario_id="pairing_dining_group",
            prompt="Create a conventional dining-table and chair grouping.",
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=[f"obj_{i:03d}" for i in range(5)],
            event_relation="functional_grouping",
            claims_factory=lambda p: self._l3_empty_claims(),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._set_xy(s, "obj_001", 1.0, 1.0),
                    self._set_xy(s, "obj_002", 2.0, 1.0),
                    self._set_xy(s, "obj_003", 3.0, 1.0),
                    self._set_xy(s, "obj_004", 4.0, 1.0),
                ),
                "ambiguous": lambda s: self._set_xy(s, "obj_004", 6.4, 4.8),
            },
            review_question="Do the chairs and table form one functionally coherent dining group?",
            required_visible_facts=["table perimeter", "all chair positions", "access and orientation of chairs toward the table"],
            delta_descriptions={
                "valid": {"operation": "identity_control"},
                "invalid": {"operation": "detach_all_chairs_from_table_group"},
                "ambiguous": {"operation": "detach_one_chair_from_table_group"},
            },
        )

        # Pairing-4: category coexistence beside a bed.
        base = self.scene(
            [
                self.object("0_SM_Bed", 3.0, 4.3, rotation_z=180.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 4.45, 4.4),
                self.object(
                    "0_SM_Deco029",
                    4.45,
                    4.4,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
            ],
            width=7.0,
            depth=6.0,
        )
        drafts += self._triplet(
            metric="object_pairing_consistency",
            scenario_id="pairing_bedside_object",
            prompt="Create a conventional bedside furniture grouping.",
            prompt_granularity="coarse_grained",
            base_scene=base,
            target_ids=["obj_000", "obj_001"],
            event_relation="category_coexistence",
            claims_factory=lambda p: self._l3_empty_claims(),
            reference_factory=lambda p: ([], []),
            mutators={
                "valid": lambda s: None,
                "invalid": lambda s: (
                    self._replace_asset(
                        s,
                        "obj_001",
                        "11_SM_Microwawe",
                        category="microwave",
                    ),
                    self._place_on_top(s, "obj_002", "obj_001"),
                ),
                "ambiguous": lambda s: (
                    self._replace_asset(
                        s,
                        "obj_001",
                        "0_SM_Deco023_02",
                        scale=1.6,
                        category="storage",
                    ),
                    self._place_on_top(s, "obj_002", "obj_001"),
                ),
            },
            review_question="Does the designated object beside the bed coherently belong in a conventional bedside grouping?",
            required_visible_facts=[
                "bed identity",
                "category and shape of the designated bedside object",
                "local arrangement and surrounding bedroom context",
            ],
            delta_descriptions={
                "valid": {"operation": "bedside_table_control"},
                "invalid": {"operation": "replace_bedside_object", "replacement_role": "microwave"},
                "ambiguous": {"operation": "replace_bedside_object", "replacement_role": "generic_storage"},
            },
        )
        for draft in drafts:
            draft.raw_coherence_label = draft.proposed_label
        return drafts

    def _build_style_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []
        # Each case emphasizes shape language, because catalog texture/material
        # fidelity must be confirmed after Blender rendering.
        specs: list[dict[str, Any]] = []

        base = self.scene(
            [
                self.object("0_modern_arm_chair_01_2k_packed", 2.3, 3.0),
                self.object("0_modern_arm_chair_01_2k_packed", 5.7, 3.0),
                self.object("0_SM_Modern_Table", 4.0, 3.1, scale=0.65),
                self.object("15_SM_Floor_Lamp", 4.0, 5.0),
            ],
            width=8.0,
            depth=6.5,
        )
        specs.append(
            {
                "id": "style_modern_reading",
                "prompt": "Create a visually coherent modern reading area.",
                "scene": base,
                "target": "obj_001",
                "invalid_asset": "0_rockingchair_01_2k_packed",
                "ambiguous_asset": "a_SM_Sofa_01b",
                "question": "Does the designated chair share a coherent visual style with the modern reading group?",
            }
        )

        base = self.scene(
            [
                self.object("0_SM_Bed", 3.2, 4.5, rotation_z=180.0),
                self.object("0_painted_wooden_nightstand_2k_packed", 1.55, 4.7),
                self.object("0_painted_wooden_nightstand_2k_packed", 4.85, 4.7),
                self.object(
                    "0_SM_Deco029",
                    1.55,
                    4.7,
                    base_z=0.616,
                    support_parent="obj_001",
                ),
                self.object(
                    "0_SM_Deco029",
                    4.85,
                    4.7,
                    base_z=0.616,
                    support_parent="obj_002",
                ),
            ],
            width=7.0,
            depth=6.0,
        )
        specs.append(
            {
                "id": "style_bedroom_set",
                "prompt": "Create a visually coherent contemporary bedroom set.",
                "scene": base,
                "target": "obj_000",
                "invalid_asset": "0_gothic_bed_01_2k_packed",
                "ambiguous_asset": "d_1000003404997",
                "question": "Does the designated bed share a coherent visual style with the surrounding bedroom set?",
            }
        )

        base = self.scene(
            [
                self.object("0_steel_frame_shelves_03_2k_packed", 1.4, 5.2, scale=0.75),
                self.object("0_SM_Modern_Table", 4.4, 4.8),
                self.object("0_SM_Chair_Sec001", 4.4, 3.4),
                self.object(
                    "41_ComputerSet_03",
                    4.4,
                    4.8,
                    base_z=0.78,
                    support_parent="obj_001",
                ),
            ],
            width=8.0,
            depth=6.5,
        )
        specs.append(
            {
                "id": "style_industrial_office",
                "prompt": "Create a visually coherent industrial-modern office.",
                "scene": base,
                "target": "obj_001",
                "invalid_asset": "0_SM_Desk_thr001",
                "ambiguous_asset": "a_SM_desk_compiled",
                "supported_child": "obj_003",
                "question": "Does the designated desk share a coherent visual style with the industrial-modern office?",
            }
        )

        base = self.dining_layout()
        specs.append(
            {
                "id": "style_dining_chairs",
                "prompt": "Create a visually coherent dining furniture set.",
                "scene": base,
                "target": "obj_004",
                "invalid_asset": "43_outsidechair01",
                "ambiguous_asset": "b_37",
                "question": "Does the designated chair share a coherent visual style with the rest of the dining set?",
            }
        )

        for spec in specs:
            target = spec["target"]
            drafts += self._triplet(
                metric="style_consistency",
                scenario_id=spec["id"],
                prompt=spec["prompt"],
                prompt_granularity="coarse_grained",
                base_scene=spec["scene"],
                target_ids=[target] + [obj["id"] for obj in spec["scene"]["objects"] if obj["id"] != target],
                event_relation="visual_style_match",
                claims_factory=lambda p: self._l3_empty_claims(),
                reference_factory=lambda p: ([], []),
                mutators={
                    "valid": lambda s: None,
                    "invalid": lambda s, t=target, jid=spec["invalid_asset"], child=spec.get("supported_child"): self._replace_asset_and_reseat(
                        s,
                        t,
                        jid,
                        supported_child_id=child,
                    ),
                    "ambiguous": lambda s, t=target, jid=spec["ambiguous_asset"], child=spec.get("supported_child"): self._replace_asset_and_reseat(
                        s,
                        t,
                        jid,
                        supported_child_id=child,
                    ),
                },
                review_question=spec["question"],
                required_visible_facts=[
                    "designated object's full silhouette and shape language",
                    "at least two surrounding comparison objects",
                    "consistent lighting and material-render context",
                ],
                delta_descriptions={
                    "valid": {"operation": "coordinated_asset_control"},
                    "invalid": {"operation": "replace_designated_asset", "replacement_jid": spec["invalid_asset"]},
                    "ambiguous": {"operation": "replace_designated_asset", "replacement_jid": spec["ambiguous_asset"]},
                },
                expected_cross_metric_effects=[],
            )
        for draft in drafts:
            draft.raw_coherence_label = draft.proposed_label
        return drafts

    def _authorization_claim_and_deviation(
        self,
        metric: str,
        prompt: str,
        *,
        source_span: str,
        target_ids: list[str],
        relation: str,
    ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], str]:
        claims = self._empty_claims()
        if metric == "style_consistency":
            claim_id = "attribute::chair::ornate_contrast"
            claims["explicit_attributes"].append(
                {
                    "claim_id": claim_id,
                    "claim_family": "explicit_attributes",
                    "required": True,
                    "target_ids": list(target_ids),
                    "attribute": "visual_style",
                    "expected": {"value": "ornate_contrast"},
                    **self._span(prompt, source_span),
                }
            )
        else:
            claim_id = (
                "oor::chair_much_larger_than_table"
                if metric == "scale_consistency"
                else "oor::chair_faces_away_from_desk"
            )
            claims["oor"].append(
                {
                    "claim_id": claim_id,
                    "claim_family": "oor",
                    "required": True,
                    "relation_id": claim_id,
                    "relation_type": relation,
                    "target_ids": list(target_ids),
                    "expected": {"satisfied": True},
                    **self._span(prompt, source_span),
                }
            )
        deviation = {
            "deviation_id": f"dev::{metric}::{relation}",
            "metric": metric,
            "target_ids": list(target_ids),
            "relation": relation,
            "source": "explicit_prompt_requirement",
            "prompt_span": source_span,
            "prompt_span_start": prompt.find(source_span),
            "prompt_span_end": prompt.find(source_span) + len(source_span),
            "source_claim_id": claim_id,
        }
        return claims, [deviation], claim_id

    def _build_authorization_drafts(self) -> list[CaseDraft]:
        drafts: list[CaseDraft] = []

        designs = [
            {
                "metric": "scale_consistency",
                "group": "auth_scale_chair_table",
                "neutral_prompt": "Place a dining chair beside a dining table.",
                "authorized_prompt": (
                    "Place a dining chair beside a dining table, and make the chair intentionally "
                    "much larger than the table as a surreal focal object."
                ),
                "span": "make the chair intentionally much larger than the table",
                "relation": "much_larger_than",
                "event_targets": ["obj_001", "obj_000"],
                "deviation_targets": ["obj_001", "obj_000"],
                "base_claim_id": "oor::chair_beside_table",
                "base_relation": "beside",
                "base_span": "Place a dining chair beside a dining table",
                "base_subject": "obj_001",
                "base_object": "obj_000",
                "base": self.scene(
                    [
                        self.object("b_47", 5.0, 3.2),
                        self.object("d_1000004220137", 3.35, 3.2, rotation_z=-90.0),
                    ],
                    width=10.0,
                    depth=7.0,
                    height=4.5,
                ),
                "deviate": lambda s: self._scale_about_base(s, "obj_001", 3.15),
                "question": "After applying only any prompt-authorized exemption, should this chair/table scale relation be penalized by L3?",
                "facts": ["complete chair and table silhouettes", "shared floor plane", "original prompt authorization text"],
            },
            {
                "metric": "object_pairing_consistency",
                "group": "auth_pairing_chair_desk",
                "neutral_prompt": "Place a desk chair directly in front of a writing desk.",
                "authorized_prompt": (
                    "Place a desk chair directly in front of a writing desk, but intentionally "
                    "make the chair face away from the desk."
                ),
                "span": "intentionally make the chair face away from the desk",
                "relation": "faces_away_from",
                "event_targets": ["obj_001", "obj_000"],
                "deviation_targets": ["obj_001", "obj_000"],
                "base_claim_id": "oor::chair_directly_in_front_of_desk",
                "base_relation": "directly_in_front_of",
                "base_span": "Place a desk chair directly in front of a writing desk",
                "base_subject": "obj_001",
                "base_object": "obj_000",
                "base": self.scene(
                    [
                        self.object("0_SM_Modern_Table", 4.0, 4.4),
                        self.object("b_8", 4.0, 3.2, rotation_z=0.0),
                        self.object(
                            "41_ComputerSet_03",
                            4.0,
                            4.4,
                            base_z=0.78,
                            support_parent="obj_000",
                        ),
                    ],
                    width=8.0,
                    depth=6.5,
                ),
                "deviate": lambda s: self._set_yaw(s, "obj_001", 180.0),
                "question": "After applying only any prompt-authorized exemption, should this chair/desk orientation be penalized by L3?",
                "facts": ["chair front/back orientation", "desk working side", "original prompt authorization text"],
            },
            {
                "metric": "style_consistency",
                "group": "auth_style_chair_contrast",
                "neutral_prompt": "Place a lounge chair beside a side table in a modern reading area.",
                "authorized_prompt": (
                    "Place a lounge chair beside a side table in a modern reading area, and make "
                    "the chair intentionally ornate so it contrasts with the other minimal furniture."
                ),
                "span": "make the chair intentionally ornate",
                "relation": "ornate_contrast",
                "event_targets": ["obj_000", "obj_001", "obj_002"],
                "deviation_targets": ["obj_000"],
                "base_claim_id": "oor::chair_beside_side_table",
                "base_relation": "beside",
                "base_span": "Place a lounge chair beside a side table",
                "base_subject": "obj_000",
                "base_object": "obj_001",
                "base": self.scene(
                    [
                        self.object("0_modern_arm_chair_01_2k_packed", 3.0, 3.0),
                        self.object("0_painted_wooden_nightstand_2k_packed", 4.1, 3.0),
                        self.object("15_SM_Floor_Lamp", 4.35, 4.45),
                    ],
                    width=8.0,
                    depth=6.5,
                ),
                "deviate": lambda s: self._replace_asset(
                    s,
                    "obj_000",
                    "a_SM_ChairVintage",
                ),
                "question": "After applying only any prompt-authorized exemption, should this chair-style contrast be penalized by L3?",
                "facts": ["chair shape language", "minimal comparison furniture", "original prompt authorization text"],
            },
        ]

        for design in designs:
            visual_scenes: dict[str, dict[str, Any]] = {}
            for deviation_state in ("absent", "present"):
                scene = self.clone_scene(design["base"])
                if deviation_state == "present":
                    design["deviate"](scene)
                visual_scenes[deviation_state] = scene

            # Stable group-local visual source tokens.  The actual opaque case ID
            # is assigned later; this token proves which prompt siblings must
            # produce byte-identical renders after scene-id normalization.
            for prompt_auth in ("absent", "present"):
                for scene_dev in ("absent", "present"):
                    prompt = (
                        design["neutral_prompt"]
                        if prompt_auth == "absent"
                        else design["authorized_prompt"]
                    )
                    if prompt_auth == "present":
                        anomaly_claims, deviations, claim_id = self._authorization_claim_and_deviation(
                            design["metric"],
                            prompt,
                            source_span=design["span"],
                            target_ids=design["deviation_targets"],
                            relation=design["relation"],
                        )
                    else:
                        anomaly_claims, deviations, claim_id = self._empty_claims(), [], None

                    base_claims = self._oor_claims(
                        prompt,
                        claim_id=design["base_claim_id"],
                        relation_type=design["base_relation"],
                        target_ids=[design["base_subject"], design["base_object"]],
                        source_span=design["base_span"],
                    )
                    claims = self._merge_claim_sets(base_claims, anomaly_claims)
                    base_reference = self._reference_pair(
                        design["base_claim_id"],
                        design["base_relation"],
                        design["base_subject"],
                        design["base_object"],
                        design["base_span"],
                    )[0]
                    anomaly_reference = (
                        [
                            {
                                "relation_id": claim_id,
                                "type": design["relation"],
                                "raw_relation": design["span"],
                                "subject_id": design["deviation_targets"][0],
                                "object_id": design["deviation_targets"][1],
                                "claim_state": "confirmed",
                            }
                        ]
                        if claim_id and design["metric"] != "style_consistency"
                        else []
                    )

                    if prompt_auth == "absent" and scene_dev == "absent":
                        final_label, raw, applied, l2 = "valid", "valid", False, "not_applicable"
                        reason = "ordinary_scene_is_coherent"
                    elif prompt_auth == "absent" and scene_dev == "present":
                        final_label, raw, applied, l2 = "invalid", "invalid", False, "not_applicable"
                        reason = "unrequested_scene_inconsistency"
                    elif prompt_auth == "present" and scene_dev == "absent":
                        final_label, raw, applied, l2 = "valid", "valid", False, "invalid"
                        reason = "requested_deviation_missing_but_generic_scene_coherent"
                    else:
                        final_label, raw, applied, l2 = "valid", "invalid", True, "valid"
                        reason = "prompt_authorized_deviation"

                    drafts.append(
                        CaseDraft(
                            metric=design["metric"],
                            scenario_id=design["group"],
                            prompt=prompt,
                            prompt_granularity="fine_grained",
                            scene=self.clone_scene(visual_scenes[scene_dev]),
                            proposed_label=final_label,
                            target_ids=list(design["event_targets"]),
                            event_relation=design["relation"],
                            claims=claims,
                            reference_oor=base_reference + anomaly_reference,
                            reference_oar=[],
                            review_question=design["question"],
                            gt_basis="prompt_authorization_2x2_pending_human",
                            required_visible_facts=list(design["facts"]),
                            declared_delta={
                                "operation": "prompt_authorization_2x2",
                                "prompt_authorization": prompt_auth,
                                "scene_deviation": scene_dev,
                                "only_scene_change": (
                                    "none" if scene_dev == "absent" else design["relation"]
                                ),
                            },
                            difficulty="authorization_boundary",
                            asset_policy=deepcopy(ASSET_POLICY),
                            authorized_deviations=deviations,
                            source_scene_ids=[],
                            design_role="prompt_authorization_2x2",
                            counterfactual_group_id=design["group"],
                            prompt_authorization=prompt_auth,
                            scene_deviation=scene_dev,
                            visual_source_case_id=None,
                            raw_coherence_label=raw,
                            authorization_applied=applied,
                            l2_anomaly_label=l2,
                            resolution_reason=reason,
                            expected_cross_metric_effects=["oor"] if design["metric"] != "style_consistency" else ["explicit_attributes"],
                        )
                    )
        return drafts

    # ------------------------------------------------------------------
    # Artifact materialization
    # ------------------------------------------------------------------
    def _write_case(self, draft: CaseDraft) -> None:
        self.case_counter += 1
        case_id = f"case_{self.case_counter:04d}"
        event_id = f"event_{self.case_counter:05d}"
        fixture_rel = Path("fixtures") / case_id
        fixture = self.out_root / fixture_rel
        fixture.mkdir(parents=True, exist_ok=False)

        scene = deepcopy(draft.scene)
        scene["scene_id"] = case_id
        scene["request_id"] = case_id
        scene["metadata"]["asset_policy"] = deepcopy(draft.asset_policy)
        scene["metadata"]["contract_state"] = "draft_pending_human"

        geometry_hash = _geometry_hash(scene)
        visual_source = self.geometry_to_first_case.setdefault(geometry_hash, case_id)
        semantic_group_key = draft.counterfactual_group_id or draft.scenario_id
        group_id = self.group_id_map.setdefault(
            semantic_group_key, f"group_{len(self.group_id_map) + 1:03d}"
        )

        request = self._scene_request(
            case_id,
            scene,
            draft.prompt,
            draft.prompt_granularity,
            draft.asset_policy,
            draft.authorized_deviations,
        )
        annotation = self._reference_annotation(case_id, scene, draft)
        contract = {
            "contract_version": "specification_contract_v1",
            "source": "benchmark_owned",
            "frozen": False,
            "claim_scope": "metric_event_slice",
            "exhaustive_prompt_parse": False,
            "request_id": case_id,
            "scene_type": str(request["scene_type"]),
            "claims": deepcopy(draft.claims),
            "authorized_deviations": deepcopy(draft.authorized_deviations),
            "review": {
                "status": "pending_human",
                "reviewer": None,
                "reviewed_at": None,
            },
        }
        plan = self._object_plan(case_id, scene, draft)
        assets = self._asset_selection(case_id, scene)
        deviations_artifact = {
            "schema_version": "authorized_deviations_bundle_v1",
            "case_id": case_id,
            "request_id": case_id,
            "review_status": "pending_human",
            "authorized_deviations": deepcopy(draft.authorized_deviations),
        }

        grouping_case = {
            "room": {
                "boundary": deepcopy(scene["boundary"]),
                "width": float(scene["boundary"][1][0]),
                "depth": float(scene["boundary"][2][1]),
                "regions": deepcopy(scene["metadata"].get("semantic_regions") or []),
            },
            "objects": [
                {
                    "id": obj["id"],
                    "source_region_id": obj.get("source_region_id"),
                }
                for obj in scene["objects"]
            ],
            "visible_relations": self._grouping_visible_relations(draft),
            "visible_attachments": [],
        }
        grouping_report = build_object_grouping_report(
            scene, grouping_case, deepcopy(self.grouping_config)
        )
        grouping = {
            "schema_version": "object_grouping_report_snapshot_v1",
            "case_id": case_id,
            "policy": deepcopy(GROUPING_POLICY),
            "grouping_policy": deepcopy(GROUPING_POLICY),
            "metric_verdict": False,
            "object_groups": deepcopy(grouping_report["object_groups"]),
            "resolved_grouping_config": deepcopy(
                grouping_report["resolved_grouping_config"]
            ),
            "omitted_edges": deepcopy(grouping_report["omitted_edges"]),
            "cross_group_relations": deepcopy(
                grouping_report["cross_group_relations"]
            ),
            "report": grouping_report,
        }

        source_claim_ids = [
            claim["claim_id"]
            for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES
            for claim in draft.claims.get(family, [])
        ]
        canonical_metric = _canonical_metric_for_draft(draft)
        metric_events = {
            "schema_version": "non_l1_metric_events_v1",
            "case_id": case_id,
            "events": [
                {
                    "event_id": event_id,
                    "metric": draft.metric,
                    "canonical_metric": canonical_metric,
                    "level": _metric_level(draft.metric),
                    "canonical_level": _canonical_level(draft.metric),
                    "subfamily": _metric_subfamily(draft.metric),
                    "target_ids": list(draft.target_ids),
                    "relation": draft.event_relation,
                    "source_claim_ids": source_claim_ids,
                    "group_ids": self._target_group_ids(grouping_report, draft.target_ids),
                    "evaluation_purpose": "visual_evidence_sufficiency_then_metric_judgement",
                    "judge_context_allowlist": [
                        "original_prompt",
                        "parsed_prompt_requirements",
                        "authorized_deviations",
                        "target_ids",
                        "relevant_global_visual_evidence",
                        "relevant_local_visual_evidence",
                        "asset_policy",
                        "deterministic_router_evidence_when_applicable",
                    ],
                    "judge_context_forbidden": [
                        "metric_gt",
                        "proposed_semantic_label",
                        "construction_manifest",
                        "review_answer",
                        "source_scene_identity",
                        "local_filesystem_paths",
                        "asset_provenance",
                        "credentials",
                    ],
                }
            ],
        }

        final_label = "pending_review"
        gt_event: dict[str, Any] = {
            "event_id": event_id,
            "metric": draft.metric,
            "canonical_metric": canonical_metric,
            "level": _metric_level(draft.metric),
            "canonical_level": _canonical_level(draft.metric),
            "subfamily": _metric_subfamily(draft.metric),
            "target_ids": list(draft.target_ids),
            "relation": draft.event_relation,
            "semantic_label": final_label,
            "proposed_semantic_label": draft.proposed_label,
            "label_status": "pending_human",
            "accuracy_eligible": False,
            "difficulty": draft.difficulty,
            "gt_basis": draft.gt_basis,
            "source_claim_ids": source_claim_ids,
            "review_question": draft.review_question,
        }
        if draft.metric in L3_METRICS:
            gt_event.update(
                {
                    "raw_coherence_label": draft.raw_coherence_label or draft.proposed_label,
                    "authorization_applied": bool(draft.authorization_applied),
                    "effective_label_after_authorization": draft.proposed_label,
                    "resolution_reason": draft.resolution_reason
                    or "generic_scene_quality_prior",
                    "matched_deviation_id": (
                        draft.authorized_deviations[0].get("deviation_id")
                        if draft.authorization_applied and draft.authorized_deviations
                        else None
                    ),
                }
            )
        if draft.l2_anomaly_label is not None:
            gt_event["paired_l2_anomaly_proposed_label"] = draft.l2_anomaly_label
        metric_gt = {
            "schema_version": "non_l1_metric_gt_v1",
            "case_id": case_id,
            "label_status": "pending_human",
            "accuracy_eligible": False,
            "events": [gt_event],
        }

        evidence_expectations = self._evidence_expectations(case_id, event_id, draft)
        provenance = {
            "schema_version": "non_l1_case_provenance_v1",
            "case_id": case_id,
            "construction": "controlled_local_programmatic_fixture",
            "builder": "scripts/build_cal_dataset2_non_l1.py",
            "build_date": BUILD_DATE,
            "source_scene_ids": list(draft.source_scene_ids),
            "catalog_path": self.catalog_path.relative_to(REPO_ROOT).as_posix(),
            "asset_policy": deepcopy(draft.asset_policy),
            "label_origin": "assistant_proposed_pending_human",
            "renderer_invoked": False,
            "vlm_invoked": False,
            "external_api_invoked": False,
        }
        review = {
            "schema_version": "non_l1_human_review_v1",
            "case_id": case_id,
            "status": "pending_human",
            "required": True,
            "reviewer": None,
            "reviewed_at": None,
            "questions": [
                {
                    "question_id": "prompt_compatibility",
                    "question": (
                        "Is the prompt natural and appropriate for its intended "
                        "difficulty, and is the tested metric claim exactly "
                        "represented by this contract slice?"
                    ),
                    "answer": None,
                },
                {
                    "question_id": "identity_mapping",
                    "question": "Do all target IDs refer to the intended visible objects?",
                    "answer": None,
                },
                {
                    "question_id": "semantic_label",
                    "question": draft.review_question,
                    "answer": None,
                },
                {
                    "question_id": "confound_check",
                    "question": "Does the controlled edit avoid a material L1 or unrelated-metric confound?",
                    "answer": None,
                },
            ],
            "promotion_effect": (
                "Only explicit human approval may freeze the contract and make the event accuracy-eligible."
            ),
        }

        public_payloads = {
            "generated_scene.json": scene,
            "scene_request.json": request,
            "object_plan.json": plan,
            "asset_selection.json": assets,
            "reference_annotation.json": annotation,
            "specification_contract.json": contract,
            "authorized_deviations.json": deviations_artifact,
            "object_grouping_report.json": grouping,
            "metric_events.json": metric_events,
        }
        private_payloads = {
            "metric_gt.json": metric_gt,
            "evidence_expectations.json": evidence_expectations,
            "provenance.json": provenance,
            "review.json": review,
        }

        for filename, payload in {**public_payloads, **private_payloads}.items():
            _write_json(fixture / filename, payload)

        invariant_hashes = {
            "scene_geometry_sha256": geometry_hash,
            "prompt_sha256": _sha256_text(draft.prompt),
            "asset_selection_sha256": _json_hash(assets),
            "grouping_partition_sha256": _grouping_partition_hash(grouping),
        }
        artifact_hashes = {
            filename: _sha256(fixture / filename)
            for filename in sorted({**public_payloads, **private_payloads})
        }
        construction = {
            "schema_version": "non_l1_construction_manifest_v1",
            "case_id": case_id,
            "event_id": event_id,
            "metric": draft.metric,
            "canonical_metric": _canonical_metric_for_draft(draft),
            "canonical_metric": canonical_metric,
            "scenario_id": draft.scenario_id,
            "design_role": draft.design_role,
            "counterfactual_group_id": group_id,
            "prompt_authorization": draft.prompt_authorization,
            "scene_deviation": draft.scene_deviation,
            "visual_source_case_id": visual_source,
            "declared_delta": deepcopy(draft.declared_delta),
            "preserved_invariants": [
                "room shell within each counterfactual family",
                "coordinate convention",
                "renderer configuration deferred and therefore unchanged",
                "fixed-catalog asset policy",
                "one metric event per case",
            ],
            "expected_cross_metric_effects": list(draft.expected_cross_metric_effects or []),
            "l1_sanity_status": (
                "yaw_proxy_bounds_overlap_and_support_preflight_passed_"
                "requires_rendered_mesh_audit"
            ),
            "human_review_status": "pending_human",
            "accuracy_eligible": False,
            "invariant_hashes": invariant_hashes,
            "artifact_hashes": artifact_hashes,
        }
        _write_json(fixture / "construction_manifest.json", construction)

        self._construction_preflight(
            case_id=case_id,
            draft=draft,
            scene=scene,
            request=request,
            plan=plan,
            assets=assets,
            annotation=annotation,
            contract=contract,
            grouping=grouping,
            metric_events=metric_events,
            gt=metric_gt,
        )

        preview_rel = Path("review/previews") / f"{case_id}.svg"
        _write_text(self.out_root / preview_rel, _scene_svg(scene, draft.target_ids))

        record = {
            "case_id": case_id,
            "fixture_dir": fixture_rel.as_posix(),
            "metric": draft.metric,
            "canonical_metric": canonical_metric,
            "level": _metric_level(draft.metric),
            "canonical_level": _canonical_level(draft.metric),
            "subfamily": _metric_subfamily(draft.metric),
            "prompt_granularity": draft.prompt_granularity,
            "scenario_id": draft.scenario_id,
            "design_role": draft.design_role,
            "counterfactual_group_id": group_id,
            "prompt_authorization": draft.prompt_authorization,
            "scene_deviation": draft.scene_deviation,
            "visual_source_case_id": visual_source,
            "event_id": event_id,
            "object_count": len(scene["objects"]),
            "target_count": len(draft.target_ids),
            "label_status": "pending_human",
            "proposed_semantic_label": draft.proposed_label,
            "accuracy_eligible": False,
            "review_status": "pending_human",
            "preview": preview_rel.as_posix(),
            "invariant_hashes": invariant_hashes,
        }
        self.case_records.append(record)
        self.review_rows.append(
            {
                **record,
                "prompt": draft.prompt,
                "review_question": draft.review_question,
                "required_visible_facts": " | ".join(draft.required_visible_facts),
                "declared_delta": json.dumps(draft.declared_delta, ensure_ascii=False, sort_keys=True),
            }
        )
        for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES:
            for claim in contract["claims"][family]:
                self.prompt_rows.append(
                    {
                        "case_id": case_id,
                        "claim_id": claim["claim_id"],
                        "claim_family": family,
                        "source_span": claim.get("source_span", ""),
                        "source_span_start": claim.get("source_span_start", ""),
                        "source_span_end": claim.get("source_span_end", ""),
                        "span_exact": _claim_span_is_exact(draft.prompt, claim),
                    }
                )
        self.counterfactual_rows.append(
            {
                "case_id": case_id,
                "metric": draft.metric,
                "scenario_id": draft.scenario_id,
                "counterfactual_group_id": group_id,
                "design_role": draft.design_role,
                "proposed_semantic_label": draft.proposed_label,
                "prompt_authorization": draft.prompt_authorization or "",
                "scene_deviation": draft.scene_deviation or "",
                "visual_source_case_id": visual_source,
                "scene_geometry_sha256": geometry_hash,
                "prompt_sha256": invariant_hashes["prompt_sha256"],
                "declared_delta": json.dumps(draft.declared_delta, ensure_ascii=False, sort_keys=True),
            }
        )

    def _scene_request(
        self,
        case_id: str,
        scene: dict[str, Any],
        prompt: str,
        granularity: str,
        asset_policy: dict[str, Any],
        deviations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        width = max(point[0] for point in scene["boundary"]) - min(
            point[0] for point in scene["boundary"]
        )
        depth = max(point[1] for point in scene["boundary"]) - min(
            point[1] for point in scene["boundary"]
        )
        dimensions = {
            "width": float(width),
            "depth": float(depth),
            "height": float(scene["scene_height"]),
        }
        return {
            "request_id": case_id,
            "instruction": prompt,
            "scene_type": str(scene["scene_type"]),
            "structure": False,
            "prompt_granularity": granularity,
            "asset_policy": deepcopy(asset_policy),
            "authorized_deviations": deepcopy(deviations),
            "metadata": {
                "generator_skipped": True,
                "reference_annotation_visibility": "benchmark_private",
                "semantic_gt_visibility": "benchmark_private",
                "claim_activation": "specification_contract",
                "specification_contract_scope": "metric_event_slice",
            },
            "room": {
                "boundary": deepcopy(scene["boundary"]),
                "height": float(scene["scene_height"]),
                "unit": "meter",
                "dimensions": dimensions,
                "explicit_dimensions": dimensions,
                "dimension_provenance": {
                    axis: "explicit_calibration_fixture" for axis in dimensions
                },
                "resolution_policy": "room_dimension_policy_v1",
                "topology": "rectangular_enclosed_room",
                "floor_z": 0.0,
                "regions": deepcopy(scene["metadata"].get("semantic_regions") or []),
            },
        }

    def _reference_annotation(
        self, case_id: str, scene: dict[str, Any], draft: CaseDraft
    ) -> dict[str, Any]:
        relation_targets = {
            str(value)
            for relation in [*draft.reference_oor, *draft.reference_oar]
            for key in ("subject_id", "object_id", "subject_ids", "object_ids", "member_ids")
            for value in (
                relation.get(key)
                if isinstance(relation.get(key), list)
                else [relation.get(key)]
            )
            if value
        }
        objects = []
        for obj in scene["objects"]:
            objects.append(
                {
                    "id": obj["id"],
                    "category": obj["category"],
                    "description": obj["description"],
                    "count": 1,
                    "claim_state": (
                        "confirmed" if obj["id"] in relation_targets else "not_mentioned"
                    ),
                    "provenance": {
                        "origin": "controlled_calibration_fixture",
                        "plan_object_id": obj["id"],
                    },
                }
            )
        return {
            "annotation_version": "reference_annotation_v1",
            "validation_status": "draft",
            "source": "model_assisted",
            "request_id": case_id,
            "scene_type": str(scene["scene_type"]),
            "inventory_policy": "open_world",
            "objects": objects,
            "oor_relations": deepcopy(draft.reference_oor),
            "oar_relations": deepcopy(draft.reference_oar),
            "room_constraints": {"claim_state": "not_mentioned"},
            "provenance": {
                "origin": "assistant_authored_pending_human",
                "leaderboard_ground_truth": False,
            },
            "review": {
                "status": "pending",
                "reviewer": None,
                "reviewed_at": None,
            },
        }

    def _object_plan(
        self, case_id: str, scene: dict[str, Any], draft: CaseDraft
    ) -> dict[str, Any]:
        relations = []
        for relation in draft.reference_oor:
            projected = {
                "family": "oor",
                **{
                    key: deepcopy(value)
                    for key, value in relation.items()
                    if key
                    in {
                        "relation_id",
                        "type",
                        "raw_relation",
                        "subject_id",
                        "object_id",
                        "subject_ids",
                        "object_ids",
                        "member_ids",
                        "axis",
                        "direction",
                    }
                },
            }
            relations.append(projected)
        for relation in draft.reference_oar:
            relations.append(
                {
                    "family": "oar",
                    "relation_id": relation["relation_id"],
                    "type": relation["type"],
                    "subject_id": relation["subject_id"],
                    "target": relation["architectural_element"],
                    "architectural_element": relation["architectural_element"],
                }
            )
        explicit_claims = [
            claim["source_span"]
            for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES
            for claim in draft.claims.get(family, [])
            if claim.get("source_span")
        ]
        return {
            "request_id": case_id,
            "scene_type": str(scene["scene_type"]),
            "scene_description": draft.prompt,
            "prompt_granularity": draft.prompt_granularity,
            "explicit_claims": explicit_claims,
            "objects": [
                {
                    "id": obj["id"],
                    "role": "",
                    "category": obj["category"],
                    "description": obj["description"],
                    "count": 1,
                    "placement_intent": {
                        "absolute_relations": [],
                        "relative_relations": [],
                    },
                    "metadata": {
                        "description_source": "fixed_catalog.short_desc",
                        "conversion_source": "controlled_calibration_fixture",
                    },
                }
                for obj in scene["objects"]
            ],
            "global_constraints": explicit_claims,
            "relations": relations,
            "metadata": {
                "conversion": {
                    "mode": "offline_assistant_authored",
                    "runtime_benchmark_component": False,
                    "generator_visible": False,
                    "review_status": "pending_human",
                },
                "literal_claims_only": True,
            },
        }

    def _asset_selection(
        self, case_id: str, scene: dict[str, Any]
    ) -> dict[str, Any]:
        records = []
        for obj in scene["objects"]:
            asset = self._asset(obj["jid"])
            selected = {
                "jid": obj["jid"],
                "category": obj["category"],
                "retrieval_category": asset.category,
                "desc": asset.desc,
                "short_desc": asset.short_desc,
                "size": list(obj["size"]),
                "catalog_size": list(asset.size),
                "asset_ref": deepcopy(obj["asset_ref"]),
                "asset_proxy": deepcopy(obj["asset_proxy"]),
                "metadata": {
                    "interactive": False,
                    "inner_placement": asset.inner_placement,
                    "align_to_wall_normal": asset.align_to_wall_normal,
                    "scaling_strategy": asset.scaling_strategy,
                    "catalog_scale_factors": deepcopy(
                        obj["metadata"].get("catalog_scale_factors")
                    ),
                },
            }
            records.append(
                {
                    "object_id": obj["id"],
                    "object_spec": {
                        "role": "",
                        "category": obj["category"],
                        "description": obj["description"],
                        "estimated_size": list(obj["size"]),
                        "count": 1,
                    },
                    "retrieval_query": {
                        "description": obj["description"],
                        "category": obj["category"],
                        "size_constraint": None,
                    },
                    "selected_asset": selected,
                    "candidates": [{**deepcopy(selected), "score": 1.0}],
                    "selection_action": "select",
                    "selection_decision": {
                        "action": "select",
                        "selected_jid": obj["jid"],
                        "reason": "frozen calibration asset",
                        "generation_request": None,
                    },
                    "selection_reason": "frozen calibration asset",
                }
            )
        return {
            "request_id": case_id,
            "asset_policy": deepcopy(ASSET_POLICY),
            "objects": records,
        }

    @staticmethod
    def _grouping_visible_relations(draft: CaseDraft) -> list[dict[str, Any]]:
        records = []
        for relation in draft.reference_oor:
            if relation.get("subject_id") and relation.get("object_id"):
                records.append(
                    {
                        "subject": relation["subject_id"],
                        "object": relation["object_id"],
                        "type": relation["type"],
                    }
                )
        return records

    @staticmethod
    def _target_group_ids(
        grouping_report: dict[str, Any], target_ids: list[str]
    ) -> list[str]:
        target_set = set(target_ids)
        return [
            group["group_id"]
            for group in grouping_report.get("object_groups", [])
            if target_set.intersection(group.get("object_ids", []))
        ]

    def _evidence_expectations(
        self, case_id: str, event_id: str, draft: CaseDraft
    ) -> dict[str, Any]:
        policy = _evidence_policy_for_metric(draft.metric)
        return {
            "schema_version": "non_l1_evidence_expectations_v1",
            "case_id": case_id,
            "event_id": event_id,
            "metric": draft.metric,
            "truth_type": "visual_evidence_sufficiency_not_scene_validity",
            "review_status": "pending_human",
            "required_visible_facts": list(draft.required_visible_facts),
            "candidate_arm_ids": list(policy["candidate_arm_ids"]),
            "production_default_under_test": deepcopy(policy["production_default_under_test"]),
            "minimum_packet_requirements": deepcopy(policy["minimum_packet_requirements"]),
            "human_questions": [
                "Can the packet support a verdict without hidden geometry?",
                "Are every designated target and the relevant contextual reference simultaneously identifiable?",
                "Would a reasonable judge need an additional view or active-camera fallback?",
            ],
            "outbound_context_allowlist": [
                "original_prompt",
                "parsed_prompt_requirements",
                "authorized_deviations",
                "target_ids",
                "relevant_global_visual_evidence",
                "relevant_local_visual_evidence",
                "asset_policy",
                "deterministic_router_evidence_when_applicable",
            ],
            "no_predeclared_winning_arm": True,
            "forbidden_judge_context": [
                "ground_truth_artifacts",
                "construction_truth",
                "proposed_outcome",
                "human_audit_answer",
            ],
        }

    def _construction_preflight(
        self,
        *,
        case_id: str,
        draft: CaseDraft,
        scene: dict[str, Any],
        request: dict[str, Any],
        plan: dict[str, Any],
        assets: dict[str, Any],
        annotation: dict[str, Any],
        contract: dict[str, Any],
        grouping: dict[str, Any],
        metric_events: dict[str, Any],
        gt: dict[str, Any],
    ) -> None:
        validate_generated_scene(scene)
        validate_scene_request(request)
        validate_object_plan(plan)
        validate_asset_selection(assets)
        validate_reference_annotation(annotation)
        validate_specification_contract(
            contract, valid_object_ids={obj["id"] for obj in scene["objects"]}
        )
        validate_authorized_deviations(
            draft.authorized_deviations,
            metric_normalizer=normalize_metric_name,
            allowed_metrics=SCENE_QUALITY_INTERFACE_METRICS,
        )
        if scene["scene_id"] != case_id or request["request_id"] != case_id:
            raise DatasetBuildError(f"{case_id}: request/scene identity mismatch")
        if len(metric_events["events"]) != 1 or len(gt["events"]) != 1:
            raise DatasetBuildError(f"{case_id}: expected exactly one metric event")
        scene_ids = {obj["id"] for obj in scene["objects"]}
        if not set(draft.target_ids) <= scene_ids:
            raise DatasetBuildError(f"{case_id}: target IDs are not a scene subset")
        for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES:
            for claim in contract["claims"][family]:
                if not _claim_span_is_exact(draft.prompt, claim):
                    raise DatasetBuildError(
                        f"{case_id}: claim {claim['claim_id']} has non-exact prompt span"
                    )
        claim_ids = {
            claim["claim_id"]
            for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES
            for claim in contract["claims"][family]
        }
        for deviation in draft.authorized_deviations:
            if deviation["source_claim_id"] not in claim_ids:
                raise DatasetBuildError(
                    f"{case_id}: deviation source claim does not exist"
                )
            start = deviation["prompt_span_start"]
            end = deviation["prompt_span_end"]
            if draft.prompt[start:end] != deviation["prompt_span"]:
                raise DatasetBuildError(
                    f"{case_id}: deviation prompt span is not exact"
                )
            if not set(deviation["target_ids"]) <= scene_ids:
                raise DatasetBuildError(
                    f"{case_id}: deviation references missing target"
                )
        assigned = [
            object_id
            for group in grouping["report"]["object_groups"]
            for object_id in group["object_ids"]
        ]
        if sorted(assigned) != sorted(scene_ids) or len(assigned) != len(set(assigned)):
            raise DatasetBuildError(f"{case_id}: grouping is not an exact partition")
        for obj in scene["objects"]:
            _validate_object_room_bounds(scene, obj, case_id)
            asset_dir = self.asset_root / obj["jid"]
            for suffix in (".fbx", ".ply", "_metadata.json"):
                path = asset_dir / (
                    f"{obj['jid']}{suffix}"
                    if suffix != "_metadata.json"
                    else f"{obj['jid']}_metadata.json"
                )
                if not path.is_file():
                    raise DatasetBuildError(f"{case_id}: missing asset file {path}")
        self._validate_proxy_overlap_and_support(
            case_id=case_id,
            draft=draft,
            scene=scene,
        )
        if gt["accuracy_eligible"] is not False or gt["label_status"] != "pending_human":
            raise DatasetBuildError(f"{case_id}: draft GT lifecycle violation")

    def _validate_proxy_overlap_and_support(
        self,
        *,
        case_id: str,
        draft: CaseDraft,
        scene: dict[str, Any],
    ) -> None:
        objects = list(scene["objects"])
        by_id = {obj["id"]: obj for obj in objects}
        proxy_tolerance = 0.01
        support_tolerance = 0.002

        for index, first in enumerate(objects):
            first_bounds = _object_proxy_bounds(first)
            for second in objects[index + 1 :]:
                if {first["category"], second["category"]} & {"rug", "carpet"}:
                    continue
                if (
                    first.get("support_parent") == second["id"]
                    or second.get("support_parent") == first["id"]
                ):
                    continue
                second_bounds = _object_proxy_bounds(second)
                overlaps = (
                    min(first_bounds[1], second_bounds[1])
                    - max(first_bounds[0], second_bounds[0]),
                    min(first_bounds[3], second_bounds[3])
                    - max(first_bounds[2], second_bounds[2]),
                    min(first_bounds[5], second_bounds[5])
                    - max(first_bounds[4], second_bounds[4]),
                )
                if all(value > proxy_tolerance for value in overlaps):
                    raise DatasetBuildError(
                        f"{case_id}: proxy overlap confound between "
                        f"{first['id']} and {second['id']}: {overlaps}"
                    )

        for obj in objects:
            bounds = _object_proxy_bounds(obj)
            bottom = bounds[4]
            if bottom <= support_tolerance:
                continue
            parent_id = obj.get("support_parent")
            if parent_id:
                parent = by_id.get(str(parent_id))
                if parent is None:
                    raise DatasetBuildError(
                        f"{case_id}: {obj['id']} has missing support_parent {parent_id}"
                    )
                parent_bounds = _object_proxy_bounds(parent)
                gap = bottom - parent_bounds[5]
                xy_overlap = (
                    min(bounds[1], parent_bounds[1]) - max(bounds[0], parent_bounds[0]),
                    min(bounds[3], parent_bounds[3]) - max(bounds[2], parent_bounds[2]),
                )
                if abs(gap) > support_tolerance or min(xy_overlap) <= 0:
                    raise DatasetBuildError(
                        f"{case_id}: {obj['id']} support contact with {parent_id} "
                        f"is inconsistent (gap={gap}, xy_overlap={xy_overlap})"
                    )
                continue

            # The tested OAR subject is intentionally supported by architecture,
            # not another scene object.  Other wall-mounted catalog assets are
            # allowed only when their proxy is close to a room plane.
            if draft.metric == "oar" and obj["id"] in draft.target_ids:
                continue
            width = max(point[0] for point in scene["boundary"])
            depth = max(point[1] for point in scene["boundary"])
            near_wall = min(
                abs(bounds[0]),
                abs(width - bounds[1]),
                abs(bounds[2]),
                abs(depth - bounds[3]),
            ) <= 0.08
            if near_wall and obj["category"] in {
                "clock",
                "picture",
                "frame",
                "art",
                "artwork",
                "tv",
            }:
                continue
            raise DatasetBuildError(
                f"{case_id}: {obj['id']} is elevated without an explicit "
                "object or architecture support"
            )

    def _write_dataset_files(self) -> None:
        metric_counts = Counter(record["metric"] for record in self.case_records)
        role_counts = Counter(record["design_role"] for record in self.case_records)
        proposed_counts = Counter(
            record["proposed_semantic_label"] for record in self.case_records
        )
        if set(metric_counts) != set(METRICS):
            raise DatasetBuildError(f"metric coverage mismatch: {dict(metric_counts)}")
        expected_counts = {
            **{metric: BASE_CASES_PER_METRIC for metric in L2_METRICS},
            **{metric: BASE_CASES_PER_METRIC + 4 for metric in L3_METRICS},
        }
        if dict(metric_counts) != expected_counts:
            raise DatasetBuildError(
                f"metric counts {dict(metric_counts)} != {expected_counts}"
            )
        if role_counts != Counter(
            {"base_metric_case": 96, "prompt_authorization_2x2": 12}
        ):
            raise DatasetBuildError(f"design-role counts are {dict(role_counts)}")

        blind_case_records = [
            {
                key: deepcopy(value)
                for key, value in record.items()
                if key
                in {
                    "case_id",
                    "fixture_dir",
                    "metric",
                    "canonical_metric",
                    "level",
                    "canonical_level",
                    "subfamily",
                    "prompt_granularity",
                    "event_id",
                    "object_count",
                    "target_count",
                    "label_status",
                    "accuracy_eligible",
                    "review_status",
                    "preview",
                    "invariant_hashes",
                }
            }
            for record in _blind_review_order(self.case_records)
        ]
        cases_payload = {
            "schema_version": "non_l1_calibration_case_index_v1",
            "dataset_id": DATASET_ID,
            "case_count": len(blind_case_records),
            "review_mode": "blind_first_pass",
            "cases": blind_case_records,
        }
        _write_json(self.out_root / "cases.json", cases_payload)

        profile = {
            "schema_version": "non_l1_evaluation_profile_v1",
            "profile_id": "cal_dataset2_claim_driven_v2",
            "activation_mode": "specification_contract",
            "prompt_granularity_role": "metadata_and_reporting_slice",
            "levels_in_scope": ["L2_specification_fidelity", "L3_scene_quality"],
            "levels_excluded": {
                "L1_physical_plausibility": "excluded_from_scoring_but_used_as_confound_preflight",
                "L4_task_functionality": "deferred_until_downstream_task_family_is_fixed",
            },
            "metrics": list(METRICS),
            "legacy_case_slice_metrics": list(METRICS),
            "canonical_metrics": [
                "oor",
                "oar",
                "functional_semantic_fidelity",
                "scale_consistency",
                "object_pairing_consistency",
                "style_consistency",
            ],
            "canonical_hierarchy": {
                "L2": {
                    "explicit_relations": ["oor", "oar"],
                    "high_level": ["functional_semantic_fidelity"],
                },
                "L3": {
                    "semantic_coherence": [
                        "scale_consistency",
                        "object_pairing_consistency",
                    ],
                    "perceptual_visual_quality": ["style_consistency"],
                },
            },
            "compatibility_notes": [
                "OOR/OAR cases remain fine_grained because the current runtime still gates their execution there.",
                "High-level L2 and canonical L3 interfaces are structurally defined but their final judges are not implemented.",
                "Legacy holistic visual_quality cannot honor authorized deviations and is compatibility-only.",
                "Prompt granularity and asset strategy are independent axes.",
                "Each specification contract is an explicit metric-event slice, not an exhaustive parse of every prompt clause.",
                "Legacy room/intent/area case labels are components of functional_semantic_fidelity.",
                "Legacy pairing orientation/group-function cases route to OOR or functional_semantic_fidelity; only category_coexistence remains L3 pairing.",
            ],
            "asset_policy": deepcopy(ASSET_POLICY),
            "grouping_policy": deepcopy(GROUPING_POLICY),
        }
        _write_json(
            self.out_root / "configs/evaluation_profile_claim_driven_v2.json",
            profile,
        )
        _write_json(
            self.out_root / "configs/evidence_arms.json",
            _evidence_arms_config(),
        )

        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "created_at": BUILD_DATE,
            "status": "draft_pending_human_review",
            "accuracy_eligible": False,
            "case_count": len(self.case_records),
            "event_count": len(self.case_records),
            "one_metric_event_per_case": True,
            "metric_counts": dict(metric_counts),
            "design_role_counts": dict(role_counts),
            "proposed_label_counts": dict(proposed_counts),
            "human_audit": {
                "status": "pending",
                "required": True,
                "mode": "blind_first_pass",
                "order": "deterministic_hash_permutation_v1",
                "construction_metadata_hidden": True,
                "review_index": "review/index.html",
                "review_queue": "review/review_queue.tsv",
                "review_protocol": "review/REVIEW_PROTOCOL.md",
                "construction_proposals": "validation/construction_proposals.tsv",
                "promotion_not_performed": True,
            },
            "construction_design": {
                "base_metric_cases": {
                    "count": 96,
                    "per_metric": {
                        "scenarios": 4,
                        "variants_per_scenario": ["valid", "invalid", "ambiguous"],
                    },
                },
                "prompt_authorization_2x2": {
                    "count": 12,
                    "metrics": list(L3_METRICS),
                    "cells": [
                        "authorization_absent__deviation_absent",
                        "authorization_absent__deviation_present",
                        "authorization_present__deviation_absent",
                        "authorization_present__deviation_present",
                    ],
                },
            },
            "frozen_variables": [
                "canonical coordinate frame",
                "fixed local asset catalog",
                "one metric event per case",
                "no VLM or renderer during construction",
                "claim-driven activation contract",
                "human-review lifecycle",
            ],
            "known_limitations": [
                "Style cases emphasize shape language; rendered material/color visibility still requires audit.",
                "Asset forward-axis conventions in orientation cases require rendered human confirmation.",
                "AABB room preflight does not replace mesh-level L1 validation.",
                "High-level L2 and canonical L3 runtime judges remain future implementations.",
            ],
            "security": {
                "credentials_used": False,
                "remote_calls": False,
                "api_calls": False,
                "forbidden_context_separated": True,
            },
            "entrypoints": {
                "builder": "Support/scripts/build_cal_dataset2_non_l1.py",
                "validator": "scripts/validate_cal_dataset2_non_l1.py",
            },
        }
        _write_json(self.out_root / "dataset_manifest.json", manifest)

        self._write_review_files()
        self._write_validation_inventory()
        _write_text(self.out_root / "README.md", _dataset_readme())
        self._refresh_file_inventory()

    def _write_review_files(self) -> None:
        blind_rows = _blind_review_order(self.review_rows)
        review_fields = [
            "case_id",
            "event_id",
            "metric",
            "level",
            "prompt_granularity",
            "label_status",
            "object_count",
            "target_count",
            "prompt",
            "review_question",
            "required_visible_facts",
            "preview",
            "human_semantic_label",
            "prompt_compatible",
            "target_mapping_correct",
            "needs_render_check",
            "notes",
        ]
        _write_tsv(
            self.out_root / "review/review_queue.tsv",
            blind_rows,
            review_fields,
        )
        _write_tsv(
            self.out_root / "validation/prompt_claim_audit.tsv",
            self.prompt_rows,
            [
                "case_id",
                "claim_id",
                "claim_family",
                "source_span",
                "source_span_start",
                "source_span_end",
                "span_exact",
            ],
        )
        _write_tsv(
            self.out_root / "validation/counterfactual_audit.tsv",
            self.counterfactual_rows,
            [
                "case_id",
                "metric",
                "scenario_id",
                "counterfactual_group_id",
                "design_role",
                "prompt_authorization",
                "scene_deviation",
                "visual_source_case_id",
                "scene_geometry_sha256",
                "prompt_sha256",
                "declared_delta",
            ],
        )
        _write_text(self.out_root / "review/index.html", _review_html(blind_rows))
        _write_text(
            self.out_root / "review/REVIEW_PROTOCOL.md",
            _review_protocol(),
        )
        _write_tsv(
            self.out_root / "validation/construction_proposals.tsv",
            self.review_rows,
            [
                "case_id",
                "event_id",
                "metric",
                "scenario_id",
                "proposed_semantic_label",
                "prompt_authorization",
                "scene_deviation",
            ],
        )

    def _write_validation_inventory(self) -> None:
        fixture_files = sorted(
            path
            for path in (self.out_root / "fixtures").rglob("*")
            if path.is_file()
        )
        inventory = {
            "schema_version": "non_l1_file_inventory_v1",
            "dataset_id": DATASET_ID,
            "file_count": len(fixture_files),
            "files": [
                {
                    "path": path.relative_to(self.out_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in fixture_files
            ],
        }
        _write_json(self.out_root / "validation/file_inventory.json", inventory)
        _write_tsv(
            self.out_root / "validation/case_inventory.tsv",
            self.case_records,
            [
                "case_id",
                "fixture_dir",
                "metric",
                "level",
                "subfamily",
                "prompt_granularity",
                "scenario_id",
                "design_role",
                "counterfactual_group_id",
                "prompt_authorization",
                "scene_deviation",
                "visual_source_case_id",
                "event_id",
                "object_count",
                "target_count",
                "label_status",
                "proposed_semantic_label",
                "accuracy_eligible",
                "review_status",
                "preview",
            ],
        )
        metric_rows = [
            {
                "case_id": record["case_id"],
                "event_id": record["event_id"],
                "metric": record["metric"],
                "level": record["level"],
                "subfamily": record["subfamily"],
                "design_role": record["design_role"],
                "label_status": record["label_status"],
            }
            for record in self.case_records
        ]
        _write_tsv(
            self.out_root / "validation/metric_inventory.tsv",
            metric_rows,
            [
                "case_id",
                "event_id",
                "metric",
                "level",
                "subfamily",
                "design_role",
                "label_status",
            ],
        )

    def _refresh_file_inventory(self) -> None:
        files = sorted(
            path
            for path in self.out_root.rglob("*")
            if path.is_file()
            and path.relative_to(self.out_root).as_posix()
            not in {
                "validation/file_inventory.json",
                "validation/validation_report.json",
            }
        )
        _write_json(
            self.out_root / "validation/file_inventory.json",
            {
                "schema_version": "non_l1_file_inventory_v1",
                "dataset_id": DATASET_ID,
                "file_count": len(files),
                "files": [
                    {
                        "path": path.relative_to(self.out_root).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in files
                ],
            },
        )


# ----------------------------------------------------------------------
# Global helpers
# ----------------------------------------------------------------------
def _read_catalog(path: Path) -> dict[str, AssetRecord]:
    result: dict[str, AssetRecord] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            jid = str(row.get("name_en") or "").strip()
            if not jid:
                continue
            try:
                size = tuple(float(value) for value in str(row["bbx"]).split(","))
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetBuildError(f"invalid catalog size for {jid}") from exc
            # The legacy catalog contains a few unusable zero-extent records.
            # Keep them out of the calibration asset closure; requesting one
            # later will fail explicitly through ``Builder._asset``.
            if len(size) != 3 or any(value <= 0 for value in size):
                continue
            result[jid] = AssetRecord(
                jid=jid,
                category=str(row.get("category") or "object"),
                short_desc=str(row.get("short_desc") or jid.replace("_", " ")),
                desc=str(row.get("caption_en") or row.get("short_desc") or jid),
                size=(float(size[0]), float(size[1]), float(size[2])),
                scaling_strategy=str(row.get("scaling_strategy") or "") or None,
                inner_placement=str(row.get("inner_placement") or "0") == "1",
                align_to_wall_normal=str(row.get("alignToWallNormal") or "0") == "1",
            )
    return result


def _metric_level(metric: str) -> str:
    return "L2" if metric in L2_METRICS else "L3"


def _canonical_level(metric: str) -> str:
    return (
        "L2_specification_fidelity"
        if metric in L2_METRICS
        else "L3_scene_quality"
    )


def _metric_subfamily(metric: str) -> str:
    if metric in {"oor", "oar"}:
        return "fine_grained_relation_fidelity"
    if metric in {
        "room_scene_type",
        "broad_semantic_intent",
        "required_functional_areas",
    }:
        return "coarse_grained_intent_fidelity"
    if metric in {"scale_consistency", "object_pairing_consistency"}:
        return "semantic_coherence"
    return "perceptual_visual_quality"


def _canonical_metric_for_draft(draft: CaseDraft) -> str:
    if draft.metric in {
        "room_scene_type",
        "broad_semantic_intent",
        "required_functional_areas",
    }:
        return "functional_semantic_fidelity"
    if draft.metric != "object_pairing_consistency":
        return draft.metric
    source_claim_ids = {
        str(claim.get("claim_id") or "")
        for family in ACCEPTED_SPECIFICATION_CLAIM_FAMILIES
        for claim in draft.claims.get(family, [])
    }
    if any(claim_id.startswith("oor::") for claim_id in source_claim_ids):
        return "oor"
    if draft.event_relation == "category_coexistence":
        return "object_pairing_consistency"
    return "functional_semantic_fidelity"


def _evidence_policy_for_metric(metric: str) -> dict[str, Any]:
    common_arms = [
        "global_context_raw",
        "local_raw",
        "local_contour",
        "global_plus_local_raw",
        "adaptive_global_then_local",
    ]
    if metric in {
        "room_scene_type",
        "broad_semantic_intent",
        "required_functional_areas",
    }:
        default = {
            "evidence_strategy": "global_only",
            "global_policy": {
                "view_family": "wall_occlusion_aware_room_perspective",
                "image_budget": 2,
                "top_down": False,
                "perspective_diversity_required": True,
            },
            "local_policy": None,
        }
        minimum = {
            "global_views": 2,
            "local_views": 0,
            "must_preserve_room_context": True,
        }
    elif metric == "scale_consistency":
        default = {
            "evidence_strategy": "script_screen_then_local",
            "global_policy": {
                "view_family": "wall_occlusion_aware_room_perspective",
                "image_budget": 1,
                "top_down": False,
            },
            "local_policy": {
                "camera_scope": "object_local",
                "image_budget": 2,
                "presentation": "raw",
            },
        }
        minimum = {
            "global_views": 1,
            "local_views": 1,
            "must_show_targets_with_shared_scale_or_function_context": True,
        }
    elif metric == "object_pairing_consistency":
        default = {
            "evidence_strategy": "global_and_local",
            "global_policy": {
                "view_family": "wall_occlusion_aware_room_perspective",
                "image_budget": 1,
                "top_down": False,
            },
            "local_policy": {
                "camera_scope": "group_local",
                "image_budget": 2,
                "presentation": "raw",
                "prerequisite": "object_grouping_report",
            },
        }
        minimum = {
            "global_views": 1,
            "local_views": 1,
            "must_show_group_member_categories_and_roles": True,
        }
    elif metric == "style_consistency":
        default = {
            "evidence_strategy": "global_screen_then_local",
            "global_policy": {
                "view_family": "global_top",
                "image_budget": 1,
                "top_down": True,
            },
            "local_policy": {
                "camera_scope": "group_local",
                "trigger_states": ["suspicious", "insufficient_evidence"],
                "image_budget": 2,
                "presentation": "raw",
            },
        }
        minimum = {
            "global_views": 1,
            "local_views_when_triggered": 1,
            "must_show_target_and_style_comparators": True,
        }
    else:
        default = {
            "evidence_strategy": "global_and_local",
            "global_policy": {
                "view_family": "wall_occlusion_aware_room_perspective",
                "image_budget": 1,
                "top_down": False,
            },
            "local_policy": {
                "camera_scope": "pair_local",
                "image_budget": 2,
                "presentation": "contour",
            },
        }
        minimum = {
            "global_views": 1,
            "local_views": 2,
            "must_show_targets_and_relation_reference": True,
        }
        if metric == "oar":
            minimum["must_include_room_axis_and_architecture_plane_legend"] = True
    return {
        "candidate_arm_ids": common_arms,
        "production_default_under_test": default,
        "minimum_packet_requirements": minimum,
    }


def _evidence_arms_config() -> dict[str, Any]:
    return {
        "schema_version": "non_l1_evidence_arms_v1",
        "purpose": "visual_evidence_sufficiency_ablation_not_metric_gt",
        "packet_components": [
            {
                "arm_id": "global_context_raw",
                "evidence_strategy": "global_only",
                "global_views": 2,
                "local_views": 0,
                "presentation": "raw",
                "global_view_family": "wall_occlusion_aware_room_perspective",
            },
            {
                "arm_id": "local_raw",
                "evidence_strategy": "script_screen_then_local",
                "global_views": 0,
                "local_views": 2,
                "presentation": "raw",
                "role": "ablation_packet_component_not_production_strategy",
            },
            {
                "arm_id": "local_contour",
                "evidence_strategy": "script_screen_then_local",
                "global_views": 0,
                "local_views": 2,
                "presentation": "contour",
                "role": "ablation_packet_component_not_production_strategy",
            },
            {
                "arm_id": "global_plus_local_raw",
                "evidence_strategy": "global_and_local",
                "global_views": 1,
                "local_views": 2,
                "presentation": "raw",
            },
            {
                "arm_id": "adaptive_global_then_local",
                "evidence_strategy": "global_screen_then_local",
                "global_views": 2,
                "local_views": {"minimum": 0, "maximum": 2},
                "presentation": "raw",
                "local_trigger_states": ["suspicious", "insufficient_evidence"],
            },
        ],
        "frozen_across_arms": [
            "scene",
            "prompt",
            "target identities",
            "renderer",
            "judge model within a comparison",
            "judge rubric",
        ],
        "no_predeclared_winner": True,
    }


def _claim_span_is_exact(prompt: str, claim: dict[str, Any]) -> bool:
    span = claim.get("source_span")
    start = claim.get("source_span_start")
    end = claim.get("source_span_end")
    return (
        isinstance(span, str)
        and isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(prompt)
        and prompt[start:end] == span
    )


def _validate_object_room_bounds(
    scene: dict[str, Any], obj: dict[str, Any], case_id: str
) -> None:
    width = max(point[0] for point in scene["boundary"])
    depth = max(point[1] for point in scene["boundary"])
    height = float(scene["scene_height"])
    # Conservative yaw-aware world AABB for the rectangular object proxy.
    bounds = _object_proxy_bounds(obj)
    tolerance = 1.0e-5
    if (
        bounds[0] < -tolerance
        or bounds[1] > width + tolerance
        or bounds[2] < -tolerance
        or bounds[3] > depth + tolerance
        or bounds[4] < -tolerance
        or bounds[5] > height + tolerance
    ):
        raise DatasetBuildError(
            f"{case_id}: {obj['id']} proxy bounds {bounds} exceed room {(width, depth, height)}"
        )


def _object_proxy_bounds(obj: dict[str, Any]) -> tuple[float, ...]:
    yaw = math.radians(float(obj["rotation"][2]))
    sx, sy, sz = [float(value) for value in obj["size"]]
    world_sx = abs(math.cos(yaw)) * sx + abs(math.sin(yaw)) * sy
    world_sy = abs(math.sin(yaw)) * sx + abs(math.cos(yaw)) * sy
    x, y, z = [float(value) for value in obj["center"]]
    return (
        x - world_sx / 2.0,
        x + world_sx / 2.0,
        y - world_sy / 2.0,
        y + world_sy / 2.0,
        z - sz / 2.0,
        z + sz / 2.0,
    )


def _normalized_scene_geometry(scene: dict[str, Any]) -> dict[str, Any]:
    return {
        "boundary": scene["boundary"],
        "scene_height": scene["scene_height"],
        "objects": [
            {
                "id": obj["id"],
                "category": obj["category"],
                "jid": obj["jid"],
                "asset_ref": obj.get("asset_ref"),
                "asset_proxy": obj.get("asset_proxy"),
                "size": obj["size"],
                "center": obj["center"],
                "rotation": obj["rotation"],
                "geometry_provenance": obj.get("geometry_provenance"),
                "support_parent": obj.get("support_parent"),
            }
            for obj in scene["objects"]
        ],
        "relations": scene.get("relations", []),
        "oor_relations": scene.get("oor_relations", []),
        "oar_relations": scene.get("oar_relations", []),
    }


def _geometry_hash(scene: dict[str, Any]) -> str:
    return _json_hash(_normalized_scene_geometry(scene))


def _asset_identity_hash(scene: dict[str, Any]) -> str:
    return _json_hash(
        [
            {
                "id": obj["id"],
                "jid": obj["jid"],
                "size": obj["size"],
            }
            for obj in scene["objects"]
        ]
    )


def _grouping_partition_hash(report: dict[str, Any]) -> str:
    groups = [
        sorted(str(object_id) for object_id in group.get("object_ids", []))
        for group in report.get("object_groups", [])
        if isinstance(group, dict)
    ]
    groups.sort()
    return _json_hash(groups)


def _json_hash(value: Any) -> str:
    data = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _blind_review_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministic, label-independent permutation for first-pass review."""

    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (
                f"{DATASET_ID}:blind-semantic-pass-v1:"
                f"{row.get('case_id', '')}"
            ).encode("utf-8")
        ).hexdigest(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if row.get(field) is None
                        else json.dumps(row[field], ensure_ascii=False)
                        if isinstance(row.get(field), (dict, list))
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def _scene_svg(scene: dict[str, Any], target_ids: list[str]) -> str:
    width_m = max(point[0] for point in scene["boundary"])
    depth_m = max(point[1] for point in scene["boundary"])
    canvas_w = 760
    canvas_h = max(420, int(canvas_w * depth_m / width_m))
    margin = 36
    scale = min((canvas_w - 2 * margin) / width_m, (canvas_h - 2 * margin) / depth_m)
    target_set = set(target_ids)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {canvas_w} {canvas_h}" width="{canvas_w}" height="{canvas_h}">',
        "<style>"
        ".room{fill:#f7f7f4;stroke:#333;stroke-width:3}"
        ".obj{fill:#c8d7e8;stroke:#35546d;stroke-width:1.5}"
        ".target{fill:#ffd7a8;stroke:#b45f06;stroke-width:3}"
        ".label{font:12px sans-serif;fill:#111}"
        ".arrow{stroke:#111;stroke-width:2;marker-end:url(#a)}"
        "</style>",
        '<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" '
        'orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#111"/></marker></defs>',
        f'<rect class="room" x="{margin}" y="{margin}" '
        f'width="{width_m * scale:.2f}" height="{depth_m * scale:.2f}"/>',
    ]
    for obj in scene["objects"]:
        x, y, _ = obj["center"]
        sx, sy, _ = obj["size"]
        px = margin + (x - sx / 2.0) * scale
        py = margin + (depth_m - y - sy / 2.0) * scale
        cls = "target" if obj["id"] in target_set else "obj"
        cx = margin + x * scale
        cy = margin + (depth_m - y) * scale
        angle = -float(obj["rotation"][2])
        parts.append(
            f'<g transform="rotate({angle:.2f} {cx:.2f} {cy:.2f})">'
            f'<rect class="{cls}" x="{px:.2f}" y="{py:.2f}" '
            f'width="{sx * scale:.2f}" height="{sy * scale:.2f}" rx="3"/>'
            "</g>"
        )
        arrow_len = max(16.0, min(42.0, sy * scale * 0.45))
        yaw = math.radians(float(obj["rotation"][2]))
        ax = cx + arrow_len * math.sin(yaw)
        ay = cy - arrow_len * math.cos(yaw)
        parts.append(
            f'<line class="arrow" x1="{cx:.2f}" y1="{cy:.2f}" '
            f'x2="{ax:.2f}" y2="{ay:.2f}"/>'
        )
        label = html.escape(f"{obj['id']} · {obj['short_desc']}")
        parts.append(
            f'<text class="label" x="{cx + 5:.2f}" y="{cy - 5:.2f}">{label}</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _review_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        facts = "".join(
            f"<li>{html.escape(item)}</li>"
            for item in str(row["required_visible_facts"]).split(" | ")
        )
        cards.append(
            f"""
<article class="card" data-metric="{html.escape(row['metric'])}">
  <header>
    <span class="case">{html.escape(row['case_id'])}</span>
    <span class="metric">{html.escape(row['metric'])}</span>
  </header>
  <div class="grid">
    <img src="../{html.escape(row['preview'])}" alt="Top-down construction preview">
    <section>
      <h3>Prompt</h3><p>{html.escape(row['prompt'])}</p>
      <h3>Human question</h3><p>{html.escape(row['review_question'])}</p>
      <h3>Required visible facts</h3><ul>{facts}</ul>
      <p class="meta">granularity={html.escape(row['prompt_granularity'])}</p>
    </section>
  </div>
</article>"""
        )
    filter_buttons = "".join(
        (
            "<button onclick=\"filterCards('"
            + html.escape(metric)
            + "')\">"
            + html.escape(metric)
            + "</button>"
        )
        for metric in METRICS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>cal_dataset2 non-L1 human review</title>
<style>
body{{margin:0;background:#f2f1ed;color:#1d1d1b;font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1320px;margin:auto;padding:24px}}
h1{{margin:0 0 8px}} .notice{{background:#fff3cd;border:1px solid #d6b656;padding:12px}}
.filters{{position:sticky;top:0;background:#f2f1ed;padding:12px 0;z-index:2}}
button{{margin:3px;padding:7px 10px;border:1px solid #777;background:white;border-radius:6px}}
.card{{background:white;border:1px solid #ccc;border-radius:10px;margin:18px 0;overflow:hidden}}
header{{display:flex;gap:12px;align-items:center;padding:10px 14px;background:#20252b;color:white}}
.case{{font-weight:700}} .metric{{color:#b8daf7}}
.grid{{display:grid;grid-template-columns:minmax(420px,56%) 1fr;gap:18px;padding:16px}}
img{{width:100%;border:1px solid #aaa;background:#fafafa}} h3{{margin:10px 0 3px}}
pre{{white-space:pre-wrap;background:#f6f6f6;padding:8px}} .meta{{color:#666}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<h1>cal_dataset2 · Non-L1 evidence calibration</h1>
<p class="notice"><strong>Blind first-pass review.</strong> Construction proposals
and controlled-delta metadata are deliberately hidden, and case order is
deterministically shuffled. The SVG is a geometry audit, not final Blender evidence.</p>
<div class="filters">
<button onclick="filterCards('all')">all</button>
{filter_buttons}
</div>
{''.join(cards)}
</main>
<script>
function filterCards(metric){{
 document.querySelectorAll('.card').forEach(c=>c.style.display=(metric==='all'||c.dataset.metric===metric)?'block':'none');
}}
</script></body></html>
"""


def _dataset_readme() -> str:
    return """# cal_dataset2_non_l1_evidence

This is the human-review draft for calibrating visual evidence across every
currently planned VLM-involved metric outside L1.

## Scope

- L2 fine-grained relation fidelity: `oor`, `oar`
- L2 high-level `functional_semantic_fidelity`: room/scene type, broad
  visual-functional intent, required areas, and prompt-specified local function
  are components of one family
- L3 semantic coherence: `scale_consistency`,
  category/role-only `object_pairing_consistency` after grouping
- L3 perceptual visual quality: `style_consistency`

L1 is excluded from scoring here, but proxy room-bound and asset-closure checks
are run to reduce physical confounds. L4 is deferred until downstream task
families are fixed.

## Composition

- 96 base cases: four scenarios per metric, each with three controlled variants
  (`valid`, `invalid`, `ambiguous`) stored only as proposed labels.
- 12 L3 prompt-authorization cases: one complete 2×2 design per L3 metric
  (`prompt authorization present/absent × scene deviation present/absent`).
- 108 cases and 108 metric events total.

All public case IDs are opaque. Every case contains exactly one metric event.

## Human review status

Nothing is official ground truth yet:

- `reference_annotation.validation_status = draft`
- `specification_contract.frozen = false`
- `metric_gt.semantic_label = pending_review`
- `metric_gt.proposed_semantic_label` is only a construction proposal
- `accuracy_eligible = false`

Start at [`review/index.html`](review/index.html), which is a blind first-pass
view with construction proposals, scenario roles, controlled deltas, and sibling
ordering hidden. `review/review_queue.tsv` uses the same shuffled order and has
blank columns for human decisions. Read
[`review/REVIEW_PROTOCOL.md`](review/REVIEW_PROTOCOL.md) before labeling. Do not
inspect `metric_gt.json`,
`construction_manifest.json`, `validation/prompt_claim_audit.tsv`,
`validation/counterfactual_audit.tsv`, or
`validation/construction_proposals.tsv` until that first pass is complete.
Record decisions in the review queue or a separate human approval artifact.
Do not edit/freeze the contracts merely because the builder validates.

## Important boundaries

- Prompt granularity and asset strategy are independent.
- OOR/OAR cases are marked `fine_grained` for current runtime compatibility.
- High-level L2 and canonical L3 judges remain structurally defined placeholders.
- `object_grouping_report.json` is an evidence partition, never semantic GT.
- Historical case-slice metric names are preserved for reproducibility.
  `canonical_metric` records current ownership: pairing orientation/function
  slices route to L2/OOR; only category coexistence stays L3 pairing.
- Scene validity and visual-evidence sufficiency are separate artifacts.
- Prompt-authorized deviations are exact, claim-linked, target-specific, and
  relation-specific; they do not exempt unrelated inconsistencies.
- `local_raw` and `local_contour` are ablation packet components, not standalone
  production strategies.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  scripts/validate_cal_dataset2_non_l1.py
```

The validator checks artifact contracts, prompt spans, target identities, asset
closure, grouping partitions, counterfactual invariants, lifecycle state,
authorization 2×2 completeness, file hashes, and public label leakage.
"""


def _review_protocol() -> str:
    return """# Blind human-review protocol

This is the first-pass construction audit. It does not freeze ground truth and
does not evaluate a final rendered evidence packet.

## Use only these files during the blind pass

1. `index.html`
2. `review_queue.tsv`

Do not inspect files under `fixtures/` or `validation/` until the first-pass
labels have been committed. Case order is a deterministic hash permutation so
counterfactual siblings are not shown as predictable valid/invalid/ambiguous
triplets.

## Fill these columns

- `human_semantic_label`: one of `valid`, `invalid`, or `ambiguous`.
- `prompt_compatible`: `yes` only if the prompt naturally expresses the tested
  metric claim and does not accidentally specify a different metric as its
  primary question.
- `target_mapping_correct`: `yes` only if the highlighted target IDs are the
  objects or architecture references needed to decide the tested claim.
- `needs_render_check`: `yes` when the canonical SVG cannot establish an
  asset-facing direction, material/style cue, attachment/contact, or perceptual
  scale boundary.
- `notes`: concise reason, especially for `ambiguous`, incompatible prompts,
  target errors, or required render checks.

## Semantic-label rule

- `valid`: the tested metric claim is clearly satisfied.
- `invalid`: the tested metric claim is clearly violated.
- `ambiguous`: reasonable reviewers can disagree about the scene semantics
  even with adequate visual access.

Do not use `ambiguous` merely because the SVG lacks perceptual evidence. Mark
`needs_render_check=yes` instead. Scene correctness and evidence sufficiency are
separate labels.

For L3, apply only an explicit, target-scoped prompt authorization. For example,
an intentionally oversized object or ornate contrast named by the prompt should
not be penalized by the corresponding L3 metric. The exemption never applies to
an unrelated object, relation, or inconsistency.

## Metric boundary

- `oor`: the tested object-to-object relation.
- `oar`: the tested object-to-architecture attachment/relation.
- `functional_semantic_fidelity`: one prompt-conditioned family. Global evidence
  checks room type, broad intent, and required areas. Group-local functionality
  is checked only when explicitly specified by the prompt.
- `scale_consistency`: perceptual scale coherence relative to relevant context.
- `object_pairing_consistency`: category/role compatibility of members delivered
  by the grouping algorithm. Position, angle, orientation, and functional
  arrangement are out of scope.
- `style_consistency`: perceptual style coherence, after exact prompt-authorized
  contrast is accounted for.

For functional-semantic and L3 judgements, `invalid` requires one or more
significant, explicitly identified, visible in-scope defects. Minor variation
or subjective preference is valid when evidence is sufficient. Missing evidence
is unresolved, never valid.

## After the blind pass

Only after responses are saved may the reviewer inspect:

- `validation/construction_proposals.tsv`
- `validation/counterfactual_audit.tsv`
- `validation/prompt_claim_audit.tsv`
- per-case `construction_manifest.json` and `metric_gt.json`

Disagreements then become adjudication items. Human approval is a separate step;
it is the only step allowed to freeze contracts or promote labels to official
ground truth.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "Support" / "datasets" / DATASET_ID,
    )
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    Builder(out_root).build()
    print(
        json.dumps(
            {
                "dataset": str(out_root),
                "case_count": CASE_COUNT,
                "status": "draft_pending_human_review",
                "review_index": str(out_root / "review/index.html"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
