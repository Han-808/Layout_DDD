"""P0b object-object collision metric with OBB broad phase and mesh narrow phase."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    footprint_overlap_area,
    get_obb_corners,
    normalize_objects,
    z_interval_overlap,
)
from benchmark.evaluator.generic_validity.mesh_collision import evaluate_mesh_pair
from benchmark.evaluator.generic_validity.mesh_geometry import (
    TRIANGLE_MESH_REPRESENTATION,
    geometry_entry_for_object,
    geometry_unavailable_reason,
    is_usable_triangle_mesh,
    load_triangle_mesh,
)
from benchmark.evaluator.generic_validity.obb_sat import obb_sat_test
from benchmark.visual_judge.p0b import (
    COLLISION_CANDIDATE_SELECTION_POLICY,
    LocalViewProvider,
    adjudicate_p0b_event,
)
from benchmark.visual_judge.contracts import (
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


COLLISION_EVALUATOR_VERSION = "collision_p0b_v3"
DEFAULT_COLLISION_CONFIG = {
    "enabled": True,
    "official_mode": False,
    "detector_only": False,
    "obb_sat_eps": 1.0e-6,
    "mesh_enclosure_eps_m": 1.0e-4,
    "separation_threshold_m": 0.02,
    "zero_penetration_contact_policy": "route_vlm",
    "tangent_plane_contact_policy": "route_vlm",
    "tangent_plane_max_thickness_m": 0.002,
    "tangent_contact_tolerance_m": 0.001,
    # This is deliberately not a generic small-collision exemption. It only
    # surfaces a narrow support-interface regime to the semantic Judge: a
    # shallow overlap with a horizontal floor layer may represent compression
    # of a compliant covering that independent rigid meshes cannot encode.
    "shallow_surface_layer_policy": "route_vlm",
    "shallow_surface_layer_max_thickness_m": 0.03,
    "shallow_surface_layer_max_overlap_m": 0.0125,
    "shallow_surface_layer_floor_tolerance_m": 0.005,
    "shallow_surface_layer_max_thinness_ratio": 0.05,
    "shallow_surface_layer_min_horizontal_alignment": 0.98,
    # Narrow deterministic certificates for geometry artefacts at ordinary
    # support interfaces.  Neither is a generic small-collision tolerance:
    # the minimum-overlap axis must be gravity-aligned and the two objects must
    # form a bottom/top support interface.  Floor-covering relief additionally
    # requires explicit rug/carpet/mat semantics.
    "support_interface_contact_policy": "direct_valid",
    "support_interface_max_penetration_m": 0.005,
    "support_interface_min_gravity_alignment": 0.98,
    "compliant_floor_covering_policy": "direct_valid",
    "compliant_floor_covering_max_thickness_m": 0.15,
    "compliant_floor_covering_max_overlap_m": 0.15,
    "compliant_floor_covering_floor_tolerance_m": 0.01,
    "compliant_floor_covering_min_horizontal_alignment": 0.95,
    "score_mode": "invalid_pair_count_over_objects",
}


class CollisionEvaluationError(RuntimeError):
    """Raised when official collision evaluation cannot complete adjudication."""


def check_collision(
    scene: dict,
    config: dict | None = None,
    *,
    collision_geometry: dict | None = None,
    prompt: str | None = None,
    relationships: list[dict] | dict | None = None,
    render_evidence: list[str] | None = None,
    vlm_judge: object | None = None,
    local_view_provider: LocalViewProvider | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_COLLISION_CONFIG, **(config or {})}
    _validate_collision_config(cfg)
    objects, object_errors = normalize_objects(scene)
    num_objects = len(objects)
    geometry_base_dir = _geometry_base_dir(collision_geometry)
    if num_objects == 0:
        return _empty_collision_report(object_errors)

    pairs: list[dict[str, Any]] = []
    collision_object_ids: set[str] = set()
    requires_vlm_count = 0
    adjudication_failures: list[str] = []
    mesh_enclosure_cache: dict[int, dict[str, Any]] = {}

    for obj_a, obj_b in combinations(objects, 2):
        pair = _evaluate_pair(
            scene=scene,
            obj_a=obj_a,
            obj_b=obj_b,
            cfg=cfg,
            collision_geometry=collision_geometry,
            geometry_base_dir=geometry_base_dir,
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
            mesh_enclosure_cache=mesh_enclosure_cache,
        )
        pairs.append(pair)
        if pair.get("requires_vlm"):
            requires_vlm_count += 1
        if pair.get("adjudication_error"):
            adjudication_failures.append(str(pair["adjudication_error"]))
        if pair.get("final_verdict") == "invalid":
            collision_object_ids.update([obj_a.id, obj_b.id])

    invalid_count = sum(1 for pair in pairs if pair.get("final_verdict") == "invalid")
    resolved_pairs = [pair for pair in pairs if pair.get("final_verdict") in {"valid", "invalid"}]
    official_mode = bool(cfg.get("official_mode"))
    detector_only = bool(cfg.get("detector_only"))

    if adjudication_failures and official_mode:
        raise CollisionEvaluationError("; ".join(adjudication_failures))
    if requires_vlm_count and official_mode and vlm_judge is None:
        raise CollisionEvaluationError(
            "collision events require P0b VLM adjudication in official mode, but no judge is configured"
        )

    unresolved_vlm_count = sum(
        1
        for pair in pairs
        if pair.get("requires_vlm") and pair.get("final_verdict") not in {"valid", "invalid"}
    )

    if detector_only:
        score = None
        status = "detector_only"
    elif unresolved_vlm_count:
        score = None
        status = "requires_vlm"
    elif not resolved_pairs:
        score = 1.0
        status = "checked"
    else:
        collision_rate = min(float(invalid_count) / float(max(num_objects, 1)), 1.0)
        score = float(1.0 - collision_rate)
        status = "checked"

    obb_only = sum(1 for pair in pairs if pair.get("evidence_level") == "obb")
    mesh_level = sum(1 for pair in pairs if pair.get("evidence_level") == "mesh")
    return {
        "metric": "collision",
        "evaluator_version": COLLISION_EVALUATOR_VERSION,
        "status": status,
        "score": score,
        "official_mode": official_mode,
        "detector_only": detector_only,
        "score_mode": str(cfg["score_mode"]),
        "shallow_surface_layer_policy": {
            "policy": str(cfg["shallow_surface_layer_policy"]),
            "decision_authority": "vlm_judge",
            "automatic_exemption": False,
            "max_layer_thickness_m": float(
                cfg["shallow_surface_layer_max_thickness_m"]
            ),
            "max_overlap_m": float(
                cfg["shallow_surface_layer_max_overlap_m"]
            ),
            "floor_tolerance_m": float(
                cfg["shallow_surface_layer_floor_tolerance_m"]
            ),
            "max_thinness_ratio": float(
                cfg["shallow_surface_layer_max_thinness_ratio"]
            ),
            "min_horizontal_alignment": float(
                cfg[
                    "shallow_surface_layer_min_horizontal_alignment"
                ]
            ),
        },
        "support_interface_contact_policy": {
            "policy": str(cfg["support_interface_contact_policy"]),
            "max_penetration_m": float(
                cfg["support_interface_max_penetration_m"]
            ),
            "min_gravity_alignment": float(
                cfg["support_interface_min_gravity_alignment"]
            ),
            "automatic_exemption": True,
            "scope": "gravity_aligned_bottom_top_support_interface_only",
        },
        "compliant_floor_covering_policy": {
            "policy": str(cfg["compliant_floor_covering_policy"]),
            "automatic_exemption": True,
            "scope": "explicit_floor_covering_semantics_and_bounded_overlap_only",
            "max_layer_thickness_m": float(
                cfg["compliant_floor_covering_max_thickness_m"]
            ),
            "max_overlap_m": float(
                cfg["compliant_floor_covering_max_overlap_m"]
            ),
        },
        "collision_count": invalid_count,
        "collision_pair_count": invalid_count,
        "collision_object_count": len(collision_object_ids),
        "collision_rate": None if score is None else float(1.0 - score),
        "requires_vlm_count": requires_vlm_count,
        "resolved_pair_count": len(resolved_pairs),
        "num_objects": num_objects,
        "pairs": pairs,
        "object_errors": object_errors,
        "coverage": {
            "pair_count": len(pairs),
            "obb_evidence_pairs": obb_only,
            "mesh_evidence_pairs": mesh_level,
            "direct_valid_pairs": sum(1 for pair in pairs if str(pair.get("route") or "").startswith("direct_valid")),
            "vlm_adjudicated_pairs": sum(1 for pair in pairs if pair.get("route") == "vlm_adjudicated"),
        },
        "notes": [
            "Collision is static object-object surface interpenetration only; floor/wall/ceiling penetration belongs to OOB/OAR.",
            "Deterministic geometry is evidence, not the final semantic judge.",
            "Candidates are proposed by a high-recall detector; selection carries no verdict prior.",
            "Only final invalid pairs count as collisions; candidate overlap alone is not penalized.",
            "Exact coplanar zero-penetration contact may be accepted directly when the frozen profile enables it.",
            "A definitive gravity-aligned penetration of at most the frozen support-interface tolerance may be accepted only when it occurs at an upper object's base and a lower object's top.",
            "A thin floor-level rug, carpet, or mat may absorb a bounded vertical overlap without creating a Collision defect; category/description semantics and geometry must both agree.",
            "A bounded shallow overlap at a horizontal floor-layer support interface is surfaced to the VLM as "
            "context, never as an automatic exemption or a generic small-collision tolerance.",
            "Intended decorative attachment or assembly may be Collision-valid after VLM adjudication; relationship claims alone never auto-exempt a pair.",
            "The frozen score denominator is canonical object count.",
        ],
    }


def _evaluate_pair(
    *,
    scene: dict,
    obj_a,
    obj_b,
    cfg: dict[str, Any],
    collision_geometry: dict | None,
    geometry_base_dir: Path | None,
    prompt: str | None,
    relationships: list[dict] | dict | None,
    render_evidence: list[str] | None,
    vlm_judge: object | None,
    local_view_provider: LocalViewProvider | None,
    mesh_enclosure_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    obb = obb_sat_test(obj_a, obj_b, eps=float(cfg.get("obb_sat_eps", 1.0e-6)))
    diagnostics = {
        "xy_overlap_area": float(footprint_overlap_area(obj_a, obj_b)),
        "z_overlap": float(z_interval_overlap(obj_a, obj_b)),
    }
    entry_a = geometry_entry_for_object(collision_geometry, obj_a.id)
    entry_b = geometry_entry_for_object(collision_geometry, obj_b.id)
    mesh_a = is_usable_triangle_mesh(entry_a, base_dir=geometry_base_dir)
    mesh_b = is_usable_triangle_mesh(entry_b, base_dir=geometry_base_dir)
    geometry_provenance = {
        "object_a": _geometry_provenance(scene, obj_a.id, obj_a, entry_a),
        "object_b": _geometry_provenance(scene, obj_b.id, obj_b, entry_b),
    }

    pair = {
        "object_a": obj_a.id,
        "object_b": obj_b.id,
        "obb_evidence": obb,
        "diagnostics": diagnostics,
        "geometry_provenance": geometry_provenance,
        "mesh_enclosure_evidence": None,
        "mesh_evidence": None,
        "evidence_level": "obb",
        "route": None,
        "requires_vlm": False,
        "final_verdict": None,
        "affects_collision_score": False,
        "judge_result": None,
        "adjudication_error": None,
        "scoring_geometry": _collision_scoring_geometry(obj_a, obj_b, obb),
    }

    enclosure_evidence = None
    enclosure_safe = False
    # Any direct-valid route that relies on supplied complete mesh geometry must
    # first establish that the mesh is expressed in the canonical object frame.
    # This guard is needed for both OBB-separated and OBB-overlapping pairs: a
    # stale/translated mesh can otherwise manufacture a false separation in the
    # narrow phase even though the canonical objects overlap.
    if obb.get("obb_certifiably_separated") or (mesh_a and mesh_b):
        enclosure_evidence = {
            "object_a": _cached_mesh_enclosure_evidence(
                obj_a,
                entry_a,
                mesh_usable=mesh_a,
                base_dir=geometry_base_dir,
                eps=float(cfg.get("mesh_enclosure_eps_m", 1.0e-4)),
                cache=mesh_enclosure_cache,
            ),
            "object_b": _cached_mesh_enclosure_evidence(
                obj_b,
                entry_b,
                mesh_usable=mesh_b,
                base_dir=geometry_base_dir,
                eps=float(cfg.get("mesh_enclosure_eps_m", 1.0e-4)),
                cache=mesh_enclosure_cache,
            ),
        }
        pair["mesh_enclosure_evidence"] = enclosure_evidence
        enclosure_safe = all(
            bool(item.get("safe_for_obb_separation")) for item in enclosure_evidence.values()
        )

    if obb.get("obb_certifiably_separated"):
        if enclosure_safe:
            pair.update(
                {
                    "route": "direct_valid_obb_separated",
                    "final_verdict": "valid",
                    "affects_collision_score": True,
                }
            )
            return pair

    if _is_direct_valid_zero_penetration_contact(
        obb,
        enclosure_safe=enclosure_safe,
        config=cfg,
    ):
        pair.update(
            {
                "route": "direct_valid_zero_penetration_contact",
                "final_verdict": "valid",
                "affects_collision_score": True,
            }
        )
        return pair

    mesh_evidence = None
    if mesh_a and mesh_b:
        mesh_evidence = evaluate_mesh_pair(
            obj_a.id,
            obj_b.id,
            entry_a,
            entry_b,
            base_dir=geometry_base_dir,
            separation_threshold_m=float(cfg.get("separation_threshold_m", 0.02)),
        )
        pair["mesh_evidence"] = mesh_evidence
        pair["evidence_level"] = "mesh"
        if mesh_evidence.get("mesh_reliable_for_separation") and enclosure_safe:
            pair.update(
                {
                    "route": "direct_valid_mesh_separated",
                    "final_verdict": "valid",
                    "affects_collision_score": True,
                }
            )
            return pair
        tangent_certificate = _tangent_plane_contact_certificate(
            obj_a,
            obj_b,
            obb=obb,
            mesh_evidence=mesh_evidence,
            enclosure_safe=enclosure_safe,
            config=cfg,
        )
        if tangent_certificate is not None:
            pair["tangent_plane_contact_evidence"] = tangent_certificate
            pair.update(
                {
                    "route": "direct_valid_tangent_plane_contact",
                    "final_verdict": "valid",
                    "affects_collision_score": True,
                }
            )
            return pair
        support_interface_certificate = (
            _support_interface_contact_certificate(
                obj_a,
                obj_b,
                obb=obb,
                mesh_evidence=mesh_evidence,
                enclosure_safe=enclosure_safe,
                config=cfg,
            )
        )
        if support_interface_certificate is not None:
            pair["support_interface_contact_evidence"] = (
                support_interface_certificate
            )
            pair.update(
                {
                    "route": "direct_valid_support_interface_contact",
                    "final_verdict": "valid",
                    "affects_collision_score": True,
                }
            )
            return pair
        floor_covering_certificate = (
            _compliant_floor_covering_certificate(
                scene,
                obj_a,
                obj_b,
                obb=obb,
                mesh_evidence=mesh_evidence,
                config=cfg,
            )
        )
        if floor_covering_certificate is not None:
            pair["compliant_floor_covering_evidence"] = (
                floor_covering_certificate
            )
            pair.update(
                {
                    "route": "direct_valid_compliant_floor_covering",
                    "final_verdict": "valid",
                    "affects_collision_score": True,
                }
            )
            return pair
        shallow_layer_evidence = (
            _shallow_surface_layer_overlap_evidence(
                scene,
                obj_a,
                obj_b,
                obb=obb,
                mesh_evidence=mesh_evidence,
                config=cfg,
            )
        )
        if shallow_layer_evidence is not None:
            pair["shallow_surface_layer_overlap_evidence"] = (
                shallow_layer_evidence
            )

    pair["requires_vlm"] = True
    if bool(cfg.get("detector_only")):
        return pair

    if vlm_judge is None:
        return pair

    event = {
        "object_a": obj_a.id,
        "object_b": obj_b.id,
        "object_ids": [obj_a.id, obj_b.id],
        "evidence_level": pair["evidence_level"],
        "candidate_selection_policy": COLLISION_CANDIDATE_SELECTION_POLICY,
    }
    detector_evidence = {
        "candidate_selection_policy": COLLISION_CANDIDATE_SELECTION_POLICY,
        "obb": obb,
        "mesh": mesh_evidence,
        "diagnostics": diagnostics,
        "geometry_provenance": geometry_provenance,
        "mesh_enclosure": pair.get("mesh_enclosure_evidence"),
        "shallow_surface_layer_overlap": pair.get(
            "shallow_surface_layer_overlap_evidence"
        ),
        "support_interface_contact": pair.get(
            "support_interface_contact_evidence"
        ),
        "compliant_floor_covering": pair.get(
            "compliant_floor_covering_evidence"
        ),
        "closest_points": mesh_evidence.get("closest_points") if isinstance(mesh_evidence, dict) else None,
        "focus_region": mesh_evidence.get("focus_region") if isinstance(mesh_evidence, dict) else None,
        "extracted_relationships_are_claims_only": True,
    }
    try:
        judge_result = adjudicate_p0b_event(
            metric="collision",
            event=event,
            prompt=str(prompt or ""),
            relationships=relationships,
            scene=scene,
            detector_evidence=detector_evidence,
            judge=vlm_judge,
            object_ids=[obj_a.id, obj_b.id],
            overview_render_evidence=list(render_evidence or []),
            local_view_provider=local_view_provider,
        )
    except EvidenceControlUnresolvedError as exc:
        pair["route"] = "unresolved"
        pair["evidence_control"] = exc.result.to_dict()
        return pair
    except Exception as exc:
        pair["adjudication_error"] = f"{type(exc).__name__}: {exc}"
        pair["route"] = "vlm_adjudication_failed"
        schema_audit = response_schema_audit_from_exception(exc)
        if schema_audit is not None:
            pair["adjudication_failure_audit"] = schema_audit
        if bool(cfg.get("official_mode")):
            raise CollisionEvaluationError(pair["adjudication_error"]) from exc
        return pair

    verdict = str(judge_result.get("verdict"))
    pair.update(
        {
            "route": "vlm_adjudicated",
            "final_verdict": verdict,
            "affects_collision_score": True,
            "judge_result": deepcopy(judge_result),
        }
    )
    return pair


def _empty_collision_report(object_errors: dict[str, str]) -> dict[str, Any]:
    return {
        "metric": "collision",
        "evaluator_version": COLLISION_EVALUATOR_VERSION,
        "status": "not_applicable",
        "score": None,
        "reason": "no_physical_objects",
        "collision_count": 0,
        "collision_pair_count": 0,
        "collision_object_count": 0,
        "collision_rate": None,
        "requires_vlm_count": 0,
        "resolved_pair_count": 0,
        "num_objects": 0,
        "pairs": [],
        "object_errors": object_errors,
        "coverage": {
            "pair_count": 0,
            "obb_evidence_pairs": 0,
            "mesh_evidence_pairs": 0,
            "direct_valid_pairs": 0,
            "vlm_adjudicated_pairs": 0,
        },
        "notes": [],
    }


def _collision_scoring_geometry(
    obj_a: Any,
    obj_b: Any,
    obb: dict[str, Any],
) -> dict[str, Any]:
    """Persist the projected thicknesses needed for post-hoc severity."""

    axis = obb.get("minimum_overlap_axis")
    depth = obb.get("minimum_overlap_depth_proxy_m")
    if not isinstance(axis, list) or len(axis) != 3:
        return {
            "penetration_depth_m": depth,
            "minimum_overlap_axis": axis,
            "projected_thickness_a_m": None,
            "projected_thickness_b_m": None,
        }
    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1.0e-12:
        thickness_a = thickness_b = None
    else:
        vector /= norm
        thickness_a = 2.0 * sum(
            abs(float(np.dot(np.asarray(obj_a.R)[:, index], vector)))
            * float(np.asarray(obj_a.half)[index])
            for index in range(3)
        )
        thickness_b = 2.0 * sum(
            abs(float(np.dot(np.asarray(obj_b.R)[:, index], vector)))
            * float(np.asarray(obj_b.half)[index])
            for index in range(3)
        )
    return {
        "penetration_depth_m": depth,
        "minimum_overlap_axis": [float(value) for value in vector],
        "projected_thickness_a_m": thickness_a,
        "projected_thickness_b_m": thickness_b,
        "thin_object_threshold_m": 0.04,
    }


def _geometry_provenance(scene: dict, object_id: str, normalized: Any, entry: dict | None) -> Any:
    if isinstance(entry, dict):
        runtime = entry.get("geometry_source") or entry.get("representation")
        if runtime is not None:
            return runtime
    for item in scene.get("objects", []) if isinstance(scene, dict) else []:
        if isinstance(item, dict) and str(item.get("id")) == str(object_id):
            static = item.get("geometry_provenance")
            if static is not None:
                return static
    asset_ref = getattr(normalized, "asset_ref", None)
    return asset_ref.get("source_db") if isinstance(asset_ref, dict) else None


def _validate_collision_config(config: dict[str, Any]) -> None:
    if bool(config.get("official_mode")) and bool(config.get("detector_only")):
        raise ValueError("collision.official_mode and collision.detector_only are mutually exclusive")
    eps = float(config.get("obb_sat_eps", 1.0e-6))
    enclosure_eps = float(config.get("mesh_enclosure_eps_m", 1.0e-4))
    separation = float(config.get("separation_threshold_m", 0.02))
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("collision.obb_sat_eps must be a finite non-negative number")
    if not math.isfinite(enclosure_eps) or enclosure_eps < 0.0:
        raise ValueError("collision.mesh_enclosure_eps_m must be a finite non-negative number")
    if not math.isfinite(separation) or separation < 0.0:
        raise ValueError("collision.separation_threshold_m must be a finite non-negative number")
    if config.get("score_mode") != "invalid_pair_count_over_objects":
        raise ValueError("collision.score_mode must be 'invalid_pair_count_over_objects'")
    if config.get("zero_penetration_contact_policy") not in {
        "route_vlm",
        "direct_valid",
    }:
        raise ValueError(
            "collision.zero_penetration_contact_policy must be "
            "'route_vlm' or 'direct_valid'"
        )
    if config.get("tangent_plane_contact_policy") not in {
        "route_vlm",
        "direct_valid",
    }:
        raise ValueError(
            "collision.tangent_plane_contact_policy must be "
            "'route_vlm' or 'direct_valid'"
        )
    if config.get("shallow_surface_layer_policy") not in {
        "disabled",
        "route_vlm",
    }:
        raise ValueError(
            "collision.shallow_surface_layer_policy must be "
            "'disabled' or 'route_vlm'"
        )
    if config.get("support_interface_contact_policy") not in {
        "disabled",
        "direct_valid",
    }:
        raise ValueError(
            "collision.support_interface_contact_policy must be "
            "'disabled' or 'direct_valid'"
        )
    if config.get("compliant_floor_covering_policy") not in {
        "disabled",
        "direct_valid",
    }:
        raise ValueError(
            "collision.compliant_floor_covering_policy must be "
            "'disabled' or 'direct_valid'"
        )
    for name in (
        "tangent_plane_max_thickness_m",
        "tangent_contact_tolerance_m",
        "shallow_surface_layer_max_thickness_m",
        "shallow_surface_layer_max_overlap_m",
        "shallow_surface_layer_floor_tolerance_m",
        "shallow_surface_layer_max_thinness_ratio",
        "support_interface_max_penetration_m",
        "compliant_floor_covering_max_thickness_m",
        "compliant_floor_covering_max_overlap_m",
        "compliant_floor_covering_floor_tolerance_m",
    ):
        value = float(config.get(name, DEFAULT_COLLISION_CONFIG[name]))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"collision.{name} must be a finite non-negative number"
            )
    alignment = float(
        config.get(
            "shallow_surface_layer_min_horizontal_alignment",
            DEFAULT_COLLISION_CONFIG[
                "shallow_surface_layer_min_horizontal_alignment"
            ],
        )
    )
    if not math.isfinite(alignment) or not 0.0 <= alignment <= 1.0:
        raise ValueError(
            "collision.shallow_surface_layer_min_horizontal_alignment "
            "must be a finite number in [0, 1]"
        )
    support_alignment = float(
        config.get(
            "support_interface_min_gravity_alignment",
            DEFAULT_COLLISION_CONFIG[
                "support_interface_min_gravity_alignment"
            ],
        )
    )
    if (
        not math.isfinite(support_alignment)
        or not 0.0 <= support_alignment <= 1.0
    ):
        raise ValueError(
            "collision.support_interface_min_gravity_alignment must be "
            "a finite number in [0, 1]"
        )
    covering_alignment = float(
        config.get(
            "compliant_floor_covering_min_horizontal_alignment",
            DEFAULT_COLLISION_CONFIG[
                "compliant_floor_covering_min_horizontal_alignment"
            ],
        )
    )
    if (
        not math.isfinite(covering_alignment)
        or not 0.0 <= covering_alignment <= 1.0
    ):
        raise ValueError(
            "collision.compliant_floor_covering_min_horizontal_alignment "
            "must be a finite number in [0, 1]"
        )


def _is_direct_valid_zero_penetration_contact(
    obb: dict[str, Any],
    *,
    enclosure_safe: bool,
    config: dict[str, Any],
) -> bool:
    """Accept a tangent contact only when complete meshes stay inside the OBBs."""

    if config.get("zero_penetration_contact_policy") != "direct_valid":
        return False
    if not enclosure_safe or obb.get("obb_certifiably_separated"):
        return False
    depth = obb.get("minimum_overlap_depth_proxy_m")
    if depth is None:
        return False
    epsilon = float(config.get("obb_sat_eps", 1.0e-6))
    return -epsilon <= float(depth) <= epsilon


def _tangent_plane_contact_certificate(
    obj_a: Any,
    obj_b: Any,
    *,
    obb: dict[str, Any],
    mesh_evidence: dict[str, Any],
    enclosure_safe: bool,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Certify a thin plane touching, rather than slicing, another object.

    This is intentionally narrower than a generic small-penetration exemption.
    The other OBB must stay on one side of the thin plane's centre sheet, so a
    wall/decoration plane passing through the middle of an object still routes
    to the VLM.
    """

    if config.get("tangent_plane_contact_policy") != "direct_valid":
        return None
    if not enclosure_safe:
        return None
    if mesh_evidence.get("surface_intersection") is not True:
        return None
    intersection = mesh_evidence.get("intersection")
    if not isinstance(intersection, dict) or not bool(
        intersection.get("definitive")
    ):
        return None
    depth = obb.get("minimum_overlap_depth_proxy_m")
    if depth is None:
        return None
    tolerance = float(config["tangent_contact_tolerance_m"])
    if float(depth) > tolerance:
        return None
    maximum_thickness = float(config["tangent_plane_max_thickness_m"])
    for plane, other in ((obj_a, obj_b), (obj_b, obj_a)):
        half = np.asarray(plane.half, dtype=float)
        full_sizes = 2.0 * half
        thin_axes = np.flatnonzero(full_sizes <= maximum_thickness)
        if not len(thin_axes):
            continue
        axis = int(thin_axes[np.argmin(full_sizes[thin_axes])])
        normal = np.asarray(plane.R, dtype=float)[:, axis]
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        plane_center = float(np.dot(np.asarray(plane.center), normal))
        plane_half = float(half[axis])
        other_center = float(np.dot(np.asarray(other.center), normal))
        other_radius = float(
            np.sum(
                np.abs(np.asarray(other.R, dtype=float).T @ normal)
                * np.asarray(other.half, dtype=float)
            )
        )
        lower = other_center - other_radius
        upper = other_center + other_radius
        negative_side = upper <= plane_center + plane_half + tolerance
        positive_side = lower >= plane_center - plane_half - tolerance
        if not (negative_side or positive_side):
            continue
        return {
            "certificate": "thin_plane_tangent_same_side_v1",
            "plane_object_id": str(plane.id),
            "other_object_id": str(other.id),
            "thin_axis": axis,
            "plane_thickness_m": float(full_sizes[axis]),
            "other_projected_interval_m": [float(lower), float(upper)],
            "plane_projected_interval_m": [
                float(plane_center - plane_half),
                float(plane_center + plane_half),
            ],
            "side": "negative" if negative_side else "positive",
            "overlap_depth_proxy_m": float(depth),
            "contact_tolerance_m": tolerance,
        }
    return None


def _support_interface_contact_certificate(
    obj_a: Any,
    obj_b: Any,
    *,
    obb: dict[str, Any],
    mesh_evidence: dict[str, Any],
    enclosure_safe: bool,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Certify only a tiny vertical overlap at a bottom/top interface.

    Asset meshes frequently include millimetre-scale bevels or fitted contact
    surfaces.  Treating those as free-space interpenetration creates false
    positives such as a mug entering a counter by 0.05 mm.  This certificate
    is intentionally topology-specific: reliable meshes must intersect, the
    SAT minimum-overlap axis must align with gravity, and one object's bottom
    must meet the other object's top.  Sideways or deep overlaps still route
    to the semantic Judge.
    """

    if config.get("support_interface_contact_policy") != "direct_valid":
        return None
    if not enclosure_safe or mesh_evidence.get("surface_intersection") is not True:
        return None
    intersection = mesh_evidence.get("intersection")
    if not isinstance(intersection, dict) or not bool(
        intersection.get("definitive")
    ):
        return None
    minimum_axis = np.asarray(
        obb.get("minimum_overlap_axis") or [], dtype=float
    )
    if minimum_axis.shape != (3,) or not np.all(np.isfinite(minimum_axis)):
        return None
    axis_norm = float(np.linalg.norm(minimum_axis))
    if axis_norm <= 1.0e-12:
        return None
    gravity_alignment = abs(float(minimum_axis[2])) / axis_norm
    if gravity_alignment < float(
        config["support_interface_min_gravity_alignment"]
    ):
        return None
    depth = obb.get("minimum_overlap_depth_proxy_m")
    if depth is None:
        return None
    penetration = float(depth)
    maximum_penetration = float(
        config["support_interface_max_penetration_m"]
    )
    if penetration <= 0.0 or penetration > maximum_penetration:
        return None

    for lower, upper in ((obj_a, obj_b), (obj_b, obj_a)):
        lower_corners = np.asarray(get_obb_corners(lower), dtype=float)
        upper_corners = np.asarray(get_obb_corners(upper), dtype=float)
        lower_top = float(np.max(lower_corners[:, 2]))
        lower_bottom = float(np.min(lower_corners[:, 2]))
        upper_top = float(np.max(upper_corners[:, 2]))
        upper_bottom = float(np.min(upper_corners[:, 2]))
        interface_penetration = lower_top - upper_bottom
        if interface_penetration <= 0.0:
            continue
        if interface_penetration > maximum_penetration + 1.0e-9:
            continue
        # The putative load must sit predominantly above the support; this
        # rejects a thin vertical slice through another object's mid-volume.
        if upper_bottom < lower_bottom or upper_top <= lower_top:
            continue
        if float(upper.center[2]) <= float(lower.center[2]):
            continue
        xy_overlap = float(footprint_overlap_area(lower, upper))
        if xy_overlap <= 0.0:
            continue
        return {
            "schema_version": "support_interface_contact_certificate_v1",
            "certificate": "gravity_aligned_bottom_top_interface",
            "support_object_id": str(lower.id),
            "load_object_id": str(upper.id),
            "interface_penetration_m": interface_penetration,
            "sat_overlap_depth_proxy_m": penetration,
            "minimum_overlap_axis_gravity_alignment": gravity_alignment,
            "xy_overlap_area_m2": xy_overlap,
            "maximum_allowed_penetration_m": maximum_penetration,
        }
    return None


def _compliant_floor_covering_certificate(
    scene: dict[str, Any],
    obj_a: Any,
    obj_b: Any,
    *,
    obb: dict[str, Any],
    mesh_evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Certify bounded relief from an explicit floor-covering asset.

    Some catalog rugs use a folded-fabric proxy whose bounding box is thicker
    than a physical textile.  That may be a Scale/Placement concern, but an
    ordinary chair or table resting into that relief is not a rigid-object
    collision.  The exemption therefore remains semantic *and* geometric:
    explicit covering semantics, floor contact, horizontal orientation, and
    vertical-only overlap are all mandatory.
    """

    if config.get("compliant_floor_covering_policy") != "direct_valid":
        return None
    if mesh_evidence.get("surface_intersection") is not True:
        return None
    intersection = mesh_evidence.get("intersection")
    if not isinstance(intersection, dict) or not bool(
        intersection.get("definitive")
    ):
        return None
    minimum_alignment = float(
        config["compliant_floor_covering_min_horizontal_alignment"]
    )
    minimum_axis = np.asarray(
        obb.get("minimum_overlap_axis") or [], dtype=float
    )
    axis_norm = (
        float(np.linalg.norm(minimum_axis))
        if minimum_axis.shape == (3,)
        and np.all(np.isfinite(minimum_axis))
        else 0.0
    )
    minimum_axis_gravity_alignment = (
        abs(float(minimum_axis[2])) / axis_norm
        if axis_norm > 1.0e-12
        else None
    )
    floor_z = _scene_floor_z(scene)
    maximum_thickness = float(
        config["compliant_floor_covering_max_thickness_m"]
    )
    maximum_overlap = float(
        config["compliant_floor_covering_max_overlap_m"]
    )
    floor_tolerance = float(
        config["compliant_floor_covering_floor_tolerance_m"]
    )
    numerical_eps = max(
        float(config.get("obb_sat_eps", 1.0e-6)), 1.0e-9
    )

    for layer, other in ((obj_a, obj_b), (obj_b, obj_a)):
        if not _is_compliant_floor_covering(layer):
            continue
        corners = np.asarray(get_obb_corners(layer), dtype=float)
        thickness = float(np.max(corners[:, 2]) - np.min(corners[:, 2]))
        if thickness <= 0.0 or thickness > maximum_thickness:
            continue
        full_sizes = 2.0 * np.asarray(layer.half, dtype=float)
        thin_axis = int(np.argmin(full_sizes))
        layer_normal = np.asarray(layer.R, dtype=float)[:, thin_axis]
        layer_normal_alignment = abs(float(layer_normal[2])) / max(
            float(np.linalg.norm(layer_normal)), 1.0e-12
        )
        if layer_normal_alignment < minimum_alignment:
            continue
        if abs(float(layer.bottom_z) - floor_z) > floor_tolerance:
            continue
        overlap = float(z_interval_overlap(layer, other))
        if overlap <= 0.0 or overlap > maximum_overlap:
            continue
        if overlap > thickness + numerical_eps:
            continue
        if float(other.bottom_z) < floor_z - numerical_eps:
            continue
        if float(other.bottom_z) > float(layer.top_z) + numerical_eps:
            continue
        if float(other.top_z) <= float(layer.top_z) + numerical_eps:
            continue
        return {
            "schema_version": "compliant_floor_covering_certificate_v1",
            "certificate": "semantic_compliant_floor_covering_relief",
            "layer_object_id": str(layer.id),
            "other_object_id": str(other.id),
            "layer_semantics": _object_semantic_text(layer),
            "semantic_source": "category_or_description",
            "layer_thickness_m": thickness,
            "vertical_overlap_m": overlap,
            "floor_z_m": floor_z,
            "layer_bottom_z_m": float(layer.bottom_z),
            "layer_top_z_m": float(layer.top_z),
            "other_bottom_z_m": float(other.bottom_z),
            "minimum_overlap_axis_gravity_alignment": (
                minimum_axis_gravity_alignment
            ),
            "layer_normal_gravity_alignment": layer_normal_alignment,
            "maximum_considered_layer_thickness_m": maximum_thickness,
            "maximum_considered_overlap_m": maximum_overlap,
        }
    return None


def _is_compliant_floor_covering(obj: Any) -> bool:
    text = _object_semantic_text(obj)
    tokens = set(text.replace("-", " ").replace("_", " ").split())
    if tokens & {"rug", "carpet", "doormat"}:
        return True
    if "mat" in tokens:
        return True
    return any(
        phrase in text
        for phrase in (
            "floor covering",
            "floor rug",
            "area rug",
            "bath mat",
            "yoga mat",
        )
    )


def _object_semantic_text(obj: Any) -> str:
    return " ".join(
        str(value or "").strip().lower()
        for value in (
            getattr(obj, "category", None),
            getattr(obj, "retrieval_category", None),
            getattr(obj, "desc", None),
            getattr(obj, "short_desc", None),
        )
        if str(value or "").strip()
    )


def _shallow_surface_layer_overlap_evidence(
    scene: dict[str, Any],
    obj_a: Any,
    obj_b: Any,
    *,
    obb: dict[str, Any],
    mesh_evidence: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Describe a narrow, non-verdict floor-layer contact regime.

    Independent rigid meshes cannot express local compression. A load-bearing
    object whose base remains at the substrate may therefore intersect the
    finite thickness of a compliant floor covering. Geometry alone cannot
    establish compliance, so this helper never returns a verdict. It only
    records when the topology and magnitude are narrow enough for the VLM to
    consider that semantic explanation.
    """

    if config.get("shallow_surface_layer_policy") != "route_vlm":
        return None
    if mesh_evidence.get("surface_intersection") is not True:
        return None
    intersection = mesh_evidence.get("intersection")
    if not isinstance(intersection, dict) or not bool(
        intersection.get("definitive")
    ):
        return None

    maximum_overlap = float(
        config["shallow_surface_layer_max_overlap_m"]
    )
    overlap = float(z_interval_overlap(obj_a, obj_b))
    if overlap <= 0.0 or overlap > maximum_overlap:
        return None

    minimum_axis = np.asarray(
        obb.get("minimum_overlap_axis") or [],
        dtype=float,
    )
    if minimum_axis.shape != (3,) or not np.all(
        np.isfinite(minimum_axis)
    ):
        return None
    axis_norm = float(np.linalg.norm(minimum_axis))
    if axis_norm <= 1.0e-12:
        return None
    gravity_alignment = abs(float(minimum_axis[2]) / axis_norm)
    minimum_alignment = float(
        config["shallow_surface_layer_min_horizontal_alignment"]
    )
    if gravity_alignment < minimum_alignment:
        return None

    floor_z = _scene_floor_z(scene)
    floor_tolerance = float(
        config["shallow_surface_layer_floor_tolerance_m"]
    )
    maximum_thickness = float(
        config["shallow_surface_layer_max_thickness_m"]
    )
    maximum_ratio = float(
        config["shallow_surface_layer_max_thinness_ratio"]
    )
    numerical_eps = max(
        float(config.get("obb_sat_eps", 1.0e-6)),
        1.0e-9,
    )

    for layer, other in ((obj_a, obj_b), (obj_b, obj_a)):
        full_sizes = 2.0 * np.asarray(layer.half, dtype=float)
        thin_axis = int(np.argmin(full_sizes))
        thickness = float(full_sizes[thin_axis])
        if thickness <= 0.0 or thickness > maximum_thickness:
            continue
        planar_sizes = np.delete(full_sizes, thin_axis)
        minimum_planar_size = float(np.min(planar_sizes))
        if minimum_planar_size <= 0.0:
            continue
        thinness_ratio = thickness / minimum_planar_size
        if thinness_ratio > maximum_ratio:
            continue
        layer_normal = np.asarray(layer.R, dtype=float)[:, thin_axis]
        normal_alignment = abs(float(layer_normal[2])) / max(
            float(np.linalg.norm(layer_normal)),
            1.0e-12,
        )
        if normal_alignment < minimum_alignment:
            continue
        if abs(float(layer.bottom_z) - floor_z) > floor_tolerance:
            continue

        # The other object may enter the finite surface layer, but it may not
        # cross the substrate or slice through the layer from the side.
        substrate_crossing = max(
            floor_z - float(other.bottom_z),
            0.0,
        )
        if substrate_crossing > numerical_eps:
            continue
        if float(other.bottom_z) > float(layer.top_z) + numerical_eps:
            continue
        if float(other.top_z) <= float(layer.top_z) + numerical_eps:
            continue
        if overlap > thickness + numerical_eps:
            continue

        return {
            "schema_version": (
                "bounded_shallow_surface_layer_overlap_v1"
            ),
            "classification": (
                "shallow_support_interface_overlap_candidate"
            ),
            "layer_object_id": str(layer.id),
            "other_object_id": str(other.id),
            "layer_thickness_m": thickness,
            "vertical_overlap_m": overlap,
            "overlap_fraction_of_layer_thickness": overlap / thickness,
            "layer_thin_axis": thin_axis,
            "layer_thinness_ratio": thinness_ratio,
            "layer_normal_gravity_alignment": normal_alignment,
            "minimum_overlap_axis_gravity_alignment": (
                gravity_alignment
            ),
            "floor_z_m": floor_z,
            "layer_bottom_z_m": float(layer.bottom_z),
            "layer_top_z_m": float(layer.top_z),
            "other_bottom_z_m": float(other.bottom_z),
            "substrate_crossing_m": substrate_crossing,
            "maximum_considered_overlap_m": maximum_overlap,
            "semantic_question": (
                "Does the visual and category evidence support a compliant "
                "surface layer compressing at an ordinary load/support "
                "interface?"
            ),
            "decision_authority": "vlm_judge",
            "carries_validity_prior": False,
            "automatic_exemption": False,
        }
    return None


def _scene_floor_z(scene: dict[str, Any]) -> float:
    room = scene.get("room")
    if isinstance(room, dict):
        value = room.get("floor_z")
        try:
            floor_z = float(value)
        except (TypeError, ValueError):
            floor_z = 0.0
        if math.isfinite(floor_z):
            return floor_z
    return 0.0


def _geometry_base_dir(collision_geometry: dict | None) -> Path | None:
    if not isinstance(collision_geometry, dict):
        return None
    manifest_path = collision_geometry.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        return Path(manifest_path).expanduser().resolve().parent
    return None


def _cached_mesh_enclosure_evidence(
    obj: Any,
    entry: dict[str, Any] | None,
    *,
    mesh_usable: bool,
    base_dir: Path | None,
    eps: float,
    cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Guard an OBB separation certificate against wrong-frame mesh artifacts.

    With no complete triangle mesh, the canonical OBB is the available physical
    representation and remains a valid certificate.  When a complete mesh is
    supplied, however, every loaded world-space vertex must lie inside that OBB;
    otherwise the pair is ambiguous and continues to mesh/VLM adjudication.
    Results are cached once per object so the normal renderer-exported PLY path
    is not reloaded for every separated pair.
    """

    # Use object identity, not the submitted ID, for cache safety. Submission
    # validation rejects duplicate IDs, but the lower-level diagnostic API may
    # still receive them with different canonical transforms.
    cache_key = id(obj)
    if cache_key in cache:
        return cache[cache_key]
    claimed_complete_mesh = bool(
        isinstance(entry, dict)
        and entry.get("representation") == TRIANGLE_MESH_REPRESENTATION
        and entry.get("complete") is True
    )
    if not mesh_usable and claimed_complete_mesh:
        result = {
            "status": "mesh_enclosure_unavailable",
            "safe_for_obb_separation": False,
            "tolerance_m": float(eps),
            "error": geometry_unavailable_reason(entry),
        }
        cache[cache_key] = result
        return result
    if not mesh_usable:
        result = {
            "status": "not_applicable_canonical_obb_representation",
            "safe_for_obb_separation": True,
            "tolerance_m": float(eps),
        }
        cache[cache_key] = result
        return result
    try:
        mesh = load_triangle_mesh(entry or {}, base_dir=base_dir)
        vertices = np.asarray(mesh.get("vertices"), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
            raise ValueError("mesh contains no Nx3 vertices")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh contains non-finite vertices")
        local = (vertices - np.asarray(obj.center, dtype=float)) @ np.asarray(obj.R, dtype=float)
        excess = np.abs(local) - np.asarray(obj.half, dtype=float)
        outside_mask = np.any(excess > float(eps), axis=1)
        maximum_excess = max(0.0, float(np.max(excess)))
        enclosed = not bool(np.any(outside_mask))
        result = {
            "status": "verified_inside" if enclosed else "outside_canonical_obb",
            "safe_for_obb_separation": enclosed,
            "vertex_count": int(len(vertices)),
            "outside_vertex_count": int(np.count_nonzero(outside_mask)),
            "maximum_excess_m": maximum_excess,
            "tolerance_m": float(eps),
            "loader": mesh.get("loader"),
        }
    except Exception as exc:
        result = {
            "status": "mesh_enclosure_unavailable",
            "safe_for_obb_separation": False,
            "tolerance_m": float(eps),
            "error": f"{type(exc).__name__}: {exc}",
        }
    cache[cache_key] = result
    return result
