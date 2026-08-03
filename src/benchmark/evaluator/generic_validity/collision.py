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


COLLISION_EVALUATOR_VERSION = "collision_p0b_v2"
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
    for name in (
        "tangent_plane_max_thickness_m",
        "tangent_contact_tolerance_m",
    ):
        value = float(config.get(name, DEFAULT_COLLISION_CONFIG[name]))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"collision.{name} must be a finite non-negative number"
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
