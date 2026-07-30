"""P0b support-gap metric: downward support probe with conservative VLM adjudication.

Support evaluates *positive* clearance along gravity: a generated object that should
rest on a supporting surface but instead has an unexplained vertical gap (floating),
including a gap inherited through an otherwise touching support stack.
Volume penetration and sinking are owned by Collision/OOB and are never turned into a
support penalty here. Static stability, centre-of-mass, affordances, and physics
simulation are out of scope. Standard gravity is ``[0, 0, -1]`` in the canonical
meter/Z-up frame.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    get_obb_corners,
    get_room_boundary,
    get_scene_height,
    normalize_objects,
    point_in_polygon_2d,
    ray_intersects_obb,
    sample_bottom_face_points,
)
from benchmark.evaluator.generic_validity.mesh_geometry import (
    geometry_entry_for_object,
    geometry_unavailable_reason,
    is_usable_triangle_mesh,
    load_triangle_mesh,
)
from benchmark.visual_judge.p0b import LocalViewProvider, adjudicate_p0b_event
from benchmark.visual_judge.runtime import EvidenceControlUnresolvedError


SUPPORT_EVALUATOR_VERSION = "support_p0b_v7"
GRAVITY = [0.0, 0.0, -1.0]
SUPPORT_CANDIDATE_SELECTION_POLICY = "high_recall_candidate_no_label_prior"
GROUNDED_SUPPORT_POLICY = "fixed_point_tolerance_contact_path_to_floor_v2"

# Scale-aware contact thresholds (meters). Small positive clearances within the
# hard band are treated as ordinary geometry/render fitting tolerance, matching
# the earlier Support policy. The cap prevents a legacy configuration override
# from turning an arbitrarily large gap into a direct-valid certificate.
# near_min (0.04) > hard_max (0.035) keeps the default contact/borderline/strong
# bands well ordered.
HARD_CONTACT_TOLERANCE_BASE_M = 0.02
HARD_CONTACT_TOLERANCE_PER_SIZE_Z = 0.005
HARD_CONTACT_TOLERANCE_MIN_M = 0.02
HARD_CONTACT_TOLERANCE_MAX_M = 0.035
DIRECT_CONTACT_TOLERANCE_CAP_M = HARD_CONTACT_TOLERANCE_MAX_M
NEAR_SUPPORT_TOLERANCE_BASE_M = 0.03
NEAR_SUPPORT_TOLERANCE_PER_SIZE_Z = 0.03
NEAR_SUPPORT_TOLERANCE_MIN_M = 0.04
NEAR_SUPPORT_TOLERANCE_MAX_M = 0.08

GAP_BAND_CONTACT = "contact"
GAP_BAND_BORDERLINE = "borderline_positive_clearance"
GAP_BAND_STRONG = "strong_positive_clearance"
GAP_BAND_UNKNOWN = "unknown_clearance"

DEFAULT_SUPPORT_CONFIG = {
    "enabled": True,
    "official_mode": False,
    "detector_only": False,
    # ``contact_tolerance_m`` is intentionally absent. If supplied for legacy
    # compatibility, it pins the contact/candidate band; deterministic contact
    # remains capped at DIRECT_CONTACT_TOLERANCE_CAP_M.
    "base_band_tolerance_m": 0.02,
    "minimum_contact_count": 1,
    "bottom_sample_grid": [4, 4],
    "max_representative_samples": 8,
    "mesh_bounds_tolerance_m": 0.03,
    "mesh_center_tolerance_m": 0.05,
}

SUPPORT_VLM_INSTRUCTION = (
    "Judge only whether the generated object or its supporting assembly has an unexplained "
    "positive support gap along gravity, or is instead physically supported, attached, suspended, or intentionally "
    "allowed to float. The object was proposed by a high-recall detector; being routed here "
    "carries no invalid prior. Invalid means only an unexplained positive support gap. A low "
    "contact fraction, an empty center ray, sparse contacts from legs/feet/frames, and support "
    "split across multiple targets are not invalid. A negative gap, sinking, object-object "
    "penetration, and out-of-bounds crossing are never Support-invalid; they belong to "
    "Collision/OOB, and an object may be Collision-invalid while remaining Support-valid. "
    "Collision and Support may both be invalid only when there is independent positive-gap "
    "evidence. Do not penalize using a different support target than the natural-language "
    "request; prompt-target compliance belongs to OOR/OAR. Prompt relationships may explain "
    "attachment or suspension but are claims, not proof. Local contact with an object whose "
    "grounded ancestry is unproven is not by itself proof of valid support; inspect the entire "
    "visible support chain. Use the prompt, relationships, "
    "architecture clearances, object semantics, and renders to decide. Treat exact detector "
    "distances as evidence and return exactly one binary verdict."
)

SUPPORT_NOTES = [
    "Support is a downward support-gap probe: unexplained positive clearance along gravity only.",
    "Sinking, negative gap, and volume penetration belong to Collision/OOB and never reduce the support score.",
    "Collision, OOB, and Support are independent multi-label metrics; a Collision-invalid object is not automatically Support-invalid.",
    "Direct contact uses a scale-aware 0.02-0.035 m tolerance (hard = clamp(0.02 + 0.005*size_z, 0.02, 0.035)); larger positive gaps route to VLM.",
    "A legacy fixed contact_tolerance_m override may narrow the band but cannot widen deterministic contact beyond 0.035 m.",
    "Positive gaps are banded against a scale-aware near tolerance (near = clamp(0.03 + 0.03*size_z, 0.04, 0.08)).",
    "Contact samples come only from the lowest base-contact band, not the full footprint or body envelope.",
    "Any reliable floor contact, or object contact with certified grounded ancestry, establishes non-floating support; contact fraction and the center ray are diagnostic only.",
    "Object contact bypasses VLM only when a fixed-point tolerance-contact graph proves a path to the floor.",
    "Floating stacks, ungrounded contact components, and contact cycles route to VLM; they are never direct-invalid.",
    "Sparse leg/frame contacts and support split across multiple targets are valid Support evidence; stability is evaluated elsewhere.",
    "Missing contact, attachment/suspension, or degraded geometry requires a binary VLM verdict; "
    "Support has no deterministic direct-invalid route.",
    "Mesh evidence must be consistent with the canonical object frame or it is rejected and explicitly routed as degraded evidence.",
    "No static stability, center-of-mass, affordance, or physics simulation is used.",
]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hard_contact_tolerance(size_z_m: float) -> float:
    return _clamp(
        HARD_CONTACT_TOLERANCE_BASE_M + HARD_CONTACT_TOLERANCE_PER_SIZE_Z * float(size_z_m),
        HARD_CONTACT_TOLERANCE_MIN_M,
        HARD_CONTACT_TOLERANCE_MAX_M,
    )


def _near_support_tolerance(size_z_m: float) -> float:
    return _clamp(
        NEAR_SUPPORT_TOLERANCE_BASE_M + NEAR_SUPPORT_TOLERANCE_PER_SIZE_Z * float(size_z_m),
        NEAR_SUPPORT_TOLERANCE_MIN_M,
        NEAR_SUPPORT_TOLERANCE_MAX_M,
    )


def _resolve_object_thresholds(
    size_z_m: float,
    fixed_contact_tolerance_m: float | None,
) -> tuple[float, float]:
    """Return the configured/default ``(hard, near)`` support thresholds.

    The hard threshold discovers candidate surfaces and, subject to the
    deterministic cap, certifies ordinary contact tolerance. Scale-aware mode
    derives both values from ``size_z``. A legacy ``contact_tolerance_m`` pins
    the hard band; the near threshold is raised so ``near >= hard``.
    """

    near = _near_support_tolerance(size_z_m)
    if fixed_contact_tolerance_m is not None:
        hard = float(fixed_contact_tolerance_m)
        near = max(near, hard)
    else:
        hard = _hard_contact_tolerance(size_z_m)
    return hard, near


def _direct_contact_tolerance(hard_contact_tolerance_m: float) -> float:
    """Bound deterministic contact while retaining legacy candidate overrides."""

    return _clamp(float(hard_contact_tolerance_m), 0.0, DIRECT_CONTACT_TOLERANCE_CAP_M)


def _classify_gap_band(
    *,
    contact_hit_count: int,
    minimum_contact_count: int,
    minimum_positive_clearance_m: float | None,
    near_support_tolerance_m: float,
) -> str:
    if contact_hit_count >= minimum_contact_count:
        return GAP_BAND_CONTACT
    if minimum_positive_clearance_m is None:
        return GAP_BAND_UNKNOWN
    if float(minimum_positive_clearance_m) <= float(near_support_tolerance_m) + 1.0e-9:
        return GAP_BAND_BORDERLINE
    return GAP_BAND_STRONG


class SupportEvaluationError(RuntimeError):
    """Raised when official support evaluation cannot complete required adjudication."""


def check_support(
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
    cfg = {**DEFAULT_SUPPORT_CONFIG, **(config or {})}
    _validate_support_config(cfg)
    if not bool(cfg.get("enabled", True)):
        return disabled_support_report()

    objects, object_errors = normalize_objects(scene)
    num_objects = len(objects)
    if num_objects == 0:
        return _empty_support_report(object_errors)

    fixed_contact_tolerance = (
        float(cfg["contact_tolerance_m"]) if cfg.get("contact_tolerance_m") is not None else None
    )
    threshold_mode = (
        "fixed_tolerance_contact"
        if fixed_contact_tolerance is not None
        else "scale_aware_tolerance_contact"
    )
    base_band_tol = float(cfg.get("base_band_tolerance_m", 0.02))
    minimum_contact_count = int(cfg.get("minimum_contact_count", 1))
    legacy_contact_fraction_threshold = (
        float(cfg["contact_fraction_threshold"])
        if cfg.get("contact_fraction_threshold") is not None
        else None
    )
    grid = _grid(cfg.get("bottom_sample_grid", [4, 4]))
    max_rep = int(cfg.get("max_representative_samples", 8))
    mesh_bounds_tolerance = float(cfg.get("mesh_bounds_tolerance_m", 0.03))
    mesh_center_tolerance = float(cfg.get("mesh_center_tolerance_m", 0.05))
    official_mode = bool(cfg.get("official_mode"))
    detector_only = bool(cfg.get("detector_only"))

    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    has_boundary = boundary.ndim == 2 and len(boundary) >= 3

    base_dir = _geometry_base_dir(collision_geometry)
    mesh_cache = _build_mesh_cache(
        objects,
        collision_geometry,
        base_dir,
        bounds_tolerance_m=mesh_bounds_tolerance,
        center_tolerance_m=mesh_center_tolerance,
    )
    grounding_analysis = _analyze_grounded_contact_graph(
        objects=objects,
        mesh_cache=mesh_cache,
        boundary=boundary,
        has_boundary=has_boundary,
        fixed_contact_tolerance=fixed_contact_tolerance,
        base_band_tol=base_band_tol,
        minimum_contact_count=minimum_contact_count,
        grid=grid,
    )

    records: list[dict[str, Any]] = []
    requires_vlm_count = 0
    adjudication_failures: list[str] = []

    for obj in objects:
        record = _evaluate_object(
            scene=scene,
            obj=obj,
            objects=objects,
            mesh_cache=mesh_cache,
            grounding_analysis=grounding_analysis,
            boundary=boundary,
            has_boundary=has_boundary,
            fixed_contact_tolerance=fixed_contact_tolerance,
            base_band_tol=base_band_tol,
            minimum_contact_count=minimum_contact_count,
            legacy_contact_fraction_threshold=legacy_contact_fraction_threshold,
            grid=grid,
            max_rep=max_rep,
            cfg=cfg,
            prompt=prompt,
            relationships=relationships,
            render_evidence=render_evidence,
            vlm_judge=vlm_judge,
            local_view_provider=local_view_provider,
        )
        records.append(record)
        if record.get("requires_vlm"):
            requires_vlm_count += 1
        if record.get("adjudication_error"):
            adjudication_failures.append(str(record["adjudication_error"]))

    valid_count = sum(1 for record in records if record.get("final_verdict") == "valid")
    invalid_count = sum(1 for record in records if record.get("final_verdict") == "invalid")
    resolved = [record for record in records if record.get("final_verdict") in {"valid", "invalid"}]
    unresolved_vlm_count = sum(
        1
        for record in records
        if record.get("requires_vlm") and record.get("final_verdict") not in {"valid", "invalid"}
    )

    if adjudication_failures and official_mode:
        raise SupportEvaluationError("; ".join(adjudication_failures))
    if requires_vlm_count and official_mode and vlm_judge is None:
        raise SupportEvaluationError(
            "support events require P0b VLM adjudication in official mode, but no judge is configured"
        )

    if detector_only:
        score = None
        status = "detector_only"
    elif unresolved_vlm_count:
        score = None
        status = "requires_vlm"
    else:
        score = float(valid_count) / float(max(num_objects, 1))
        status = "checked"

    direct_valid = sum(1 for record in records if str(record.get("route") or "").startswith("direct_valid"))
    vlm_adjudicated = sum(1 for record in records if record.get("route") == "vlm_adjudicated")
    return {
        "metric": "support",
        "evaluator_version": SUPPORT_EVALUATOR_VERSION,
        "status": status,
        "score": score,
        "enabled": True,
        "official_mode": official_mode,
        "detector_only": detector_only,
        "gravity": list(GRAVITY),
        "candidate_selection_policy": SUPPORT_CANDIDATE_SELECTION_POLICY,
        "grounded_support_policy": GROUNDED_SUPPORT_POLICY,
        "certified_grounded_object_ids": list(grounding_analysis["certified_grounded_object_ids"]),
        "unresolved_grounding_object_ids": list(grounding_analysis["unresolved_grounding_object_ids"]),
        "support_contact_graph_edges": deepcopy(grounding_analysis["contact_graph_edges"]),
        "threshold_mode": threshold_mode,
        "direct_contact_tolerance_bounds_m": [
            HARD_CONTACT_TOLERANCE_MIN_M,
            HARD_CONTACT_TOLERANCE_MAX_M,
        ],
        "direct_contact_tolerance_cap_m": DIRECT_CONTACT_TOLERANCE_CAP_M,
        "fixed_contact_tolerance_m": fixed_contact_tolerance,
        "legacy_contact_tolerance_affects_direct_valid": fixed_contact_tolerance is not None,
        "hard_contact_tolerance_bounds_m": [HARD_CONTACT_TOLERANCE_MIN_M, HARD_CONTACT_TOLERANCE_MAX_M],
        "near_support_tolerance_bounds_m": [NEAR_SUPPORT_TOLERANCE_MIN_M, NEAR_SUPPORT_TOLERANCE_MAX_M],
        "hard_contact_tolerance_formula": {
            "base_m": HARD_CONTACT_TOLERANCE_BASE_M,
            "per_size_z": HARD_CONTACT_TOLERANCE_PER_SIZE_Z,
            "min_m": HARD_CONTACT_TOLERANCE_MIN_M,
            "max_m": HARD_CONTACT_TOLERANCE_MAX_M,
        },
        "near_support_tolerance_formula": {
            "base_m": NEAR_SUPPORT_TOLERANCE_BASE_M,
            "per_size_z": NEAR_SUPPORT_TOLERANCE_PER_SIZE_Z,
            "min_m": NEAR_SUPPORT_TOLERANCE_MIN_M,
            "max_m": NEAR_SUPPORT_TOLERANCE_MAX_M,
        },
        "base_band_tolerance_m": base_band_tol,
        "minimum_contact_count": minimum_contact_count,
        "contact_fraction_threshold": legacy_contact_fraction_threshold,
        "contact_fraction_affects_route": False,
        "bottom_sample_grid": list(grid),
        "mesh_bounds_tolerance_m": mesh_bounds_tolerance,
        "mesh_center_tolerance_m": mesh_center_tolerance,
        "num_objects": num_objects,
        "evaluated_object_count": num_objects,
        "valid_support_object_count": valid_count,
        "supported_object_count": valid_count,
        "unsupported_object_count": invalid_count,
        "requires_vlm_count": requires_vlm_count,
        "resolved_object_count": len(resolved),
        "objects": records,
        "object_errors": object_errors,
        "coverage": {
            "object_count": num_objects,
            "direct_valid_objects": direct_valid,
            "vlm_adjudicated_objects": vlm_adjudicated,
            "mesh_evidence_objects": sum(1 for record in records if record.get("evidence_level") == "mesh"),
            "mixed_evidence_objects": sum(1 for record in records if record.get("evidence_level") == "mixed"),
            "obb_evidence_objects": sum(1 for record in records if record.get("evidence_level") == "obb"),
        },
        "notes": list(SUPPORT_NOTES),
    }


def _analyze_grounded_contact_graph(
    *,
    objects,
    mesh_cache: dict[str, dict[str, Any]],
    boundary: np.ndarray,
    has_boundary: bool,
    fixed_contact_tolerance: float | None,
    base_band_tol: float,
    minimum_contact_count: int,
    grid: tuple[int, int],
) -> dict[str, Any]:
    """Certify tolerance-contact paths to the floor with an order-independent pass.

    Local object contact is insufficient for a one-sided direct-valid decision:
    the contacted object may itself float, or several penetrating objects may
    form an ungrounded cycle. This fixed-point graph seeds reliable floor
    contacts within the scale-aware hard tolerance, then propagates through
    equally bounded object contacts. Wall and ceiling attachment never seed the graph because their
    semantic validity remains VLM-owned.
    """

    snapshots: dict[str, dict[str, Any]] = {}
    for obj in objects:
        source_cache = mesh_cache.get(obj.id) or {}
        size_z = float(np.asarray(obj.size, dtype=float)[2])
        candidate_tol, _ = _resolve_object_thresholds(size_z, fixed_contact_tolerance)
        direct_contact_tolerance = _direct_contact_tolerance(candidate_tol)
        lower_envelope_points, _, _, _ = _bottom_samples(obj, source_cache, grid)
        sample_points, _ = _base_contact_band(lower_envelope_points, base_band_tol)
        hits = [
            _nearest_support(
                point,
                objects,
                mesh_cache,
                boundary,
                has_boundary,
                candidate_tol,
                direct_contact_tolerance,
                obj.id,
            )
            for point in sample_points
        ]
        contact_hits = [hit for hit in hits if hit["contact"]]
        degraded_reasons = _geometry_degraded_reasons(
            source_id=obj.id,
            source_cache=source_cache,
            contact_hits=contact_hits,
        )
        contact_target_ids = sorted(
            {
                str(hit["target"])
                for hit in contact_hits
                if hit.get("target") not in {None, "floor"}
            }
        )
        snapshots[obj.id] = {
            "local_tolerance_contact_count": len(contact_hits),
            "direct_contact_tolerance_m": direct_contact_tolerance,
            "floor_contact_hit_count": sum(hit.get("target") == "floor" for hit in contact_hits),
            "contact_target_ids": contact_target_ids,
            "geometry_degraded_reasons": degraded_reasons,
            "locally_certifiable_contact": (
                len(contact_hits) >= minimum_contact_count and not degraded_reasons
            ),
        }

    grounded_paths: dict[str, list[str]] = {}
    for object_id in sorted(snapshots):
        snapshot = snapshots[object_id]
        if snapshot["locally_certifiable_contact"] and snapshot["floor_contact_hit_count"] > 0:
            grounded_paths[object_id] = [object_id, "floor"]

    changed = True
    while changed:
        changed = False
        for object_id in sorted(snapshots):
            if object_id in grounded_paths:
                continue
            snapshot = snapshots[object_id]
            if not snapshot["locally_certifiable_contact"]:
                continue
            grounded_targets = sorted(
                target_id
                for target_id in snapshot["contact_target_ids"]
                if target_id in grounded_paths
            )
            if not grounded_targets:
                continue
            target_id = grounded_targets[0]
            grounded_paths[object_id] = [object_id, *grounded_paths[target_id]]
            changed = True

    contact_edges = {
        object_id: list(snapshot["contact_target_ids"])
        for object_id, snapshot in snapshots.items()
    }
    object_results: dict[str, dict[str, Any]] = {}
    for object_id in sorted(snapshots):
        snapshot = snapshots[object_id]
        reachable = _reachable_contact_object_ids(object_id, contact_edges)
        cycle_reachable = _contact_cycle_reachable(object_id, contact_edges)
        certified = object_id in grounded_paths
        if certified:
            grounding_status = "certified_tolerance_contact_path_to_floor"
        elif snapshot["geometry_degraded_reasons"]:
            grounding_status = "degraded_geometry_requires_vlm"
        elif snapshot["local_tolerance_contact_count"] < minimum_contact_count:
            grounding_status = "no_reliable_tolerance_contact"
        elif snapshot["contact_target_ids"]:
            grounding_status = "local_object_contact_without_ground_path"
        else:
            grounding_status = "tolerance_contact_without_certified_architecture_path"
        object_results[object_id] = {
            **snapshot,
            "certified_grounded_support": certified,
            "grounding_status": grounding_status,
            "grounded_support_path": list(grounded_paths.get(object_id) or []),
            "reachable_contact_object_ids": reachable,
            "ungrounded_contact_cycle_reachable": bool(cycle_reachable and not certified),
        }

    return {
        "policy": GROUNDED_SUPPORT_POLICY,
        "certified_grounded_object_ids": sorted(grounded_paths),
        "unresolved_grounding_object_ids": sorted(set(snapshots) - set(grounded_paths)),
        "contact_graph_edges": [
            {"source_object_id": source_id, "target_object_id": target_id}
            for source_id in sorted(contact_edges)
            for target_id in contact_edges[source_id]
        ],
        "objects": object_results,
    }


def _reachable_contact_object_ids(
    source_id: str,
    contact_edges: dict[str, list[str]],
) -> list[str]:
    pending = list(contact_edges.get(source_id) or [])
    reachable: set[str] = set()
    while pending:
        target_id = pending.pop()
        if target_id == source_id or target_id in reachable:
            continue
        reachable.add(target_id)
        pending.extend(contact_edges.get(target_id) or [])
    return sorted(reachable)


def _contact_cycle_reachable(
    source_id: str,
    contact_edges: dict[str, list[str]],
) -> bool:
    def visit(node: str, active: set[str], finished: set[str]) -> bool:
        if node in active:
            return True
        if node in finished:
            return False
        active.add(node)
        for target_id in contact_edges.get(node) or []:
            if visit(target_id, active, finished):
                return True
        active.remove(node)
        finished.add(node)
        return False

    return visit(source_id, set(), set())


def _evaluate_object(
    *,
    scene: dict,
    obj,
    objects,
    mesh_cache: dict[str, dict[str, Any]],
    grounding_analysis: dict[str, Any],
    boundary: np.ndarray,
    has_boundary: bool,
    fixed_contact_tolerance: float | None,
    base_band_tol: float,
    minimum_contact_count: int,
    legacy_contact_fraction_threshold: float | None,
    grid: tuple[int, int],
    max_rep: int,
    cfg: dict[str, Any],
    prompt: str | None,
    relationships: list[dict] | dict | None,
    render_evidence: list[str] | None,
    vlm_judge: object | None,
    local_view_provider: LocalViewProvider | None,
) -> dict[str, Any]:
    source_cache = mesh_cache.get(obj.id) or {}
    grounding = deepcopy((grounding_analysis.get("objects") or {}).get(obj.id) or {})
    size_z = float(np.asarray(obj.size, dtype=float)[2])
    hard_contact_tolerance, near_support_tolerance = _resolve_object_thresholds(
        size_z,
        fixed_contact_tolerance,
    )
    candidate_tol = hard_contact_tolerance
    direct_contact_tolerance = _direct_contact_tolerance(hard_contact_tolerance)
    lower_envelope_points, source_representation, source_sample_method, source_center_point = _bottom_samples(
        obj,
        source_cache,
        grid,
    )
    sample_points, base_min_z = _base_contact_band(lower_envelope_points, base_band_tol)
    hits = [
        _nearest_support(
            point,
            objects,
            mesh_cache,
            boundary,
            has_boundary,
            candidate_tol,
            direct_contact_tolerance,
            obj.id,
        )
        for point in sample_points
    ]
    sample_count = len(hits)
    contact_hits = [hit for hit in hits if hit["contact"]]
    contact_hit_count = len(contact_hits)
    contact_fraction = float(contact_hit_count) / float(max(sample_count, 1))
    unsupported_sample_count = max(0, sample_count - contact_hit_count)

    if source_center_point is None:
        center_hit = {
            "target": None,
            "target_representation": None,
            "height_m": None,
            "gap_m": None,
            "position": [float(obj.center[0]), float(obj.center[1]), float(obj.bottom_z)],
            "evidence_source": "source_center_column_empty",
            "target_geometry_degraded_reason": None,
            "contact": False,
        }
    else:
        center_hit = _nearest_support(
            source_center_point,
            objects,
            mesh_cache,
            boundary,
            has_boundary,
            candidate_tol,
            direct_contact_tolerance,
            obj.id,
        )
    center_source_available = source_center_point is not None
    center_ray_supported = bool(center_hit["contact"])

    gaps = [float(hit["gap_m"]) for hit in hits if hit["gap_m"] is not None]
    contact_gaps = [float(hit["gap_m"]) for hit in contact_hits if hit["gap_m"] is not None]
    # A positive support gap is a floating (non-contact) sample with a measured
    # nearest surface. Contact samples (including deep floor penetration, which is
    # a Collision/OOB concern) and samples with no surface below are excluded.
    positive_clearances = [
        float(hit["gap_m"])
        for hit in hits
        if hit["gap_m"] is not None and not hit["contact"]
    ]
    minimum_positive_clearance_m = min(positive_clearances) if positive_clearances else None
    normalized_minimum_positive_clearance = (
        float(minimum_positive_clearance_m) / max(size_z, 1.0e-6)
        if minimum_positive_clearance_m is not None
        else None
    )
    gap_band = _classify_gap_band(
        contact_hit_count=contact_hit_count,
        minimum_contact_count=minimum_contact_count,
        minimum_positive_clearance_m=minimum_positive_clearance_m,
        near_support_tolerance_m=near_support_tolerance,
    )
    contact_targets = sorted({str(hit["target"]) for hit in contact_hits if hit["target"] is not None})
    per_target_hit_counts: dict[str, int] = {}
    for hit in contact_hits:
        key = str(hit["target"]) if hit["target"] is not None else "none"
        per_target_hit_counts[key] = per_target_hit_counts.get(key, 0) + 1
    candidate_support_ids = sorted(
        {
            str(hit["target"])
            for hit in hits
            if hit["target"] is not None and str(hit["target"]) != "floor"
        }
        | {
            str(target_id)
            for target_id in grounding.get("reachable_contact_object_ids", [])
            if str(target_id) != obj.id
        }
    )
    evidence_level = _evidence_level(hits, source_representation=source_representation)
    geometry_provenance = _geometry_provenance(scene, obj.id, mesh_cache.get(obj.id))
    architecture_plane_clearances = _architecture_plane_clearances(
        scene,
        obj,
        boundary,
        has_boundary,
    )
    architecture_contact_candidates = _architecture_contact_candidates(
        architecture_plane_clearances,
        tolerance_m=candidate_tol,
    )
    lower_envelope_zs = [float(point[2]) for point in lower_envelope_points]
    geometry_degraded_reasons = _geometry_degraded_reasons(
        source_id=obj.id,
        source_cache=source_cache,
        contact_hits=contact_hits,
    )
    measured_support_modes = _measured_support_modes(contact_hits)
    routing_reasons: list[str] = []
    if contact_hit_count < minimum_contact_count:
        routing_reasons.append("no_reliable_base_contact")
    if gap_band in {GAP_BAND_BORDERLINE, GAP_BAND_STRONG, GAP_BAND_UNKNOWN}:
        routing_reasons.append(gap_band)
    if geometry_degraded_reasons:
        routing_reasons.append("geometry_evidence_degraded")
    if (
        contact_hit_count >= minimum_contact_count
        and not geometry_degraded_reasons
        and not bool(grounding.get("certified_grounded_support"))
    ):
        routing_reasons.append("object_contact_without_certified_ground_path")
        if bool(grounding.get("ungrounded_contact_cycle_reachable")):
            routing_reasons.append("ungrounded_contact_cycle")
    if architecture_contact_candidates and contact_hit_count < minimum_contact_count:
        routing_reasons.append("possible_architecture_attachment")

    record: dict[str, Any] = {
        "object_id": obj.id,
        "lower_envelope_sample_count": len(lower_envelope_points),
        "base_contact_sample_count": sample_count,
        "sample_count": sample_count,
        "base_contact_hit_count": contact_hit_count,
        "contact_hit_count": contact_hit_count,
        "unsupported_base_sample_count": unsupported_sample_count,
        "base_contact_fraction": contact_fraction,
        "contact_fraction": contact_fraction,
        "contact_fraction_affects_route": False,
        "size_z_m": size_z,
        "contact_tolerance_m": direct_contact_tolerance,
        "direct_contact_tolerance_m": direct_contact_tolerance,
        # Retained as a report compatibility alias; this is a tolerance band,
        # not a frozen numerical epsilon in support_p0b_v7.
        "direct_contact_epsilon_m": direct_contact_tolerance,
        "support_candidate_tolerance_m": candidate_tol,
        "hard_contact_tolerance_m": hard_contact_tolerance,
        "legacy_contact_tolerance_affects_direct_valid": fixed_contact_tolerance is not None,
        "near_support_tolerance_m": near_support_tolerance,
        "gap_band": gap_band,
        "base_band_tolerance_m": base_band_tol,
        "base_min_z_m": base_min_z,
        "lower_envelope_z_span_m": (
            float(max(lower_envelope_zs) - min(lower_envelope_zs))
            if lower_envelope_zs
            else None
        ),
        "minimum_contact_count": minimum_contact_count,
        "contact_fraction_threshold": legacy_contact_fraction_threshold,
        "gap_statistics_m": _gap_statistics(gaps),
        "contact_gap_statistics_m": _gap_statistics(contact_gaps),
        "positive_clearance_statistics_m": _gap_statistics(positive_clearances),
        "minimum_positive_clearance_m": minimum_positive_clearance_m,
        "normalized_minimum_positive_clearance": normalized_minimum_positive_clearance,
        "support_targets": contact_targets,
        "per_target_hit_counts": per_target_hit_counts,
        "measured_support_modes": measured_support_modes,
        "candidate_support_object_ids": candidate_support_ids,
        "grounded_support_policy": GROUNDED_SUPPORT_POLICY,
        "grounded_support_required_for_direct_valid": True,
        "certified_grounded_support": bool(grounding.get("certified_grounded_support")),
        "grounding_status": grounding.get("grounding_status") or "unknown",
        "grounded_support_path": list(grounding.get("grounded_support_path") or []),
        "grounding_contact_target_ids": list(grounding.get("contact_target_ids") or []),
        "reachable_grounding_contact_object_ids": list(
            grounding.get("reachable_contact_object_ids") or []
        ),
        "ungrounded_contact_cycle_reachable": bool(
            grounding.get("ungrounded_contact_cycle_reachable")
        ),
        "center_source_available": center_source_available,
        "center_ray_supported": center_ray_supported,
        "center_ray_affects_route": False,
        "source_representation": source_representation,
        "source_sample_method": source_sample_method,
        "contact_sample_method": "lowest_base_contact_band",
        "source_mesh_load_error": source_cache.get("load_error"),
        "source_mesh_frame_validation": deepcopy(source_cache.get("frame_validation")),
        "geometry_evidence_degraded": bool(geometry_degraded_reasons),
        "geometry_degraded_reasons": geometry_degraded_reasons,
        "evidence_level": evidence_level,
        "geometry_provenance": geometry_provenance,
        "architecture_plane_clearances_m": architecture_plane_clearances,
        "architecture_contact_candidates": architecture_contact_candidates,
        "representative_samples": _representative_samples(hits, center_hit, max_rep),
        "routing_reasons": routing_reasons,
        "requires_vlm": False,
        "route": None,
        "final_verdict": None,
        "affects_support_score": False,
        "judge_result": None,
        "adjudication_error": None,
    }

    direct_valid = (
        contact_hit_count >= minimum_contact_count
        and not geometry_degraded_reasons
        and bool(grounding.get("certified_grounded_support"))
    )
    if direct_valid:
        record.update(
            {
                "route": "direct_valid_contact",
                "direct_valid_reason": "reliable_contact_with_certified_ground_path",
                "final_verdict": "valid",
                "affects_support_score": True,
            }
        )
        return record

    record["requires_vlm"] = True
    if bool(cfg.get("detector_only")):
        return record
    if vlm_judge is None:
        return record

    event = {
        "object_id": obj.id,
        "object_ids": [obj.id, *candidate_support_ids],
        "architecture_element": "floor_walls_ceiling_and_supports",
        "candidate_selection_policy": SUPPORT_CANDIDATE_SELECTION_POLICY,
        "gap_band": gap_band,
        "measured_support_modes": measured_support_modes,
        "architecture_contact_candidates": architecture_contact_candidates,
    }
    detector_evidence = _detector_evidence(
        obj=obj,
        record=record,
        direct_contact_tolerance=direct_contact_tolerance,
        candidate_tol=candidate_tol,
        base_band_tol=base_band_tol,
        minimum_contact_count=minimum_contact_count,
        legacy_contact_fraction_threshold=legacy_contact_fraction_threshold,
        candidate_support_ids=candidate_support_ids,
        objects=objects,
        mesh_cache=mesh_cache,
    )
    try:
        judge_result = adjudicate_p0b_event(
            metric="support",
            event=event,
            prompt=str(prompt or ""),
            relationships=relationships,
            scene=scene,
            detector_evidence=detector_evidence,
            judge=vlm_judge,
            object_ids=[obj.id, *candidate_support_ids],
            overview_render_evidence=list(render_evidence or []),
            local_view_provider=local_view_provider,
        )
    except EvidenceControlUnresolvedError as exc:
        record["route"] = "unresolved"
        record["evidence_control"] = exc.result.to_dict()
        return record
    except Exception as exc:
        record["adjudication_error"] = f"{type(exc).__name__}: {exc}"
        record["route"] = "vlm_adjudication_failed"
        if bool(cfg.get("official_mode")):
            raise SupportEvaluationError(record["adjudication_error"]) from exc
        return record

    verdict = str(judge_result.get("verdict"))
    record.update(
        {
            "route": "vlm_adjudicated",
            "final_verdict": verdict,
            "affects_support_score": True,
            "judge_result": deepcopy(judge_result),
        }
    )
    return record


def _detector_evidence(
    *,
    obj,
    record: dict[str, Any],
    direct_contact_tolerance: float,
    candidate_tol: float,
    base_band_tol: float,
    minimum_contact_count: int,
    legacy_contact_fraction_threshold: float | None,
    candidate_support_ids: list[str],
    objects,
    mesh_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_id = {other.id: other for other in objects}
    candidate_support_objects = []
    for candidate_id in candidate_support_ids:
        other = by_id.get(candidate_id)
        if other is None:
            continue
        candidate_support_objects.append(
            {
                "id": other.id,
                "category": other.category,
                "description": other.desc,
                "representation": _representation(mesh_cache.get(other.id)),
                "geometry_evidence_degraded": bool((mesh_cache.get(other.id) or {}).get("degraded")),
                "mesh_frame_validation": deepcopy((mesh_cache.get(other.id) or {}).get("frame_validation")),
            }
        )
    return {
        "detector": SUPPORT_EVALUATOR_VERSION,
        "support_instruction": SUPPORT_VLM_INSTRUCTION,
        "candidate_selection_policy": SUPPORT_CANDIDATE_SELECTION_POLICY,
        "grounded_support_policy": record["grounded_support_policy"],
        "grounded_support_required_for_direct_valid": record[
            "grounded_support_required_for_direct_valid"
        ],
        "certified_grounded_support": record["certified_grounded_support"],
        "grounding_status": record["grounding_status"],
        "grounded_support_path": record["grounded_support_path"],
        "grounding_contact_target_ids": record["grounding_contact_target_ids"],
        "reachable_grounding_contact_object_ids": record[
            "reachable_grounding_contact_object_ids"
        ],
        "ungrounded_contact_cycle_reachable": record[
            "ungrounded_contact_cycle_reachable"
        ],
        "gravity": list(GRAVITY),
        "base_contact_fraction": record["base_contact_fraction"],
        "contact_fraction": record["contact_fraction"],
        "contact_fraction_affects_route": False,
        "size_z_m": record["size_z_m"],
        "contact_tolerance_m": direct_contact_tolerance,
        "direct_contact_tolerance_m": direct_contact_tolerance,
        "direct_contact_epsilon_m": direct_contact_tolerance,
        "support_candidate_tolerance_m": candidate_tol,
        "hard_contact_tolerance_m": record["hard_contact_tolerance_m"],
        "legacy_contact_tolerance_affects_direct_valid": record[
            "legacy_contact_tolerance_affects_direct_valid"
        ],
        "near_support_tolerance_m": record["near_support_tolerance_m"],
        "gap_band": record["gap_band"],
        "normalized_minimum_positive_clearance": record["normalized_minimum_positive_clearance"],
        "base_band_tolerance_m": base_band_tol,
        "minimum_contact_count": minimum_contact_count,
        "contact_fraction_threshold": legacy_contact_fraction_threshold,
        "lower_envelope_sample_count": record["lower_envelope_sample_count"],
        "base_contact_sample_count": record["base_contact_sample_count"],
        "base_contact_hit_count": record["base_contact_hit_count"],
        "sample_count": record["sample_count"],
        "contact_hit_count": record["contact_hit_count"],
        "unsupported_base_sample_count": record["unsupported_base_sample_count"],
        "base_min_z_m": record["base_min_z_m"],
        "lower_envelope_z_span_m": record["lower_envelope_z_span_m"],
        "center_source_available": record["center_source_available"],
        "center_ray_supported": record["center_ray_supported"],
        "center_ray_affects_route": False,
        "source_representation": record["source_representation"],
        "source_sample_method": record["source_sample_method"],
        "contact_sample_method": record["contact_sample_method"],
        "source_mesh_load_error": record["source_mesh_load_error"],
        "source_mesh_frame_validation": record["source_mesh_frame_validation"],
        "geometry_evidence_degraded": record["geometry_evidence_degraded"],
        "geometry_degraded_reasons": record["geometry_degraded_reasons"],
        "gap_statistics_m": record["gap_statistics_m"],
        "contact_gap_statistics_m": record["contact_gap_statistics_m"],
        "positive_clearance_statistics_m": record["positive_clearance_statistics_m"],
        "minimum_positive_clearance_m": record["minimum_positive_clearance_m"],
        "support_targets": record["support_targets"],
        "per_target_hit_counts": record["per_target_hit_counts"],
        "measured_support_modes": record["measured_support_modes"],
        "representative_ray_hits": record["representative_samples"],
        "architecture_plane_clearances_m": record["architecture_plane_clearances_m"],
        "architecture_contact_candidates": record["architecture_contact_candidates"],
        "routing_reasons": record["routing_reasons"],
        "evidence_level": record["evidence_level"],
        "evaluated_object": {
            "id": obj.id,
            "category": obj.category,
            "description": obj.desc,
            "center": [float(value) for value in np.asarray(obj.center, dtype=float)],
            "size": [float(value) for value in np.asarray(obj.size, dtype=float)],
            "rotation_degrees": [float(value) for value in np.asarray(obj.rotation, dtype=float)],
            "geometry_provenance": record["geometry_provenance"],
        },
        "candidate_support_objects": candidate_support_objects,
        "extracted_relationships_are_claims_only": True,
    }


def _base_contact_band(
    lower_envelope_points: list[np.ndarray],
    tolerance_m: float,
) -> tuple[list[np.ndarray], float | None]:
    """Select only geometry that can plausibly form the object's lowest base.

    A lower envelope still contains raised surfaces such as a tabletop underside
    in XY cells without legs. Measuring contact over that entire envelope treats
    ordinary negative space as floating. The base band keeps samples within a
    frozen vertical tolerance of the object's global lowest sampled point.
    """

    if not lower_envelope_points:
        return [], None
    minimum_z = min(float(point[2]) for point in lower_envelope_points)
    upper_z = minimum_z + float(tolerance_m) + 1.0e-9
    return (
        [point for point in lower_envelope_points if float(point[2]) <= upper_z],
        minimum_z,
    )


def _architecture_plane_clearances(
    scene: dict,
    obj,
    boundary: np.ndarray,
    has_boundary: bool,
) -> dict[str, float | None]:
    """Signed OBB clearances to the frozen room planes in meters.

    Positive values are inside the room, zero is contact, and negative values
    cross the plane. These measurements let the VLM distinguish an unsupported
    object from a plausible wall or ceiling attachment without changing OOB's
    independent verdict.
    """

    corners = np.asarray(get_obb_corners(obj), dtype=float)
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    result: dict[str, float | None] = {
        "west": None,
        "east": None,
        "south": None,
        "north": None,
        "floor": float(minimum[2]),
        "ceiling": float(get_scene_height(scene) - maximum[2]),
    }
    if has_boundary:
        room_minimum = np.min(boundary[:, :2], axis=0)
        room_maximum = np.max(boundary[:, :2], axis=0)
        result.update(
            {
                "west": float(minimum[0] - room_minimum[0]),
                "east": float(room_maximum[0] - maximum[0]),
                "south": float(minimum[1] - room_minimum[1]),
                "north": float(room_maximum[1] - maximum[1]),
            }
        )
    return result


def _architecture_contact_candidates(
    clearances: dict[str, float | None],
    *,
    tolerance_m: float,
) -> list[dict[str, Any]]:
    """Architecture planes close enough to explain attachment or suspension.

    Floor contact is measured by the downward probe. Wall and ceiling proximity
    remain semantic attachment evidence for the VLM and never auto-pass Support.
    """

    candidates: list[dict[str, Any]] = []
    for plane in ("west", "east", "south", "north", "ceiling"):
        clearance = clearances.get(plane)
        if clearance is None or abs(float(clearance)) > float(tolerance_m):
            continue
        candidates.append(
            {
                "plane": plane,
                "signed_clearance_m": float(clearance),
                "mode": "ceiling_attachment" if plane == "ceiling" else "wall_attachment",
            }
        )
    return candidates


def _geometry_degraded_reasons(
    *,
    source_id: str,
    source_cache: dict[str, Any],
    contact_hits: list[dict[str, Any]],
) -> list[str]:
    reasons: list[str] = []
    if source_cache.get("degraded"):
        reasons.append(f"source:{source_id}:{source_cache.get('load_error') or 'mesh_unavailable'}")
    for hit in contact_hits:
        target = hit.get("target")
        reason = hit.get("target_geometry_degraded_reason")
        if target not in {None, "floor"} and reason:
            reasons.append(f"target:{target}:{reason}")
    return list(dict.fromkeys(reasons))


def _measured_support_modes(contact_hits: list[dict[str, Any]]) -> list[str]:
    modes: set[str] = set()
    for hit in contact_hits:
        target = hit.get("target")
        if target == "floor":
            modes.add("floor_contact")
        elif target is not None:
            modes.add("object_contact")
    return sorted(modes)


def _bottom_samples(
    obj,
    source_cache: dict[str, Any] | None,
    grid: tuple[int, int],
) -> tuple[list[np.ndarray], str, str, np.ndarray | None]:
    """World-space bottom support sample points.

    Mesh-backed objects sample the distributed lower envelope of the mesh. The
    caller subsequently extracts the global base-contact band so raised surfaces
    such as table undersides do not enter the contact denominator. Proxy-only
    objects sample the OBB bottom face.
    """

    mesh = source_cache.get("mesh") if isinstance(source_cache, dict) else None
    if mesh is None:
        center_point = obj.center + obj.R @ np.array([0.0, 0.0, -obj.half[2]], dtype=float)
        return (
            [np.asarray(point, dtype=float) for point in sample_bottom_face_points(obj, grid)],
            "obb",
            "obb_bottom_face_grid",
            np.asarray(center_point, dtype=float),
        )

    vertices = np.asarray(mesh["vertices"], dtype=float)
    faces = np.asarray(mesh["faces"], dtype=int)
    samples_by_cell: dict[tuple[int, int], np.ndarray] = {}
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    xs = _cell_centers(float(minimum[0]), float(maximum[0]), grid[0])
    ys = _cell_centers(float(minimum[1]), float(maximum[1]), grid[1])
    for x_index, x in enumerate(xs):
        for y_index, y in enumerate(ys):
            zs = _column_surface_zs(vertices, faces, x, y)
            if zs.size:
                samples_by_cell[(x_index, y_index)] = np.array(
                    [x, y, float(np.min(zs))],
                    dtype=float,
                )

    # A cell-center ray can miss a narrow leg. Fold real vertices into the same
    # fixed grid and retain only the lowest point per cell. The sample budget is
    # therefore bounded by grid_x * grid_y and does not vary with tessellation
    # density or vertex ordering.
    for vertex in vertices:
        key = (
            _cell_index(float(vertex[0]), float(minimum[0]), float(maximum[0]), grid[0]),
            _cell_index(float(vertex[1]), float(minimum[1]), float(maximum[1]), grid[1]),
        )
        previous = samples_by_cell.get(key)
        if previous is None or float(vertex[2]) < float(previous[2]):
            samples_by_cell[key] = np.asarray(vertex, dtype=float)

    samples = [samples_by_cell[key] for key in sorted(samples_by_cell)]
    center_zs = _column_surface_zs(
        vertices,
        faces,
        float(obj.center[0]),
        float(obj.center[1]),
    )
    center_point = (
        np.array([float(obj.center[0]), float(obj.center[1]), float(np.min(center_zs))], dtype=float)
        if center_zs.size
        else None
    )
    if samples:
        return samples, "mesh", "mesh_lower_envelope", center_point

    # A loaded triangle mesh should always provide a lower point. Keep the
    # branch conservative if a third-party loader returns an empty artifact.
    fallback = obj.center + obj.R @ np.array([0.0, 0.0, -obj.half[2]], dtype=float)
    return [np.asarray(fallback, dtype=float)], "obb", "mesh_empty_obb_fallback", np.asarray(fallback, dtype=float)


def _nearest_support(
    point: np.ndarray,
    objects,
    mesh_cache: dict[str, dict[str, Any]],
    boundary: np.ndarray,
    has_boundary: bool,
    candidate_tol: float,
    direct_contact_tolerance: float,
    source_id: str,
) -> dict[str, Any]:
    x, y, z_p = float(point[0]), float(point[1]), float(point[2])
    best_h: float | None = None
    best_target: str | None = None
    best_rep: str | None = None
    best_source: str | None = None
    best_degraded_reason: str | None = None

    for other in objects:
        if other.id == source_id:
            continue
        cache = mesh_cache.get(other.id) or {}
        mesh = cache.get("mesh")
        if mesh is not None:
            bounds_min = cache.get("bounds_min")
            bounds_max = cache.get("bounds_max")
            if (
                bounds_min is not None
                and bounds_max is not None
                and (
                    x < float(bounds_min[0]) - 1.0e-9
                    or x > float(bounds_max[0]) + 1.0e-9
                    or y < float(bounds_min[1]) - 1.0e-9
                    or y > float(bounds_max[1]) + 1.0e-9
                    or float(bounds_min[2]) > z_p + candidate_tol
                )
            ):
                continue
            zs = _column_surface_zs(mesh["vertices"], mesh["faces"], x, y)
            below = zs[zs <= z_p + candidate_tol]
            if below.size == 0:
                continue
            h = float(np.max(below))
            rep, source = "mesh", "mesh_ray"
        else:
            hit = ray_intersects_obb([x, y, float(other.top_z) + 1.0], [0.0, 0.0, -1.0], other)
            if hit is None:
                continue
            h = float(hit["point"][2])
            if h > z_p + candidate_tol:
                continue
            rep, source = "obb", "obb_ray"
        if best_h is None or h > best_h:
            best_h, best_target, best_rep, best_source = h, other.id, rep, source
            best_degraded_reason = str(cache.get("load_error")) if cache.get("degraded") else None

    inside_room = (not has_boundary) or point_in_polygon_2d([x, y], boundary)
    if inside_room and 0.0 <= z_p + candidate_tol and (best_h is None or 0.0 > best_h):
        best_h, best_target, best_rep, best_source = 0.0, "floor", "architecture", "floor_plane"
        best_degraded_reason = None

    if best_h is not None:
        gap = z_p - best_h
        return {
            "target": best_target,
            "target_representation": best_rep,
            "height_m": best_h,
            "gap_m": gap,
            "position": [x, y, best_h],
            "evidence_source": best_source,
            "target_geometry_degraded_reason": best_degraded_reason,
            "contact": bool(gap <= direct_contact_tolerance),
        }
    if inside_room:
        # No support surface at/below the sample within the candidate tolerance while
        # inside the room means the bottom is deeply below the floor.
        # Penetration is a Collision/OOB concern, so it counts as contact, never floating.
        return {
            "target": "floor",
            "target_representation": "architecture",
            "height_m": 0.0,
            "gap_m": z_p,
            "position": [x, y, 0.0],
            "evidence_source": "floor_penetration",
            "target_geometry_degraded_reason": None,
            "contact": True,
        }
    return {
        "target": None,
        "target_representation": None,
        "height_m": None,
        "gap_m": None,
        "position": [x, y, z_p],
        "evidence_source": "no_surface",
        "target_geometry_degraded_reason": None,
        "contact": False,
    }


def _column_surface_zs(vertices: np.ndarray, faces: np.ndarray, x: float, y: float, eps: float = 1.0e-9) -> np.ndarray:
    """Surface z-values where the vertical line at (x, y) crosses mesh triangles.

    Vertical triangles project to zero XY area and are skipped, so walls and other
    non-horizontal faces never register as support surfaces.
    """

    if faces.size == 0:
        return np.empty(0, dtype=float)
    tris = vertices[faces]
    ax, ay, az = tris[:, 0, 0], tris[:, 0, 1], tris[:, 0, 2]
    bx, by, bz = tris[:, 1, 0], tris[:, 1, 1], tris[:, 1, 2]
    cx, cy, cz = tris[:, 2, 0], tris[:, 2, 1], tris[:, 2, 2]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    valid = np.abs(denom) > eps
    safe = np.where(valid, denom, 1.0)
    a = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / safe
    b = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / safe
    c = 1.0 - a - b
    inside = valid & (a >= -1.0e-9) & (b >= -1.0e-9) & (c >= -1.0e-9)
    if not np.any(inside):
        return np.empty(0, dtype=float)
    z = a * az + b * bz + c * cz
    return z[inside]


def _evidence_level(hits: list[dict[str, Any]], *, source_representation: str) -> str:
    object_reps = {source_representation} if source_representation in {"mesh", "obb"} else set()
    object_reps.update(
        {
        str(hit["target_representation"])
        for hit in hits
        if hit["target"] is not None and str(hit["target"]) != "floor" and hit["target_representation"] in {"mesh", "obb"}
        }
    )
    if "mesh" in object_reps and "obb" in object_reps:
        return "mixed"
    if object_reps == {"mesh"}:
        return "mesh"
    return "obb"


def _gap_statistics(gaps: list[float]) -> dict[str, Any]:
    if not gaps:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    arr = np.asarray(gaps, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _representative_samples(hits: list[dict[str, Any]], center_hit: dict[str, Any], max_n: int) -> list[dict[str, Any]]:
    reps: list[dict[str, Any]] = [{**_compact_sample(center_hit), "is_center": True}]
    remaining = max(0, int(max_n) - 1)
    if remaining and hits:
        ordered = sorted(
            range(len(hits)),
            key=lambda index: (hits[index]["gap_m"] if hits[index]["gap_m"] is not None else float("inf")),
        )
        count = min(remaining, len(ordered))
        if count == 1:
            picks = [ordered[0]]
        else:
            picks = [ordered[round(step * (len(ordered) - 1) / (count - 1))] for step in range(count)]
        seen: set[int] = set()
        for index in picks:
            if index in seen:
                continue
            seen.add(index)
            reps.append({**_compact_sample(hits[index]), "is_center": False})
    return reps


def _compact_sample(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": [float(value) for value in hit["position"]],
        "target": hit["target"],
        "target_representation": hit["target_representation"],
        "gap_m": None if hit["gap_m"] is None else float(hit["gap_m"]),
        "evidence_source": hit["evidence_source"],
        "target_geometry_degraded_reason": hit.get("target_geometry_degraded_reason"),
        "contact": bool(hit["contact"]),
    }


def _build_mesh_cache(
    objects,
    collision_geometry: dict | None,
    base_dir: Path | None,
    *,
    bounds_tolerance_m: float,
    center_tolerance_m: float,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for obj in objects:
        entry = geometry_entry_for_object(collision_geometry, obj.id)
        mesh = None
        load_error = None
        frame_validation = None
        bounds_min = None
        bounds_max = None
        mesh_expected = _mesh_evidence_expected(entry)
        if is_usable_triangle_mesh(entry, base_dir=base_dir):
            try:
                mesh = load_triangle_mesh(entry, base_dir=base_dir)
                validation_error = _loaded_mesh_validation_error(mesh)
                if validation_error is not None:
                    raise ValueError(validation_error)
                vertices = np.asarray(mesh["vertices"], dtype=float)
                frame_validation = _mesh_frame_validation(
                    vertices,
                    obj,
                    bounds_tolerance_m=bounds_tolerance_m,
                    center_tolerance_m=center_tolerance_m,
                )
                if not frame_validation["canonical_consistent"]:
                    raise ValueError(
                        "canonical mesh frame mismatch: "
                        + ", ".join(frame_validation["failure_reasons"])
                    )
                bounds_min = vertices.min(axis=0)
                bounds_max = vertices.max(axis=0)
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
                mesh = None
                bounds_min = None
                bounds_max = None
        elif mesh_expected:
            load_error = geometry_unavailable_reason(entry)
        cache[obj.id] = {
            "entry": entry,
            "mesh": mesh,
            "load_error": load_error,
            "frame_validation": frame_validation,
            "mesh_expected": mesh_expected,
            "degraded": bool(mesh_expected and mesh is None),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
        }
    return cache


def _mesh_evidence_expected(entry: dict[str, Any] | None) -> bool:
    if not isinstance(entry, dict):
        return False
    return entry.get("representation") in {"triangle_mesh", "point_cloud"}


def _mesh_frame_validation(
    vertices: np.ndarray,
    obj,
    *,
    bounds_tolerance_m: float,
    center_tolerance_m: float,
) -> dict[str, Any]:
    """Check that a baked world-space mesh still belongs to its canonical object.

    The rendered asset may be uniformly smaller than the canonical OBB, so size
    equality is intentionally not required. The mesh must remain contained by
    the canonical OBB within a frozen tolerance. Bounds-center offset is retained
    as a warning because legitimate thin or asymmetric geometry need not be
    centered. Containment still catches stale Blender parent transforms without
    rejecting valid contain-fit slack.
    """

    world = np.asarray(vertices, dtype=float)
    local = (world - np.asarray(obj.center, dtype=float)) @ np.asarray(obj.R, dtype=float)
    local_min = np.min(local, axis=0)
    local_max = np.max(local, axis=0)
    local_center = (local_min + local_max) * 0.5
    center_offset = float(np.linalg.norm(local_center))
    lower_overflow = np.maximum((-np.asarray(obj.half, dtype=float)) - local_min, 0.0)
    upper_overflow = np.maximum(local_max - np.asarray(obj.half, dtype=float), 0.0)
    overflow = np.maximum(lower_overflow, upper_overflow)
    maximum_overflow = float(np.max(overflow))
    reasons: list[str] = []
    warnings: list[str] = []
    if center_offset > center_tolerance_m:
        warnings.append(
            f"local_bounds_center_offset_m={center_offset:.6f}>{center_tolerance_m:.6f}"
        )
    if maximum_overflow > bounds_tolerance_m:
        reasons.append(
            f"canonical_obb_overflow_m={maximum_overflow:.6f}>{bounds_tolerance_m:.6f}"
        )
    return {
        "canonical_consistent": not reasons,
        "failure_reasons": reasons,
        "warnings": warnings,
        "local_bounds_min": [float(value) for value in local_min],
        "local_bounds_max": [float(value) for value in local_max],
        "local_bounds_center": [float(value) for value in local_center],
        "local_bounds_center_offset_m": center_offset,
        "canonical_obb_overflow_by_axis_m": [float(value) for value in overflow],
        "maximum_canonical_obb_overflow_m": maximum_overflow,
        "bounds_tolerance_m": float(bounds_tolerance_m),
        "center_tolerance_m": float(center_tolerance_m),
    }


def _representation(cache: dict[str, Any] | None) -> str:
    return "mesh" if isinstance(cache, dict) and cache.get("mesh") is not None else "obb"


def _geometry_provenance(scene: dict, object_id: str, cache: dict[str, Any] | None) -> Any:
    if isinstance(cache, dict) and isinstance(cache.get("entry"), dict):
        runtime_provenance = cache["entry"].get("geometry_source")
        if runtime_provenance is not None:
            return runtime_provenance
    for item in scene.get("objects", []) if isinstance(scene, dict) else []:
        if isinstance(item, dict) and str(item.get("id")) == str(object_id):
            provenance = item.get("geometry_provenance")
            if provenance is not None:
                return provenance
    return None


def _geometry_base_dir(collision_geometry: dict | None) -> Path | None:
    if not isinstance(collision_geometry, dict):
        return None
    manifest_path = collision_geometry.get("manifest_path")
    if isinstance(manifest_path, str) and manifest_path.strip():
        return Path(manifest_path).expanduser().resolve().parent
    return None


def _grid(value: object) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (max(1, int(value[0])), max(1, int(value[1])))
    return (4, 4)


def _cell_centers(minimum: float, maximum: float, count: int) -> list[float]:
    if maximum <= minimum:
        return [float(minimum)]
    step = (maximum - minimum) / float(max(1, int(count)))
    return [float(minimum + (index + 0.5) * step) for index in range(max(1, int(count)))]


def _cell_index(value: float, minimum: float, maximum: float, count: int) -> int:
    resolved_count = max(1, int(count))
    if maximum <= minimum:
        return 0
    normalized = (float(value) - float(minimum)) / (float(maximum) - float(minimum))
    return min(resolved_count - 1, max(0, int(math.floor(normalized * resolved_count))))


def _loaded_mesh_validation_error(mesh: object) -> str | None:
    if not isinstance(mesh, dict):
        return "loaded triangle mesh must be a mapping"
    try:
        vertices = np.asarray(mesh.get("vertices"), dtype=float)
        faces = np.asarray(mesh.get("faces"), dtype=int)
    except (TypeError, ValueError) as exc:
        return f"triangle mesh arrays are invalid: {exc}"
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) == 0:
        return "triangle mesh vertices must be a non-empty Nx3 array"
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
        return "triangle mesh faces must be a non-empty Mx3 array"
    if not np.all(np.isfinite(vertices)):
        return "triangle mesh vertices must be finite"
    if not np.all((faces >= 0) & (faces < len(vertices))):
        return "triangle mesh contains out-of-range face indices"
    triangles = vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    if not np.any(doubled_areas > 1.0e-12):
        return "triangle mesh contains no non-degenerate faces"
    return None


def _validate_support_config(config: dict[str, Any]) -> None:
    if bool(config.get("official_mode")) and bool(config.get("detector_only")):
        raise ValueError("support.official_mode and support.detector_only are mutually exclusive")
    # Scale-aware thresholds are frozen constants; validate their internal ordering
    # so ``near >= hard`` holds for every object size.
    if HARD_CONTACT_TOLERANCE_MIN_M > HARD_CONTACT_TOLERANCE_MAX_M:
        raise ValueError("support hard contact tolerance min must be <= max")
    if NEAR_SUPPORT_TOLERANCE_MIN_M > NEAR_SUPPORT_TOLERANCE_MAX_M:
        raise ValueError("support near support tolerance min must be <= max")
    if NEAR_SUPPORT_TOLERANCE_MIN_M < HARD_CONTACT_TOLERANCE_MAX_M:
        raise ValueError("support near tolerance must be >= hard tolerance for every size")
    # ``contact_tolerance_m`` remains an optional legacy fixed-band override.
    # Direct-valid contact is always bounded by
    # DIRECT_CONTACT_TOLERANCE_CAP_M, even if candidate discovery is wider.
    fixed_contact_tolerance = config.get("contact_tolerance_m")
    if fixed_contact_tolerance is not None:
        tolerance = float(fixed_contact_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("support.contact_tolerance_m must be a finite non-negative number")
    base_band_tolerance = float(config.get("base_band_tolerance_m", 0.02))
    mesh_bounds_tolerance = float(config.get("mesh_bounds_tolerance_m", 0.03))
    mesh_center_tolerance = float(config.get("mesh_center_tolerance_m", 0.05))
    if not math.isfinite(base_band_tolerance) or base_band_tolerance < 0.0:
        raise ValueError("support.base_band_tolerance_m must be a finite non-negative number")
    if not math.isfinite(mesh_bounds_tolerance) or mesh_bounds_tolerance < 0.0:
        raise ValueError("support.mesh_bounds_tolerance_m must be a finite non-negative number")
    if not math.isfinite(mesh_center_tolerance) or mesh_center_tolerance < 0.0:
        raise ValueError("support.mesh_center_tolerance_m must be a finite non-negative number")
    legacy_threshold = config.get("contact_fraction_threshold")
    if legacy_threshold is not None:
        threshold = float(legacy_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("support.contact_fraction_threshold must be between 0 and 1")
    raw_minimum_contact_count = config.get("minimum_contact_count", 1)
    if isinstance(raw_minimum_contact_count, bool) or not isinstance(raw_minimum_contact_count, int):
        raise ValueError("support.minimum_contact_count must be a positive integer")
    if int(raw_minimum_contact_count) <= 0:
        raise ValueError("support.minimum_contact_count must be a positive integer")
    grid = config.get("bottom_sample_grid", [4, 4])
    if not isinstance(grid, (list, tuple)) or len(grid) < 2:
        raise ValueError("support.bottom_sample_grid must contain two positive integers")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in grid[:2]):
        raise ValueError("support.bottom_sample_grid must contain two positive integers")
    grid_x, grid_y = int(grid[0]), int(grid[1])
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError("support.bottom_sample_grid must contain two positive integers")
    raw_max_representative = config.get("max_representative_samples", 8)
    if isinstance(raw_max_representative, bool) or not isinstance(raw_max_representative, int):
        raise ValueError("support.max_representative_samples must be a positive integer")
    max_representative = int(raw_max_representative)
    if max_representative <= 0:
        raise ValueError("support.max_representative_samples must be a positive integer")


def disabled_support_report() -> dict[str, Any]:
    return {
        "metric": "support",
        "evaluator_version": SUPPORT_EVALUATOR_VERSION,
        "status": "not_applicable",
        "score": None,
        "enabled": False,
        "reason": "disabled_by_configuration",
        "num_objects": 0,
        "evaluated_object_count": 0,
        "valid_support_object_count": 0,
        "supported_object_count": 0,
        "unsupported_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": {},
        "coverage": _empty_coverage(),
        "notes": list(SUPPORT_NOTES),
    }


def _empty_support_report(object_errors: dict[str, str]) -> dict[str, Any]:
    return {
        "metric": "support",
        "evaluator_version": SUPPORT_EVALUATOR_VERSION,
        "status": "not_applicable",
        "score": None,
        "enabled": True,
        "reason": "no_physical_objects",
        "num_objects": 0,
        "evaluated_object_count": 0,
        "valid_support_object_count": 0,
        "supported_object_count": 0,
        "unsupported_object_count": 0,
        "requires_vlm_count": 0,
        "resolved_object_count": 0,
        "objects": [],
        "object_errors": object_errors,
        "coverage": _empty_coverage(),
        "notes": list(SUPPORT_NOTES),
    }


def _empty_coverage() -> dict[str, Any]:
    return {
        "object_count": 0,
        "direct_valid_objects": 0,
        "vlm_adjudicated_objects": 0,
        "mesh_evidence_objects": 0,
        "mixed_evidence_objects": 0,
        "obb_evidence_objects": 0,
    }
