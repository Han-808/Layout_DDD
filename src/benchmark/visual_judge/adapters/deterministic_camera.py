from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any, Callable, Mapping

from benchmark.rendering.camera_pose import (
    DEFAULT_CAMERA_CANDIDATE_POLICY,
    generate_camera_pose_candidates,
    normalize_camera_candidate_policy,
    resolve_camera_pose_mode,
)
from benchmark.visual_judge.adapters.legacy_camera import (
    camera_selection_result_from_value,
)
from benchmark.visual_judge.camera_dsl import (
    CameraConstraintSet,
    active_constraint_references,
)
from benchmark.visual_judge.camera_repair import CameraRepairPlan
from benchmark.visual_judge.camera_ranking import (
    DeterministicCameraRankingConfig,
)
from benchmark.visual_judge.camera_targets import (
    merge_authoritative_target_ids,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
)


DETERMINISTIC_SUPPORTED_OBSERVATIONS = frozenset(
    {
        # These observations have explicit candidate rejection checks below.
        # Other DSL tokens remain available to VLM selection, but must not be
        # treated as deterministically verified by this adapter.
        "target_visible",
        "joint_visibility",
    }
)


class NoFeasibleCameraCandidates(ValueError):
    """Normal geometry-search exhaustion from an injected candidate generator."""


class DeterministicLocalCameraSelector:
    """Geometry-only candidate generation, filtering, and ranking adapter."""

    backend = "deterministic"
    validated_internal_candidate_bank = True

    def __init__(
        self,
        *,
        candidate_generator: Callable[..., list[dict[str, Any]]] = (
            generate_camera_pose_candidates
        ),
        candidate_policy: str = DEFAULT_CAMERA_CANDIDATE_POLICY,
        feature_enricher: (
            Callable[
                [dict[str, Any], CameraSelectionRequest],
                dict[str, Any],
            ]
            | None
        ) = None,
        ranking_config: (
            DeterministicCameraRankingConfig
            | Mapping[str, Any]
            | None
        ) = None,
        ranking_config_sources: Mapping[str, str] | None = None,
    ) -> None:
        if not callable(candidate_generator):
            raise TypeError(
                "deterministic candidate_generator must be callable"
            )
        self.candidate_generator = candidate_generator
        self.candidate_policy = normalize_camera_candidate_policy(
            candidate_policy
        )
        self.feature_enricher = feature_enricher
        self.ranking_config = DeterministicCameraRankingConfig.from_value(
            ranking_config
        )
        if ranking_config_sources is None:
            supplied_keys = (
                set(ranking_config)
                if isinstance(ranking_config, Mapping)
                else (
                    set(self.ranking_config.to_dict())
                    if ranking_config is not None
                    else set()
                )
            )
            source_values = {
                key: (
                    "dependency_injection"
                    if key in supplied_keys
                    else "default"
                )
                for key in self.ranking_config.to_dict()
            }
        else:
            source_values = ranking_config_sources
        unknown_sources = set(source_values) - set(
            self.ranking_config.to_dict()
        )
        if unknown_sources:
            raise ValueError(
                "unknown deterministic ranking source fields: "
                f"{sorted(unknown_sources)}"
            )
        self.ranking_config_sources = {
            key: str(source_values.get(key) or "default")
            for key in self.ranking_config.to_dict()
        }

    def select(
        self,
        request: CameraSelectionRequest,
    ) -> CameraSelectionResult:
        constraints = CameraConstraintSet.from_value(
            request.constraints,
            known_target_ids=_known_target_ids(request),
        )
        unsupported_observations = sorted(
            (
                set(constraints.required_observations)
                | set(constraints.preserved_observations)
            )
            - DETERMINISTIC_SUPPORTED_OBSERVATIONS
        )
        if unsupported_observations:
            candidate_ids = [
                str(candidate.get("id") or "").strip()
                for candidate in request.candidate_views
                if isinstance(candidate, dict)
                and str(candidate.get("id") or "").strip()
            ]
            return camera_selection_result_from_value(
                {
                    "outcome": "no_feasible_candidate",
                    "attempted_candidate_ids": candidate_ids,
                    "rejected_candidates": [
                        {
                            "candidate_id": candidate_id,
                            "reason_codes": [
                                "observation_not_supported_by_deterministic_selector"
                            ],
                            "failed_constraints": unsupported_observations,
                            "features": {
                                "deterministic_capability": "unsupported"
                            },
                        }
                        for candidate_id in candidate_ids
                    ],
                    "reason_codes": [
                        "observation_not_supported_by_deterministic_selector"
                    ],
                    "reason": (
                        "the deterministic selector cannot verify required "
                        "observations: "
                        + ", ".join(unsupported_observations)
                    ),
                    "provenance": {
                        "strategy": "geometry_visibility_diversity_v1",
                        "supported_observations": sorted(
                            DETERMINISTIC_SUPPORTED_OBSERVATIONS
                        ),
                        "unsupported_observations": unsupported_observations,
                        "candidate_generation_skipped": True,
                    },
                },
                request=request,
                backend=self.backend,
            )
        started = perf_counter()
        generation_error: str | None = None
        generation_outcome = "controller_candidate_bank"
        if request.candidate_views:
            candidates = list(deepcopy(request.candidate_views))
            generation_source = "controller_candidate_bank"
        else:
            try:
                candidates = self.candidate_generator(
                    _candidate_generation_request(
                        request,
                        constraints=constraints,
                    ),
                    max_candidates=_positive_int(
                        request.budget.get("candidate_budget", 8),
                        "deterministic candidate_budget",
                    ),
                    policy=self.candidate_policy,
                )
            except NoFeasibleCameraCandidates as exc:
                candidates = []
                generation_error = f"{type(exc).__name__}: {exc}"
                generation_outcome = "no_feasible_candidate"
            except ValueError as exc:
                if not _is_explicit_geometry_exhaustion(exc):
                    raise
                candidates = []
                generation_error = f"{type(exc).__name__}: {exc}"
                generation_outcome = "no_feasible_candidate"
            else:
                if not isinstance(candidates, list) or not all(
                    isinstance(candidate, dict) for candidate in candidates
                ):
                    raise TypeError(
                        "deterministic candidate_generator must return a list "
                        "of candidate mappings"
                    )
                generation_outcome = (
                    "generated" if candidates else "empty_candidate_bank"
                )
            generation_source = (
                f"{self.candidate_generator.__module__}."
                f"{self.candidate_generator.__qualname__}"
            )
        generation_seconds = max(0.0, perf_counter() - started)
        candidates = candidates[
            : _positive_int(
                request.budget.get("candidate_budget", 8),
                "deterministic candidate_budget",
            )
        ]
        validation_request = replace(
            request,
            candidate_views=tuple(deepcopy(candidates)),
        )

        attempted_before = {
            str(value)
            for value in request.context.get("attempted_view_ids", [])
            if str(value).strip()
        }
        rejected: list[dict[str, Any]] = []
        ranked: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
        examined_ids: list[str] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("id") or "").strip()
            if not candidate_id:
                raise ValueError(
                    "deterministic candidate requires a non-empty id"
                )
            if candidate_id in examined_ids:
                raise ValueError(
                    "deterministic candidate ids must be unique"
                )
            examined_ids.append(candidate_id)
            features = _candidate_features(
                candidate,
                request=request,
                constraints=constraints,
            )
            if self.feature_enricher is not None:
                enriched = self.feature_enricher(
                    deepcopy(candidate),
                    request,
                )
                if not isinstance(enriched, dict):
                    raise ValueError(
                        "deterministic feature_enricher must return a mapping"
                    )
                features.update(deepcopy(enriched))
            reasons = _candidate_rejection_reasons(
                candidate,
                constraints=constraints,
                attempted_before=attempted_before,
                features=features,
            )
            if reasons:
                rejected.append(
                    {
                        "candidate_id": candidate_id,
                        "reason_codes": reasons,
                        "failed_constraints": _failed_constraints(
                            reasons,
                            constraints=constraints,
                        ),
                        "features": features,
                    }
                )
                continue
            score = _candidate_score(
                features,
                constraints=constraints,
                ranking=self.ranking_config,
            )
            ranked.append(
                (
                    -score,
                    candidate_id,
                    candidate,
                    {**features, "ranking_score": score},
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1]))
        limit = min(
            _positive_int(
                request.budget.get("max_views_per_round"),
                "deterministic max_views_per_round",
            ),
            len(ranked),
        )
        selected_records = ranked[:limit]
        provenance = {
            "strategy": "geometry_visibility_diversity_v1",
            "ranking_parameters": self.ranking_config.to_dict(),
            "ranking_parameter_sources": deepcopy(
                self.ranking_config_sources
            ),
            "candidate_generation_source": generation_source,
            "candidate_policy": self.candidate_policy,
            "candidate_generation_time_seconds": generation_seconds,
            "candidate_count": len(candidates),
            "filtered_candidate_count": len(ranked),
            "candidate_features": {
                candidate_id: features
                for _, candidate_id, _, features in ranked
            },
            "generation_error": generation_error,
            "generation_outcome": generation_outcome,
            "reuses_existing_camera_algorithms": [
                "generate_camera_pose_candidates",
                "metric_specific_camera_pose_mode",
                "local_geometry_and_proxy_framing",
            ],
        }
        if not selected_records:
            codes = _no_feasible_reason_codes(
                rejected,
                generation_error=generation_error,
                had_candidates=bool(candidates),
            )
            return camera_selection_result_from_value(
                {
                    "outcome": "no_feasible_candidate",
                    "attempted_candidate_ids": examined_ids,
                    "rejected_candidates": rejected,
                    "reason_codes": codes,
                    "reason": (
                        "no unattempted camera candidate satisfies the "
                        "technical constraints"
                    ),
                    "provenance": provenance,
                },
                request=validation_request,
                backend=self.backend,
            )

        selected = [
            deepcopy(candidate)
            for _, _, candidate, _ in selected_records
        ]
        selected_ids = [
            candidate_id
            for _, candidate_id, _, _ in selected_records
        ]
        return camera_selection_result_from_value(
            {
                "outcome": "selected",
                "selected_view_ids": selected_ids,
                "selected_views": selected,
                "attempted_candidate_ids": examined_ids,
                "rejected_candidates": rejected,
                "reason_codes": ["geometry_ranked_selection"],
                "reason": (
                    "selected the highest-ranked feasible, unattempted "
                    "camera candidates"
                ),
                "provenance": provenance,
            },
            request=validation_request,
            backend=self.backend,
        )


class DeterministicCameraRepairSolver:
    """Realize one trusted repair plan with the local deterministic selector."""

    def __init__(
        self,
        selector: DeterministicLocalCameraSelector | None = None,
    ) -> None:
        self.selector = selector or DeterministicLocalCameraSelector()

    def realize(
        self,
        request: CameraSelectionRequest,
        plan: CameraRepairPlan,
    ) -> CameraSelectionResult:
        constraints = CameraConstraintSet.from_value(
            request.constraints,
            known_target_ids=_known_target_ids(request),
        )
        plan.validate_against(constraints)
        relaxed = set(plan.relaxed_constraints)
        required = tuple(
            value
            for value in constraints.required_observations
            if value not in relaxed
        )
        preserved = tuple(
            value
            for value in constraints.preserved_observations
            if value not in relaxed
        )
        delegated_observations = tuple(
            value
            for value in (*required, *preserved)
            if value not in DETERMINISTIC_SUPPORTED_OBSERVATIONS
        )
        solver_required = tuple(
            value
            for value in required
            if value in DETERMINISTIC_SUPPORTED_OBSERVATIONS
        )
        if not solver_required:
            # The VLM selected only a trusted repair-plan objective. Geometry
            # realization still needs a technically checkable framing target;
            # the post-render Judge, not this selector, verifies the delegated
            # semantic observation.
            solver_required = ("target_visible",)
        solver_preserved = tuple(
            value
            for value in preserved
            if value in DETERMINISTIC_SUPPORTED_OBSERVATIONS
            and value not in solver_required
        )
        realized_constraints = CameraConstraintSet(
            target_ids=constraints.target_ids,
            required_observations=solver_required,
            preserved_observations=solver_preserved,
            preferred_view_families=(
                plan.preferred_view_families
                or constraints.preferred_view_families
            ),
            forbidden_view_families=(
                constraints.forbidden_view_families
            ),
            min_projected_coverage=(
                None
                if "min_projected_coverage" in relaxed
                else constraints.min_projected_coverage
            ),
            require_joint_visibility=(
                constraints.require_joint_visibility
                and "require_joint_visibility" not in relaxed
                and "joint_visibility" not in relaxed
                and "joint_visibility" in solver_required
            ),
            require_global_anchor=(
                False
            ),
            relaxable_constraints=(),
            metric=constraints.metric,
            view_goal=constraints.view_goal,
            metadata={
                **deepcopy(constraints.metadata),
                "selected_repair_plan": plan.to_dict(),
                "realization": "deterministic_local_camera_selector",
                "original_required_observations": list(required),
                "unverified_observations_requiring_post_render_judge": list(
                    delegated_observations
                ),
            },
        )
        budget = deepcopy(request.budget)
        budget["max_views_per_round"] = min(
            int(budget["max_views_per_round"]),
            plan.required_view_count,
        )
        return self.selector.select(
            replace(
                request,
                constraints=realized_constraints.to_dict(),
                budget=budget,
                context={
                    **deepcopy(request.context),
                    "selected_repair_plan_id": plan.plan_id,
                },
            )
        )


def _candidate_generation_request(
    request: CameraSelectionRequest,
    *,
    constraints: CameraConstraintSet,
) -> dict[str, Any]:
    try:
        resolved_mode = resolve_camera_pose_mode(
            "auto",
            request.metric,
        )
    except ValueError:
        resolved_mode = "visibility_ranked"
    group_scope = request.context.get("group_scope")
    event = deepcopy(request.context.get("event") or {})
    if isinstance(group_scope, dict):
        event.setdefault("group_id", group_scope.get("group_id"))
        event.setdefault(
            "focus_region",
            deepcopy(group_scope.get("target_bounds")),
        )
        event.setdefault(
            "object_ids",
            list(group_scope.get("member_ids") or constraints.target_ids),
        )
    return {
        "metric": request.metric,
        "scene": deepcopy(request.scene),
        "object_ids": list(constraints.target_ids),
        "target_ids": list(constraints.target_ids),
        "group_scope": (
            deepcopy(group_scope)
            if isinstance(group_scope, dict)
            else None
        ),
        "target_bounds": deepcopy(
            request.context.get("target_bounds")
        ),
        "focus_center": deepcopy(
            request.context.get("focus_center")
        ),
        "target_extent": deepcopy(
            request.context.get("target_extent")
        ),
        "detector_evidence": deepcopy(
            request.context.get("detector_evidence") or {}
        ),
        "event": event,
        "_resolved_camera_pose_mode": resolved_mode,
        "_camera_render": deepcopy(
            request.context.get("camera_render") or {}
        ),
    }


def _candidate_rejection_reasons(
    candidate: dict[str, Any],
    *,
    constraints: CameraConstraintSet,
    attempted_before: set[str],
    features: dict[str, Any],
) -> list[str]:
    candidate_id = str(candidate.get("id") or "")
    reasons: list[str] = []
    if candidate_id in attempted_before:
        reasons.append("candidate_already_attempted")
    if features.get("camera_pose_verifiable") is not True:
        reasons.append("camera_pose_unverifiable")
    if features.get("geometry_feasibility_verified") is not True:
        reasons.append("geometry_feasibility_unverified")
    elif features.get("geometry_feasibility") is False:
        reasons.append("geometry_infeasible")
    framing = candidate.get("proxy_framing")
    if (
        isinstance(framing, dict)
        and framing.get("proxy_bounds_fit") is False
        and _requires_target_visibility(constraints)
    ):
        reasons.append("target_proxy_out_of_frame")
    family = _view_family(candidate)
    if (
        constraints.forbidden_view_families
        and family in constraints.forbidden_view_families
    ):
        reasons.append("forbidden_view_family")
    if (
        constraints.preferred_view_families
        and candidate.get("strict_view_family") is True
        and family not in constraints.preferred_view_families
    ):
        reasons.append("required_view_family_missing")
    if _requires_target_visibility(constraints):
        target_visibility = features.get("target_visibility_estimate")
        if target_visibility is None:
            reasons.append("target_visibility_unverified")
        elif target_visibility is False:
            reasons.append("target_not_visible")
        coverage = features.get("projected_coverage_estimate")
        if coverage is None:
            reasons.append("projected_coverage_unverified")
        elif (
            constraints.min_projected_coverage is not None
            and coverage < constraints.min_projected_coverage
        ):
            reasons.append("projected_coverage_insufficient")
    if _requires_joint_visibility(constraints):
        joint_visibility = features.get("joint_visibility_estimate")
        if joint_visibility is None:
            reasons.append("joint_visibility_unverified")
        elif joint_visibility is False:
            reasons.append("joint_visibility_unavailable")
    return list(dict.fromkeys(reasons))


def _candidate_features(
    candidate: dict[str, Any],
    *,
    request: CameraSelectionRequest,
    constraints: CameraConstraintSet,
) -> dict[str, Any]:
    visibility = _visibility(candidate)
    coverage, coverage_source = _coverage_estimate(
        candidate,
        visibility=visibility,
    )
    target_ids = {
        str(value)
        for value in (
            candidate.get("target_object_ids")
            or candidate.get("target_ids")
            or []
        )
    }
    required_targets = set(constraints.target_ids)
    proxy_framing = candidate.get("proxy_framing")
    proxy_fit = (
        proxy_framing.get("proxy_bounds_fit")
        if isinstance(proxy_framing, dict)
        else None
    )
    target_visibility = _boolean_measurement(
        visibility,
        candidate,
        keys=("target_visible",),
    )
    if (
        target_visibility is None
        and required_targets
        and required_targets <= target_ids
        and proxy_fit is True
    ):
        target_visibility = True
    joint_visibility = _boolean_measurement(
        visibility,
        candidate,
        keys=("jointly_visible", "joint_visibility"),
    )
    if (
        joint_visibility is None
        and len(required_targets) > 1
        and required_targets <= target_ids
        and proxy_fit is True
    ):
        joint_visibility = True
    feasibility_verified, geometry_feasibility = _geometry_feasibility(
        candidate
    )
    return {
        "camera_pose_verifiable": _camera_pose_verifiable(candidate),
        "geometry_feasibility_verified": feasibility_verified,
        "geometry_feasibility": geometry_feasibility,
        "target_visibility_estimate": target_visibility,
        "joint_visibility_estimate": joint_visibility,
        "projected_coverage_estimate": coverage,
        "projected_coverage_source": coverage_source,
        "contact_support_boundary_cue_estimate": (
            1.0
            if candidate.get("support_contact_focus")
            or candidate.get("focus_plane_flag")
            else 0.0
        ),
        "distance_from_previous_camera_poses": (
            _pose_diversity(candidate, request.existing_visual_evidence)
        ),
        "view_family": _view_family(candidate),
        "view_family_match": (
            not constraints.preferred_view_families
            or _view_family(candidate)
            in constraints.preferred_view_families
        ),
        "predicted_occlusion": visibility.get(
            "predicted_occlusion",
            "unknown",
        ),
        "proxy_bounds_fit": (
            proxy_fit
        ),
    }


def _candidate_score(
    features: dict[str, Any],
    *,
    constraints: CameraConstraintSet,
    ranking: DeterministicCameraRankingConfig,
) -> float:
    score = 0.0
    score += (
        ranking.geometry_feasible_bonus
        if features["geometry_feasibility"] is True
        and features["geometry_feasibility_verified"] is True
        else ranking.geometry_unverified_penalty
    )
    score += (
        ranking.target_visibility_bonus
        if features["target_visibility_estimate"] is True
        else 0.0
    )
    if _requires_joint_visibility(constraints):
        score += (
            ranking.joint_visibility_bonus
            if features["joint_visibility_estimate"] is True
            else 0.0
        )
    coverage = features["projected_coverage_estimate"]
    if isinstance(coverage, (int, float)):
        score += (
            min(max(float(coverage), 0.0), 1.0)
            * ranking.projected_coverage_weight
        )
    score += (
        float(features["contact_support_boundary_cue_estimate"])
        * ranking.contact_cue_weight
    )
    score += min(
        float(features["distance_from_previous_camera_poses"]),
        ranking.pose_diversity_cap,
    ) * ranking.pose_diversity_weight
    if features["view_family_match"]:
        score += ranking.view_family_match_bonus
    if features["predicted_occlusion"] is False:
        score += ranking.unoccluded_bonus
    return score


def _coverage_estimate(
    candidate: dict[str, Any],
    *,
    visibility: dict[str, Any],
) -> tuple[float | None, str | None]:
    for source in (
        visibility,
        candidate.get("proxy_framing"),
        candidate,
    ):
        if not isinstance(source, dict):
            continue
        for key in (
            "projected_coverage",
            "projected_coverage_estimate",
            "target_pixel_fraction",
        ):
            value = source.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                return float(value), key
    framing = candidate.get("proxy_framing")
    if isinstance(framing, dict):
        x = framing.get("max_abs_ndc_x")
        y = framing.get("max_abs_ndc_y")
        if (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and math.isfinite(float(x))
            and float(x) >= 0.0
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and math.isfinite(float(y))
            and float(y) >= 0.0
            and framing.get("all_corners_in_front") is not False
        ):
            return (
                min(1.0, max(0.0, float(x) * float(y))),
                "proxy_framing_ndc_bounds",
            )
    return None, None


def _failed_constraints(
    reasons: list[str],
    *,
    constraints: CameraConstraintSet,
) -> list[str]:
    """Map technical rejection evidence onto active, relaxable DSL references."""

    active = active_constraint_references(constraints)
    failed: list[str] = []

    def add(reference: str) -> None:
        if reference in active and reference not in failed:
            failed.append(reference)

    reason_set = set(reasons)
    if reason_set & {
        "target_not_visible",
        "target_visibility_unverified",
        "target_proxy_out_of_frame",
    }:
        add("target_visible")
    if reason_set & {
        "projected_coverage_insufficient",
        "projected_coverage_unverified",
    }:
        if "min_projected_coverage" in active:
            add("min_projected_coverage")
        else:
            add("target_visible")
    if reason_set & {
        "joint_visibility_unavailable",
        "joint_visibility_unverified",
    }:
        add("joint_visibility")
    return failed


def _requires_target_visibility(
    constraints: CameraConstraintSet,
) -> bool:
    return any(target_id != "scene" for target_id in constraints.target_ids)


def _requires_joint_visibility(
    constraints: CameraConstraintSet,
) -> bool:
    return constraints.require_joint_visibility or "joint_visibility" in {
        *constraints.required_observations,
        *constraints.preserved_observations,
    }


def _camera_pose_verifiable(candidate: dict[str, Any]) -> bool:
    pose = candidate.get("pose")
    source = pose if isinstance(pose, dict) else candidate
    location = _vector3(source.get("location"))
    target = _vector3(source.get("target"))
    lens = source.get("lens_mm")
    if location is None or target is None or location == target:
        return False
    return (
        isinstance(lens, (int, float))
        and not isinstance(lens, bool)
        and math.isfinite(float(lens))
        and float(lens) > 0.0
    )


def _geometry_feasibility(
    candidate: dict[str, Any],
) -> tuple[bool, bool | None]:
    explicit = candidate.get("feasible")
    if isinstance(explicit, bool):
        return True, explicit
    details = candidate.get("feasibility")
    if not isinstance(details, dict):
        return False, None
    explicit = details.get("feasible")
    if isinstance(explicit, bool):
        return True, explicit
    if str(candidate.get("candidate_policy") or "") != "local":
        return False, None
    interval = details.get("feasible_distance_interval_m")
    actual = details.get("actual_distance_m")
    if (
        not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in interval
        )
        or float(interval[0]) > float(interval[1])
        or isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not float(interval[0]) <= float(actual) <= float(interval[1])
    ):
        return False, None
    return True, True


def _boolean_measurement(
    *sources: dict[str, Any],
    keys: tuple[str, ...],
) -> bool | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                return value
    return None


def _visibility(candidate: dict[str, Any]) -> dict[str, Any]:
    value = candidate.get("visibility")
    return value if isinstance(value, dict) else {}


def _view_family(candidate: dict[str, Any]) -> str:
    explicit = (
        candidate.get("view_family")
        or candidate.get("family")
        or candidate.get("focus_kind")
    )
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    candidate_id = str(candidate.get("id") or "").lower()
    if "contact" in candidate_id or candidate.get("support_contact_focus"):
        return "contact_plane_oblique"
    if "top" in candidate_id:
        return "global_top"
    if "global" in candidate_id:
        return "global_perspective"
    if "side" in candidate_id:
        return "separation_side"
    return "metric_local"


def _pose_diversity(
    candidate: dict[str, Any],
    evidence: tuple[Any, ...],
) -> float:
    location = _vector3(candidate.get("location"))
    if location is None:
        return 0.0
    distances: list[float] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        pose = item.get("pose")
        source = pose if isinstance(pose, dict) else item
        other = _vector3(source.get("location"))
        if other is None:
            continue
        distances.append(
            math.sqrt(
                sum((first - second) ** 2 for first, second in zip(location, other))
            )
        )
    return min(distances) if distances else 1.0


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _known_target_ids(
    request: CameraSelectionRequest,
) -> tuple[str, ...]:
    raw = request.context.get("known_target_ids")
    if raw is not None and not isinstance(raw, (list, tuple)):
        raise ValueError("known_target_ids must be a JSON list")
    values = [
        str(value)
        for value in (raw or ())
        if str(value).strip()
    ]
    return merge_authoritative_target_ids(values, request.scene)


def _no_feasible_reason_codes(
    rejected: list[dict[str, Any]],
    *,
    generation_error: str | None,
    had_candidates: bool,
) -> list[str]:
    codes = {
        str(code)
        for item in rejected
        for code in item.get("reason_codes", [])
    }
    if generation_error:
        codes.add("no_feasible_candidate")
        codes.add("candidate_generation_infeasible")
    elif not had_candidates:
        codes.add("no_feasible_candidate")
        codes.add("candidate_bank_empty")
    elif codes == {"candidate_already_attempted"}:
        codes.add("candidate_ranking_exhausted")
    else:
        codes.add("no_feasible_candidate")
    return sorted(codes)


def _is_explicit_geometry_exhaustion(error: ValueError) -> bool:
    """Recognize the frozen generator's documented normal exhaustion signal."""

    message = str(error)
    return (
        message == "camera candidate generation produced no valid poses"
        or message.startswith(
            "feasible camera candidate generation could not satisfy the exact "
            "bank size:"
        )
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
