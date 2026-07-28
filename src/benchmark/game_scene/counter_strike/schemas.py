"""Frozen JSON Schemas for the Counter-Strike benchmark declarations.

The benchmark profile is authored as YAML and each corpus case contract is
authored as JSON, but both are validated with JSON Schema after parsing.  The
schemas live with the loader so the package has no runtime dependency on a
repository-relative schema path.
"""

from __future__ import annotations

from typing import Any


def _closed_object(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(required or properties),
        "properties": properties,
    }


def _number(
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    return schema


def _integer(
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "minimum": minimum}
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _weight() -> dict[str, Any]:
    return _number(minimum=0.0, maximum=1.0)


def _weighted_metric(properties: dict[str, Any]) -> dict[str, Any]:
    return _closed_object({"weight": _weight(), **properties})


COUNTER_STRIKE_BENCHMARK_CONFIG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:layout-ddd:counter-strike-benchmark-config:v1",
    **_closed_object(
        {
            "profile_version": {
                "const": "counter_strike_static_spatial_benchmark_v1"
            },
            "status": {"const": "frozen"},
            "scene_family": {"const": "counter_strike_static_arena"},
            "scope": {"const": "static_3d_environment_only"},
            "composite": _closed_object(
                {
                    "require_complete_coverage": {"const": True},
                    "layer_weights": _closed_object(
                        {
                            "l1_physical_plausibility": _weight(),
                            "l3_scene_quality": _weight(),
                            "l4_static_spatial_design": _weight(),
                        }
                    ),
                    "canonical_metric_weights": _closed_object(
                        {
                            "collision": _weight(),
                            "navigability": _weight(),
                            "style_consistency": _weight(),
                        }
                    ),
                }
            ),
            "player_profile": _closed_object(
                {
                    "agent_radius_m": _number(exclusive_minimum=0.0),
                    "standing_height_m": _number(exclusive_minimum=0.0),
                    "standing_eye_height_m": _number(exclusive_minimum=0.0),
                    "crouching_eye_height_m": _number(exclusive_minimum=0.0),
                    "step_over_height_m": _number(minimum=0.0),
                    "run_speed_mps": _number(exclusive_minimum=0.0),
                }
            ),
            "static_world": _closed_object(
                {
                    "grid_resolution_m": _number(exclusive_minimum=0.0),
                    "max_grid_cells": _integer(minimum=1),
                    "connectivity": {"enum": [4, 8]},
                    "boundary_source": {
                        "const": "non_flat_structure_envelope"
                    },
                    "spawn_snap_radius_m": _number(minimum=0.0),
                    "perimeter_band_m": _number(minimum=0.0),
                }
            ),
            "visual_evidence": _closed_object(
                {
                    "global_view_family": {
                        "const": "canonical_high_oblique_pair_v1"
                    },
                    "global_image_budget": _integer(minimum=1),
                    "regional_view_family": {
                        "const": "canonical_style_region_quadrants_v1"
                    },
                    "regional_image_budget": _integer(minimum=1),
                    "active_fallback": _closed_object(
                        {
                            "enabled": {"type": "boolean"},
                            "implementation": {
                                "const": "frozen_browser_regional_bank"
                            },
                            "selector_judge_decoupled": {"const": True},
                            "max_extra_views": _integer(minimum=0),
                            "exhaustion_status": {"const": "unresolved"},
                        }
                    ),
                    "judge": _closed_object(
                        {
                            "repeats": _integer(minimum=1),
                            "require_repeat_agreement": {"type": "boolean"},
                            "confidence_is_not_calibration": {"const": True},
                        }
                    ),
                }
            ),
            "l4_metrics": _closed_object(
                {
                    "zone_clarity": _weighted_metric(
                        {
                            "backend": {
                                "const": "deterministic_topology_plus_vlm"
                            },
                            "required_roles": {
                                "const": [
                                    "team_a_spawn",
                                    "team_b_spawn",
                                    "preparation",
                                    "main_engagement",
                                    "flank",
                                ]
                            },
                            "score_components": _closed_object(
                                {
                                    "deterministic": _weight(),
                                    "perceptual": _weight(),
                                }
                            ),
                            "valid_threshold": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "min_clear_roles": _integer(minimum=0, maximum=5),
                        }
                    ),
                    "route_structure": _weighted_metric(
                        {
                            "backend": {"const": "deterministic"},
                            "candidate_path_budget": _integer(minimum=1),
                            "required_main_routes": _integer(minimum=1),
                            "required_flank_routes": _integer(minimum=0),
                            "max_main_detour_ratio": _number(
                                exclusive_minimum=0.0
                            ),
                            "min_flank_detour_ratio": _number(
                                exclusive_minimum=0.0
                            ),
                            "max_flank_detour_ratio": _number(
                                exclusive_minimum=0.0
                            ),
                            "max_route_overlap": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "route_separation_m": _number(minimum=0.0),
                        }
                    ),
                    "spawn_balance": _weighted_metric(
                        {
                            "backend": {"const": "deterministic"},
                            "engagement_selection": {
                                "const": (
                                    "maximin_traffic_distance_balance"
                                )
                            },
                            "travel_ratio_tolerance": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "exposure_ratio_tolerance": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "contact_distance_ratio_tolerance": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "initial_cross_spawn_los_invalid": {
                                "type": "boolean"
                            },
                            "exposure_ray_count": _integer(minimum=1),
                            "exposure_ray_length_m": _number(
                                exclusive_minimum=0.0
                            ),
                        }
                    ),
                    "landmark_legibility": _weighted_metric(
                        {
                            "backend": {"const": "vlm_global_then_regional"},
                            "required_landmarks": _integer(minimum=1),
                            "require_nameable_visual_cue": {"type": "boolean"},
                            "require_spatially_distinct_regions": {
                                "type": "boolean"
                            },
                            "valid_threshold": _number(
                                minimum=0.0, maximum=1.0
                            ),
                        }
                    ),
                    "cover_diversity": _weighted_metric(
                        {
                            "backend": {
                                "const": (
                                    "deterministic_cover_assemblies_plus_vlm"
                                )
                            },
                            "score_components": _closed_object(
                                {
                                    "deterministic": _weight(),
                                    "perceptual": _weight(),
                                }
                            ),
                            "valid_threshold": _number(
                                minimum=0.0, maximum=1.0
                            ),
                            "min_cover_assemblies": _integer(minimum=1),
                            "min_cover_forms": _integer(minimum=1),
                            "min_height_bands": _integer(minimum=1),
                            "min_width_bands": _integer(minimum=1),
                            "min_arrangement_types": _integer(minimum=1),
                            "adjacency_distance_m": _number(
                                exclusive_minimum=0.0
                            ),
                            "minimum_component_height_m": _number(
                                minimum=0.0
                            ),
                            "minimum_occlusion_height_m": _number(
                                exclusive_minimum=0.0
                            ),
                            "maximum_base_offset_m": _number(minimum=0.0),
                            "assembly_merge_distance_m": _number(minimum=0.0),
                            "assembly_vertical_gap_m": _number(minimum=0.0),
                            "stepped_height_delta_m": _number(minimum=0.0),
                            "arrangement_neighbor_distance_m": _number(
                                exclusive_minimum=0.0
                            ),
                            "duplicate_position_tolerance_m": _number(
                                minimum=0.0
                            ),
                            "duplicate_scale_ratio_tolerance": _number(
                                minimum=0.0, maximum=1.0
                            ),
                        }
                    ),
                }
            ),
            "failure_semantics": _closed_object(
                {
                    "invalid_contract": {"const": "case_invalid"},
                    "missing_spawn_metadata": {"const": "case_invalid"},
                    "dynamic_entity_unresolved": {"const": "unresolved"},
                    "insufficient_visual_evidence": {"const": "unresolved"},
                    "judge_disagreement": {"const": "unresolved"},
                    "algorithm_failure": {"const": "metric_failed"},
                    "computed_design_failure": {
                        "const": "evaluated_zero_allowed"
                    },
                }
            ),
        }
    ),
}


_SPAWN_POINT_SCHEMA = {
    "type": "array",
    "minItems": 3,
    "maxItems": 3,
    "prefixItems": [_number(), _number(), _number()],
    "items": False,
}

_TEAM_SPAWN_SCHEMA = _closed_object(
    {
        "points": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": _SPAWN_POINT_SCHEMA,
        },
        "jitter_radius_m": _number(minimum=0.0),
    }
)

COUNTER_STRIKE_CASE_CONTRACT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:layout-ddd:counter-strike-case-contract:v1",
    **_closed_object(
        {
            "contract_version": {"const": "counter_strike_case_contract_v1"},
            "case_id": {
                "type": "string",
                "minLength": 1,
                "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]*$",
            },
            "corpus_id": {"type": "string", "minLength": 1},
            "source_frame": _closed_object(
                {
                    "up_axis": {"const": "y"},
                    "unit_scale": _number(exclusive_minimum=0.0),
                }
            ),
            "source_assertions": {
                "type": "array",
                "minItems": 1,
                "items": _closed_object(
                    {
                        "path": {
                            "type": "string",
                            "minLength": 1,
                            "pattern": "^(?!/)(?!.*(?:^|/)\\.\\.(?:/|$)).+$",
                        },
                        "sha256": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "evidence": {"type": "string", "minLength": 1},
                    }
                ),
            },
            "team_spawns": _closed_object(
                {
                    "team_a": _TEAM_SPAWN_SCHEMA,
                    "team_b": _TEAM_SPAWN_SCHEMA,
                }
            ),
            "annotation": _closed_object(
                {
                    "source": {
                        "const": "benchmark_audited_source_declaration"
                    },
                    "score_authority": {"const": False},
                }
            ),
        }
    ),
}
