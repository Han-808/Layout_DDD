"""Strict configuration contract for the browser-Game ingestion route.

``game_mode`` is not a second evaluator.  It owns only the path from a browser
game source to frozen canonical artifacts, then references the same canonical
L0--L4 profile consumed by every other submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.evaluator.profile import L1, L2, L3, L4, resolve_evaluation_profile
from benchmark.utils.io import load_yaml


GAME_MODE_VERSION = "game_mode_v1"
GAME_MODE = "game"
GAME_WORKFLOW = "canonical_l0_l4"
GAME_INPUT_RUNTIME = "threejs"
GAME_INGESTION_ADAPTER = "game_scene_probe_v1"
GAME_CONTROLLED_RENDER_CONTRACT = "threejs_direct_webgl_renderer_v1"
GAME_OBJECTIZATION_STRATEGY = "visible_mesh_then_geometric_collapse"
GAME_CASE_BUNDLE_BUILDER = "canonical_game_case_bundle_v1"
GAME_SCENE_FAMILY = "counter_strike_static_arena"
GAME_ACTIVE_METRICS = frozenset(
    {"collision", "navigability", "style_consistency"}
)


class GameModeConfigError(ValueError):
    """Raised when a Game route config is incomplete or claims unsupported behavior."""


@dataclass(frozen=True)
class GameModeConfig:
    """Resolved and validated Game route configuration."""

    path: Path
    raw: dict[str, Any]
    evaluation_profile_path: Path
    evaluation_profile: dict[str, Any]

    @property
    def entry_html(self) -> str:
        return str(self.raw["input"]["entry_html"])

    @property
    def renderer_kwargs(self) -> dict[str, Any]:
        capture = self.raw["capture"]
        geometry = self.raw["geometry"]
        objectization = self.raw["objectization"]
        return {
            "width": int(capture["width"]),
            "height": int(capture["height"]),
            "seed": int(capture["seed"]),
            "step_ms": float(capture["step_ms"]),
            "warmup_frames": int(capture["warmup_frames"]),
            "views": tuple(dict(item) for item in capture["states"]),
            "controlled_camera": dict(capture["controlled_camera"]),
            "max_vertices_per_object": int(geometry["max_vertices_per_object"]),
            "unit_scale": float(geometry["unit_scale"]),
            "exclude_camera_descendants": bool(
                objectization["exclude_camera_descendants"]
            ),
            "drop_non_physical_meshes": bool(objectization["drop_non_physical_meshes"]),
            "collapse_contained_meshes": bool(
                objectization["collapse_contained_meshes"]
            ),
            "timeout_seconds": int(capture["timeout_seconds"]),
        }

    @property
    def instruction(self) -> str:
        return str(self.raw["evaluation"]["instruction"])

    @property
    def default_visual_style_spec_path(self) -> Path:
        raw = str(self.raw["evaluation"]["default_visual_style_spec"])
        return (self.path.parent / raw).resolve()


def load_game_mode_config(path: str | Path) -> GameModeConfig:
    """Load the checked-in route config and reject inert or unsupported fields."""

    config_path = Path(path).expanduser().resolve()
    raw = load_yaml(config_path)
    if not isinstance(raw, dict):
        raise GameModeConfigError("game_mode config must be a YAML object")
    _require_exact_keys(
        raw,
        {
            "route_version",
            "mode",
            "workflow",
            "input",
            "ingestion",
            "capture",
            "objectization",
            "geometry",
            "evaluation",
            "failure_semantics",
        },
        "game_mode",
    )
    _require_value(raw, "route_version", GAME_MODE_VERSION, "game_mode")
    _require_value(raw, "mode", GAME_MODE, "game_mode")
    _require_value(raw, "workflow", GAME_WORKFLOW, "game_mode")
    _validate_input(raw["input"])
    _validate_ingestion(raw["ingestion"])
    _validate_capture(raw["capture"])
    _validate_objectization(raw["objectization"])
    _validate_geometry(raw["geometry"])
    profile_path = _validate_evaluation(raw["evaluation"], config_path)
    _validate_failure_semantics(raw["failure_semantics"])

    profile = load_yaml(profile_path)
    if not isinstance(profile, dict):
        raise GameModeConfigError(
            f"game_mode evaluation profile is not a YAML object: {profile_path}"
        )
    resolved = resolve_evaluation_profile(profile)
    if resolved != profile:
        raise GameModeConfigError(
            "game_mode evaluation profile must be fully resolved and frozen"
        )
    active = {
        name
        for layer in (L1, L2, L3, L4)
        for name, metric in profile[layer].get("metrics", {}).items()
        if isinstance(metric, dict) and metric.get("enabled") is True
    }
    if active != GAME_ACTIVE_METRICS:
        raise GameModeConfigError(
            "game_mode profile must enable exactly collision, navigability, "
            "and style_consistency; "
            f"got {sorted(active)}"
        )
    return GameModeConfig(
        path=config_path,
        raw=raw,
        evaluation_profile_path=profile_path,
        evaluation_profile=profile,
    )


def _validate_input(value: Any) -> None:
    data = _mapping(value, "game_mode.input")
    _require_exact_keys(
        data,
        {"kind", "runtime", "entry_html", "unsupported_runtime_policy"},
        "game_mode.input",
    )
    _require_value(data, "kind", "browser_game", "game_mode.input")
    _require_value(data, "runtime", GAME_INPUT_RUNTIME, "game_mode.input")
    _require_value(
        data,
        "unsupported_runtime_policy",
        "reject_not_ingestable",
        "game_mode.input",
    )
    _non_empty_string(data["entry_html"], "game_mode.input.entry_html")


def _validate_ingestion(value: Any) -> None:
    data = _mapping(value, "game_mode.ingestion")
    _require_exact_keys(
        data,
        {
            "adapter",
            "instrumentation",
            "scene_selection",
            "controlled_render_contract",
            "serve_over_loopback_http",
            "external_three_policy",
        },
        "game_mode.ingestion",
    )
    _require_value(data, "adapter", GAME_INGESTION_ADAPTER, "game_mode.ingestion")
    _require_value(
        data,
        "instrumentation",
        "load_time_scene_constructor_hook",
        "game_mode.ingestion",
    )
    _require_value(
        data,
        "scene_selection",
        "richest_registered_scene",
        "game_mode.ingestion",
    )
    _require_value(
        data,
        "controlled_render_contract",
        GAME_CONTROLLED_RENDER_CONTRACT,
        "game_mode.ingestion",
    )
    _require_true(data["serve_over_loopback_http"], "game_mode.ingestion.serve_over_loopback_http")
    _require_value(
        data,
        "external_three_policy",
        "exact_version_replacement_or_original",
        "game_mode.ingestion",
    )


def _validate_capture(value: Any) -> None:
    data = _mapping(value, "game_mode.capture")
    _require_exact_keys(
        data,
        {
            "width",
            "height",
            "seed",
            "step_ms",
            "warmup_frames",
            "timeout_seconds",
            "states",
            "controlled_camera",
        },
        "game_mode.capture",
    )
    _bounded_int(data["width"], "game_mode.capture.width", minimum=64, maximum=4096)
    _bounded_int(data["height"], "game_mode.capture.height", minimum=64, maximum=4096)
    _bounded_int(data["seed"], "game_mode.capture.seed", minimum=1, maximum=2**32 - 1)
    _positive_number(data["step_ms"], "game_mode.capture.step_ms")
    _bounded_int(
        data["warmup_frames"],
        "game_mode.capture.warmup_frames",
        minimum=0,
        maximum=100000,
    )
    _bounded_int(
        data["timeout_seconds"],
        "game_mode.capture.timeout_seconds",
        minimum=1,
        maximum=3600,
    )
    states = data["states"]
    if not isinstance(states, list) or not states:
        raise GameModeConfigError("game_mode.capture.states must be a non-empty list")
    names: set[str] = set()
    for index, state in enumerate(states):
        path = f"game_mode.capture.states[{index}]"
        item = _mapping(state, path)
        _require_exact_keys(item, {"name", "step_frames"}, path)
        name = _non_empty_string(item["name"], f"{path}.name")
        if name in names:
            raise GameModeConfigError(f"{path}.name duplicates {name!r}")
        names.add(name)
        _bounded_int(item["step_frames"], f"{path}.step_frames", minimum=0, maximum=100000)
    controlled = _mapping(
        data["controlled_camera"],
        "game_mode.capture.controlled_camera",
    )
    _require_exact_keys(
        controlled,
        {
            "enabled",
            "required",
            "view_family",
            "image_budget",
            "style_local_fallback_enabled",
            "style_local_view_family",
            "style_local_image_budget",
            "canvas_only",
            "include_authored_camera_diagnostics",
            "unsupported_render_pipeline",
        },
        "game_mode.capture.controlled_camera",
    )
    for key in (
        "enabled",
        "required",
        "canvas_only",
        "include_authored_camera_diagnostics",
        "style_local_fallback_enabled",
    ):
        _require_true(
            controlled[key],
            f"game_mode.capture.controlled_camera.{key}",
        )
    _require_value(
        controlled,
        "view_family",
        "canonical_high_oblique_pair_v1",
        "game_mode.capture.controlled_camera",
    )
    _bounded_int(
        controlled["image_budget"],
        "game_mode.capture.controlled_camera.image_budget",
        minimum=2,
        maximum=2,
    )
    _require_value(
        controlled,
        "style_local_view_family",
        "canonical_style_region_quadrants_v1",
        "game_mode.capture.controlled_camera",
    )
    _bounded_int(
        controlled["style_local_image_budget"],
        "game_mode.capture.controlled_camera.style_local_image_budget",
        minimum=1,
        maximum=4,
    )
    _require_value(
        controlled,
        "unsupported_render_pipeline",
        "fail_not_ingestable",
        "game_mode.capture.controlled_camera",
    )


def _validate_objectization(value: Any) -> None:
    data = _mapping(value, "game_mode.objectization")
    _require_exact_keys(
        data,
        {
            "strategy",
            "exclude_non_mesh_nodes",
            "exclude_invisible_meshes",
            "exclude_camera_descendants",
            "drop_non_physical_meshes",
            "collapse_contained_meshes",
            "compound_group_internal_collision",
            "report_individualization",
        },
        "game_mode.objectization",
    )
    _require_value(
        data,
        "strategy",
        GAME_OBJECTIZATION_STRATEGY,
        "game_mode.objectization",
    )
    for key in (
        "exclude_non_mesh_nodes",
        "exclude_invisible_meshes",
        "exclude_camera_descendants",
        "drop_non_physical_meshes",
        "collapse_contained_meshes",
        "report_individualization",
    ):
        _require_true(data[key], f"game_mode.objectization.{key}")
    _require_value(
        data,
        "compound_group_internal_collision",
        "not_evaluated",
        "game_mode.objectization",
    )


def _validate_geometry(value: Any) -> None:
    data = _mapping(value, "game_mode.geometry")
    _require_exact_keys(
        data,
        {
            "source",
            "source_up_axis",
            "unit_scale",
            "max_vertices_per_object",
            "truncation_policy",
            "narrow_phase",
            "containment_backend",
        },
        "game_mode.geometry",
    )
    _require_value(
        data,
        "source",
        "world_baked_buffer_geometry",
        "game_mode.geometry",
    )
    _require_value(data, "source_up_axis", "y", "game_mode.geometry")
    _positive_number(data["unit_scale"], "game_mode.geometry.unit_scale")
    _bounded_int(
        data["max_vertices_per_object"],
        "game_mode.geometry.max_vertices_per_object",
        minimum=3,
        maximum=10_000_000,
    )
    _require_value(
        data,
        "truncation_policy",
        "bounds_proxy_and_report",
        "game_mode.geometry",
    )
    _require_value(data, "narrow_phase", "trimesh_fcl", "game_mode.geometry")
    _require_value(
        data,
        "containment_backend",
        "trimesh_rtree",
        "game_mode.geometry",
    )


def _validate_evaluation(value: Any, config_path: Path) -> Path:
    data = _mapping(value, "game_mode.evaluation")
    _require_exact_keys(
        data,
        {
            "profile",
            "case_bundle_builder",
            "active_metrics",
            "scene_family",
            "instruction",
            "default_visual_style_spec",
            "style_spec_policy",
            "intentional_contact_policy",
        },
        "game_mode.evaluation",
    )
    raw_profile = _non_empty_string(data["profile"], "game_mode.evaluation.profile")
    profile_path = (config_path.parent / raw_profile).resolve()
    if not profile_path.is_file():
        raise GameModeConfigError(
            f"game_mode evaluation profile does not exist: {profile_path}"
        )
    _require_value(
        data,
        "case_bundle_builder",
        GAME_CASE_BUNDLE_BUILDER,
        "game_mode.evaluation",
    )
    active = data["active_metrics"]
    if not isinstance(active, list) or set(active) != GAME_ACTIVE_METRICS:
        raise GameModeConfigError(
            "game_mode.evaluation.active_metrics must contain exactly "
            "collision, navigability, and style_consistency"
        )
    _require_value(
        data,
        "scene_family",
        GAME_SCENE_FAMILY,
        "game_mode.evaluation",
    )
    _non_empty_string(data["instruction"], "game_mode.evaluation.instruction")
    raw_style_spec = _non_empty_string(
        data["default_visual_style_spec"],
        "game_mode.evaluation.default_visual_style_spec",
    )
    style_spec_path = (config_path.parent / raw_style_spec).resolve()
    if not style_spec_path.is_file():
        raise GameModeConfigError(
            "game_mode default visual style spec does not exist: "
            f"{style_spec_path}"
        )
    _require_value(
        data,
        "style_spec_policy",
        "required_benchmark_owned",
        "game_mode.evaluation",
    )
    _require_value(
        data,
        "intentional_contact_policy",
        "conditional_vlm_intentional_assembly_allowed",
        "game_mode.evaluation",
    )
    return profile_path


def _validate_failure_semantics(value: Any) -> None:
    data = _mapping(value, "game_mode.failure_semantics")
    expected = {
        "unsupported_runtime": "not_ingestable",
        "unsupported_render_pipeline": "not_ingestable",
        "missing_probe": "failed",
        "incomplete_geometry": "reported_not_silently_valid",
        "unresolved_metric": "unresolved",
    }
    _require_exact_keys(data, set(expected), "game_mode.failure_semantics")
    for key, expected_value in expected.items():
        _require_value(data, key, expected_value, "game_mode.failure_semantics")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GameModeConfigError(f"{path} must be a YAML object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    if set(value) != expected:
        raise GameModeConfigError(
            f"{path} has unknown keys {sorted(set(value) - expected)} or "
            f"missing keys {sorted(expected - set(value))}"
        )


def _require_value(
    value: dict[str, Any],
    key: str,
    expected: Any,
    path: str,
) -> None:
    if value.get(key) != expected:
        raise GameModeConfigError(f"{path}.{key} must be {expected!r}")


def _require_true(value: Any, path: str) -> None:
    if value is not True:
        raise GameModeConfigError(f"{path} must be true")


def _non_empty_string(value: Any, path: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise GameModeConfigError(f"{path} must be a non-empty string")
    return text


def _bounded_int(value: Any, path: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GameModeConfigError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise GameModeConfigError(f"{path} must be between {minimum} and {maximum}")
    return value


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise GameModeConfigError(f"{path} must be a positive number")
    return float(value)
