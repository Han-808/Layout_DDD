from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROMPT_ONLY = "prompt_only"
COMPACT_OBJECTS = "compact_objects"
COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS = "compact_objects_with_estimated_relations"
COMPACT_OBJECTS_WITH_SPATIAL_CUES = "compact_objects_with_spatial_cues"
FULL_METADATA_BUDGETED = "full_metadata_budgeted"


@dataclass(frozen=True)
class InputModeSpec:
    name: str
    description: str
    is_main: bool
    is_debug: bool
    includes_required_objects: bool
    includes_bbox_sizes: bool
    includes_spatial_cues: bool
    includes_full_metadata: bool
    includes_source_layout_hints: bool
    default_for_benchmark: bool


INPUT_MODE_SPECS: dict[str, InputModeSpec] = {
    PROMPT_ONLY: InputModeSpec(
        name=PROMPT_ONLY,
        description="Natural-language scene baseline with no model-visible required object list.",
        is_main=True,
        is_debug=False,
        includes_required_objects=False,
        includes_bbox_sizes=False,
        includes_spatial_cues=False,
        includes_full_metadata=False,
        includes_source_layout_hints=False,
        default_for_benchmark=True,
    ),
    COMPACT_OBJECTS: InputModeSpec(
        name=COMPACT_OBJECTS,
        description="Compact object-list prompt with room proxy, object ids, categories, bbox sizes, and current source-derived layout hints.",
        is_main=True,
        is_debug=False,
        includes_required_objects=True,
        includes_bbox_sizes=True,
        includes_spatial_cues=False,
        includes_full_metadata=False,
        includes_source_layout_hints=True,
        default_for_benchmark=True,
    ),
    COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS: InputModeSpec(
        name=COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
        description=(
            "Compact object-list prompt plus deterministic estimated spatial cues. "
            "The cues are generated from geometry/metadata heuristics and are not HSSD ground-truth relations."
        ),
        is_main=True,
        is_debug=False,
        includes_required_objects=True,
        includes_bbox_sizes=True,
        includes_spatial_cues=True,
        includes_full_metadata=False,
        includes_source_layout_hints=True,
        default_for_benchmark=True,
    ),
    FULL_METADATA_BUDGETED: InputModeSpec(
        name=FULL_METADATA_BUDGETED,
        description="Debug-only token-heavy prompt preserving richer source metadata under compact budgets.",
        is_main=False,
        is_debug=True,
        includes_required_objects=True,
        includes_bbox_sizes=True,
        includes_spatial_cues=True,
        includes_full_metadata=True,
        includes_source_layout_hints=True,
        default_for_benchmark=False,
    ),
}

INPUT_MODE_ALIASES = {
    COMPACT_OBJECTS_WITH_SPATIAL_CUES: COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
}

MAIN_INPUT_MODES = (
    PROMPT_ONLY,
    COMPACT_OBJECTS,
    COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
)
DEBUG_INPUT_MODES = (FULL_METADATA_BUDGETED,)
INPUT_REPRESENTATION_MODES = set(INPUT_MODE_SPECS)
ACCEPTED_INPUT_REPRESENTATION_MODES = set(INPUT_MODE_SPECS) | set(INPUT_MODE_ALIASES)

DEFAULT_REPRESENTATION_BY_INPUT_LEVEL = {
    "prompt_only": PROMPT_ONLY,
    "structured_basic": COMPACT_OBJECTS,
    "structured_relation": COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
}


def canonicalize_input_mode(name: object) -> str:
    if not isinstance(name, str):
        raise ValueError(f"Input mode must be a string, got {type(name).__name__}.")
    canonical = INPUT_MODE_ALIASES.get(name, name)
    if canonical not in INPUT_MODE_SPECS:
        available = ", ".join(sorted(ACCEPTED_INPUT_REPRESENTATION_MODES))
        raise ValueError(f"Unsupported input mode '{name}'. Available: {available}")
    return canonical


def get_input_mode_spec(name: object) -> InputModeSpec:
    return INPUT_MODE_SPECS[canonicalize_input_mode(name)]


def list_main_input_modes() -> list[str]:
    return list(MAIN_INPUT_MODES)


def list_debug_input_modes() -> list[str]:
    return list(DEBUG_INPUT_MODES)


def list_all_input_modes(*, include_aliases: bool = False) -> list[str]:
    modes = list(INPUT_MODE_SPECS)
    if include_aliases:
        modes.extend(INPUT_MODE_ALIASES)
    return modes


def representation_mode_for_level(input_level: object, override: object = None) -> str:
    if isinstance(override, str):
        try:
            mode = canonicalize_input_mode(override)
        except ValueError:
            mode = ""
        if mode:
            if input_level == "prompt_only":
                return PROMPT_ONLY
            return mode
    return DEFAULT_REPRESENTATION_BY_INPUT_LEVEL.get(str(input_level), COMPACT_OBJECTS)


def resolve_input_representation_mode(case_or_config: dict[str, Any], default: str | None = None) -> str:
    candidates = [
        case_or_config.get("scene_representation_mode"),
        case_or_config.get("input_representation_mode"),
        case_or_config.get("model_input_mode"),
    ]
    source = case_or_config.get("source")
    if isinstance(source, dict):
        candidates.extend(
            [
                source.get("scene_representation_mode"),
                source.get("input_representation_mode"),
                source.get("model_input_mode"),
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                return canonicalize_input_mode(candidate)
            except ValueError:
                continue
    if default:
        return canonicalize_input_mode(default)
    return representation_mode_for_level(case_or_config.get("input_level"))


def prompt_includes_relations(mode: str) -> bool:
    spec = get_input_mode_spec(mode)
    return spec.includes_spatial_cues or spec.includes_full_metadata
