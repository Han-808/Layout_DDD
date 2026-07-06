from __future__ import annotations

from benchmark.input_modes import (
    COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS,
    FULL_METADATA_BUDGETED,
    canonicalize_input_mode,
    get_input_mode_spec,
    list_debug_input_modes,
    list_main_input_modes,
)


def test_main_input_modes_are_the_default_benchmark_modes() -> None:
    assert list_main_input_modes() == [
        "prompt_only",
        "compact_objects",
        "compact_objects_with_estimated_relations",
    ]
    assert FULL_METADATA_BUDGETED not in list_main_input_modes()
    assert list_debug_input_modes() == [FULL_METADATA_BUDGETED]


def test_spatial_cues_alias_resolves_to_existing_mode_name() -> None:
    assert canonicalize_input_mode("compact_objects_with_spatial_cues") == COMPACT_OBJECTS_WITH_ESTIMATED_RELATIONS
    spec = get_input_mode_spec("compact_objects_with_spatial_cues")
    assert spec.includes_spatial_cues is True
    assert spec.is_main is True


def test_full_metadata_budgeted_is_debug_only() -> None:
    spec = get_input_mode_spec(FULL_METADATA_BUDGETED)
    assert spec.is_debug is True
    assert spec.is_main is False
    assert spec.default_for_benchmark is False
