from __future__ import annotations

from benchmark.assets.facing import (
    CATALOG_FACING_CONTRACT_VERSION,
    DEFAULT_DIRECTED_FUNCTIONAL_SIDE,
    benchmark_catalog_facing_contract,
    resolve_catalog_functional_side,
    yaw_degrees_for_world_heading,
)
from benchmark.visual_judge.usable_surface import (
    CATALOG_CONTRACT_USABLE_SURFACE_DETECTOR_BACKEND,
    CatalogContractThenVLMUsableSurfaceDetector,
    build_usable_surface_detector,
)


class _Decoder:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.model = None

    def decode_usable_surface(self, request: dict) -> dict:
        self.calls.append(request)
        raise AssertionError("catalog contract should avoid the VLM decoder")


def _catalog_object() -> dict:
    return {
        "id": "television",
        "jid": "a_SM_TV_01",
        "category": "television",
        "asset_ref": {
            "source_db": "frozen_imaginarium_catalog",
            "asset_key": "a_SM_TV_01",
        },
    }


def test_catalog_facing_contract_and_cardinal_yaws_are_consistent() -> None:
    contract = benchmark_catalog_facing_contract()

    assert contract["contract_version"] == CATALOG_FACING_CONTRACT_VERSION
    assert contract["default_directed_functional_side"] == (
        DEFAULT_DIRECTED_FUNCTIONAL_SIDE
    )
    assert yaw_degrees_for_world_heading((0, -1)) == 0.0
    assert yaw_degrees_for_world_heading((1, 0)) == 90.0
    assert yaw_degrees_for_world_heading((0, 1)) == 180.0
    assert yaw_degrees_for_world_heading((-1, 0)) == -90.0
    assert contract["cardinal_yaw_examples_deg"] == {
        "face_world_neg_y": 0.0,
        "face_world_pos_x": 90.0,
        "face_world_pos_y": 180.0,
        "face_world_neg_x": -90.0,
    }


def test_catalog_side_resolution_is_scoped_and_supports_explicit_overrides(
) -> None:
    default = resolve_catalog_functional_side(
        _catalog_object(),
        surface_roles=["display_side"],
    )
    assert default is not None
    assert default["side_id"] == "local_neg_y"
    assert default["resolution_source"] == "catalog_default"

    overridden = resolve_catalog_functional_side(
        _catalog_object(),
        surface_roles=["display_side"],
        overrides={"a_SM_TV_01": {"display_side": "local_pos_x"}},
    )
    assert overridden is not None
    assert overridden["side_id"] == "local_pos_x"
    assert overridden["resolution_source"] == "explicit_role_override"

    external = _catalog_object()
    external["asset_ref"]["source_db"] = "generated"
    assert resolve_catalog_functional_side(
        external,
        surface_roles=["display_side"],
    ) is None


def test_materialized_fixed_catalog_requires_imaginarium_snapshot_provenance(
) -> None:
    materialized = _catalog_object()
    materialized["asset_ref"]["source_db"] = "fixed_catalog"
    materialized["metadata"] = {
        "materialization": {
            "catalog_snapshot_id": "imaginarium_catalog_7d645ab70de0c6fd",
        }
    }

    resolved = resolve_catalog_functional_side(
        materialized,
        surface_roles=["display_side"],
    )
    assert resolved is not None
    assert resolved["side_id"] == "local_neg_y"
    assert resolved["source_resolution"] == (
        "materialized_catalog_snapshot_provenance"
    )

    materialized["metadata"]["materialization"][
        "catalog_snapshot_id"
    ] = "another_catalog_123"
    assert resolve_catalog_functional_side(
        materialized,
        surface_roles=["display_side"],
    ) is None


def test_conflicting_role_overrides_defer_to_visual_fallback() -> None:
    assert resolve_catalog_functional_side(
        _catalog_object(),
        surface_roles=["display_side", "control_side"],
        overrides={
            "a_SM_TV_01": {
                "display_side": "local_neg_y",
                "control_side": "local_pos_x",
            }
        },
    ) is None


def test_default_detector_resolves_catalog_side_without_preview_or_vlm() -> None:
    decoder = _Decoder()
    detector = build_usable_surface_detector(decoder=decoder)

    assert isinstance(
        detector,
        CatalogContractThenVLMUsableSurfaceDetector,
    )
    assert detector.implementation_id == (
        CATALOG_CONTRACT_USABLE_SURFACE_DETECTOR_BACKEND
    )
    result = detector.resolve_without_previews(
        {
            "scene_id": "scene",
            "target_id": "television",
            "target_category": "television",
            "surface_roles": ["display_side"],
            "object_record": _catalog_object(),
        }
    )

    assert result is not None
    assert result["surfaces"][0]["side_id"] == "local_neg_y"
    assert result["provenance"]["decoder_called"] is False
    assert decoder.calls == []
    manifest = detector.manifest()
    assert manifest["catalog_facing_contract_version"] == (
        CATALOG_FACING_CONTRACT_VERSION
    )
    assert manifest["resolution_order"] == [
        "benchmark_catalog_contract",
        "vlm_trusted_side_fallback",
    ]
