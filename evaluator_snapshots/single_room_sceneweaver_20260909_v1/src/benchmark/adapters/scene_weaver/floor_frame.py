"""Measured native-floor normalization with explicit architecture limitations."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path

from benchmark.adapters.common.geometry import finite_float
from benchmark.scene_io.validate import ArtifactValidationError


FLOOR_FRAME_VERSION = "sceneweaver_measured_floor_frame_v1"


def apply_native_floor_frame(scene: dict, config: Mapping, layout_path: Path) -> dict:
    """Translate uniformly from measured architecture, never from furniture.

    Preserve the public room contract. A native clear-height mismatch requires
    diagnostic opt-in; translation cannot repair the input architecture.
    """
    evidence = config.get("sceneweaver_native_floor_frame")
    if evidence is None:
        if config.get("sceneweaver_native_size_semantics") == "released_object_dimensions_rounded_2dp":
            raise ArtifactValidationError("released SceneWeaver output requires measured native floor evidence")
        return scene
    if not isinstance(evidence, Mapping) or evidence.get("schema_version") != FLOOR_FRAME_VERSION:
        raise ArtifactValidationError("invalid SceneWeaver native floor evidence schema")
    if evidence.get("measurement_source") != "saved_blend_architecture_world_vertices":
        raise ArtifactValidationError("SceneWeaver floor must be measured from native architecture geometry")
    if evidence.get("layout_sha256") != hashlib.sha256(layout_path.read_bytes()).hexdigest():
        raise ArtifactValidationError("SceneWeaver floor evidence layout hash mismatch")
    hc = scene["metadata"]["harness_compatibility"]
    if evidence.get("selected_iteration") != hc.get("selected_iteration"):
        raise ArtifactValidationError("SceneWeaver floor evidence iteration mismatch")
    blend_hash = str(evidence.get("native_blend_sha256") or "")
    if len(blend_hash) != 64 or any(c not in "0123456789abcdef" for c in blend_hash):
        raise ArtifactValidationError("SceneWeaver floor evidence requires native blend SHA256")
    floor_z = finite_float(evidence.get("floor_z_m"), "native floor z")
    ceiling_z = finite_float(evidence.get("ceiling_z_m"), "native ceiling z")
    tolerance = 1.0e-5
    for name, z in (("floor", floor_z), ("ceiling", ceiling_z)):
        plane = evidence.get(name)
        if not isinstance(plane, Mapping) or not str(plane.get("object_name") or "").endswith("." + name):
            raise ArtifactValidationError(f"SceneWeaver {name} measurement lacks architecture identity")
        lo = finite_float(plane.get("world_z_min_m"), name + " z min")
        hi = finite_float(plane.get("world_z_max_m"), name + " z max")
        if hi < lo or hi - lo > tolerance or not lo - tolerance <= z <= hi + tolerance:
            raise ArtifactValidationError(f"SceneWeaver {name} is not a single horizontal plane")
    clear_height = ceiling_z - floor_z
    if clear_height <= 0:
        raise ArtifactValidationError("SceneWeaver native clear height must be positive")
    public_height = finite_float(scene["scene_height"], "canonical scene height")
    mismatch = abs(clear_height - public_height) > tolerance
    if mismatch and config.get("sceneweaver_allow_room_height_mismatch") is not True:
        raise ArtifactValidationError(
            "SceneWeaver native clear height differs from public room height; "
            "diagnostic re-export requires sceneweaver_allow_room_height_mismatch=True"
        )
    cc = hc["coordinate_conversion"]
    if cc.get("floor_frame") is not None or abs(float(cc["origin_shift"][2])) > tolerance:
        raise ArtifactValidationError("SceneWeaver native floor shift was already applied or is ambiguous")
    architecture = scene["metadata"].get("architecture_contract") or {}
    for value in (architecture.get("floor_z"), (architecture.get("floor") or {}).get("z"),
                  (architecture.get("room") or {}).get("floor_z")):
        if value is not None and abs(finite_float(value, "canonical floor")) > tolerance:
            raise ArtifactValidationError("SceneWeaver floor normalization requires canonical floor z=0")
    result = copy.deepcopy(scene)
    for obj in result["objects"]:
        obj["center"][2] = finite_float(obj["center"][2], "object center z") - floor_z
    hc = result["metadata"]["harness_compatibility"]
    hc["coordinate_conversion"]["origin_shift"][2] = -floor_z
    hc["coordinate_conversion"]["floor_frame"] = {
        **copy.deepcopy(dict(evidence)),
        "policy": "uniform_translation_preserve_native_relative_poses_and_clearances",
        "canonical_floor_z_m": 0.0,
        "translated_native_ceiling_z_m": clear_height,
        "public_ceiling_z_m": public_height,
        "native_clear_height_matches_public_room": not mismatch,
        "public_architecture_preserved": True,
    }
    if mismatch:
        hc["room_contract_match"] = "native_clear_height_mismatch_diagnostic_only"
        hc["full_architecture_compatibility_qualified"] = False
    return result
