import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re

from benchmark.adapters.common.adapter import (
    SINGLE_ROOM_HARNESS_CAPABILITIES,
    HarnessConverterAdapter,
)
from benchmark.adapters.common.execution import artifact_sha256
from benchmark.adapters.common.native_input import (
    public_asset_selection,
    public_generator_input,
    public_instruction,
    public_room,
    public_scene_type,
)
from benchmark.adapters.scene_weaver.converter import (
    convert_scene_weaver,
    discover_layout_iterations,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


class SceneWeaverAdapter(HarnessConverterAdapter):
    """Adapter for SceneWeaver's final record_scene/layout_<iter>.json."""

    name = "scene_weaver"
    output_schema = "sceneweaver_layout_v1"
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES
    executable_integration = True
    native_input_filename = "sceneweaver_request.json"
    native_input_schema = "sceneweaver_public_runner_request_v1"
    default_native_artifact = "{upstream_output_dir}/sceneweaver_native"
    converter = staticmethod(convert_scene_weaver)

    def build_native_input(self, method_input: dict, config: dict) -> dict:
        visible = public_generator_input(method_input)
        if isinstance(visible.get("self_reflection"), Mapping):
            raise ArtifactValidationError(
                "SceneWeaver executable integration never accepts benchmark "
                "evaluation feedback; evaluate native iterations offline"
            )
        room = public_room(method_input)
        dimensions = room.get("dimensions")
        dimensions = dimensions if isinstance(dimensions, Mapping) else {}
        try:
            count = int(config.get("sceneweaver_count", 1))
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                "sceneweaver_count must be a positive integer"
            ) from exc
        if count <= 0:
            raise ArtifactValidationError(
                "sceneweaver_count must be a positive integer"
            )
        structure = visible.get("structure")
        structure = structure if isinstance(structure, Mapping) else {}
        object_plan = structure.get("object_plan")
        public_plan = (
            _without_asset_locators(object_plan)
            if isinstance(object_plan, Mapping)
            else None
        )
        public_plan_sha256 = (
            hashlib.sha256(
                json.dumps(
                    public_plan,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if public_plan is not None
            else None
        )
        return {
            "schema_version": self.native_input_schema,
            "prompt": public_instruction(method_input),
            "scene_type": public_scene_type(method_input),
            "count": count,
            "benchmark_room": {
                "boundary": room.get("boundary"),
                "roomsize": [dimensions.get("width"), dimensions.get("depth")],
                "height": room.get("height"),
                "unit": room.get("unit", "meter"),
            },
            "asset_selection": _without_asset_locators(
                public_asset_selection(method_input)
            ),
            "public_object_plan": public_plan,
            "public_object_plan_sha256": public_plan_sha256,
            "feedback_source": "native_sceneweaver_only",
        }

    def inspect_native_artifact(
        self,
        native_artifact_path: Path,
        config: dict,
    ) -> dict:
        del config
        iterations = discover_layout_iterations(native_artifact_path)
        if not iterations:
            raise ArtifactValidationError(
                "SceneWeaver native artifact contains no layout_<iteration>.json"
            )
        inventory = []
        artifact_root = (
            native_artifact_path
            if native_artifact_path.is_dir()
            else native_artifact_path.parent
        )
        for iteration, layout_path in sorted(iterations.items()):
            digest, _ = artifact_sha256(layout_path)
            iteration_pattern = re.compile(
                rf"(?:^|_){iteration}(?:_|\.|$)"
            )
            related = sorted(
                path.resolve().as_posix()
                for path in artifact_root.rglob("*")
                if path.is_file()
                and path != layout_path
                and iteration_pattern.search(path.name)
            )
            inventory.append(
                {
                    "iteration": iteration,
                    "layout_path": layout_path.resolve().as_posix(),
                    "layout_sha256": digest,
                    "related_artifacts": related,
                }
            )
        return {
            "sceneweaver_available_iterations": sorted(iterations),
            "sceneweaver_iteration_artifacts": inventory,
            "benchmark_feedback_used_by_native_loop": False,
        }

    def enrich_conversion_config(self, config: dict) -> dict:
        cfg = super().enrich_conversion_config(config)
        if isinstance(cfg.get("asset_bindings"), Mapping):
            return cfg
        path_value = cfg.get("asset_bindings_path")
        run_metadata = getattr(self, "last_run_metadata", None)
        auxiliary = (
            run_metadata.get("preserved_auxiliary_artifacts")
            if isinstance(run_metadata, dict)
            else None
        )
        if not path_value and isinstance(auxiliary, Mapping):
            item = auxiliary.get("asset_bindings")
            if isinstance(item, Mapping):
                path_value = item.get("path")
        if not path_value:
            return cfg
        loaded = read_json(path_value)
        if isinstance(loaded, Mapping) and isinstance(
            loaded.get("asset_bindings"), Mapping
        ):
            loaded = loaded["asset_bindings"]
        if not isinstance(loaded, Mapping):
            raise ArtifactValidationError(
                "SceneWeaver asset_bindings_path must contain a binding mapping"
            )
        cfg["asset_bindings"] = dict(loaded)
        cfg["asset_bindings_path"] = str(path_value)
        return cfg


def _without_asset_locators(value):
    """Keep public IDs/metadata while withholding host-local cache paths."""

    private_keys = {
        "asset_root",
        "cache_path",
        "file_path",
        "mesh_path",
        "mesh_uri",
        "metadata_path",
        "metadata_uri",
        "path",
    }
    if isinstance(value, Mapping):
        return {
            str(key): _without_asset_locators(item)
            for key, item in value.items()
            if str(key) not in private_keys
        }
    if isinstance(value, list):
        return [_without_asset_locators(item) for item in value]
    return deepcopy(value)


__all__ = ["SceneWeaverAdapter"]
