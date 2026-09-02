"""Base adapter for externally executed scene-generation harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from benchmark.adapters.base import (
    AdapterCapabilities,
    GenerationAdapter,
    SceneCompatibilityRequirements,
)
from benchmark.adapters.common.assets import (
    ASSET_RESOLUTION_EXACT_ONLY,
    asset_resolution_policy,
    load_asset_provider,
)
from benchmark.adapters.common.geometry import boundary_model
from benchmark.adapters.output_routing import OUTPUT_CONVERTER
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
)
from benchmark.nl_scene.generation_input import build_generator_visible_payload
from benchmark.scene_io.normalize import bind_scene_to_generation_request, normalize_scene
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import write_json


HarnessConverter = Callable[[Path, dict, dict, Any], dict]


SINGLE_ROOM_HARNESS_CAPABILITIES = AdapterCapabilities(
    input_modes=(
        "natural_language_direct",
        "natural_language_structured",
        "structured_assets",
    ),
    asset_support="optional",
    input_types=(I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE),
    native_output_types=(O1_OBJECT_STATE,),
    evaluator_output_types=(O1_OBJECT_STATE,),
    room_models=("single_room",),
    boundary_models=("axis_aligned_rectangle",),
    architecture_features=(),
    geometry_fidelity=("bbox", "mesh_optional"),
    preserves_asset_identity=True,
)


class HarnessConverterAdapter(GenerationAdapter):
    """Common file/directory adapter with an injected pure converter function."""

    name = "external_harness"
    output_schema = "external_harness_output"
    output_ingestion_kind = OUTPUT_CONVERTER
    converter: HarnessConverter | None = None
    capabilities = SINGLE_ROOM_HARNESS_CAPABILITIES

    def prepare_input(
        self,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        validate_generation_input(generation_input)
        contract = self.resolve_io_contract(generation_input, config=config)
        policy = asset_resolution_policy(config or {})
        scene_compatibility = self.resolve_scene_compatibility(
            generation_input,
            config=config,
        )
        return write_json(
            Path(out_dir) / "method_input.json",
            {
                "protocol": self.output_schema,
                "harness": self.name,
                "io_contract": contract.as_dict(),
                "asset_resolution_policy": policy,
                "scene_compatibility": scene_compatibility.as_dict(),
                "generator_input": build_generator_visible_payload(generation_input),
            },
        )

    def run_generation(
        self,
        method_input_path: Path,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        cfg = config or {}
        runner = cfg.get("runner")
        if runner is not None:
            return Path(
                runner(
                    method_input_path=Path(method_input_path),
                    out_dir=Path(out_dir),
                    config=cfg,
                )
            )
        output_path = cfg.get("raw_output_path") or cfg.get(f"{self.name}_output_path")
        if output_path:
            return Path(str(output_path))
        raise NotImplementedError(
            f"{self.name} adapter requires config.runner, config.raw_output_path, "
            "or an externally supplied --method-output"
        )

    def convert_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        if self.converter is None:
            raise NotImplementedError(f"Adapter {self.name!r} has no converter")
        validate_generation_input(generation_input)
        cfg = dict(config or {})
        policy = asset_resolution_policy(cfg)
        cfg["asset_resolution_policy"] = policy
        scene_compatibility = self.resolve_scene_compatibility(
            generation_input,
            config=cfg,
        )
        source_path = Path(method_output_path)
        provider = load_asset_provider(cfg, source_path=source_path)
        scene = self.converter(source_path, generation_input, cfg, provider)
        if not isinstance(scene, dict):
            raise ArtifactValidationError(
                f"{self.name} converter must return a canonical scene JSON object"
            )
        metadata = scene.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        compatibility = metadata.get("harness_compatibility")
        compatibility = compatibility if isinstance(compatibility, dict) else {}
        metadata["harness_compatibility"] = {
            **compatibility,
            "asset_resolution_policy": policy,
            "scene_compatibility_requirements": scene_compatibility.as_dict(),
        }
        scene["metadata"] = metadata
        normalized = normalize_scene(
            bind_scene_to_generation_request(scene, generation_input),
            asset_csv=cfg.get("asset_csv") or cfg.get("asset_csv_path"),
            asset_root=cfg.get("asset_root"),
            enrich_assets=bool(cfg.get("enrich_assets", False)),
        )
        metadata = normalized.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized["metadata"] = {
            **metadata,
            "native_output_type": O1_OBJECT_STATE,
            "output_adapter": self.name,
        }
        output_path = write_json(Path(out_dir) / "generated_scene.json", normalized)
        self.last_parse_metadata = {
            "harness": self.name,
            "native_schema": self.output_schema,
            "raw_artifact_path": source_path.resolve().as_posix(),
            "canonical_output_path": output_path.resolve().as_posix(),
            "asset_provider_configured": provider is not None,
            "asset_resolution_policy": policy,
            "scene_compatibility_requirements": scene_compatibility.as_dict(),
        }
        return output_path

    def resolve_scene_compatibility(
        self,
        generation_input: dict,
        config: dict | None = None,
    ) -> SceneCompatibilityRequirements:
        """Resolve and enforce the semantic contract before conversion."""

        cfg = config or {}
        request = generation_input.get("scene_request")
        request = request if isinstance(request, dict) else {}
        room = request.get("room")
        room = room if isinstance(room, dict) else {}
        boundary = room.get("boundary")
        observed_boundary = (
            boundary_model(boundary)
            if isinstance(boundary, list) and len(boundary) >= 3
            else "axis_aligned_rectangle"
        )
        room_model = str(cfg.get("room_model") or "single_room").strip()
        room_bearing_artifacts = (
            generation_input,
            request,
            room,
            generation_input.get("object_plan"),
            generation_input.get("asset_selection"),
        )
        if any(
            isinstance(artifact, dict)
            and isinstance(artifact.get("rooms"), list)
            for artifact in room_bearing_artifacts
        ):
            room_model = "multi_room"
        architecture_features = _string_tuple(
            cfg.get("required_architecture_features"),
            field="required_architecture_features",
            default=(),
        )
        geometry_fidelity = _string_tuple(
            cfg.get("required_geometry_fidelity"),
            field="required_geometry_fidelity",
            default=("bbox",),
        )
        policy = asset_resolution_policy(cfg)
        try:
            requirements = SceneCompatibilityRequirements(
                room_models=(room_model,),
                boundary_models=(observed_boundary,),
                architecture_features=architecture_features,
                geometry_fidelity=geometry_fidelity,
                preserves_asset_identity=(policy == ASSET_RESOLUTION_EXACT_ONLY),
            )
            self.capabilities.require_scene_compatibility(requirements)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"Adapter {self.name!r} is not evaluator-compatible for this scene: {exc}"
            ) from exc
        self.last_scene_compatibility = requirements.as_dict()
        return requirements


def _string_tuple(
    value: Any,
    *,
    field: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ArtifactValidationError(f"adapter_config.{field} must be a string list")
    return tuple(str(item).strip() for item in value)


__all__ = [
    "HarnessConverter",
    "HarnessConverterAdapter",
    "SINGLE_ROOM_HARNESS_CAPABILITIES",
]
