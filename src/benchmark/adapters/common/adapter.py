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
from benchmark.adapters.common.execution import (
    execute_external_harness,
    preserve_supplied_native_artifact,
    update_execution_result,
    verify_preserved_native_artifact,
)
from benchmark.adapters.output_routing import OUTPUT_CONVERTER
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
)
from benchmark.nl_scene.generation_input import build_generator_visible_payload
from benchmark.scene_io.normalize import bind_scene_to_generation_request, normalize_scene
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import read_json, write_json


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
    executable_integration = False
    native_input_filename = "native_input.json"
    native_input_schema = "external_harness_native_input_v1"
    default_native_artifact: str | None = None
    default_native_artifact_glob: str | None = None

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
        request = generation_input.get("scene_request")
        request = request if isinstance(request, dict) else {}
        method_input = {
            "protocol": self.output_schema,
            "harness": self.name,
            "request_id": str(generation_input.get("request_id") or ""),
            "public_request": {
                "scene_type": str(request.get("scene_type") or "room"),
            },
            "io_contract": contract.as_dict(),
            "asset_resolution_policy": policy,
            "scene_compatibility": scene_compatibility.as_dict(),
            "generator_input": build_generator_visible_payload(generation_input),
        }
        native_input_path: Path | None = None
        if self.executable_integration:
            native_input = self.build_native_input(method_input, config or {})
            native_input_path = write_json(
                Path(out_dir) / "native_input" / self.native_input_filename,
                native_input,
            )
            method_input["execution_input"] = {
                "schema": self.native_input_schema,
                "path": native_input_path.resolve().as_posix(),
            }
        method_input_path = write_json(Path(out_dir) / "method_input.json", method_input)
        self.last_preparation_metadata = {
            "method_input_path": method_input_path.resolve().as_posix(),
            "native_input_path": (
                native_input_path.resolve().as_posix()
                if native_input_path is not None
                else None
            ),
            "native_input_schema": (
                self.native_input_schema if native_input_path is not None else None
            ),
        }
        return method_input_path

    def build_native_input(
        self,
        method_input: dict,
        config: dict,
    ) -> Any:
        """Build a public, method-specific upstream request artifact."""

        del method_input, config
        raise NotImplementedError(
            f"Adapter {self.name!r} has no executable native-input builder"
        )

    def run_generation(
        self,
        method_input_path: Path,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        cfg = config or {}
        if cfg.get("execution") is not None and not self.executable_integration:
            raise NotImplementedError(
                f"Adapter {self.name!r} is compatibility-only and has no executable "
                "integration profile"
            )
        native_input_path = self._prepared_native_input_path(method_input_path)
        preserved, metadata = execute_external_harness(
            adapter_name=self.name,
            method_input_path=Path(method_input_path),
            native_input_path=native_input_path,
            out_dir=Path(out_dir),
            config=cfg,
            default_native_artifact=self.default_native_artifact,
            default_native_artifact_glob=self.default_native_artifact_glob,
        )
        source_value = metadata.get("source_native_artifact_path")
        if source_value:
            self._original_native_source_path = Path(str(source_value)).resolve()
        self._preserved_native_source_path = preserved.resolve()
        self._record_native_audit(preserved, cfg, metadata)
        return preserved

    def preserve_supplied_native_output(
        self,
        method_output_path: Path,
        method_input_path: Path,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        """Snapshot a Mode-A artifact before strict canonical conversion."""

        source = Path(method_output_path).expanduser().resolve()
        self._original_native_source_path = source
        native_input = self._prepared_native_input_path(
            method_input_path,
            required=False,
        )
        preserved, metadata = preserve_supplied_native_artifact(
            adapter_name=self.name,
            source_path=source,
            method_input_path=Path(method_input_path),
            native_input_path=native_input,
            out_dir=Path(out_dir),
            config=config or {},
        )
        self._preserved_native_source_path = preserved.resolve()
        self._record_native_audit(preserved, config or {}, metadata)
        return preserved

    def verify_preserved_native_output(
        self,
        canonical_scene_path: Path,
    ) -> None:
        metadata = getattr(self, "last_run_metadata", None)
        if isinstance(metadata, dict):
            verify_preserved_native_artifact(
                metadata,
                canonical_scene_path=Path(canonical_scene_path),
            )

    def inspect_native_artifact(
        self,
        native_artifact_path: Path,
        config: dict,
    ) -> dict[str, Any]:
        """Return adapter-specific, audit-only native artifact inventory."""

        del native_artifact_path, config
        return {}

    def _record_native_audit(
        self,
        native_artifact_path: Path,
        config: dict,
        metadata: dict[str, Any],
    ) -> None:
        try:
            audit_fields = self.inspect_native_artifact(
                native_artifact_path,
                config,
            )
        except BaseException as exc:
            failure = {
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            update_execution_result(metadata, failure)
            self.last_run_metadata = metadata
            raise
        if audit_fields:
            update_execution_result(metadata, audit_fields)
        self.last_run_metadata = metadata

    def enrich_conversion_config(self, config: dict) -> dict:
        """Resolve preserved sidecars without changing conversion semantics."""

        cfg = dict(config)
        run_metadata = getattr(self, "last_run_metadata", None)
        auxiliary = (
            run_metadata.get("preserved_auxiliary_artifacts")
            if isinstance(run_metadata, dict)
            else None
        )
        if isinstance(auxiliary, dict):
            for name, config_key in (
                ("asset_manifest", "asset_manifest_path"),
                ("asset_ids", "asset_ids_path"),
                ("asset_bindings", "asset_bindings_path"),
                ("scene_config", "scene_config_path"),
            ):
                item = auxiliary.get(name)
                if isinstance(item, dict) and item.get("path"):
                    cfg[config_key] = str(item["path"])
        original = getattr(self, "_original_native_source_path", None)
        if isinstance(original, Path):
            preserved = getattr(self, "_preserved_native_source_path", None)
            base = (
                preserved
                if original.is_dir() and isinstance(preserved, Path)
                else original.parent
            )
            for key in (
                "asset_manifest_path",
                "asset_ids_path",
                "asset_bindings_path",
                "scene_config_path",
            ):
                value = cfg.get(key)
                if not value:
                    continue
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    cfg[key] = (base / path).resolve().as_posix()
        return cfg

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
        cfg = self.enrich_conversion_config(dict(config or {}))
        policy = asset_resolution_policy(cfg)
        cfg["asset_resolution_policy"] = policy
        scene_compatibility = self.resolve_scene_compatibility(
            generation_input,
            config=cfg,
        )
        source_path = Path(method_output_path)
        try:
            provider = load_asset_provider(cfg, source_path=source_path)
            scene = self.converter(source_path, generation_input, cfg, provider)
        except BaseException as exc:
            self._record_conversion_failure(exc)
            raise
        if not isinstance(scene, dict):
            error = ArtifactValidationError(
                f"{self.name} converter must return a canonical scene JSON object"
            )
            self._record_conversion_failure(error)
            raise error
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
        try:
            normalized = normalize_scene(
                bind_scene_to_generation_request(scene, generation_input),
                asset_csv=cfg.get("asset_csv") or cfg.get("asset_csv_path"),
                asset_root=cfg.get("asset_root"),
                enrich_assets=bool(cfg.get("enrich_assets", False)),
            )
        except BaseException as exc:
            self._record_conversion_failure(exc)
            raise
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
        run_metadata = getattr(self, "last_run_metadata", None)
        if isinstance(run_metadata, dict):
            update_execution_result(
                run_metadata,
                {
                    "conversion_status": "completed",
                    "converter_metadata": self.last_parse_metadata,
                },
            )
        return output_path

    def _record_conversion_failure(self, error: BaseException) -> None:
        run_metadata = getattr(self, "last_run_metadata", None)
        if not isinstance(run_metadata, dict):
            return
        update_execution_result(
            run_metadata,
            {
                "conversion_status": "failed",
                "conversion_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            },
        )

    def _prepared_native_input_path(
        self,
        method_input_path: Path,
        *,
        required: bool | None = None,
    ) -> Path:
        must_exist = self.executable_integration if required is None else required
        metadata = getattr(self, "last_preparation_metadata", None)
        value = metadata.get("native_input_path") if isinstance(metadata, dict) else None
        if not value:
            method_input = read_json(Path(method_input_path))
            execution_input = (
                method_input.get("execution_input")
                if isinstance(method_input, dict)
                else None
            )
            value = (
                execution_input.get("path")
                if isinstance(execution_input, dict)
                else None
            )
        if value:
            path = Path(str(value)).expanduser().resolve()
            if not path.is_file():
                raise ArtifactValidationError(
                    f"prepared native input is missing: {path}"
                )
            return path
        if must_exist:
            raise ArtifactValidationError(
                f"Adapter {self.name!r} has no prepared native execution input"
            )
        return Path(method_input_path).resolve()

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
