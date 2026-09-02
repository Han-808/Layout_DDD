from __future__ import annotations

import hashlib
from pathlib import Path

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.adapters.output_routing import OUTPUT_CONVERTER
from benchmark.adapters.catalog_placement.converter import (
    build_catalog_instance_registry,
    convert_catalog_placement_to_scene,
    extract_catalog_placement,
)
from benchmark.adapters.catalog_placement.prompt import (
    CATALOG_PLACEMENT_VERSION,
    build_catalog_placement_method_input,
)
from benchmark.io_contracts import (
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
    O3_SCENE_PACKAGE,
)
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import read_json, write_json


class CatalogPlacementAdapter(GenerationAdapter):
    """One-shot generator for rigid instances from a frozen selected catalog."""

    name = "catalog_placement"
    output_schema = CATALOG_PLACEMENT_VERSION
    output_ingestion_kind = OUTPUT_CONVERTER
    capabilities = AdapterCapabilities(
        input_modes=("structured_assets",),
        asset_support="required",
        input_types=(I2_NATURAL_LANGUAGE_STRUCTURE,),
        native_output_types=(O1_OBJECT_STATE,),
        evaluator_output_types=(O1_OBJECT_STATE, O3_SCENE_PACKAGE),
        room_models=("single_room",),
        boundary_models=("axis_aligned_rectangle",),
        architecture_features=(),
        geometry_fidelity=("bbox", "mesh_optional"),
        preserves_asset_identity=True,
    )
    executable_integration = True

    def __init__(self) -> None:
        self.last_run_metadata: dict = {}
        self.last_parse_metadata: dict = {}

    def prepare_input(
        self, generation_input: dict, out_dir: Path, config: dict | None = None
    ) -> Path:
        validate_generation_input(generation_input)
        self.resolve_io_contract(generation_input, config=config)
        method_input = build_catalog_placement_method_input(generation_input)
        return write_json(Path(out_dir) / "method_input.json", method_input)

    def run_generation(
        self, method_input_path: Path, out_dir: Path, config: dict | None = None
    ) -> Path:
        cfg = config or {}
        if "api_key" in cfg:
            raise ValueError(
                "catalog_placement adapter config must not contain literal api_key; "
                "use api_key_env instead"
            )
        endpoint = str(cfg.get("endpoint") or "").strip()
        model_id = str(cfg.get("model") or cfg.get("model_id") or "").strip()
        if not endpoint:
            raise ValueError("catalog_placement adapter requires config.endpoint")
        if not model_id:
            raise ValueError(
                "catalog_placement adapter requires config.model or config.model_id"
            )
        requested_repairs = int(cfg.get("schema_repair_attempts", 0) or 0)
        if requested_repairs > 0:
            raise ValueError(
                "catalog_placement schema repair is disabled; malformed generator output "
                "must fail closed. Set schema_repair_attempts to 0."
            )

        method_input = read_json(method_input_path)
        client = OpenAICompatibleModel(
            name=str(cfg.get("name") or "catalog_placement_generator"),
            endpoint=endpoint,
            model_id=model_id,
            api_key_env=cfg.get("api_key_env"),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=(
                int(cfg["max_tokens"])
                if cfg.get("max_tokens") is not None
                else 4096
            ),
            context_length=(
                int(cfg["context_length"])
                if cfg.get("context_length") is not None
                else None
            ),
            timeout_seconds=int(cfg.get("timeout_seconds", 300)),
            response_format_json=bool(cfg.get("response_format_json", True)),
            max_retries=int(cfg.get("max_retries", 1)),
            retry_backoff_seconds=float(cfg.get("retry_backoff_seconds", 1.0)),
            max_tokens_field=str(cfg.get("max_tokens_field", "max_tokens")),
            send_temperature=bool(cfg.get("send_temperature", True)),
            require_api_key=(
                bool(cfg["require_api_key"])
                if cfg.get("require_api_key") is not None
                else None
            ),
        )
        response_text = client.chat_messages(
            method_input["messages"],
            response_format_json=bool(cfg.get("response_format_json", True)),
            call_type="scene_generation",
            case={
                "case_id": method_input.get("request_id"),
                "input_mode": method_input.get("input_mode"),
            },
        )
        raw_response_path = Path(out_dir) / "model_response.txt"
        raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        raw_response_path.write_text(response_text, encoding="utf-8")
        raw_response_sha256 = hashlib.sha256(raw_response_path.read_bytes()).hexdigest()
        request_metadata_path = write_json(
            Path(out_dir) / "model_request_metadata.json",
            client.last_request_metadata,
        )
        schema_repair = {
            "attempted": False,
            "attempt_count": 0,
            "policy": "disabled_fail_closed",
        }
        try:
            placement = extract_catalog_placement(
                response_text,
                public_slot_ids=method_input.get("public_slot_ids", []),
                require_slot_binding=True,
            )
        except ArtifactValidationError as error:
            self.last_run_metadata = {
                "provider": "openai_compatible",
                "endpoint": endpoint,
                "model": model_id,
                "output_schema": self.output_schema,
                "raw_response_path": raw_response_path.as_posix(),
                "raw_response_sha256": raw_response_sha256,
                "request_metadata_path": request_metadata_path.as_posix(),
                "schema_repair": schema_repair,
                "schema_failure": str(error),
            }
            raise ArtifactValidationError(
                "catalog_placement_v1 generator output failed strict validation; "
                f"preserved raw response at {raw_response_path.as_posix()}: {error}"
            ) from error
        output_path = write_json(
            Path(out_dir) / "catalog_placement_output.json", placement
        )
        native_artifact_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        self.last_run_metadata = {
            "provider": "openai_compatible",
            "endpoint": endpoint,
            "model": model_id,
            "output_schema": self.output_schema,
            "raw_response_path": raw_response_path.as_posix(),
            "raw_response_sha256": raw_response_sha256,
            "native_artifact_path": output_path.as_posix(),
            "native_artifact_sha256": native_artifact_sha256,
            "request_metadata_path": request_metadata_path.as_posix(),
            "schema_repair": schema_repair,
        }
        return output_path

    def parse_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        source_path = Path(method_output_path)
        raw_path = Path(out_dir) / "catalog_placement_raw_artifact.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(source_path.read_bytes())
        try:
            payload = read_json(source_path)
            placement = extract_catalog_placement(payload)
            scene = convert_catalog_placement_to_scene(placement, generation_input)
            registry = build_catalog_instance_registry(placement, generation_input)
        except (ArtifactValidationError, ValueError) as error:
            self.last_parse_metadata = {
                "output_schema": self.output_schema,
                "raw_artifact_path": raw_path.as_posix(),
                "validation_failure": str(error),
            }
            if isinstance(error, ArtifactValidationError):
                raise
            raise ArtifactValidationError(str(error)) from error
        registry_path = write_json(
            Path(out_dir) / "instance_registry.json", registry
        )
        scene_path = write_json(Path(out_dir) / "generated_scene.json", scene)
        self.last_parse_metadata = {
            "output_schema": self.output_schema,
            "raw_artifact_path": raw_path.as_posix(),
            "instance_registry_path": registry_path.as_posix(),
            "canonical_output_path": scene_path.as_posix(),
        }
        return scene_path
