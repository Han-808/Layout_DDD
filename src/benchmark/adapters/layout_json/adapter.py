from __future__ import annotations

from pathlib import Path

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
)
from benchmark.adapters.layout_json.converter import convert_layout_json_to_scene, extract_layout_json
from benchmark.adapters.layout_json.prompt import LAYOUT_JSON_VERSION, build_layout_json_method_input
from benchmark.models.json_response import ModelResponseError
from benchmark.models.openai_compatible_model import OpenAICompatibleModel
from benchmark.scene_io.validate import ArtifactValidationError, validate_generation_input
from benchmark.utils.io import read_json, write_json


class LayoutJsonAdapter(GenerationAdapter):
    """One-shot LLM generator using layout_json_v1 as its native output."""

    name = "layout_json"
    output_schema = LAYOUT_JSON_VERSION
    capabilities = AdapterCapabilities(
        input_modes=("natural_language_direct", "natural_language_structured", "structured_assets"),
        asset_support="optional",
        input_types=(I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE),
        native_output_types=(O1_OBJECT_STATE,),
        evaluator_output_types=(O1_OBJECT_STATE,),
    )

    def __init__(self) -> None:
        self.last_run_metadata: dict = {}

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        validate_generation_input(generation_input)
        method_input = build_layout_json_method_input(generation_input)
        return write_json(Path(out_dir) / "method_input.json", method_input)

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        if "api_key" in cfg:
            raise ValueError(
                "layout_json adapter config must not contain literal api_key; use api_key_env instead"
            )
        endpoint = str(cfg.get("endpoint") or "").strip()
        model_id = str(cfg.get("model") or cfg.get("model_id") or "").strip()
        if not endpoint:
            raise ValueError("layout_json adapter requires config.endpoint")
        if not model_id:
            raise ValueError("layout_json adapter requires config.model or config.model_id")
        requested_repairs = int(cfg.get("schema_repair_attempts", 0) or 0)
        if requested_repairs > 0:
            # Frozen P0a: the current benchmark adapter is format-only and must
            # never issue an LLM/VLM schema-repair request. Reject rather than
            # silently ignore so callers cannot expect repair behavior.
            raise ValueError(
                "layout_json schema repair is disabled in the current benchmark path "
                "(frozen P0a: the adapter is format-only and issues zero schema-repair "
                "model calls). Set schema_repair_attempts to 0."
            )
        method_input = read_json(method_input_path)
        client = OpenAICompatibleModel(
            name=str(cfg.get("name") or "layout_json_generator"),
            endpoint=endpoint,
            model_id=model_id,
            api_key_env=cfg.get("api_key_env"),
            temperature=float(cfg.get("temperature", 0.0)),
            max_tokens=int(cfg["max_tokens"]) if cfg.get("max_tokens") is not None else 4096,
            context_length=int(cfg["context_length"]) if cfg.get("context_length") is not None else None,
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
        # Always preserve the raw generator output before any validation so a
        # schema failure keeps the exact bytes the generator produced.
        raw_response_path.write_text(response_text, encoding="utf-8")
        schema_repair = {
            "attempted": False,
            "attempt_count": 0,
            "policy": "disabled_format_only_p0a",
            "reason": (
                "current benchmark adapter is format-only; malformed output is a schema "
                "failure and no LLM/VLM schema-repair request is issued"
            ),
        }
        try:
            layout = extract_layout_json(response_text)
        except (ArtifactValidationError, ModelResponseError) as error:
            self.last_run_metadata = {
                "provider": "openai_compatible",
                "endpoint": endpoint,
                "model": model_id,
                "output_schema": self.output_schema,
                "raw_response_path": raw_response_path.as_posix(),
                "schema_repair": schema_repair,
                "schema_failure": str(error),
            }
            raise ArtifactValidationError(
                "layout_json_v1 generator output failed schema validation and schema repair is "
                f"disabled in the current benchmark path; preserved raw response at "
                f"{raw_response_path.as_posix()}: {error}"
            ) from error
        layout_path = write_json(Path(out_dir) / "layout_json_output.json", layout)
        request_metadata_path = write_json(Path(out_dir) / "model_request_metadata.json", client.last_request_metadata)
        self.last_run_metadata = {
            "provider": "openai_compatible",
            "endpoint": endpoint,
            "model": model_id,
            "output_schema": self.output_schema,
            "raw_response_path": raw_response_path.as_posix(),
            "request_metadata_path": request_metadata_path.as_posix(),
            "schema_repair": schema_repair,
        }
        return layout_path

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        payload = read_json(method_output_path)
        layout = extract_layout_json(payload)
        scene = convert_layout_json_to_scene(layout, generation_input)
        return write_json(Path(out_dir) / "generated_scene.json", scene)
