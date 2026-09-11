"""Base adapter for externally executed scene-generation harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.adapters.common.assets import load_asset_provider
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


class HarnessConverterAdapter(GenerationAdapter):
    """Common file/directory adapter with an injected pure converter function."""

    name = "external_harness"
    output_schema = "external_harness_output"
    output_ingestion_kind = OUTPUT_CONVERTER
    converter: HarnessConverter | None = None
    capabilities = AdapterCapabilities(
        input_modes=("natural_language_direct", "natural_language_structured", "structured_assets"),
        asset_support="optional",
        input_types=(I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE),
        native_output_types=(O1_OBJECT_STATE,),
        evaluator_output_types=(O1_OBJECT_STATE,),
    )

    def prepare_input(
        self,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        validate_generation_input(generation_input)
        contract = self.resolve_io_contract(generation_input, config=config)
        return write_json(
            Path(out_dir) / "method_input.json",
            {
                "protocol": self.output_schema,
                "harness": self.name,
                "io_contract": contract.as_dict(),
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
        cfg = dict(config or {})
        source_path = Path(method_output_path)
        provider = load_asset_provider(cfg, source_path=source_path)
        scene = self.converter(source_path, generation_input, cfg, provider)
        if not isinstance(scene, dict):
            raise ArtifactValidationError(
                f"{self.name} converter must return a canonical scene JSON object"
            )
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
        }
        return output_path


__all__ = ["HarnessConverter", "HarnessConverterAdapter"]
