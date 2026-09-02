from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from benchmark.adapters.base import (
    AdapterCapabilities,
    GenerationAdapter,
    OutputMaterializationRequired,
)
from benchmark.adapters.output_routing import OUTPUT_LOADER
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O1_OBJECT_STATE,
    O2_SCENE_PROGRAM,
    O3_SCENE_PACKAGE,
    GeneratorIOContract,
)
from benchmark.nl_scene.generation_input import build_generator_visible_payload
from benchmark.scene_io.normalize import bind_scene_to_generation_request, normalize_scene
from benchmark.scene_io.validate import validate_generation_input, validate_scene_package
from benchmark.utils.io import read_json, write_json


class SceneProgramExecutor(Protocol):
    """Runtime hook for Blender, an engine API, MCP, or another tool host."""

    def execute(
        self,
        *,
        program_path: Path,
        out_dir: Path,
        generator_input: dict,
        evaluator_output_type: str,
        vlm_assistance: dict,
        vlm_assistant: Any | None,
        config: dict,
    ) -> str | Path: ...


class SceneProgramAdapter(GenerationAdapter):
    """O2 adapter boundary; concrete generators supply a runner and executor."""

    name = "scene_program"
    output_schema = "scene_program_v1"
    # The native program is executed first; the exported O1/O3 artifact then
    # enters through the existing canonical loader path.
    output_ingestion_kind = OUTPUT_LOADER
    capabilities = AdapterCapabilities(
        input_modes=("natural_language_direct", "natural_language_structured", "structured_assets"),
        asset_support="optional",
        input_types=(I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE),
        native_output_types=(O2_SCENE_PROGRAM,),
        evaluator_output_types=(O1_OBJECT_STATE, O3_SCENE_PACKAGE),
        vlm_assistance_stages=("program_generation", "program_execution", "state_export"),
    )

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        validate_generation_input(generation_input)
        io_contract = self.resolve_io_contract(generation_input, config=config)
        return write_json(
            Path(out_dir) / "method_input.json",
            {
                "protocol": self.output_schema,
                "io_contract": io_contract.as_dict(),
                "generator_input": build_generator_visible_payload(generation_input),
                "vlm_assistance": self.resolve_vlm_assistance(config),
            },
        )

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        runner = cfg.get("runner")
        if runner is not None:
            result = runner(method_input_path=Path(method_input_path), out_dir=Path(out_dir), config=cfg)
            return Path(result)
        program_path = cfg.get("program_path") or cfg.get("raw_output_path")
        if program_path:
            return Path(program_path)
        raise NotImplementedError(
            "scene_program adapter requires config.runner, config.program_path, or an externally supplied method output"
        )

    def execute_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        *,
        contract: GeneratorIOContract,
        config: dict | None = None,
    ) -> Path:
        cfg = dict(config or {})
        vlm_assistance = self.resolve_vlm_assistance(cfg)
        executor: Any = cfg.get("executor")
        if executor is not None:
            execute = getattr(executor, "execute", executor)
            result = execute(
                program_path=Path(method_output_path),
                out_dir=Path(out_dir),
                generator_input=build_generator_visible_payload(generation_input),
                evaluator_output_type=contract.evaluator_output_type,
                vlm_assistance=vlm_assistance,
                vlm_assistant=cfg.get("vlm_assistant"),
                config=cfg,
            )
            return Path(result)
        exported_path = cfg.get("exported_scene_path") or cfg.get("executed_scene_path")
        if exported_path:
            return Path(exported_path)
        request_path = write_json(
            Path(out_dir) / "execution_request.json",
            {
                "protocol": self.output_schema,
                "program_path": Path(method_output_path).as_posix(),
                "evaluator_output_type": contract.evaluator_output_type,
                "vlm_assistance": vlm_assistance,
                "status": "executor_required",
            },
        )
        raise OutputMaterializationRequired(
            "scene_program output requires config.executor or config.exported_scene_path; "
            f"handoff written to {request_path}"
        )

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        payload = read_json(method_output_path)
        scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else payload
        normalized = normalize_scene(
            bind_scene_to_generation_request(scene, generation_input),
            asset_csv=cfg.get("asset_csv") or cfg.get("asset_csv_path"),
            asset_root=cfg.get("asset_root"),
            enrich_assets=bool(cfg.get("enrich_assets", False)),
        )
        contract = self.resolve_io_contract(generation_input, config=cfg)
        if contract.evaluator_output_type == O3_SCENE_PACKAGE:
            official_mode = bool(cfg.get("official_mode") or cfg.get("require_fixed_catalog"))
            catalog_snapshot_id = str(cfg.get("catalog_snapshot_id") or "").strip()
            if official_mode and not catalog_snapshot_id:
                raise ValueError("official O3 scene-program exports require config.catalog_snapshot_id")
            validate_scene_package(
                normalized,
                allowed_asset_ids=cfg.get("allowed_asset_ids"),
                require_fixed_catalog=official_mode,
            )
        metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        normalized["metadata"] = {
            **metadata,
            "native_output_type": O2_SCENE_PROGRAM,
            "output_adapter": self.name,
            "asset_catalog_snapshot_id": (
                str(cfg.get("catalog_snapshot_id"))
                if contract.evaluator_output_type == O3_SCENE_PACKAGE and cfg.get("catalog_snapshot_id")
                else None
            ),
            "fixed_catalog_enforced": bool(
                contract.evaluator_output_type == O3_SCENE_PACKAGE
                and (cfg.get("official_mode") or cfg.get("require_fixed_catalog"))
            ),
        }
        return write_json(Path(out_dir) / "generated_scene.json", normalized)
