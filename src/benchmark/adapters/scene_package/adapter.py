from __future__ import annotations

from pathlib import Path

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    I2_NATURAL_LANGUAGE_STRUCTURE,
    O3_SCENE_PACKAGE,
)
from benchmark.nl_scene.generation_input import build_generator_visible_payload
from benchmark.scene_io.normalize import bind_scene_to_generation_request, normalize_scene
from benchmark.scene_io.validate import validate_generation_input, validate_scene_package
from benchmark.utils.io import read_json, write_json


class ScenePackageAdapter(GenerationAdapter):
    """O3 boundary for object-indexed scenes grounded in a fixed asset catalog."""

    name = "scene_package"
    output_schema = "scene_package_v1"
    capabilities = AdapterCapabilities(
        input_modes=("natural_language_direct", "natural_language_structured", "structured_assets"),
        asset_support="optional",
        input_types=(I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE),
        native_output_types=(O3_SCENE_PACKAGE,),
        evaluator_output_types=(O3_SCENE_PACKAGE,),
        vlm_assistance_stages=("scene_generation", "asset_selection", "package_validation"),
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
        raw_output_path = cfg.get("raw_output_path") or cfg.get("scene_package_path")
        if raw_output_path:
            return Path(raw_output_path)
        raise NotImplementedError(
            "scene_package adapter requires config.runner, config.scene_package_path, or an externally supplied method output"
        )

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        official_mode = bool(cfg.get("official_mode") or cfg.get("require_fixed_catalog"))
        catalog_snapshot_id = str(cfg.get("catalog_snapshot_id") or "").strip()
        if official_mode and not catalog_snapshot_id:
            raise ValueError("official O3 scene packages require config.catalog_snapshot_id")
        payload = read_json(method_output_path)
        scene = payload.get("scene") if isinstance(payload.get("scene"), dict) else payload
        normalized = normalize_scene(
            bind_scene_to_generation_request(scene, generation_input),
            asset_csv=cfg.get("asset_csv") or cfg.get("asset_csv_path"),
            asset_root=cfg.get("asset_root"),
            enrich_assets=bool(cfg.get("enrich_assets", False)),
        )
        validate_scene_package(
            normalized,
            allowed_asset_ids=cfg.get("allowed_asset_ids"),
            require_fixed_catalog=official_mode,
        )
        metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        normalized["metadata"] = {
            **metadata,
            "native_output_type": O3_SCENE_PACKAGE,
            "output_adapter": self.name,
            "asset_catalog_snapshot_id": catalog_snapshot_id or None,
            "fixed_catalog_enforced": official_mode,
        }
        return write_json(Path(out_dir) / "generated_scene.json", normalized)
