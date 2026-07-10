from __future__ import annotations

from pathlib import Path

from benchmark.adapters.base import AdapterCapabilities, GenerationAdapter
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.validate import validate_generation_input
from benchmark.utils.io import read_json, write_json


class PassthroughAdapter(GenerationAdapter):
    name = "passthrough"
    capabilities = AdapterCapabilities(
        input_modes=("natural_language_direct", "natural_language_structured", "structured_assets"),
        asset_support="optional",
    )

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        validate_generation_input(generation_input)
        return write_json(Path(out_dir) / "generation_input.json", generation_input)

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        raise NotImplementedError("Passthrough adapter does not run generation. Provide --generated-scene.")

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        scene = read_json(method_output_path)
        normalized = normalize_scene(
            scene,
            asset_csv=cfg.get("asset_csv") or cfg.get("asset_csv_path"),
            asset_root=cfg.get("asset_root"),
            enrich_assets=bool(cfg.get("enrich_assets", False)),
        )
        return write_json(Path(out_dir) / "generated_scene.json", normalized)
