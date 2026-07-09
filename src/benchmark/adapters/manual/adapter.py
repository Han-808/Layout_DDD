from __future__ import annotations

from pathlib import Path

from benchmark.adapters.base import GenerationAdapter
from benchmark.scene_io.normalize import normalize_scene
from benchmark.scene_io.validate import validate_generation_input
from benchmark.utils.io import read_json, write_json


class ManualAdapter(GenerationAdapter):
    name = "manual"

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        validate_generation_input(generation_input)
        return write_json(Path(out_dir) / "method_input.json", generation_input)

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        cfg = config or {}
        raw_output = cfg.get("raw_output_path") or cfg.get("generated_scene")
        if raw_output:
            return Path(raw_output)
        raise NotImplementedError("Manual adapter does not run generation unless config.raw_output_path is provided.")

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
