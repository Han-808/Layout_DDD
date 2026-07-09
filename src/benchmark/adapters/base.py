from __future__ import annotations

from pathlib import Path


class GenerationAdapter:
    """Base class for method-specific generation adapters."""

    name: str = "base"

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Convert canonical generation_input into method-specific input."""

        raise NotImplementedError

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        """Run an internal generator or external method and return raw output."""

        raise NotImplementedError(f"Adapter {self.name!r} does not implement generation.")

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Convert method-specific output into canonical generated_scene.json."""

        raise NotImplementedError
