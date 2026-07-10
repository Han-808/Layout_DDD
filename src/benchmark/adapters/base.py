from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ASSET_SUPPORT_VALUES = {"required", "optional", "unsupported", "unknown"}


@dataclass(frozen=True)
class AdapterCapabilities:
    """Generator-facing capabilities used by harness routing."""

    input_modes: tuple[str, ...] = ("natural_language_direct",)
    asset_support: str = "unknown"

    def __post_init__(self) -> None:
        if self.asset_support not in ASSET_SUPPORT_VALUES:
            raise ValueError(
                f"asset_support must be one of {sorted(ASSET_SUPPORT_VALUES)}, got {self.asset_support!r}"
            )

    def as_dict(self) -> dict:
        return {
            "input_modes": list(self.input_modes),
            "asset_support": self.asset_support,
        }


class GenerationAdapter:
    """Base class for method-specific generation adapters."""

    name: str = "base"
    capabilities = AdapterCapabilities()

    def prepare_input(self, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Convert canonical generation_input into method-specific input."""

        raise NotImplementedError

    def run_generation(self, method_input_path: Path, out_dir: Path, config: dict | None = None) -> Path:
        """Run an internal generator or external method and return raw output."""

        raise NotImplementedError(f"Adapter {self.name!r} does not implement generation.")

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Convert method-specific output into canonical generated_scene.json."""

        raise NotImplementedError
