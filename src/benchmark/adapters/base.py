from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.io_contracts import (
    I1_NATURAL_LANGUAGE,
    INPUT_TYPES,
    O1_OBJECT_STATE,
    OUTPUT_TYPES,
    EVALUATOR_OUTPUT_TYPES,
    GeneratorIOContract,
    resolve_generation_io_contract,
)
from benchmark.vlm_assistance import VLMAssistanceBudget


ASSET_SUPPORT_VALUES = {"required", "optional", "unsupported", "unknown"}


def _validate_values(name: str, values: tuple[str, ...], allowed: set[str]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported values: {unknown}")


@dataclass(frozen=True)
class AdapterCapabilities:
    """Generator-facing capabilities used by harness routing."""

    input_modes: tuple[str, ...] = ("natural_language_direct",)
    asset_support: str = "unknown"
    input_types: tuple[str, ...] = (I1_NATURAL_LANGUAGE,)
    native_output_types: tuple[str, ...] = (O1_OBJECT_STATE,)
    evaluator_output_types: tuple[str, ...] = (O1_OBJECT_STATE,)
    vlm_assistance_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.asset_support not in ASSET_SUPPORT_VALUES:
            raise ValueError(
                f"asset_support must be one of {sorted(ASSET_SUPPORT_VALUES)}, got {self.asset_support!r}"
            )
        _validate_values("input_types", self.input_types, INPUT_TYPES)
        _validate_values("native_output_types", self.native_output_types, OUTPUT_TYPES)
        _validate_values("evaluator_output_types", self.evaluator_output_types, EVALUATOR_OUTPUT_TYPES)

    def as_dict(self) -> dict:
        return {
            "input_modes": list(self.input_modes),
            "asset_support": self.asset_support,
            "input_types": list(self.input_types),
            "native_output_types": list(self.native_output_types),
            "evaluator_output_types": list(self.evaluator_output_types),
            "vlm_assistance_stages": list(self.vlm_assistance_stages),
        }

    def accepts(self, contract: GeneratorIOContract) -> bool:
        return (
            contract.input_type in self.input_types
            and contract.native_output_type in self.native_output_types
            and contract.evaluator_output_type in self.evaluator_output_types
        )


class OutputMaterializationRequired(RuntimeError):
    """Raised when native generator output still needs an external executor."""


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

    def resolve_io_contract(self, generation_input: dict, config: dict | None = None) -> GeneratorIOContract:
        cfg = config or {}
        native_output_type = str(cfg.get("native_output_type") or self._default_native_output_type())
        contract = resolve_generation_io_contract(
            generation_input,
            native_output_type=native_output_type,
        )
        if not self.capabilities.accepts(contract):
            raise ValueError(
                f"Adapter {self.name!r} does not support {contract.as_dict()}; "
                f"capabilities={self.capabilities.as_dict()}"
            )
        return contract

    def resolve_vlm_assistance(self, config: dict | None = None) -> dict:
        """Resolve optional VLM assistance without invoking a model."""

        cfg = config or {}
        budget = VLMAssistanceBudget.from_mapping(cfg.get("vlm_budget"))
        stages = list(self.capabilities.vlm_assistance_stages)
        handler_configured = cfg.get("vlm_assistant") is not None
        if budget.enabled and not stages:
            raise ValueError(f"Adapter {self.name!r} does not support VLM assistance")
        if budget.enabled and not handler_configured:
            raise ValueError(
                f"Adapter {self.name!r} has a positive VLM budget but no config.vlm_assistant handler"
            )
        return {
            "supported": bool(stages),
            "stages": stages,
            "budget": budget.as_dict(),
            "enabled": budget.enabled,
            "handler_configured": handler_configured,
            "status": "configured" if budget.enabled else "disabled_by_budget",
        }

    def materialize_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
        execution_dir: Path | None = None,
    ) -> Path:
        """Execute native output when needed, then export evaluator JSON."""

        contract = self.resolve_io_contract(generation_input, config=config)
        materialization_input = Path(method_output_path)
        runtime_dir = Path(execution_dir) if execution_dir is not None else Path(out_dir)
        if contract.requires_execution:
            materialization_input = self.execute_output(
                materialization_input,
                generation_input,
                runtime_dir,
                contract=contract,
                config=config,
            )
        canonical_path = self.parse_output(
            materialization_input,
            generation_input,
            Path(out_dir),
            config=config,
        )
        self.last_materialization_metadata = {
            **contract.as_dict(),
            "native_output_path": Path(method_output_path).as_posix(),
            "executed_output_path": materialization_input.as_posix() if contract.requires_execution else None,
            "canonical_output_path": Path(canonical_path).as_posix(),
            "execution_dir": runtime_dir.as_posix(),
        }
        return Path(canonical_path)

    def execute_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        *,
        contract: GeneratorIOContract,
        config: dict | None = None,
    ) -> Path:
        """Execute O2 output and return an exported O1/O3 state artifact."""

        raise OutputMaterializationRequired(
            f"Adapter {self.name!r} needs an executor to materialize {contract.native_output_type}"
        )

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Convert method-specific output into canonical generated_scene.json."""

        raise NotImplementedError

    def _default_native_output_type(self) -> str:
        if len(self.capabilities.native_output_types) != 1:
            raise ValueError(
                f"Adapter {self.name!r} supports multiple native outputs; config.native_output_type is required"
            )
        return self.capabilities.native_output_types[0]
