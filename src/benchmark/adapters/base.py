from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.adapters.output_routing import (
    OUTPUT_CONVERTER,
    OUTPUT_LOADER,
    OutputIngestionKind,
    SceneOutputRoute,
)
from benchmark.io_contracts import (
    EVALUATOR_OUTPUT_TYPES,
    I1_NATURAL_LANGUAGE,
    INPUT_TYPES,
    O1_OBJECT_STATE,
    OUTPUT_TYPES,
    GeneratorIOContract,
    resolve_generation_io_contract,
)
from benchmark.vlm_assistance import VLMAssistanceBudget


ASSET_SUPPORT_VALUES = {"required", "optional", "unsupported", "unknown"}
ROOM_MODEL_VALUES = {"single_room", "multi_room"}
BOUNDARY_MODEL_VALUES = {"axis_aligned_rectangle", "polygon"}
ARCHITECTURE_FEATURE_VALUES = {"walls", "openings", "room_topology"}
GEOMETRY_FIDELITY_VALUES = {"bbox", "mesh_optional", "mesh"}


def _validate_values(
    name: str,
    values: tuple[str, ...],
    allowed: set[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unsupported values: {unknown}")


@dataclass(frozen=True)
class SceneCompatibilityRequirements:
    """Scene semantics required by one evaluator-bound conversion."""

    room_models: tuple[str, ...] = ("single_room",)
    boundary_models: tuple[str, ...] = ("axis_aligned_rectangle",)
    architecture_features: tuple[str, ...] = ()
    geometry_fidelity: tuple[str, ...] = ("bbox",)
    preserves_asset_identity: bool = True

    def __post_init__(self) -> None:
        _validate_values("room_models", self.room_models, ROOM_MODEL_VALUES)
        _validate_values(
            "boundary_models", self.boundary_models, BOUNDARY_MODEL_VALUES
        )
        _validate_values(
            "architecture_features",
            self.architecture_features,
            ARCHITECTURE_FEATURE_VALUES,
            allow_empty=True,
        )
        _validate_values(
            "geometry_fidelity", self.geometry_fidelity, GEOMETRY_FIDELITY_VALUES
        )

    def as_dict(self) -> dict:
        return {
            "room_models": list(self.room_models),
            "boundary_models": list(self.boundary_models),
            "architecture_features": list(self.architecture_features),
            "geometry_fidelity": list(self.geometry_fidelity),
            "preserves_asset_identity": self.preserves_asset_identity,
        }


@dataclass(frozen=True)
class AdapterCapabilities:
    """Generator-facing capabilities used by harness routing."""

    input_modes: tuple[str, ...] = ("natural_language_direct",)
    asset_support: str = "unknown"
    input_types: tuple[str, ...] = (I1_NATURAL_LANGUAGE,)
    native_output_types: tuple[str, ...] = (O1_OBJECT_STATE,)
    evaluator_output_types: tuple[str, ...] = (O1_OBJECT_STATE,)
    vlm_assistance_stages: tuple[str, ...] = ()
    room_models: tuple[str, ...] = ("single_room",)
    boundary_models: tuple[str, ...] = ("axis_aligned_rectangle",)
    architecture_features: tuple[str, ...] = ()
    geometry_fidelity: tuple[str, ...] = ("bbox",)
    preserves_asset_identity: bool = False

    def __post_init__(self) -> None:
        if self.asset_support not in ASSET_SUPPORT_VALUES:
            raise ValueError(
                f"asset_support must be one of {sorted(ASSET_SUPPORT_VALUES)}, got {self.asset_support!r}"
            )
        _validate_values("input_types", self.input_types, INPUT_TYPES)
        _validate_values("native_output_types", self.native_output_types, OUTPUT_TYPES)
        _validate_values("evaluator_output_types", self.evaluator_output_types, EVALUATOR_OUTPUT_TYPES)
        _validate_values("room_models", self.room_models, ROOM_MODEL_VALUES)
        _validate_values(
            "boundary_models", self.boundary_models, BOUNDARY_MODEL_VALUES
        )
        _validate_values(
            "architecture_features",
            self.architecture_features,
            ARCHITECTURE_FEATURE_VALUES,
            allow_empty=True,
        )
        _validate_values(
            "geometry_fidelity", self.geometry_fidelity, GEOMETRY_FIDELITY_VALUES
        )

    def as_dict(self) -> dict:
        return {
            "input_modes": list(self.input_modes),
            "asset_support": self.asset_support,
            "input_types": list(self.input_types),
            "native_output_types": list(self.native_output_types),
            "evaluator_output_types": list(self.evaluator_output_types),
            "vlm_assistance_stages": list(self.vlm_assistance_stages),
            "room_models": list(self.room_models),
            "boundary_models": list(self.boundary_models),
            "architecture_features": list(self.architecture_features),
            "geometry_fidelity": list(self.geometry_fidelity),
            "preserves_asset_identity": self.preserves_asset_identity,
        }

    def accepts(self, contract: GeneratorIOContract) -> bool:
        return (
            contract.input_type in self.input_types
            and contract.native_output_type in self.native_output_types
            and contract.evaluator_output_type in self.evaluator_output_types
        )

    def require_scene_compatibility(
        self, requirements: SceneCompatibilityRequirements
    ) -> None:
        """Fail when evaluator requirements exceed declared adapter semantics."""

        missing = {
            "room_models": sorted(set(requirements.room_models) - set(self.room_models)),
            "boundary_models": sorted(
                set(requirements.boundary_models) - set(self.boundary_models)
            ),
            "architecture_features": sorted(
                set(requirements.architecture_features)
                - set(self.architecture_features)
            ),
            "geometry_fidelity": sorted(
                set(requirements.geometry_fidelity) - set(self.geometry_fidelity)
            ),
        }
        missing = {key: value for key, value in missing.items() if value}
        if requirements.preserves_asset_identity and not self.preserves_asset_identity:
            missing["preserves_asset_identity"] = [True]
        if missing:
            raise ValueError(
                "scene/evaluation requirements exceed adapter capabilities: "
                f"missing={missing}; capabilities={self.as_dict()}"
            )


class OutputMaterializationRequired(RuntimeError):
    """Raised when native generator output still needs an external executor."""


class GenerationAdapter:
    """Base class for method-specific generation adapters."""

    name: str = "base"
    capabilities = AdapterCapabilities()
    output_ingestion_kind: OutputIngestionKind | None = None

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
        output_route = self.scene_output_route()
        canonical_path = output_route.materialize(
            materialization_input,
            generation_input,
            Path(out_dir),
            config,
        )
        self.last_materialization_metadata = {
            **contract.as_dict(),
            "output_ingestion_kind": output_route.kind,
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

    def scene_output_route(self) -> SceneOutputRoute:
        """Select exactly one canonicalization route for this adapter.

        Existing adapters retain ``parse_output`` as their implementation hook.
        New harness adapters may instead override ``load_output`` or
        ``convert_output`` directly.
        """

        if self.output_ingestion_kind == OUTPUT_LOADER:
            return SceneOutputRoute.existing_loader(self.load_output)
        if self.output_ingestion_kind == OUTPUT_CONVERTER:
            return SceneOutputRoute.converter(self.convert_output)
        raise ValueError(
            f"Adapter {self.name!r} must declare output_ingestion_kind as "
            f"{OUTPUT_LOADER!r} or {OUTPUT_CONVERTER!r}"
        )

    def load_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        """Load an already supported scene representation.

        The delegation preserves current adapters while giving future native
        loaders a dedicated override point.
        """

        return self.parse_output(method_output_path, generation_input, out_dir, config)

    def convert_output(
        self,
        method_output_path: Path,
        generation_input: dict,
        out_dir: Path,
        config: dict | None = None,
    ) -> Path:
        """Convert a harness-native representation to the canonical scene."""

        return self.parse_output(method_output_path, generation_input, out_dir, config)

    def parse_output(self, method_output_path: Path, generation_input: dict, out_dir: Path, config: dict | None = None) -> Path:
        """Legacy output hook retained for current concrete adapters."""

        raise NotImplementedError

    def _default_native_output_type(self) -> str:
        if len(self.capabilities.native_output_types) != 1:
            raise ValueError(
                f"Adapter {self.name!r} supports multiple native outputs; config.native_output_type is required"
            )
        return self.capabilities.native_output_types[0]
