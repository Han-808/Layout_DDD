"""Legacy frozen-core loading for the config-only generation CLI.

See ``docs/generation_transport_compatibility.md``.  These helpers load the
generation core only; no evaluator module is imported.
"""

from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    FrozenCoreRuntimeInputs,
    ModelMetadata,
    inspect_model_metadata,
    load_frozen_core,
    load_runtime_inputs,
    load_selected_briefs,
)

__all__ = [
    "FrozenCoreRuntimeInputs",
    "ModelMetadata",
    "inspect_model_metadata",
    "load_frozen_core",
    "load_runtime_inputs",
    "load_selected_briefs",
]
