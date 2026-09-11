from __future__ import annotations

from dataclasses import dataclass


I1_NATURAL_LANGUAGE = "i1_natural_language"
I2_NATURAL_LANGUAGE_STRUCTURE = "i2_natural_language_structure"

O1_OBJECT_STATE = "o1_object_state"
O2_SCENE_PROGRAM = "o2_scene_program"
O3_SCENE_PACKAGE = "o3_scene_package"

INPUT_TYPES = {I1_NATURAL_LANGUAGE, I2_NATURAL_LANGUAGE_STRUCTURE}
OUTPUT_TYPES = {O1_OBJECT_STATE, O2_SCENE_PROGRAM, O3_SCENE_PACKAGE}
EVALUATOR_OUTPUT_TYPES = {O1_OBJECT_STATE, O3_SCENE_PACKAGE}

__all__ = [
    "EVALUATOR_OUTPUT_TYPES",
    "GeneratorIOContract",
    "I1_NATURAL_LANGUAGE",
    "I2_NATURAL_LANGUAGE_STRUCTURE",
    "INPUT_TYPES",
    "O1_OBJECT_STATE",
    "O2_SCENE_PROGRAM",
    "O3_SCENE_PACKAGE",
    "OUTPUT_TYPES",
    "input_type_for_mode",
    "resolve_generation_io_contract",
]

_INPUT_MODE_TYPES = {
    "natural_language_direct": I1_NATURAL_LANGUAGE,
    "natural_language_structured": I2_NATURAL_LANGUAGE_STRUCTURE,
    # Asset assistance is orthogonal to the semantic input type. A fixed
    # catalog or selected assets do not create a third benchmark input type.
    "structured_assets": I2_NATURAL_LANGUAGE_STRUCTURE,
}


@dataclass(frozen=True)
class GeneratorIOContract:
    """Resolved generator protocol and evaluator boundary for one run."""

    input_type: str
    native_output_type: str
    evaluator_output_type: str
    feedback_assistance: bool = False

    def __post_init__(self) -> None:
        if self.input_type not in INPUT_TYPES:
            raise ValueError(f"Unknown input_type {self.input_type!r}")
        if self.native_output_type not in OUTPUT_TYPES:
            raise ValueError(f"Unknown native_output_type {self.native_output_type!r}")
        if self.evaluator_output_type not in EVALUATOR_OUTPUT_TYPES:
            raise ValueError(
                "evaluator_output_type must be o1_object_state or o3_scene_package; "
                "o2_scene_program must first be executed and exported"
            )

    @property
    def native_combination(self) -> str:
        return f"{self.input_type}:{self.native_output_type}"

    @property
    def requires_execution(self) -> bool:
        return self.native_output_type == O2_SCENE_PROGRAM

    def as_dict(self) -> dict:
        return {
            "input_type": self.input_type,
            "native_output_type": self.native_output_type,
            "evaluator_output_type": self.evaluator_output_type,
            "native_combination": self.native_combination,
            "requires_execution": self.requires_execution,
            "feedback_assistance": self.feedback_assistance,
        }


def input_type_for_mode(input_mode: str) -> str:
    try:
        return _INPUT_MODE_TYPES[str(input_mode)]
    except KeyError as exc:
        raise ValueError(f"Input mode {input_mode!r} has no I1/I2 contract mapping") from exc


def resolve_generation_io_contract(
    generation_input: dict,
    *,
    native_output_type: str,
) -> GeneratorIOContract:
    contract = generation_input.get("generation_contract")
    if not isinstance(contract, dict):
        raise ValueError("generation_input.generation_contract must be a JSON object")
    input_mode = str(contract.get("input_mode") or "natural_language_direct")
    inferred_input_type = input_type_for_mode(input_mode)
    declared_input_type = str(contract.get("input_type") or inferred_input_type)
    if declared_input_type != inferred_input_type:
        raise ValueError(
            f"generation_contract.input_type {declared_input_type!r} conflicts with input_mode {input_mode!r}"
        )
    evaluator_output_type = str(contract.get("evaluator_output_type") or O1_OBJECT_STATE)
    reflection = generation_input.get("self_reflection")
    return GeneratorIOContract(
        input_type=declared_input_type,
        native_output_type=native_output_type,
        evaluator_output_type=evaluator_output_type,
        feedback_assistance=isinstance(reflection, dict) and reflection.get("enabled") is True,
    )
